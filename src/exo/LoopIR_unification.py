import functools
import itertools
import re
from collections import ChainMap

import pysmt
from asdl_adt import ADT
from pysmt import shortcuts as SMT

from .LoopIR import (
    LoopIR,
    T,
    LoopIR_Do,
    FreeVars,
    Alpha_Rename,
    comparision_ops,
    LoopIR_Dependencies,
)
from .LoopIR_scheduling import SchedulingError
from .prelude import *
from .new_eff import Check_Aliasing
import exo.internal_cursors as ic


def _get_smt_solver():
    factory = pysmt.factory.Factory(pysmt.shortcuts.get_env())
    slvs = factory.all_solvers()
    if len(slvs) == 0:
        raise OSError("Could not find any SMT solvers")
    return pysmt.shortcuts.Solver(name=next(iter(slvs)))


def sanitize_str(s):
    return re.sub(r"\W", "_", s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class UnificationError(Exception):
    def __init__(self, msg):
        self._err_msg = str(msg)

    def __str__(self):
        return self._err_msg


def Get_Live_Variables(stmt_cursor):
    c = stmt_cursor
    covering_stmts = []
    while True:
        try:
            c = c.prev()
        except ic.InvalidCursorError:
            c = c.parent()
            if c == c.root():
                break
        covering_stmts.append(c._node)

    live_vars = ChainMap()

    proc = c.get_root()
    for arg in proc.args:
        live_vars[arg.name] = arg.type

    for s in reversed(covering_stmts):
        if isinstance(s, LoopIR.WindowStmt):
            live_vars[s.name] = s.rhs.type
        elif isinstance(s, LoopIR.Alloc):
            live_vars[s.name] = s.type
        elif isinstance(s, LoopIR.If):
            live_vars = live_vars.new_child()
        elif isinstance(s, LoopIR.For):
            live_vars = live_vars.new_child()
            live_vars[s.iter] = T.index

    return live_vars


def DoReplace(subproc, block_cursor):
    n_stmts = len(subproc.body)
    if len(block_cursor) < n_stmts:
        raise SchedulingError("Not enough statements to match")

    # prevent name clashes between the statement block and sub-proc
    temp_subproc = Alpha_Rename(subproc).result()
    stmts = [c._node for c in block_cursor[:n_stmts]]
    live_vars = Get_Live_Variables(block_cursor[0])
    new_args = Unification(temp_subproc, stmts, live_vars).result()

    # but don't use a different LoopIR.proc for the callsite itself
    new_call = LoopIR.Call(subproc, new_args, stmts[0].srcinfo)

    ir, fwd = block_cursor._replace([new_call])
    Check_Aliasing(ir)
    return ir, fwd


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unification "System of Equations" grammar

# the problem designates disjoint subsets of the variables used
# as "holes" (variables to solve for) and "knowns" (variables to express
# a solution as an affine combination of).  Any variable not in either of
# those lists is unknown but not permissible in a solution expression.
UEq = ADT(
    """
module UEq {
    problem = ( sym*    holes,  -- symbols the solution is requested for
                sym*    knowns, -- symbols allowed in solution expressions
                pred*   preds   -- conj of equations
              )

    pred    = Conj( pred* preds )
            | Disj( pred* preds )
            | Cases( sym case_var, pred* cases )
            | Eq( expr lhs, expr rhs )

    -- affine expressions
    expr  =  Const(int val)
          |  Var( sym name )
          |  Add( expr lhs, expr rhs )
          |  Scale( int coeff, expr e )

} """,
    {
        "sym": Sym,
    },
)


# -------------------------------------- #
# Conversion to Strings for Debug


def _str_uexpr(e, prec=0):
    etyp = type(e)
    if etyp is UEq.Const:
        return str(e.val)
    elif etyp is UEq.Var:
        return str(e.name)
    elif etyp is UEq.Add:
        s = f"{_str_uexpr(e.lhs, 0)} + {_str_uexpr(e.rhs, 1)}"
        if prec > 0:
            s = f"({s})"
        return s
    elif etyp is UEq.Scale:
        return f"{e.coeff}*{_str_uexpr(e.e, 10)}"
    else:
        assert False, "bad case"


@extclass(UEq.Const)
@extclass(UEq.Var)
@extclass(UEq.Add)
@extclass(UEq.Scale)
def __str__(self):
    return _str_uexpr(self)


del __str__


def _str_upred(p, prec=0):
    ptyp = type(p)
    if ptyp is UEq.Eq:
        return f"{p.lhs} == {p.rhs}"
    elif ptyp is UEq.Conj or ptyp is UEq.Disj:
        op = " and " if UEq.Conj else " or "
        s = op.join([_str_upred(pp, 1) for pp in p.preds])
        if prec > 0:
            s = f"({s})"
        return s
    elif ptyp is UEq.Cases:
        return f"cases({p.case_var}) " + " | ".join([f"({pred})" for pred in p.cases])
    else:
        assert False, "bad case"


@extclass(UEq.Conj)
@extclass(UEq.Cases)
@extclass(UEq.Disj)
@extclass(UEq.Eq)
def __str__(self):
    return _str_upred(self)


del __str__


@extclass(UEq.problem)
def __str__(prob):
    lines = [
        "Holes:   " + ", ".join([str(x) for x in prob.holes]),
        "Knowns:  " + ", ".join([str(x) for x in prob.knowns]),
    ]
    lines += [str(p) for p in prob.preds]
    return "\n".join(lines)


del __str__


# -------------------------------------- #
# How to solve this system of equations


@extclass(UEq.expr)
def normalize(orig_e):
    def to_nform(e):
        if isinstance(e, UEq.Const):
            return [], e.val
        elif isinstance(e, UEq.Var):
            return [(1, e.name)], 0
        elif isinstance(e, UEq.Add):
            xcs, xoff = to_nform(e.lhs)
            ycs, yoff = to_nform(e.rhs)

            def merge(xcs=xcs, ycs=ycs):
                xi, yi = 0, 0
                res = []
                while xi < len(xcs) and yi < len(ycs):
                    (xc, xv), (yc, yv) = xcs[xi], ycs[yi]
                    if xv < yv:
                        res.append((xc, xv))
                        xi += 1
                    elif yv < xv:
                        res.append((yc, yv))
                        yi += 1
                    else:
                        if xc + yc != 0:
                            res.append((xc + yc, xv))
                        xi += 1
                        yi += 1
                if xi == len(xcs):
                    res += ycs[yi:]
                elif yi == len(ycs):
                    res += xcs[xi:]
                else:
                    assert False, "bad iteration"
                return res

            return merge(), xoff + yoff
        elif isinstance(e, UEq.Scale):
            if e.coeff == 0:
                return [], 0
            cs, off = to_nform(e.e)
            return [(e.coeff * c, v) for (c, v) in cs], e.coeff * off
        else:
            assert False, "bad case"

    def from_nform(cs, off):
        e = None
        for (c, v) in cs:
            assert c != 0
            t = UEq.Var(v)
            t = t if c == 1 else UEq.Scale(c, t)
            e = t if e is None else UEq.Add(e, t)
        if e is None:
            return UEq.Const(off)
        elif off == 0:
            return e
        else:
            return UEq.Add(e, UEq.Const(off))

    cs, off = to_nform(orig_e)
    return from_nform(cs, off)


@extclass(UEq.expr)
def sub(x, y):
    return UEq.Add(x, UEq.Scale(-1, y))


@extclass(UEq.problem)
def solve(prob):
    solver = _get_smt_solver()

    known_list = prob.knowns
    known_idx = {k: i for i, k in enumerate(known_list)}
    hole_idx = {k: i for i, k in enumerate(prob.holes)}
    Nk = len(known_list)

    var_set = dict()
    case_set = dict()

    def get_var(x):
        if x in var_set:
            return var_set[x]
        else:
            vec = [SMT.Symbol(f"{repr(x)}_{repr(k)}", SMT.INT) for k in known_list] + [
                SMT.Symbol(f"{repr(x)}_const", SMT.INT)
            ]
            var_set[x] = vec
            return vec

    def get_case(x):
        if x not in case_set:
            case_set[x] = SMT.Symbol(f"{repr(x)}", SMT.INT)
        return case_set[x]

    # initialize all hole variables, ensuring they are defined
    for x in prob.holes:
        get_var(x)

    def lower_e(e):
        if isinstance(e, UEq.Const):
            return ([SMT.Int(0)] * Nk) + [SMT.Int(e.val)]
        elif isinstance(e, UEq.Var):
            if e.name in known_idx:
                one_hot = [SMT.Int(0)] * (Nk + 1)
                one_hot[known_idx[e.name]] = SMT.Int(1)
                return one_hot
            elif e.name in hole_idx:
                return get_var(e.name)
            else:
                raise UnificationError(
                    f"Unable to cancel variable '{e.name}' from both sides "
                    f"of a unification equation"
                )
        elif isinstance(e, UEq.Add):
            lhs = lower_e(e.lhs)
            rhs = lower_e(e.rhs)
            return [SMT.Plus(x, y) for x, y in zip(lhs, rhs)]
        elif isinstance(e, UEq.Scale):
            arg = lower_e(e.e)
            return [SMT.Times(SMT.Int(e.coeff), a) for a in arg]
        else:
            assert False, "bad case"

    def lower_p(p):
        if isinstance(p, UEq.Eq):
            diff = p.lhs.sub(p.rhs).normalize()
            try:
                es = lower_e(diff)
                return SMT.And(*[SMT.Equals(x, SMT.Int(0)) for x in es])
            except UnificationError:
                return SMT.FALSE()
        elif isinstance(p, UEq.Conj):
            return SMT.And(*[lower_p(pp) for pp in p.preds])
        elif isinstance(p, UEq.Disj):
            return SMT.Or(*[lower_p(pp) for pp in p.preds])
        elif isinstance(p, UEq.Cases):
            case_var = get_case(p.case_var)

            def per_case(i, c):
                pp = lower_p(c)
                is_case = SMT.Equals(case_var, SMT.Int(i))
                return SMT.And(is_case, pp)

            disj = SMT.Or(*[per_case(i, c) for i, c in enumerate(p.cases)])
            case_lo = SMT.GE(case_var, SMT.Int(0))
            case_hi = SMT.LT(case_var, SMT.Int(len(p.cases)))
            return SMT.And(disj, case_lo, case_hi)
        else:
            assert False, "bad case"

    prob_pred = SMT.And(*[lower_p(p) for p in prob.preds])
    if not solver.is_sat(prob_pred):
        return None
    else:
        solutions = dict()
        for hole_var in prob.holes:
            x_syms = get_var(hole_var)
            x_val_dict = solver.get_py_values(x_syms)
            x_vals = [x_val_dict[x_sym] for x_sym in x_syms]
            expr = None
            for xx, v in zip(known_list, x_vals):
                v = int(v)
                if v == 0:
                    continue
                elif v == 1:
                    term = UEq.Var(xx)
                else:
                    term = UEq.Scale(v, UEq.Var(xx))

                expr = term if expr is None else UEq.Add(expr, term)

            # constant offset
            off = UEq.Const(int(x_vals[-1]))
            expr = off if expr is None else UEq.Add(expr, off)

            solutions[hole_var] = expr

        # report on case decisions
        for x in case_set:
            val = solver.get_py_value(case_set[x])
            solutions[x] = int(val)

        return solutions


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unification compiler pass


class _Find_Mod_Div_Symbols(LoopIR_Do):
    def __init__(self, stmts, FV):
        self.node_to_sym = dict()  # many to one
        self.tuple_to_sym = dict()  # de-duplicating lookup
        self.sym_to_node = dict()  # pick a node for each symbol
        self.FV = FV

        self.unq_count = 0

        self.do_stmts(stmts)

    def result(self):
        return self.node_to_sym, self.sym_to_node

    @functools.lru_cache(maxsize=None)
    def tuple_memo(self, *args):
        return tuple(args)

    def do_e(self, e):
        if (
            isinstance(e, LoopIR.BinOp)
            and (e.op == "%" or e.op == "/")
            and e.type.is_indexable()
        ):
            # found a mod-div site
            tuple_node = self.tupleify(e)
            if tuple_node is None:
                raise UnificationError(
                    f"{e.srcinfo}: cannot handle this '{e.op}' operation"
                )

            # either we have already seen this expression
            if tuple_node in self.tuple_to_sym:
                sym = self.tuple_to_sym[tuple_node]
            # or we are encountering it for the first time
            else:
                opname = "mod" if e.op == "%" else "div"
                node_name = sanitize_str(str(e))
                sym = Sym(f"{opname}_{self.unq_count}_{node_name}")
                self.unq_count += 1
                self.tuple_to_sym[tuple_node] = sym
                self.sym_to_node[sym] = e

            # regardless, record which symbol we've assigned to
            # this specific AST node sub-tree
            self.node_to_sym[id(e)] = sym

        else:
            super().do_e(e)

    def tupleify(self, e):
        if isinstance(e, LoopIR.Read):
            assert len(e.idx) == 0
            if e.name not in self.FV:
                raise UnificationError(
                    f"{e.srcinfo}: "
                    f"Found bound variable '{e.name}' "
                    f"inside of mod or div operation"
                )
            else:
                return e.name
        elif isinstance(e, LoopIR.Const):
            return e.val
        elif isinstance(e, LoopIR.USub):
            return self.tuple_memo("-", self.tupleify(e.arg))
        elif isinstance(e, LoopIR.BinOp):
            return self.tuple_memo(e.op, self.tupleify(e.lhs), self.tupleify(e.rhs))
        else:
            assert False, "Bad case tupleify"


class BufVar:
    def __init__(self, name, typ, use_win=True):
        self.name = name
        self.typ = typ
        self.n_dim = len(typ.shape())

        self.solution_buf = None

        self.use_win = use_win
        # `win_dim` is the number of dimensions that are
        # point-accesses in a window expression
        # `n_dim` is the number of dimensions that are
        # interval/slice-accesses in a window expression
        # The sum of the two is the size of whatever buffer is unified against
        self.win_dim = None
        self.cases = []
        self.case_var = None

    def set_buf_solution(self, solution_buf):
        self.solution_buf = solution_buf

    def set_window_dim(self, win_dim):
        assert win_dim is not None
        do_setup = self.win_dim is None
        self.win_dim = win_dim

        if do_setup:
            self.cases = []
            self.case_var = Sym(f"{self.name}_which_case")
            full_dim = self.n_dim + win_dim

            for case_id, pt_idxs in enumerate(
                itertools.combinations(range(0, full_dim), win_dim)
            ):
                case_name = (
                    f"{self.name}_"
                    + "_".join([str(i) for i in pt_idxs])
                    + f"_cs{case_id}"
                )

                def make_pt(i):
                    return Sym(f"{case_name}_pt{i}")

                def make_interval(i):
                    return (Sym(f"{case_name}_lo{i}"), Sym(f"{case_name}_hi{i}"))

                idx_vars = [
                    (make_pt(i) if i in pt_idxs else make_interval(i))
                    for i in range(0, full_dim)
                ]

                self.cases.append(idx_vars)

    def get_sz_eq(self, UObj):
        assert self.win_dim is not None
        results = []
        for c in self.cases:
            intervals = [i for i in c if not isinstance(i, Sym)]
            assert len(intervals) == self.n_dim
            for (lo, hi), sz in zip(intervals, self.typ.shape()):
                diff = UEq.Add(UEq.Var(hi), UEq.Scale(-1, UEq.Var(lo)))
                results += [UEq.Eq(diff, UObj.to_ueq(sz))]
        return results

    def all_syms(self):
        if not self.case_var:
            return []
        else:
            xs = [self.case_var]
            for c in self.cases:
                for i in c:
                    if isinstance(i, Sym):
                        xs.append(i)
                    else:
                        xs.append(i[0])
                        xs.append(i[1])
            return xs

    def get_solution(self, UObj, ueq_solutions, srcinfo):
        buf = self.solution_buf
        buf_typ = UObj.FV[buf]
        if not self.case_var:
            return LoopIR.Read(buf, [], buf_typ, srcinfo)
        else:
            which_case = ueq_solutions[self.case_var]
            case = self.cases[which_case]

            def subtract(hi, lo):
                if isinstance(lo, LoopIR.Const) and lo.val == 0:
                    return hi
                else:
                    return LoopIR.BinOp("-", hi, lo, T.index, hi.srcinfo)

            idx = []
            win_shape = []
            for w in case:
                if isinstance(w, Sym):
                    pt = UObj.from_ueq(ueq_solutions[w], srcinfo)
                    idx.append(LoopIR.Point(pt, srcinfo))
                else:
                    lo = UObj.from_ueq(ueq_solutions[w[0]], srcinfo)
                    hi = UObj.from_ueq(ueq_solutions[w[1]], srcinfo)
                    idx.append(LoopIR.Interval(lo, hi, srcinfo))
                    win_shape.append(subtract(hi, lo))

            as_tensor = T.Tensor(win_shape, True, buf_typ.type)
            w_typ = T.Window(buf_typ, as_tensor, buf, idx)
            return LoopIR.WindowExpr(buf, idx, w_typ, srcinfo)


class Unification:
    def __init__(self, subproc, stmt_block, live_vars):
        self.equations = []
        self.stmt_block = stmt_block

        # variables for the UEq system
        # self.holes      = []
        # self.knowns     = []

        # set up different kinds of variables before we
        # begin doing the structural matching...
        # self.arg_syms       = { fa.name : True for fa in subproc.args }
        self.index_holes = [fa.name for fa in subproc.args if fa.type.is_indexable()]
        self.buf_holes = {
            fa.name: BufVar(fa.name, fa.type, fa.type.is_win())
            for fa in subproc.args
            if fa.type.is_numeric()
        }
        self.bool_holes = {fa.name: False for fa in subproc.args if fa.type == T.bool}

        self.stride_holes = {
            fa.name: False for fa in subproc.args if fa.type == T.stride
        }

        # keep track of all buffer names we might need to unify,
        # not just the unknown arguments, but also temporary allocated bufs
        # these variables should ONLY occur on the sub-procedure side
        # of the unification; no BufVars for the original code.
        self.buf_unknowns = self.buf_holes.copy()

        # get the free variables, and lookup their types
        # as well as expanding the free variable set to
        # account for dependent typing
        FV_set = FreeVars(stmt_block).result()
        self.FV = dict()

        def add_fv(x):
            assert x in live_vars, f"expected FV {x} to be live"
            typ = live_vars[x]
            self.FV[x] = typ

            def expand_e(e):
                if isinstance(e, LoopIR.Read):
                    add_fv(e.name)
                elif isinstance(e, LoopIR.USub):
                    expand_e(e.arg)
                elif isinstance(e, LoopIR.BinOp):
                    expand_e(e.lhs)
                    expand_e(e.rhs)

            if isinstance(typ, T.Tensor):
                for e in typ.hi:
                    expand_e(e)
            elif isinstance(typ, T.Window):
                for w in typ.idx:
                    if isinstance(w, LoopIR.Interval):
                        expand_e(w.lo)
                        expand_e(w.hi)
                    else:
                        expand_e(w.pt)
                add_fv(typ.src_buf)

        for x in FV_set:
            add_fv(x)

        # block-side buffer types
        self.bbuf_types = {x: self.FV[x] for x in self.FV if self.FV[x].is_numeric()}

        # self.node_syms  = None
        # self.sym_nodes  = None
        self.node_syms, self.sym_nodes = _Find_Mod_Div_Symbols(
            stmt_block, self.FV
        ).result()

        # substitutions to do of intermediate indexing variables
        self.idx_subst = dict()

        # TODO: Asserts
        # We don't have inequality in EQs IR

        # TODO: Size
        # Inequality...??

        # build up the full set of equations...
        self.unify_stmts(subproc.body, stmt_block)

        # setup the problem
        for nm in self.buf_holes:
            if self.buf_holes[nm].solution_buf is None:
                raise UnificationError(
                    f"Cannot perform unification due to an un-unused argument: {nm}"
                )

        holes = self.index_holes + [
            x for nm in self.buf_holes for x in self.buf_holes[nm].all_syms()
        ]
        knowns = [nm for nm in self.sym_nodes] + [
            nm for nm in self.FV if self.FV[nm].is_indexable()
        ]
        ueq_prob = UEq.problem(holes, knowns, self.equations)

        # solve the problem
        solutions = ueq_prob.solve()
        if solutions is None:
            raise UnificationError(f"Unification of various index expressions failed")

        # construct the solution arguments
        def get_arg(fa):
            if fa.type.is_indexable():
                return self.from_ueq(solutions[fa.name], stmt_block[0].srcinfo)
            elif fa.type == T.bool:
                if fa.name in self.bool_holes:
                    if self.bool_holes[fa.name] is False:
                        return LoopIR.Const(False, T.bool, stmt_block[0].srcinfo)
                    else:
                        return self.bool_holes[fa.name]
                else:
                    # subprocedure argument must be consistent
                    assert False, "bad case"
            elif fa.type == T.stride:
                if fa.name in self.stride_holes:
                    return self.stride_holes[fa.name]
                else:
                    raise UnificationError(f"stride argument {fa.name}" + " unused")
            else:
                assert fa.type.is_numeric()
                bufvar = self.buf_holes[fa.name]
                return bufvar.get_solution(self, solutions, stmt_block[0].srcinfo)

        self.new_args = [get_arg(fa) for fa in subproc.args]

    def err(self):
        raise TypeError("subproc and pattern don't match")

    def result(self):
        return self.new_args

    # ----------

    def to_ueq(self, e, in_subproc=False):
        insp = in_subproc
        if isinstance(e, LoopIR.Read):
            assert len(e.idx) == 0
            name = self.idx_subst[e.name] if e.name in self.idx_subst else e.name
            return UEq.Var(name)
        elif isinstance(e, LoopIR.Const):
            return UEq.Const(e.val)
        elif isinstance(e, LoopIR.USub):
            return UEq.Scale(-1, self.to_ueq(e.arg, insp))
        elif isinstance(e, LoopIR.BinOp):
            if e.op == "+":
                return UEq.Add(self.to_ueq(e.lhs, insp), self.to_ueq(e.rhs, insp))
            elif e.op == "-":
                rhs = UEq.Scale(-1, self.to_ueq(e.rhs, insp))
                return UEq.Add(self.to_ueq(e.lhs, insp), rhs)
            elif e.op == "*":
                if isinstance(e.lhs, LoopIR.Const):
                    return UEq.Scale(e.lhs.val, self.to_ueq(e.rhs, insp))
                elif isinstance(e.rhs, LoopIR.Const):
                    return UEq.Scale(e.rhs.val, self.to_ueq(e.lhs, insp))
                else:
                    assert False, "unexpected multiplication; improve the code here"
            elif e.op == "/" or e.op == "%":
                if in_subproc:
                    raise UnificationError(
                        f"unification with sub-procedures making use of "
                        f"'%' or '/' operations is not currently supported"
                    )
                else:
                    name = self.node_syms[id(e)]
                    return UEq.Var(name)
            else:
                assert False, f"bad op case: {e.op}"
        else:
            assert False, "unexpected affine expression case"

    def from_ueq(self, e, srcinfo=null_srcinfo()):
        if isinstance(e, UEq.Var):
            if e.name in self.sym_nodes:
                return self.sym_nodes[e.name]
            else:
                typ = self.FV[e.name]
                return LoopIR.Read(e.name, [], typ, srcinfo)

        elif isinstance(e, UEq.Const):
            return LoopIR.Const(e.val, T.int, srcinfo)
        elif isinstance(e, UEq.Add):
            lhs = self.from_ueq(e.lhs, srcinfo)
            rhs = self.from_ueq(e.rhs, srcinfo)
            typ = (
                lhs.type
                if rhs.type == T.int
                else rhs.type
                if lhs.type == T.int
                else lhs.type
                if rhs.type == T.size
                else rhs.type
            )
            return LoopIR.BinOp("+", lhs, rhs, typ, srcinfo)
        elif isinstance(e, UEq.Scale):
            lhs = LoopIR.Const(e.coeff, T.int, srcinfo)
            rhs = self.from_ueq(e.e, srcinfo)
            return LoopIR.BinOp("*", lhs, rhs, rhs.type, srcinfo)
        else:
            assert False, "bad case"

    # ----------

    def all_bound_e(self, be):
        if isinstance(be, LoopIR.Read):
            if be.name not in self.FV:
                return False
            return all(self.all_bound_e(i) for i in be.idx)
        elif isinstance(be, LoopIR.Const):
            return True
        elif isinstance(be, LoopIR.USub):
            return self.all_bound_e(be.arg)
        elif isinstance(be, LoopIR.BinOp):
            return self.all_bound_e(be.lhs) and self.all_bound_e(be.rhs)
        elif isinstance(be, LoopIR.BuiltIn):
            return all(self.all_bound_e(a) for a in be.args)
        else:
            assert False, "unsupported case"

    def is_exact_e(self, e0, e1):
        if type(e0) is not type(e1):
            return False
        elif isinstance(e0, LoopIR.Read):
            return e0.name == e1.name and all(
                self.is_exact_e(i0, i1) for i0, i1 in zip(e0.idx, e1.idx)
            )
        elif isinstance(e0, LoopIR.Const):
            return e0.val == e1.val
        elif isinstance(e0, LoopIR.USub):
            return self.is_exact_e(e0.arg, e1.arg)
        elif isinstance(e0, LoopIR.BinOp):
            return (
                e0.op == e1.op
                and self.is_exact_e(e0.lhs, e1.lhs)
                and self.is_exact_e(e0.rhs, e1.rhs)
            )
        elif isinstance(e0, LoopIR.BuiltIn):
            return e0.f == e1.f and all(
                self.is_exact_e(a0, a1) for a0, a1 in zip(e0.args, e1.args)
            )
        else:
            assert False, "unsupported case"

    # ----------

    def unify_affine_e(self, pa, ba):
        self.equations.append(UEq.Eq(self.to_ueq(pa, in_subproc=True), self.to_ueq(ba)))

    def unify_bool_hole(self, pe, be):
        assert pe.type == be.type == T.bool
        assert isinstance(pe, LoopIR.Read) and pe.name in self.bool_holes

        if not self.all_bound_e(be):
            raise UnificationError(
                f"Cannot unify expression {be} (@{be.srcinfo}) with the "
                f"boolean argument {pe.name} because it contains "
                f"variables that are not free in the code being replaced"
            )

        # if we haven't yet unified this name with an expression
        lookup = self.bool_holes[pe.name]
        if lookup is False:
            self.bool_holes[pe.name] = be
        elif not self.is_exact_e(be, lookup):
            raise UnificationError(
                f"Cannot unify the boolean argument {pe.name} with two "
                f"seemingly inequivalent expressions "
                f"{be} (@{be.srcinfo}) and {lookup} (@{lookup.srcinfo})"
            )

    def unify_stride_hole(self, pe, be):
        assert pe.type == be.type == T.stride
        assert isinstance(pe, LoopIR.Read) and pe.name in self.stride_holes

        # TODO: Add checks here??
        lookup = self.stride_holes[pe.name]
        if lookup is False:
            self.stride_holes[pe.name] = be

    def unify_stmts(self, proc_s, block_s):
        if len(proc_s) != len(block_s):
            ploc, bloc = "", ""
            if len(proc_s) > 0:
                ploc = f" (@{proc_s[0].srcinfo})"
            if len(block_s) > 0:
                bloc = f" (@{block_s[0].srcinfo})"
            raise UnificationError(
                f"cannot unify {len(proc_s)} statement(s){ploc} with "
                f"{len(block_s)} statement(s){bloc}"
            )
        elif len(proc_s) == 0:
            return

        ps, proc_s = proc_s[0], proc_s[1:]
        bs, block_s = block_s[0], block_s[1:]

        if type(ps) is not type(bs):
            raise UnificationError(
                f"cannot unify a {type(ps)} statement (@{ps.srcinfo}) with "
                f"a {type(bs)} statement (@{bs.srcinfo})"
            )
        elif isinstance(ps, (LoopIR.Assign, LoopIR.Reduce)):
            self.unify_e(ps.rhs, bs.rhs)
            self.unify_accesses(ps, bs)
        elif isinstance(ps, LoopIR.WriteConfig):
            if ps.config != bs.config or ps.field != bs.field:
                raise UnificationError(
                    f"cannot unify Writeconfig '{pe.config.name()}.{pe.field}' "
                    f"with Writeconfig '{be.config.name()}.{be.field}'"
                )
            self.unify_e(ps.rhs, bs.rhs)
        elif isinstance(ps, LoopIR.Pass):
            pass
        elif isinstance(ps, LoopIR.If):
            self.unify_e(ps.cond, bs.cond)
            self.unify_stmts(ps.body, bs.body)
            self.unify_stmts(ps.orelse, bs.orelse)
        elif isinstance(ps, LoopIR.For):
            # BINDING
            self.idx_subst[ps.iter] = bs.iter
            self.unify_e(ps.lo, bs.lo)
            self.unify_e(ps.hi, bs.hi)
            self.unify_stmts(ps.body, bs.body)
        elif isinstance(ps, LoopIR.Alloc):
            # introduce BufVars on the sub-procedure side of unification
            # and immediately force the solution to match the name found
            # on the original code side of unification
            pvar = BufVar(ps.name, ps.type, use_win=False)
            pvar.set_buf_solution(bs.name)
            self.buf_unknowns[ps.name] = pvar
            self.bbuf_types[bs.name] = bs.type
            self.unify_types(ps.type, bs.type, ps, bs)
        elif isinstance(ps, LoopIR.Call):
            if ps.f != bs.f:
                raise UnificationError(
                    f"cannot unify a call to '{ps.f.name()}' (@{ps.srcinfo}) "
                    f"with a call to {bs.f.name()} (@{bs.srcinfo})"
                )
            for pe, be in zip(ps.args, bs.args):
                self.unify_e(pe, be)
        elif isinstance(ps, LoopIR.WindowStmt):
            self.unify_e(ps.rhs, bs.rhs)
            # new name identification is similar to Alloc
            pvar = BufVar(ps.name, ps.rhs.type.as_tensor, use_win=False)
            pvar.set_buf_solution(bs.name)
            self.buf_unknowns[ps.name] = pvar
            self.bbuf_types[bs.name] = bs.rhs.type.as_tensor

        # tail recursion
        self.unify_stmts(proc_s, block_s)

    # directly unify two buffer names without adding any windowing
    def unify_buf_name_no_win(self, pname, bname):
        pvar = self.buf_unknowns[pname]

        if pvar.use_win:
            raise UnificationError(
                f"Cannot unify the windowed buffer '{pname}' "
                f"with the buffer '{bname}' because '{bname}' is used "
                f"in a position where windowing is not currently supported"
            )

        if pvar.solution_buf and pvar.solution_buf != bname:
            raise UnificationError(
                f"The buffer {pname} cannot be unified to both "
                f"the buffer {pvar.solution_buf}, and the buffer {bname}"
            )
        else:
            pvar.set_buf_solution(bname)

    def unify_accesses(self, pnode, bnode):
        pbuf, pidx = pnode.name, pnode.idx
        bbuf, bidx = bnode.name, bnode.idx
        pvar = self.buf_unknowns[pbuf]

        idx_gap = len(bidx) - len(pidx)
        # first, reject any numbers of indices that absolutely
        # cannot be made to work
        if idx_gap < 0:
            raise UnificationError(
                f"the access to '{pbuf}' (@{pnode.srcinfo}) has too many "
                f"indices ({len(pidx)}, compared to {len(bidx)}) to unify with "
                f"the access to '{bbuf}' (@{bnode.srcinfo})"
            )

        # handle special case of unindexed buffers used in
        # call-argument position
        if isinstance(bnode, LoopIR.Read) and len(bnode.type.shape()) > 0:
            assert len(bidx) == 0
            # we now know that bnode looks something like `x` where
            # `x` is not a scalar
            if len(pnode.type.shape()) == 0:
                raise UnificationError(
                    f"Could not unify buffer '{pbuf}' (@{pnode.srcinfo}) "
                    f"with buffer '{bbuf}' (@{bnode.srcinfo})"
                )
            else:
                assert len(pidx) == 0
                self.unify_types(pnode.type, bnode.type, pnode, bnode)
                self.unify_buf_name_no_win(pbuf, bbuf)
                return
        elif isinstance(pnode, LoopIR.Read) and len(pnode.type.shape()) > 0:
            # NOTE: bnode is not trivial b/c of the elif
            raise UnificationError(
                f"Unification of the simple call argument "
                f"'{pbuf}' (@{pnode.srcinfo}) "
                f"with a non-simple call argument "
                f"'{bbuf}' (@{bnode.srcinfo}) "
                f"is currently unsupported"
            )

        # otherwise, we can be sure that everything has been
        # accessed all the way down to a particular scalar value
        assert pnode.type.is_real_scalar() and bnode.type.is_real_scalar()

        # How to unify accesses when there is no intermediate windowing
        if not pvar.use_win:
            if idx_gap == 0:
                # with the index gap closed...
                for pi, bi in zip(pidx, bidx):
                    self.unify_affine_e(pi, bi)
                self.unify_types(pvar.typ, self.bbuf_types[bbuf], pnode, bnode)
                self.unify_buf_name_no_win(pbuf, bbuf)
            elif len(pidx) == 0 and self.is_proc_constant(bbuf):
                raise NotImplementedError("Unify buffer access with constant")
            else:
                raise UnificationError(
                    f"cannot unify the access to '{pbuf}' (@{pnode.srcinfo}) "
                    f"using {len(pidx)} indices with the access to "
                    f"'{bbuf}' (@{bnode.srcinfo}) using {len(bidx)} indices."
                )

        # Otherwise, how to unify accesses WITH windowing in the way
        else:
            if pvar.win_dim is not None and pvar.win_dim != idx_gap:
                raise UnificationError(
                    f"cannot unify '{pbuf}' (@{pnode.srcinfo}) "
                    f"with '{bbuf}' (@{bnode.srcinfo}) "
                    f"because '{pbuf}' is already being windowed down "
                    f"from a {pvar.n_dim + pvar.win_dim} dimension tensor, "
                    f"but is required to be windowed down from a "
                    f"{len(bidx)} dimension tensor here"
                )

            # set up all the case variables and the
            # equations relating windowing lo/hi expressions to
            # the resulting window's size-type expressions
            # Guard this to prevent redundant imposition of sizing equations
            if pvar.win_dim is None:
                pvar.set_buf_solution(bbuf)
                pvar.set_window_dim(idx_gap)
                self.equations += pvar.get_sz_eq(self)

            # now construct the equations relating the indexing on
            # the two sides of this access in all possible cases
            def case_conj(case_idxs):
                eqs = []
                tmp_pidx = pidx.copy()
                assert len(bidx) == len(case_idxs)
                for bi, wi in zip(bidx, case_idxs):
                    be = self.to_ueq(bi)
                    if isinstance(wi, Sym):  # point access from window
                        pe = UEq.Var(wi)
                    else:  # interval access
                        pe = UEq.Add(UEq.Var(wi[0]), self.to_ueq(tmp_pidx.pop(0)))
                    eqs.append(UEq.Eq(pe, be))

                assert len(tmp_pidx) == 0
                return UEq.Conj(eqs)

            cases = UEq.Cases(pvar.case_var, [case_conj(cidxs) for cidxs in pvar.cases])
            self.equations.append(cases)

    def unify_types(self, pt, bt, pnode, bnode):
        if pt.is_real_scalar() and bt.is_real_scalar():
            return  # success
        elif pt.is_indexable() and bt.is_indexable():
            return  # success
        elif pt == T.bool and bt == T.bool:
            return  # success
        elif pt.is_tensor_or_window() and bt.is_tensor_or_window():
            if len(pt.shape()) != len(bt.shape()):
                raise UnificationError(
                    f"cannot unify a tensor-type of "
                    f"{len(pt.shape())} dimensions (@{pnode.srcinfo}) with "
                    f"a tensor-type of {len(bt.shape())} dimensions "
                    f"(@{bnode.srcinfo})"
                )
            for psz, bsz in zip(pt.shape(), bt.shape()):
                self.unify_affine_e(psz, bsz)
        else:
            raise UnificationError(
                f"cannot unify type {pt} (@{pnode.srcinfo}) with "
                f"type {bt} (@{bnode.srcinfo})"
            )

    @staticmethod
    def comparision_to_unification_expr(e):
        def sub(lhs, rhs):
            return LoopIR.BinOp("-", lhs, rhs, T.index, null_srcinfo())

        def add1(e):
            one = LoopIR.Const(1, T.int, null_srcinfo())
            return LoopIR.BinOp("+", e, one, T.index, null_srcinfo())

        if e.op == "<" or e.op == "==":
            # rewrite `lhs op rhs` into `0 op rhs - lhs`
            return sub(e.rhs, e.lhs)
        elif e.op == "<=":
            # rewrite `lhs <= rhs` into `0 < rhs - lhs + 1`
            return add1(sub(e.rhs, e.lhs))
        elif e.op == ">":
            # rewrite `lhs > rhs` into `0 < lhs - rhs`
            return sub(e.lhs, e.rhs)
        elif e.op == ">=":
            # rewrite `lhs >= rhs` into `0 < lhs - rhs + 1`
            return add1(sub(e.lhs, e.rhs))
        else:
            assert False, f"Unreachable op found: {e.op}"

    def unify_e(self, pe, be):
        if pe.type.is_indexable() != be.type.is_indexable() or (pe.type == T.bool) != (
            be.type == T.bool
        ):
            raise UnificationError(
                f"expected expressions to have similar types:\n"
                f"  {pe}: {pe.type} [{pe.srcinfo}]\n"
                f"  {be}: {be.type} [{be.srcinfo}]"
            )
        elif pe.type.is_indexable():
            # convert to an equality
            self.unify_affine_e(pe, be)
            return
        elif (
            pe.type == T.bool
            and isinstance(pe, LoopIR.Read)
            and pe.name in self.bool_holes
        ):
            self.unify_bool_hole(pe, be)
            return
        elif (
            pe.type == T.stride
            and isinstance(pe, LoopIR.Read)
            and pe.name in self.stride_holes
        ):
            self.unify_stride_hole(pe, be)
            return

        if type(pe) is not type(be):
            raise UnificationError(
                f"cannot unify a {type(pe)} expression (@{pe.srcinfo}) with "
                f"a {type(be)} expression (@{be.srcinfo})"
            )
        elif isinstance(pe, LoopIR.Read):
            assert pe.type.is_numeric(), "unhandled expression type...?"
            self.unify_accesses(pe, be)
        elif isinstance(pe, LoopIR.Const):
            if pe.val != be.val:
                raise UnificationError(
                    f"cannot unify {pe.val} (@{pe.srcinfo}) with "
                    f"{be.val} (@{be.srcinfo})"
                )
        elif isinstance(pe, LoopIR.USub):
            self.unify_e(pe.arg, be.arg)
        elif isinstance(pe, LoopIR.BinOp):
            exprs = [pe.rhs, pe.lhs, be.rhs, be.lhs]
            if pe.op in comparision_ops and all(e.type.is_indexable() for e in exprs):
                inequality_ops = comparision_ops - {"=="}
                if pe.op == be.op or (
                    pe.op in inequality_ops and be.op in inequality_ops
                ):
                    pe_e = self.comparision_to_unification_expr(pe)
                    be_e = self.comparision_to_unification_expr(be)
                    self.unify_e(pe_e, be_e)
                    return
            if pe.op != be.op:
                raise UnificationError(
                    f"cannot unify a '{pe.op}' (@{pe.srcinfo}) with "
                    f"a '{be.op}'' (@{be.srcinfo})"
                )
            self.unify_e(pe.lhs, be.lhs)
            self.unify_e(pe.rhs, be.rhs)
        elif isinstance(pe, LoopIR.BuiltIn):
            if pe.f != be.f:
                raise UnificationError(
                    f"cannot unify builtin '{pe.f.name()}' (@{pe.srcinfo}) "
                    f"with builtin '{be.f.name()}'' (@{be.srcinfo})"
                )
            for pa, ba in zip(pe.args, be.args):
                self.unify_e(pa, ba)
        elif isinstance(pe, LoopIR.ReadConfig):
            if pe.config != be.config or pe.field != be.field:
                raise UnificationError(
                    f"cannot unify readconfig '{pe.config.name()}.{pe.field}' "
                    f"with readconfig '{be.config.name()}.{be.field}'"
                )
        elif isinstance(pe, LoopIR.WindowExpr):
            pvar = self.buf_unknowns[pe.name]

            # unify the two buffers
            self.unify_buf_name_no_win(pe.name, be.name)
            self.unify_types(pvar.typ, self.bbuf_types[be.name], pe, be)

            # unify the two windowing expressions
            if len(pe.idx) != len(be.idx):
                raise UnificationError(
                    f"cannot unify the windowing of {pe.name} (@{pe.srcinfo}) "
                    f"using {len(pe.idx)} indices with the windowing of "
                    f"{be.name} (@{be.srcinfo}) using {len(be.idx)}"
                )

            for i, (pw, bw) in enumerate(zip(pe.idx, be.idx)):
                if type(pw) is not type(bw):
                    raise UnificationError(
                        f"cannot unify the windowing of "
                        f"{pe.name} (@{pe.srcinfo}) with the windowing of "
                        f"{be.name} (@{be.srcinfo}) because one evaluates to a "
                        f"point at index {i}, while the other evaluates to an "
                        f"interval"
                    )
                elif isinstance(pw, LoopIR.Point):
                    self.unify_affine_e(pw.pt, bw.pt)
                else:
                    self.unify_affine_e(pw.lo, bw.lo)
                    self.unify_affine_e(pw.hi, bw.hi)
        else:
            assert False, f"bad case of {type(pe)}"

    def is_proc_constant(self, bbuf):
        deps = LoopIR_Dependencies(bbuf, self.stmt_block).result()
        return all(dep in self.FV for dep in deps)
