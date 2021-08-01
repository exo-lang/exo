from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *
from .LoopIR import LoopIR, T, LoopIR_Rewrite, LoopIR_Do, FreeVars
from .effectcheck import InferEffects, CheckEffects

from collections import ChainMap
import functools
import re
import itertools

import pysmt
from pysmt import shortcuts as SMT

def _get_smt_solver():
    factory = pysmt.factory.Factory(pysmt.shortcuts.get_env())
    slvs    = factory.all_solvers()
    if len(slvs) == 0: raise OSError("Could not find any SMT solvers")
    return pysmt.shortcuts.Solver(name=next(iter(slvs)))

def sanitize_str(s):
    return re.sub(r'\w','_',s)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class UnificationError(Exception):
    def __init__(self,msg):
        self._err_msg   = str(msg)
    def __str__(self):
        return self._err_msg


class DoReplace(LoopIR_Rewrite):
    def __init__(self, proc, subproc, stmt_block):
        # ensure that subproc and stmt_block match in # of statements
        n_stmts = len(subproc.body)
        if len(stmt_block) < n_stmts:
            raise SchedulingError("Not enough statements to match")
        stmt_block = stmt_block[:n_stmts]

        self.subproc        = subproc
        self.target_block   = stmt_block
        self.live_vars      = ChainMap()

        super().__init__(proc)
        # fix up effects post-hoc
        self.proc = InferEffects(self.proc).result()
        # and then check that all effect-check conditions are
        # still satisfied...
        CheckEffects(self.proc)

    def push(self):
        self.live_vars = self.live_vars.new_child()
    def pop(self):
        self.live_vars = self.live_vars.parents

    def map_fnarg(self, fa):
        self.live_vars[fa.name] = fa.type
        return super().map_fnarg(fa)

    def map_s(self, s):
        # For all leaf-statements (containing no sub-statements),
        # just return the original statement.  Bind variables when
        # necessary, and then for scoped blocks, manage scope and recursion
        styp = type(s)
        if styp is LoopIR.WindowStmt:
            self.live_vars[s.lhs] = s.rhs.type
        elif styp is LoopIR.Alloc:
            self.live_vars[s.name] = s.type
        elif styp is LoopIR.If:
            self.push()
            body = self.map_stmts(s.body)
            self.pop()
            self.push()
            orelse = self.map_stmts(s.orelse)
            self.pop()

            return [LoopIR.If( s.cond, body, orelse, s.eff, s.srcinfo )]

        elif styp is LoopIR.ForAll:
            self.push()
            self.live_vars[s.iter] = T.index
            body = self.map_stmts(s.body)
            self.pop()

            return [LoopIR.ForAll( s.iter, s.hi, body, s.eff, s.srcinfo )]

        return [s]

    def map_stmts(self, stmts):
        # see if we can find the target block in this block
        n_stmts = len(self.target_block)
        match_i = None
        for i,s in enumerate(stmts):
            if s == self.target_block[0]:
                if stmts[i:i+n_stmts] == self.target_block:
                    match_i = i
                    break

        if match_i is None:
            return super().map_stmts(stmts)
        else: # process the match
            prefix_stmts = super().map_stmts(stmts[ : match_i])

            new_args = Unification(self.subproc, stmts,
                                   self.live_vars).result()

            new_call = LoopIR.Call(self.subproc, new_args, None, stmt.srcinfo)

            return prefix_stmts + [new_call] + stmts[match_i+n_stmts : ]

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self,e):
        return e
    def map_t(self,t):
        return t
    def map_eff(self,eff):
        return eff


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unification "System of Equations" grammar

# the problem designates disjoint subsets of the variables used
# as "holes" (variables to solve for) and "knowns" (variables to express
# a solution as an affine combination of).  Any variable not in either of
# those lists is unknown but not permissible in a solution expression.
UEq = ADT("""
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

} """, {
    'sym':          lambda x: type(x) is Sym,
})


# -------------------------------------- #
# Conversion to Strings for Debug

def _str_uexpr(e, prec=0):
    etyp = type(e)
    if etyp is UEq.Const:
        return str(e.val)
    elif etyp is UEq.Var:
        return str(e.name)
    elif etyp is UEq.Add:
        s = f"{_str_uexpr(e.lhs,0)} + {_str_uexpr(e.rhs,1)}"
        if prec > 0:
            s = f"({s})"
        return s
    elif etyp is UEq.Scale:
        return f"{e.coeff}*{_str_uexpr(e.e,10)}"
    else: assert False, "bad case"

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
        op  = ' and ' if UEq.Conj else ' or '
        s   = op.join([ _str_upred(pp,1) for pp in p.preds ])
        if prec > 0:
            s = f"({s})"
        return s
    elif ptyp is UEq.Cases:
        return (f'cases({p.case_var}) '+
                ' | '.join([ f"({pred})" for pred in p.cases ]))
    else: assert False, "bad case"

@extclass(UEq.Conj)
@extclass(UEq.Disj)
@extclass(UEq.Eq)
def __str__(self):
    return _str_upred(self)
del __str__

@extclass(UEq.problem)
def __str__(prob):
    lines = [ "Holes:   "+', '.join([ str(x) for x in prob.holes ]) ,
              "Knowns:  "+', '.join([ str(x) for x in prob.knowns ]) ]
    lines += [ str(p) for p in prob.preds ]
    return '\n'.join(lines)
del __str__


# -------------------------------------- #
# How to solve this system of equations

@extclass(UEq.problem)
def solve(prob):
    solver      = _get_smt_solver()

    known_list  = prob.knowns
    known_idx   = { k : i for i,k in enumerate(known_list) }
    Nk          = len(known_list)

    var_set     = dict()
    case_set    = dict()
    def get_var(x):
        if x in var_set:
            return var_set[x]
        else:
            vec = ([ SMT.Symbol(f"{repr(x)}_{repr(k)}", SMT.INT)
                     for k in known_list ] +
                   [ SMT.Symbol(f"{repr(x)}_const", SMT.INT) ])
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
        if type(e) is UEq.Const:
            return ([SMT.Int(0)] * Nk) + [SMT.Int(e.val)]
        elif type(e) is UEq.Var:
            if e.name in known_idx:
                one_hot = [SMT.Int(0)] * (Nk+1)
                one_hot[known_idx[e.name]] = SMT.Int(1)
                return one_hot
            else:
                return get_var(e.name)
        elif type(e) is UEq.Add:
            lhs = lower_e(e.lhs)
            rhs = lower_e(e.rhs)
            return [ SMT.Plus(x,y) for x,y in zip(lhs,rhs) ]
        elif type(e) is UEq.Scale:
            arg = lower_e(e.e)
            return [ SMT.Times(e.coeff, a) for a in arg ]
        else: assert False, "bad case"

    def lower_p(p):
        if type(p) is UEq.Eq:
            lhs = lower_e(p.lhs)
            rhs = lower_e(p.rhs)
            return SMT.And(*[ SMT.Equals(x,y) for x,y in zip(lhs,rhs) ])
        elif type(p) is UEq.Conj:
            return SMT.And(*[ lower_p(pp) for pp in p.preds ])
        elif type(p) is UEq.Disj:
            return SMT.Or(*[ lower_p(pp) for pp in p.preds ])
        elif type(p) is UEq.Cases:
            case_var = get_case(p.case_var)
            def per_case(i,c):
                pp = lower_p(c)
                is_case = SMT.Equals(case_var, SMT.Int(i))
                return SMT.And(is_case, pp)
            disj    = SMT.Or(*[ per_case(i,c) for i,c in enumerate(p.cases) ])
            case_lo = SMT.GE(case_var, SMT.Int(0))
            case_hi = SMT.LT(case_var, SMT.Int(len(p.cases)))
            return SMT.And(disj, case_lo, case_hi)
        else: assert False, "bad case"

    prob_pred   = SMT.And(*[ lower_p(p) for p in prob.preds ])
    if not solver.is_sat(prob_pred):
        return None
    else:
        solutions = dict()
        for x in prob.holes:
            x_syms  = get_var(x)
            x_vals  = solver.get_py_values(x_syms)
            expr    = None
            for x,v in zip(known_list, x_vals):
                v = int(v)
                if v == 0:
                    continue
                elif v == 1:
                    term = UEq.Var(x)
                else:
                    term = UEq.Scale(v, UEq.Var(x))

                expr = term if expr is None else UEq.Add(expr, term)

            # constant offset
            off     = UEq.Const(int(x_vals[-1]))
            expr    = off if expr is None else UEq.Add(expr, off)

            solutions[x] = expr

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
        self.node_to_sym    = dict() # many to one
        self.tuple_to_sym   = dict() # de-duplicating lookup
        self.sym_to_node    = dict() # pick a node for each symbol
        self.FV             = FV

        self.unq_count      = 0

        self.do_stmts(stmts)

    def result(self):
        return self.node_to_sym, self.sym_to_node

    @functools.lru_cache(maxsize=None)
    def tuple_memo(*args):
        return tuple(*args)

    def do_e(self, e):
        if ( type(e) is LoopIR.BinOp and
             (e.op == '%' or e.op == '/') and 
             e.type.is_indexable() ):
            # found a mod-div site
            tuple_node = self.tupleify(e)
            if tuple_node is None:
                raise UnificationError(f"{e.srcinfo}: cannot handle this "+
                                       f"'{e.op}' operation")

            # either we have already seen this expression
            if tuple_node in self.tuple_to_sym:
                sym = self.tuple_to_sym[tuple_node]
            # or we are encountering it for the first time
            else:
                opname      = 'mod' if e.op == '%' else 'div'
                node_name   = sanitize_str(str(e))
                sym         = Sym(f'{opname}_{self.unq_count}_{node_name}')
                self.unq_count += 1
                self.tuple_to_sym[tuple_node] = sym
                self.sym_to_node[sym] = e

            # regardless, record which symbol we've assigned to
            # this specific AST node sub-tree
            self.node_to_sym[e] = sym

        else:
            super().do_e(e)

    def tupleify(self, e):
        if type(e) is LoopIR.Read:
            assert len(e.idx) == 0
            if e.name not in self.FV:
                raise UnificationError(f"{e.srcinfo}:"+
                        f" Found bound variable '{e.name}' inside of "+
                        f"mod or div operation")
            else:
                return e.name
        elif type(e) is LoopIR.Const:
            return e.val
        elif type(e) is LoopIR.USub:
            return self.tuple_memo('-', self.tupleify(e.arg))
        elif type(e) is LoopIR.BinOp:
            return self.tuple_memo(e.op, self.tupleify(e.lhs),
                                         self.tupleify(e.rhs))
        else: assert False, "Bad case tupleify"

    def do_eff(self, eff):
        return

class BufVar:
    def __init__(self, name, typ, use_win=True):
        self.name           = name
        self.typ            = typ
        self.n_dim          = len(typ.shape())

        self.solution_buf   = None

        self.use_win        = use_win
        # `win_dim` is the number of dimensions that are
        # point-accesses in a window expression
        # `n_dim` is the number of dimensions that are
        # interval/slice-accesses in a window expression
        # The sum of the two is the size of whatever buffer is unified against
        self.win_dim        = None
        self.cases          = []
        self.case_var       = None

    def set_buf_solution(self, solution_buf):
        self.solution_buf   = solution_buf

    def set_window_dim(self, win_dim):
        assert win_dim is not None
        do_setup            = self.win_dim is None
        self.win_dim        = win_dim

        if do_setup:
            self.cases      = []
            self.case_var   = Sym(f"{self.name}_which_case")
            full_dim        = self.n_dim + win_dim

            for case_id, pt_idxs in enumerate(
                itertools.combinations(range(0,full_dim),win_dim)):
                case_name = (f"{self.name}_"+
                             '_'.join([str(i) for i in pt_idxs])+
                             f"_cs{case_id}")

                def make_pt(i):
                    return Sym(f"{case_name}_pt{i}")
                def make_interval(i):
                    return (Sym(f"{case_name}_lo{i}"),
                            Sym(f"{case_name}_hi{i}"))

                idx_vars = [ (make_pt(i) if i in pt_idxs else
                              make_interval(i))
                             for i in range(0,full_dim) ]

                self.cases.append(idx_vars)

    def get_sz_eq(self, UObj):
        assert self.win_dim is not None
        interval_cases = [ e for e in self.cases if type(e) is not Sym ]
        assert len(interval_cases) == self.n_dim
        results = []
        for (lo,hi),sz in zip(interval_cases, self.typ.shape()):
            diff    = UEq.Add( UEq.Var(hi), UEq.Scale(-1, UEq.Var(lo)) )
            results += [UEq.Eq( diff, UObj.to_ueq(sz) )]
        return results

    def all_syms(self):
        if not self.case_var:
            return []
        else:
            xs = [ self.case_var ]
            for c in self.cases:
                for i in c:
                    if type(i) is Sym:
                        xs.append(i)
                    else:
                        xs.append(i[0])
                        xs.append(i[1])
            return xs

    def get_solution(self, UObj, ueq_solutions, srcinfo):
        if not self.case_var:
            return LoopIR.Read(self.solution_buf, [], self.typ, srcinfo)
        else:
            which_case  = ueq_solutions[self.case_var]
            case        = self.cases[which_case]

            idx         = []
            for w in case:
                if type(w) is Sym:
                    pt = UObj.from_ueq( ueq_solutions[w] )
                    idx.append(LoopIR.Point(pt, srcinfo))
                else:
                    lo = UObj.from_ueq( ueq_solutions[w[0]] )
                    hi = UObj.from_ueq( ueq_solutions[w[1]] )
                    idx.append(LoopIR.Interval(lo, hi, srcinfo))

            buf         = self.solution_buf
            w_typ       = T.Window(UObj.FV[buf], self.typ, buf, idx)
            return LoopIR.WindowExpr(buf, idx, w_typ, srcinfo)


class Unification:
    def __init__(self, subproc, stmt_block, live_vars):
        self.equations  = []

        # variables for the UEq system
        #self.holes      = []
        #self.knowns     = []

        # set up different kinds of variables before we
        # begin doing the structural matching...
        #self.arg_syms       = { fa.name : True for fa in subproc.args }
        self.index_holes    = [ fa.name for fa in subproc.args
                                        if fa.type.is_indexable() ]
        self.buf_holes      = { fa.name :
                                BufVar(fa.name, fa.type, fa.type.is_win())
                                for fa in subproc.args
                                if fa.type.is_numeric() }
        assert all( fa.type != T.bool for fa in subproc.args ), (""+
                    "code needs extension to handle boolean variables")

        # keep track of all buffer names we might need to unify,
        # not just the unknown arguments, but also temporary allocated bufs
        # these variables should ONLY occur on the sub-procedure side
        # of the unification; no BufVars for the original code.
        self.buf_unknowns   = self.buf_holes.copy()

        # get the free variables, and lookup their types
        # as well as expanding the free variable set to
        # account for dependent typing
        FV_set          = FreeVars(stmt_block).result()
        self.FV         = dict()
        def add_fv(x):
            assert x in live_vars, f"expected FV {x} to be live"
            typ         = live_vars[x]
            self.FV[x]  = typ

            def expand_e(e):
                if type(e) is LoopIR.Read:
                    add_fv(e.name)
                elif type(e) is LoopIR.USub:
                    expand_e(e.arg)
                elif type(e) is LoopIR.BinOp:
                    expand_e(e.lhs, e.rhs)

            if type(typ) is T.Tensor:
                for e in typ.hi:
                    expand_e(e)
            elif type(typ) is T.Window:
                for w in typ.idx:
                    if type(w) is LoopIR.Interval:
                        expand_e(w.lo)
                        expand_e(w.hi)
                    else:
                        expand_e(w.pt)
                add_fv(typ.src_buf)
        for x in FV_set:
            add_fv(x)

        # block-side buffer types
        self.bbuf_types     = { x : self.FV[x] for x in self.FV
                                                if self.FV[x].is_numeric() }

        #self.node_syms  = None
        #self.sym_nodes  = None
        self.node_syms, self.sym_nodes = _Find_Mod_Div_Symbols(stmt_block,
                                                               self.FV)

        # TODO: Asserts
        # We don't have inequality in EQs IR

        # TODO: Size
        # Inequality...??

        # build up the full set of equations...
        self.unify_stmts(subproc.body, stmt_block)

        # setup the problem
        for nm in self.buf_holes:
            if self.buf_holes[nm].solution_buf is None:
                raise UnificationError(f""+
                    f"Cannot perform unification due to an un-unused "+
                    f"argument: {nm}")

        holes       = (self.index_holes +
                       [ x for nm in self.buf_holes
                           for x in self.buf_holes[nm].all_syms() ])
        knowns      = [ nm for nm in self.FV ]
        ueq_prob    = UEq.problem(holes, knowns, self.equations)

        # solve the problem
        solutions   = ueq_prob.solve()
        if solutions is None:
            raise UnificationError(f""+
                f"Unification of various index expressions failed")

        # construct the solution arguments
        def get_arg(fa):
            if fa.type.is_indexable():
                return self.from_ueq(solutions[fa.name])
            else:
                assert fa.type.is_numeric()
                bufvar = self.buf_holes[fa.name]
                return bufvar.get_solution(self, solutions,
                                           stmt_block[0].srcinfo)
        new_args    = [ get_arg(fa) for fa in subproc.args ]
        return new_args


    def err(self):
        raise TypeError("subproc and pattern don't match")

    def result(self):
        return Equations.system(self.equations)


    # ----------

    def to_ueq(self, e, in_subproc=False):
        insp = in_subproc
        if type(e) is LoopIR.Read:
            assert len(e.idx) == 0
            return UEq.Var(e.name)
        elif type(e) is LoopIR.Const:
            return UEq.Const(e.val)
        elif type(e) is LoopIR.USub:
            return UEq.Scale( -1, self.to_ueq(e.arg,insp) )
        elif type(e) is LoopIR.BinOp:
            if e.op == '+':
                return UEq.Add( self.to_ueq(e.lhs,insp),
                                self.to_ueq(e.rhs,insp) )
            elif e.op == '-':
                rhs = UEq.Scale( -1, self.to_ueq(e.rhs,insp) )
                return UEq.Add( self.to_ueq(e.lhs,insp), rhs )
            elif e.op == '*':
                if type(e.lhs) is LoopIR.Const:
                    return UEq.Scale( e.lhs.val, self.to_ueq(e.rhs,insp) )
                elif type(e.rhs) is LoopIR.Const:
                    return UEq.Scale( e.rhs.val, self.to_ueq(e.lhs,insp) )
                else: assert False, ("unexpected multiplication; "+
                                     "improve the code here")
            elif e.op == '/' or e.op == '%':
                if in_subproc:
                    raise UnificationError(f""+
                        f"unification with sub-procedures making use of "
                        f"'%' or '/' operations is not currently supported")
                else:
                    name = self.node_syms[e]
                    return UEq.Var(name)
            else: assert False, f"bad op case: {e.op}"
        else: assert False, "unexpected affine expression case"

    def from_ueq(self, e, srcinfo=null_srcinfo()):
        if type(e) is UEq.Var:
            if e.name in self.sym_nodes:
                return self.sym_nodes[e.name]
            else:
                typ = self.FV[e.name]
                return LoopIR.Read(e.name, [], typ, srcinfo)

        elif type(e) is UEq.Const:
            return LoopIR.Const(e.val, T.int, srcinfo)
        elif type(e) is UEq.Add:
            lhs = self.from_ueq(e.lhs, srcinfo)
            rhs = self.from_ueq(e.rhs, srcinfo)
            typ = (lhs.type if rhs.type == T.int else
                   rhs.type if lhs.type == T.int else
                   lhs.type if rhs.type == T.size else
                   rhs.type)
            return LoopIR.BinOp('+', lhs, rhs, typ, srcinfo)
        elif type(e) is UEq.Scale:
            lhs = LoopIR.Const(e.coeff, T.int, srcinfo)
            rhs = self.from_ueq(e.e,srcinfo)
            return LoopIR.BinOp('*', lhs, rhs, rhs.type, srcinfo)
        else: assert False, "bad case"

    # ----------

    def unify_affine_e(self, pa, ba):
        self.equations.push(UEq.Eq( self.to_ueq(pa,in_subproc=True),
                                    self.to_ueq(ba) ))

    def unify_stmts(self, proc_s, block_s):
        if len(proc_s) != len(block_s):
            ploc, bloc = "",""
            if len(proc_s) > 0:
                ploc = f" (@{proc_s[0].srcinfo})"
            if len(block_s) > 0:
                bloc = f" (@{block_s[0].srcinfo})"
            raise UnificationError(f""+
                f"cannot unify {len(proc_s)} statement(s){ploc} with "+
                f"{len(block_s)} statement(s){bloc}")
        elif len(proc_s) == 0:
            return

        ps, proc_s  = proc_s[0], proc_s[1:]
        bs, block_s = block_s[0], block_s[1:]

        if type(ps) is not type(bs):
            raise UnificationError(f""+
                f"cannot unify a {type(ps)} statement (@{ps.srcinfo}) with "+
                f"a {type(bs)} statement (@{bs.srcinfo})")
        elif type(ps) is LoopIR.Assign or type(ps) is LoopIR.Reduce:
            self.unify_e(ps.rhs, bs.rhs)
            self.unify_accesses(ps, bs)
        elif type(ps) is LoopIR.Pass:
            pass
        elif type(ps) is LoopIR.If:
            self.unify_e(ps.cond, bs.cond)
            self.unify_stmts(ps.body, bs.body)
            self.unify_stmts(ps.orelse, bs.orelse)
        elif type(ps) is LoopIR.ForAll:
            # BINDING
            self.equations.push(UEq.Eq( UEq.Var(ps.iter), UEq.Var(bs.iter) ))
            self.unify_e(ps.hi, bs.hi)
            self.unify_stmts(ps.body, bs.body)
        elif type(ps) is LoopIR.Alloc:
            # introduce BufVars on the sub-procedure side of unification
            # and immediately force the solution to match the name found
            # on the original code side of unification
            pvar = BufVar(ps.name, ps.type, use_win=False)
            pvar.set_buf_solution(bs.name)
            self.buf_unknowns[ps.name] = pvar
            self.bbuf_types[bs.name] = bs.type
            self.unify_types(ps.type, bs.type)
        elif type(ps) is LoopIR.Call:
            if ps.f != bs.f:
                raise UnificationError(f""+
                    f"cannot unify a call to '{ps.f.name()}' (@{ps.srcinfo}) "+
                    f"with a call to {bs.f.name()} (@{bs.srcinfo})")
            for pe, be in zip(ps.args, bs.args):
                self.unify_e(pe, be)
        elif type(ps) is LoopIR.WindowStmt:
            self.unify_e(ps.rhs, bs.rhs)
            # new name identification is similar to Alloc
            pvar = BufVar(ps.lhs, ps.rhs.type.as_tensor, use_win=False)
            pvar.set_buf_solution(bs.lhs)
            self.buf_unknowns[ps.lhs] = pvar
            self.bbuf_types[bs.lhs]   = bs.rhs.type.as_tensor

        # tail recursion
        self.unify_stmts(proc_s, block_s)

    # directly unify two buffer names without adding any windowing
    def unify_buf_name_no_win(self, pname, bname):
        pvar = self.buf_unknowns[pname]

        if pvar.use_win:
            raise UnificationError(f""+
                f"Cannot unify the windowed buffer '{pname}' "+
                f"with the buffer '{bname}' because '{bname}' is used "+
                f"in a position where windowing is not currently supported")

        if pvar.solution_buf and pvar.solution_buf != bname:
            raise UnificationError(f""+
                f"The buffer {pname} cannot be unified to both "+
                f"the buffer {pvar.solution_buf}, and the buffer {bname}")
        else:
            pvar.set_buf_solution(bname)

    def unify_accesses(self, pnode, bnode):
        pbuf, pidx  = pnode.name, pnode.idx
        bbuf, bidx  = bnode.name, bnode.idx
        pvar        = self.buf_unknowns[pbuf]

        idx_gap = len(bidx) - len(pidx)
        # first, reject any numbers of indices that absolutely
        # cannot be made to work
        if idx_gap < 0:
            raise UnificationError(f""+
                f"the access to '{pbuf}' (@{pnode.srcinfo}) "+
                f"has too many indices "+
                f"({len(pidx)}, compared to {len(bidx)}) "+
                f"to unify with the access to '{bbuf}' (@{bnode.srcinfo})")

        # handle special case of unindexed buffers used in
        # call-argument position
        if type(bnode) is LoopIR.Read and len(bnode.type.shape()) > 0:
            assert len(bidx) == 0
            # we now know that bnode looks something like `x` where
            # `x` is not a scalar
            if len(pnode.type.shape()) == 0:
                raise UnificationError(f""+
                    f"Could not unify buffer '{pbuf}' (@{pnode.srcinfo}) "+
                    f"with buffer '{bbuf}' (@{bnode.srcinfo})")
            else:
                assert len(pidx) == 0
                self.unify_types(pnode.type, bnode.type)
                self.unify_buf_name_no_win(pbuf,bbuf)
                return
        elif type(pnode) is LoopIR.Read and len(pnode.type.shape()) > 0:
            # NOTE: bnode is not trivial b/c of the elif
            raise UnificationError(f""+
                f"Unification of the simple call argument "+
                f"'{pbuf}' (@{pnode.srcinfo}) "+
                f"with a non-simple call argument "+
                f"'{bbuf}' (@{bnode.srcinfo}) "+
                f"is currently unsupported")

        # otherwise, we can be sure that everything has been
        # accessed all the way down to a particular scalar value
        assert pnode.type.is_real_scalar() and bnode.type.is_real_scalar()


        # How to unify accesses when there is no intermediate windowing
        if not pvar.use_win:
            if idx_gap != 0:
                raise UnificationError(f""+
                    f"cannot unify "+
                    f"the access to '{pbuf}' (@{pnode.srcinfo}) "+
                    f"using {len(pidx)} indices "+
                    f"with the access to '{bbuf}' (@{bnode.srcinfo}) "+
                    f"using {len(bidx)} indices.")

            # with the index gap closed...
            for pi,bi in zip(pidx,bidx):
                self.unify_affine_e(pi,bi)
            self.unify_types(pvar.typ, self.bbuf_types[bbuf])
            self.unify_buf_name_no_win(pbuf,bbuf)

        # Otherwise, how to unify accesses WITH windowing in the way
        else:
            if pvar.win_dim is not None and pvar.win_dim != idx_gap:
                raise UnificationError(f""+
                    f"cannot unify '{pbuf}' (@{pnode.srcinfo}) "+
                    f"with '{bbuf}' (@{bnode.srcinfo}) "+
                    f"because '{pbuf}' is already being windowed down "+
                    f"from a {pvar.n_dim + pvar.win_dim} dimension tensor, "+
                    f"but is required to be windowed down from a "+
                    f"{len(bidx)} dimension tensor here")

            # set up all the case variables and the
            # equations relating windowing lo/hi expressions to
            # the resulting window's size-type expressions
            # Guard this to prevent redundant imposition of sizing equations
            if pvar.wind_dim is None:
                pvar.set_window_dim(idx_gap)
                self.equations += pvar.get_sz_eq(self)

            # now construct the equations relating the indexing on
            # the two sides of this access in all possible cases
            def case_conj(case_idxs):
                eqs     = []
                pidx    = pidx.copy()
                for bi, wi in zip(bidx,case_idxs):
                    be  = self.to_ueq(bi)
                    pe  = None
                    if type(wi) == Sym: # point access from window
                        pe  = UEq.Var(wi)
                    else: # interval access
                        pe  = UEq.Add(UEq.Var(wi[0]),
                                      self.to_ueq(pidx.pop(0)))
                    eqs.append(UEq.Eq(pe, be))

                assert len(pidx) == 0
                return UEq.Conj(eqs)

            cases = UEq.Cases(pvar.case_var,
                              [ case_conj(cidxs) for cidxs in pvar.cases ])
            self.equations.append(cases)


    def unify_types(self, pt, bt, pnode, bnode):
        if pt.is_real_scalar() and bt.is_real_scalar():
            return # success
        elif pt.is_indexable() and bt.is_indexable():
            return # success
        elif pt == T.bool and bt == T.bool:
            return # success
        elif pt.is_tensor_or_window() and bt.is_tensor_or_window():
            if len(pt.shape()) != len(bt.shape()):
                raise UnificationError(f""+
                    f"cannot unify a tensor-type of "+
                    f"{len(pt.shape())} dimensions (@{pnode.srcinfo}) with "+
                    f"a tensor-type of {len(bt.shape())} dimensions "+
                    f"(@{bnode.srcinfo})")
            for psz,bsz in zip(pt.shape(),bt.shape()):
                self.unify_affine_e(psz,bsz)
        else:
            raise UnificationError(f""+
                f"cannot unify type {pt} (@{pnode.srcinfo}) with "+
                f"type {bt} (@{bnode.srcinfo})")

    def unify_e(self, pe, be):
        if pe.type.is_indexable() != be.type.is_indexable():
            raise UnificationError(f""+
                f"expected expressions (@{pe.srcinfo} vs. @{be.srcinfo}) "+
                f"to have similar types")
        elif pe.type.is_indexable():
            # convert to an equality
            self.unify_affine_e(pe, be)

        if type(pe) != type(be):
            raise UnificationError(f""+
                f"cannot unify a {type(pe)} expression (@{pe.srcinfo}) with "+
                f"a {type(be)} expression (@{be.srcinfo})")
        elif type(pe) is LoopIR.Read:
            assert pe.type.is_numeric(), "unhandled expression type...?"
            self.unify_accesses(pe, be)
        elif type(pe) is LoopIR.Const:
            if pe.val != be.val:
                raise UnificationError(f""+
                    f"cannot unify {pe.val} (@{pe.srcinfo}) with "+
                    f"{be.val} (@{be.srcinfo})")
        elif type(pe) is LoopIR.USub:
            self.unify_e(pe.arg, be.arg)
        elif type(pe) is LoopIR.BinOp:
            if pe.op != be.op:
                raise UnificationError(f""+
                    f"cannot unify a '{pe.op}' (@{pe.srcinfo}) with "+
                    f"a '{be.op}'' (@{be.srcinfo})")
            self.unify_e(pe.lhs, be.lhs)
            self.unify_e(pe.rhs, be.rhs)
        elif type(pe) is LoopIR.BuiltIn:
            if pe.f != be.f:
                raise UnificationError(f""+
                    f"cannot unify builtin '{pe.f.name()}' (@{pe.srcinfo}) "+
                    f"with builtin '{be.f.name()}'' (@{be.srcinfo})")
            for pa,ba in zip(pe.args,be.args):
                self.unify_e(pa,ba)
        elif type(pe) is LoopIR.WindowExpr:
            pvar = self.buf_unknowns[pe.name]

            # unify the two buffers
            self.unify_buf_name_no_win(pe.name,be.name)
            self.unify_types(pvar.typ, self.bbuf_types[bbuf])

            # unify the two windowing expressions
            if len(pe.idx) != len(be.idx):
                raise UnificationError(f""+
                    f"cannot unify the windowing of "+
                    f"{pe.name} (@{pe.srcinfo}) "+
                    f"using {len(pe.idx)} indices "+
                    f"with the windowing of {be.name} (@{be.srcinfo}) "+
                    f"using {len(be.idx)}")

            for i,(pw,bw) in enumerate(zip(pe.idx,be.idx)):
                if type(pw) != type(bw):
                    raise UnificationError(f""+
                        f"cannot unify the windowing of "+
                        f"{pe.name} (@{pe.srcinfo}) "+
                        f"with the windowing of {be.name} (@{be.srcinfo}) "+
                        f"because one evaluates to a point at index {i}, "+
                        f"while the other evaluates to an interval")
                elif type(pw) == LoopIR.Point:
                    self.unify_affine_e(pw.pt, bw.pt)
                else:
                    self.unify_affine_e(pw.lo, bw.lo)
                    self.unify_affine_e(pw.hi, bw.hi)
        else: assert False, "bad case"





