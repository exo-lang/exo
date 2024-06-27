from collections import OrderedDict, ChainMap
from enum import Enum
from itertools import chain

from .LoopIR import Alpha_Rename, SubstArgs, LoopIR_Do
from .configs import reverse_config_lookup, Config
from .new_analysis_core import *
from .proc_eqv import get_repr_proc

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Useful Basic Concepts and Structured Data

_simple_proc_cache = dict()


def get_simple_proc(proc):
    orig_repr = get_repr_proc(proc)
    if orig_repr not in _simple_proc_cache:
        _simple_proc_cache[orig_repr] = Alpha_Rename(orig_repr).result()
    return _simple_proc_cache[orig_repr]


@dataclass
class APoint:
    name: Sym
    coords: list[A.expr]
    typ: Any


@dataclass
class AWinCoord:
    is_pt: bool
    val: A.expr


@dataclass
class AWin:
    name: Sym
    coords: list[AWinCoord]
    strides: list[A.expr]

    def __str__(win):
        def coordstr(c):
            op = "=" if c.is_pt else "+"
            return f"{op}{c.val}"

        coords = ""
        if len(win.coords) > 0:
            coords = "," + (",".join([coordstr(c) for c in win.coords]))
        return f"({win.name}{coords})"

    def nslots(self):
        return sum([not c.is_pt for c in self.coords])

    # compose with another AWin
    def __add__(lhs, rhs):
        assert isinstance(rhs, AWin)
        # ignore rhs.name
        # check that the number of coordinates is compatible
        assert lhs.nslots() == len(rhs.coords)
        ri = 0
        coords = []
        for lc in lhs.coords:
            if lc.is_pt:
                coords.append(lc)
            else:
                rc = rhs.coords[ri]
                ri += 1
                coords.append(AWinCoord(rc.is_pt, rc.val + lc.val))
        return AWin(lhs.name, coords, lhs.strides)

    # apply to a point
    def __call__(self, pt):
        assert isinstance(pt, APoint)
        assert self.nslots() == len(pt.coords)
        pi = 0
        coords = []
        for wc in self.coords:
            if wc.is_pt:
                coords.append(wc.val)
            else:
                coords.append(wc.val + pt.coords[pi])
                pi += 1
        return APoint(self.name, coords, pt.typ)

    # get a stride
    def get_stride(self, ndim):
        interval_strides = [s for c, s in zip(self.coords, self.strides) if not c.is_pt]
        return interval_strides[ndim]


def AWinAlloc(name, sizes):
    assert all(isinstance(sz, LoopIR.expr) for sz in sizes)
    coords = [AWinCoord(is_pt=False, val=AInt(0)) for _ in sizes]
    strides = [A.Stride(name, i, T.stride, null_srcinfo()) for i in range(len(sizes))]

    # fill out constant strides where possible
    if len(strides) > 0:
        strides[-1] = AInt(1)
        sprod = 1
        for i in reversed(range(len(sizes) - 1)):
            sz = sizes[i + 1]
            if isinstance(sz, LoopIR.Const):
                sprod *= sz.val
                strides[i] = AInt(sprod)
            else:
                break

    return AWin(name, coords, strides)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Analysis Environments


@dataclass
class TupleBinding:
    names: list[Sym]
    rhs: Any


@dataclass
class BindingList:
    names: list[Sym]
    rhs: list[Any]


@dataclass
class WinBind:
    name: Sym
    rhs: AWin


class AEnv:
    def __init__(self, name=None, expr=None, addnames=False):
        # bindings stored in order of definition
        # reverse order from application
        self.bindings = []
        self.names = set()
        if name is None:
            return
        elif isinstance(name, Sym):
            if isinstance(expr, A.expr):
                self.bindings = [BindingList([name], [expr])]
            else:
                assert isinstance(expr, AWin)
                self.bindings = [WinBind(name, expr)]

            if addnames:
                self.names.add(name)
        elif type(name) is list:
            assert isinstance(expr, A.expr)
            assert type(expr.type) is tuple and len(name) == len(expr.type)
            self.bindings = [TupleBinding(name, expr)]

            if addnames:
                self.names = self.names.union(name)
        else:
            assert False, "bad case"

    def __str__(self):
        if len(self.bindings) == 0:
            return "[]"

        def bstr(bd):
            if isinstance(bd, TupleBinding):
                nm = ",".join([str(n) for n in bd.names])
                return f"[{nm} ↦ {bd.rhs}]"
            elif isinstance(bd, WinBind):
                return f"[{bd.name} ↦ {bd.rhs}]"
            elif isinstance(bd, BindingList):
                return "".join([f"[{x} ↦ {v}]" for x, v in zip(bd.names, bd.rhs)])
            else:
                assert False, "bad case"

        return "".join([bstr(bd) for bd in self.bindings])

    def __add__(lhs, rhs):
        assert isinstance(rhs, AEnv)
        result = AEnv()
        # compression optimization
        if len(lhs.bindings) > 0 and len(rhs.bindings) > 0:
            lb, rb = lhs.bindings[-1], rhs.bindings[0]
            if isinstance(lb, BindingList) and isinstance(rb, BindingList):
                mb = BindingList(lb.names + rb.names, lb.rhs + rb.rhs)
                result.bindings = lhs.bindings[:-1] + [mb] + rhs.bindings[1:]
                result.names = lhs.names.union(rhs.names)
                return result
        # otherwise
        result.bindings = lhs.bindings + rhs.bindings
        result.names = lhs.names.union(rhs.names)
        return result

    def __call__(self, arg):
        if isinstance(arg, A.expr):
            res = arg
            for bd in reversed(self.bindings):
                if isinstance(bd, TupleBinding):
                    res = ALetTuple(bd.names, bd.rhs, res)
                elif isinstance(bd, WinBind):
                    # bind strides...
                    res = ALetStride(bd.name, bd.rhs.strides, res)
                    # TODO: Probably need to introduce let bindings
                    # for coordinates in a way to get the values of globals
                    # at the site of windowing correct
                    # raise NotImplementedError("TODO: how to handle?")
                elif isinstance(bd, BindingList):
                    res = ALet(bd.names, bd.rhs, res)
                else:
                    assert False, "bad case"
            return res
        elif isinstance(arg, list) and len(arg) == 0:
            return []
        elif isinstance(arg, list) and isinstance(arg[0], E.eff):
            return [E.BindEnv(self)] + arg
        else:
            assert False, f"Cannot apply AEnv to {type(arg)}"

    def translate_win(self, winmap):
        winmap = winmap.copy()
        for bd in self.bindings:
            if isinstance(bd, WinBind):
                win = bd.rhs
                prewin = winmap.get(win.name, None)
                if prewin is not None:
                    win = prewin + win
                winmap[bd.name] = win
        return winmap

    # allow iteration through the names bound
    # exclude WinBind
    def names_types(self):
        nmtyp = OrderedDict()
        for bd in self.bindings:
            if isinstance(bd, TupleBinding):
                for n, rt in zip(bd.names, bd.rhs.type):
                    if n in self.names:
                        assert n not in nmtyp or nmtyp[n] == rt
                        nmtyp[n] = rt
            elif isinstance(bd, WinBind):
                pass  # ignore windows for this
            elif isinstance(bd, BindingList):
                for n, r in zip(bd.names, bd.rhs):
                    if n in self.names:
                        nmtyp[n] = r.type
        return nmtyp

    # we assume this packs up a scoping level
    # and therefore drop any WinBinds in the process
    def bind_to_copies(self):
        nmtyps = self.names_types()
        orig_vars = [A.Var(nm, typ, null_srcinfo()) for nm, typ in nmtyps.items()]
        copy_vars = [
            A.Var(nm.copy(), typ, null_srcinfo()) for nm, typ in nmtyps.items()
        ]
        varmap = OrderedDict(
            {nm: copyv for (nm, _), copyv in zip(nmtyps.items(), copy_vars)}
        )

        body = A.Tuple(orig_vars, tuple(e.type for e in orig_vars), null_srcinfo())
        # wrap the tuple in let-bindings representing this entire AEnv
        body = self(body)
        # then construct an AEnv that maps the new variable symbols
        # to these values
        new_env = AEnv([v.name for v in copy_vars], body, addnames=False)
        return varmap, new_env


def aenv_join(aenvs):
    aenv = AEnv()
    for a in aenvs:
        aenv = aenv + a
    return aenv


def AEnvPar(bind_dict, addnames=False):
    if len(bind_dict) == 0:
        return AEnv()
    names, rhs, types = [], [], []
    for nm, e in bind_dict.items():
        names.append(nm)
        rhs.append(e)
        types.append(e.type)
    rhs_tuple = A.Tuple(rhs, tuple(types), rhs[0].srcinfo)
    return AEnv(names, rhs_tuple, addnames=addnames)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Basic Global Dataflow Analysis


def filter_reals(e, changeset):
    def rec(e):
        if isinstance(e, A.ConstSym):
            if e.name in changeset:
                return A.Unk(e.type, e.srcinfo)
            else:
                return e
        elif isinstance(e, (A.Not, A.USub, A.ForAll, A.Exists, A.Definitely, A.Maybe)):
            return e.update(arg=rec(e.arg))
        elif isinstance(e, A.BinOp):
            return e.update(lhs=rec(e.lhs), rhs=rec(e.rhs))
        elif isinstance(e, A.LetStrides):
            return e.update(strides=[rec(s) for s in e.strides], body=rec(e.body))
        elif isinstance(e, A.Select):
            return e.update(cond=rec(e.cond), tcase=rec(e.tcase), fcase=rec(e.fcase))
        elif isinstance(e, A.Tuple):
            return e.update(args=[rec(a) for a in e.args])
        elif isinstance(e, A.LetTuple):
            return e.update(rhs=rec(e.rhs), body=rec(e.body))
        elif isinstance(e, A.Let):
            return e.update(rhs=[rec(a) for a in e.rhs], body=rec(e.body))
        else:
            return e

    return rec(e)


def lift_e(e):
    if isinstance(e, LoopIR.WindowExpr):

        def lift_w(w):
            if isinstance(w, LoopIR.Interval):
                return AWinCoord(is_pt=False, val=lift_e(w.lo))
            elif isinstance(w, LoopIR.Point):
                return AWinCoord(is_pt=True, val=lift_e(w.pt))
            else:
                assert False, f"bad w_access case"

        # default strides
        strides = [A.Stride(e.name, i, T.stride, e.srcinfo) for i in range(len(e.idx))]
        return AWin(e.name, [lift_w(w) for w in e.idx], strides)
    else:
        if e.type.is_indexable() or e.type.is_stridable() or e.type == T.bool:
            if isinstance(e, LoopIR.Read):
                assert len(e.idx) == 0
                return A.Var(e.name, e.type, e.srcinfo)
            elif isinstance(e, LoopIR.Const):
                return A.Const(e.val, e.type, e.srcinfo)
            elif isinstance(e, LoopIR.BinOp):
                return A.BinOp(e.op, lift_e(e.lhs), lift_e(e.rhs), e.type, e.srcinfo)
            elif isinstance(e, LoopIR.USub):
                return A.USub(lift_e(e.arg), e.type, e.srcinfo)
            elif isinstance(e, LoopIR.StrideExpr):
                return A.Stride(e.name, e.dim, e.type, e.srcinfo)
            elif isinstance(e, LoopIR.ReadConfig):
                globname = e.config._INTERNAL_sym(e.field)
                return A.Var(globname, e.type, e.srcinfo)
            else:
                f"bad case: {type(e)}"
        else:
            assert e.type.is_numeric()
            if e.type.is_real_scalar():
                if isinstance(e, LoopIR.Const):
                    return A.Const(e.val, e.type, e.srcinfo)
                elif isinstance(e, LoopIR.Read):
                    return A.ConstSym(e.name, e.type, e.srcinfo)
                elif isinstance(e, LoopIR.ReadConfig):
                    globname = e.config._INTERNAL_sym(e.field)
                    return A.Var(globname, e.type, e.srcinfo)

            return A.Unk(T.err, e.srcinfo)


def lift_es(es):
    return [lift_e(e) for e in es]


# Produce a set of AExprs which occur as right-hand-sides
# of config writes.
def possible_config_writes(stmts):
    class Find_RHS(LoopIR_Do):
        def __init__(self, stmts):
            # to collect the results in
            self.writes = dict()
            self.windows = [AEnv()]

            self.do_stmts(stmts)

        def add_write(self, config, field, rhs):
            rhs = lift_e(rhs)
            rhs = self.windows[-1](rhs).simplify()
            key = config._INTERNAL_sym(field)
            if key not in self.writes:
                self.writes[key] = set()
            self.writes[key].add(rhs)

        def result(self):
            return self.writes

        # remove any candidate expressions that use the given name
        def filter(self, name):
            for key in self.writes:
                exprs = self.writes[key]
                exprs = {e for e in exprs if name not in aeFV(e)}
                if len(exprs) == 0:
                    del self.writes[key]
                else:
                    self.writes[key] = exprs

        def push(self):
            self.windows.append(self.windows[-1])

        def pop(self):
            self.windows.pop()

        def do_s(self, s):
            if isinstance(s, LoopIR.WriteConfig):
                self.add_write(s.config, s.field, s.rhs)

            elif isinstance(s, LoopIR.If):
                self.push()
                self.do_stmts(s.body)
                self.pop()
                self.push()
                self.do_stmts(s.orelse)
                self.pop()

            elif isinstance(s, LoopIR.For):
                self.push()
                self.do_stmts(s.body)
                self.filter(s.iter)
                self.pop()

            elif isinstance(s, LoopIR.WindowStmt):
                # accumulate windowing expressions
                awin = lift_e(s.rhs)
                assert isinstance(awin, AWin)
                aenv = self.windows[-1] + AEnv(s.name, awin, addnames=False)
                self.windows[-1] = aenv

            elif isinstance(s, LoopIR.Call):
                call_env = call_bindings(s.args, s.f.args)
                sub_writes = Find_RHS(s.f.body).result()
                win_env = self.windows[-1]
                for key in sub_writes:
                    if key not in self.writes:
                        self.writes[key] = set()
                    for rhs in sub_writes[key]:
                        rhs = win_env(call_env(rhs)).simplify()
                        self.writes[key].add(rhs)

            super().do_s(s)

        # short-circuiting for efficiency
        def do_e(self, e):
            pass

        def do_t(self, t):
            pass

    return Find_RHS(stmts).result()


def globenv(stmts):
    aenvs = []
    for s in stmts:
        if isinstance(s, LoopIR.WriteConfig):
            globname = s.config._INTERNAL_sym(s.field)
            rhs = lift_e(s.rhs)
            aenvs.append(AEnv(globname, rhs, addnames=True))
        elif isinstance(s, LoopIR.WindowStmt):
            win = lift_e(s.rhs)
            aenvs.append(AEnv(s.name, win))
        elif isinstance(s, LoopIR.Alloc):
            win = AWinAlloc(s.name, s.type.shape())
            aenvs.append(AEnv(s.name, win))
        elif isinstance(s, LoopIR.If):
            # extract environments for each side of the branch
            body_env = globenv(s.body)
            else_env = globenv(s.orelse)
            # get the map from old to new names, and binding env
            bvarmap, benv = body_env.bind_to_copies()
            evarmap, eenv = else_env.bind_to_copies()
            aenvs += [benv, eenv]
            oldvars = {
                nm: A.Var(nm, newv.type, s.srcinfo)
                for nm, newv in chain(bvarmap.items(), evarmap.items())
            }

            # bind the condition so it isn't duplicated
            condsym = Sym("if_cond")
            condvar = A.Var(condsym, T.bool, s.cond.srcinfo)
            cond = lift_e(s.cond)
            aenvs.append(AEnv(condsym, cond))

            # We must now construct an environment that defines the
            # new value for variables `x` among the possibilities
            newbinds = dict()
            for nm, oldv in oldvars.items():
                # default to old-value
                tcase = bvarmap.get(nm, oldv)
                fcase = evarmap.get(nm, oldv)
                val = A.Select(condvar, tcase, fcase, oldv.type, s.srcinfo)
                newbinds[nm] = val
            aenvs.append(AEnvPar(newbinds, addnames=True))

        elif isinstance(s, LoopIR.For):
            # extract environment for the body and bind its
            # results via copies of the variables
            i, j = s.iter, s.iter.copy()
            body_env = globenv(s.body)
            bvarmap, benv = body_env.bind_to_copies()
            aenvs.append(benv)

            # bind the bounds condition so it isn't duplicated
            # non_empty   = AInt(0) < lift_e(s.hi)
            def fix(x, body_x):
                bds = AAnd(lift_e(s.lo) <= AInt(i), AInt(i) < lift_e(s.hi))
                no_change = AImplies(bds, AEq(body_x, x))
                return A.ForAll(i, no_change, T.bool, s.srcinfo)

            # extract possible RHS values for config-fields
            # cfg_writes = possible_config_writes([s])
            # for cfgfld in cfg_writes:
            #     pass

            # def fix_cfg(x, rhs, body_x, lower=0):
            #     bds = AAnd(AInt(lower) <= AInt(i), AInt(i) < lift_e(s.hi))
            #     is_assigned = A.Exists(
            #         i, AAnd(bds, AEq(body_x, rhs)), T.bool, s.srcinfo
            #     )
            #     no_change_or_assign = AOr(AEq(body_x, rhs), AEq(body_x, x))
            #     AImplies

            #     no_change = 23
            #     # A.Exists(i, is_assigned, T.bool, s.srcinfo)
            #     no_change = A
            #     no_change = AImplies(bds, AEq(body_x, x))
            #     return A.ForAll(i, no_change, T.bool, s.srcinfo)

            # define the value of variables due to the first iteration alone
            # def iter0(x,body_x):
            #    non_empty   = AInt(0) < lift_e(s.hi)
            #    is_iter0    = AEq(AInt(i), AInt(0))

            # optional attempt to have tricky conditions
            # body_j_env  = AEnv(i, AInt(j)) + body_env
            # j_bvarmap, j_benv = body_j_env.bind_to_copies()
            # aenvs.append(j_benv)
            # bds_j       = AAnd(AInt(0) <= AInt(j),
            #                   AInt(j) < lift_e(s.hi))
            # def same_after(body_x,body_j_x):
            #    consistent =  A.ForAll(i, AImplies(bds,
            #                    A.ForAll(j, AImplies(bds_j,
            #                                AEq(body_x, body_j_x)),
            #                             T.bool, s.srcinfo)),
            #                    T.bool, s.srcinfo)
            #    return AAnd(non_empty, consistent)

            # Now construct an environment that defines the new
            # value for variables `x` based on fixed-point conditions
            newbinds = dict()
            for nm, bvar in bvarmap.items():
                oldvar = A.Var(nm, bvar.type, s.srcinfo)
                val = A.Select(
                    fix(oldvar, bvar),
                    oldvar,
                    A.Unk(oldvar.type, s.srcinfo),
                    oldvar.type,
                    s.srcinfo,
                )

                # j_bvar  = j_bvarmap[nm]
                # oldvar  = A.Var(nm, bvar.type, s.srcinfo)
                # val     = A.Select(fix(oldvar, bvar),
                #                   oldvar,
                #                   A.Unk(oldvar.type, s.srcinfo),
                #                   #A.Select(same_after(bvar, j_bvar),
                #                   #         bvar,
                #                   #         A.Unk(oldvar.type, s.srcinfo),
                #                   #         oldvar.type, s.srcinfo),
                #                   oldvar.type, s.srcinfo)
                newbinds[nm] = val
            aenvs.append(AEnvPar(newbinds, addnames=True))

        elif isinstance(s, LoopIR.Call):
            sub_proc = get_simple_proc(s.f)
            sub_env = globenv_proc(sub_proc)
            call_env = call_bindings(s.args, sub_proc.args)
            aenvs += [call_env, sub_env]

        else:
            pass

    return aenv_join(aenvs)


def call_bindings(call_args, sig_args):
    assert len(call_args) == len(sig_args)
    aenvs = []
    for a, fa in zip(call_args, sig_args):
        if fa.type.is_numeric():
            if isinstance(a, LoopIR.WindowExpr):
                aenvs.append(AEnv(fa.name, lift_e(a)))
            elif isinstance(a, LoopIR.ReadConfig):
                aenvs.append(AEnv(fa.name, lift_e(a)))
            else:
                assert isinstance(a, LoopIR.Read)
                # determine whether or not this is a scalar or tensor
                # we must behave as if the tensors are being windowed
                # in order to avoid stupid errors in the lowering
                shape = fa.type.shape()
                if len(shape) > 0:
                    aenvs.append(AEnv(fa.name, AWinAlloc(a.name, shape)))
                else:
                    aenvs.append(AEnv(fa.name, AWin(a.name, [], [])))
        else:
            aenvs.append(AEnv(fa.name, lift_e(a)))
    return aenv_join(aenvs)


_globenv_proc_cache = dict()


def globenv_proc(proc):
    if proc not in _globenv_proc_cache:
        _globenv_proc_cache[proc] = globenv(proc.body)
    return _globenv_proc_cache[proc]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Location Sets


def is_type_bound(val):
    if isinstance(val, (tuple, LoopIR.type)):
        return val
    raise ValidationError(Union[tuple, LoopIR.type], type(val))


LS = ADT(
    """
module LocSet {
    locset  = Empty     ()
            | Point     ( sym name,   aexpr* coords,  type? type )
            | WholeBuf  ( sym name,   int ndim )
            | Union     ( locset lhs, locset rhs )
            | Isct      ( locset lhs, locset rhs )
            | Diff      ( locset lhs, locset rhs )
            | BigUnion  ( sym   name, locset arg )
            | Filter    ( aexpr cond, locset arg )
            | LetEnv    ( aenv   env, locset arg )
            | HideAlloc ( sym   name, locset arg )
} """,
    {
        "sym": Sym,
        "aexpr": A.expr,
        "aenv": AEnv,
        "type": is_type_bound,
    },
)


ls_prec = {
    #
    "bigunion": 10,
    #
    "union": 20,
    "isct": 20,
    "diff": 20,
}


def LUnion(lhs, rhs):
    if isinstance(lhs, LS.Empty):
        return rhs
    elif isinstance(rhs, LS.Empty):
        return lhs
    else:
        return LS.Union(lhs, rhs)


def LIsct(lhs, rhs):
    if isinstance(lhs, LS.Empty) or isinstance(rhs, LS.Empty):
        return LS.Empty()
    else:
        return LS.Isct(lhs, rhs)


def LDiff(lhs, rhs):
    if isinstance(lhs, LS.Empty):
        return LS.Empty()
    elif isinstance(rhs, LS.Empty):
        return lhs
    else:
        return LS.Diff(lhs, rhs)


def LBigUnion(name, arg):
    if isinstance(arg, LS.Empty):
        return LS.Empty()
    else:
        return LS.BigUnion(name, arg)


def LFilter(cond, arg):
    if isinstance(arg, LS.Empty):
        return LS.Empty()
    else:
        return LS.Filter(cond, arg)


def LLetEnv(env, arg):
    if isinstance(arg, LS.Empty):
        return LS.Empty()
    else:
        return LS.LetEnv(env, arg)


def LHideAlloc(name, arg):
    if isinstance(arg, LS.Empty):
        return LS.Empty()
    else:
        return LS.HideAlloc(name, arg)


# pretty printing
def _lsstr(ls, prec=0):
    if isinstance(ls, LS.Empty):
        return "∅"
    elif isinstance(ls, LS.Point):
        coords = ",".join([str(a) for a in ls.coords])
        typ = f":{ls.type}" if ls.type else ""
        return f"{{({ls.name},{coords}{typ})}}"
    elif isinstance(ls, LS.WholeBuf):
        return f"{{{ls.name}|{ls.ndim}}}"
    elif isinstance(ls, (LS.Union, LS.Isct, LS.Diff)):
        if isinstance(ls, LS.Union):
            op, local_prec = "∪", ls_prec["union"]
        elif isinstance(ls, LS.Isct):
            op, local_prec = "∩", ls_prec["isct"]
        else:
            op, local_prec = "-", ls_prec["diff"]
        lhs = _lsstr(ls.lhs, prec=local_prec)
        rhs = _lsstr(ls.rhs, prec=local_prec + 1)
        if local_prec < prec:
            return f"({lhs} {op} {rhs})"
        else:
            return f"{lhs} {op} {rhs}"
    elif isinstance(ls, LS.BigUnion):
        arg = _lsstr(ls.arg, prec=30)
        return f"∪{ls.name}.{arg}"
    elif isinstance(ls, LS.Filter):
        arg = _lsstr(ls.arg)
        return f"filter({ls.cond},{arg})"
    elif isinstance(ls, LS.LetEnv):
        arg = _lsstr(ls.arg)
        return f"let({ls.env},{arg})"
    elif isinstance(ls, LS.HideAlloc):
        arg = _lsstr(ls.arg)
        return f"alloc({ls.name},{arg})"
    else:
        assert False, f"bad case: {type(ls)}"


@extclass(LS.locset)
def __str__(ls):
    return _lsstr(ls)


def is_elem(pt, ls, win_map=None, alloc_masks=None):
    # default arg
    win_map = win_map or dict()
    alloc_masks = alloc_masks or []

    if isinstance(ls, LS.Empty):
        return ABool(False)
    elif isinstance(ls, LS.Point):
        lspt = APoint(ls.name, ls.coords, ls.type)
        if ls.name in win_map:
            lspt = win_map[ls.name](lspt)

        if pt.name != lspt.name or lspt.name in alloc_masks:
            return ABool(False)
        assert len(pt.coords) == len(lspt.coords)
        eqs = [AEq(q, p) for q, p in zip(pt.coords, lspt.coords)]
        return AAnd(*eqs)
    elif isinstance(ls, LS.WholeBuf):
        bufname = ls.name
        if bufname in win_map:
            bufname = win_map[bufname].name
        return ABool(pt.name == bufname)
    elif isinstance(ls, (LS.Union, LS.Isct, LS.Diff)):
        lhs = is_elem(pt, ls.lhs, win_map=win_map, alloc_masks=alloc_masks)
        rhs = is_elem(pt, ls.rhs, win_map=win_map, alloc_masks=alloc_masks)
        if isinstance(ls, LS.Union):
            return AOr(lhs, rhs)
        elif isinstance(ls, LS.Isct):
            return AAnd(lhs, rhs)
        elif isinstance(ls, LS.Diff):
            return AAnd(lhs, ANot(rhs))
        else:
            assert False
    elif isinstance(ls, LS.BigUnion):
        arg = is_elem(pt, ls.arg, win_map=win_map, alloc_masks=alloc_masks)
        return A.Exists(ls.name, arg, T.bool, null_srcinfo())
    elif isinstance(ls, LS.Filter):
        arg = is_elem(pt, ls.arg, win_map=win_map, alloc_masks=alloc_masks)
        return AAnd(ls.cond, arg)
    elif isinstance(ls, LS.LetEnv):
        win_map = ls.env.translate_win(win_map)
        arg = is_elem(pt, ls.arg, win_map=win_map, alloc_masks=alloc_masks)
        # wrap binding around the expression
        return ls.env(arg)
    elif isinstance(ls, LS.HideAlloc):
        alloc_masks.append(ls.name)
        res = is_elem(pt, ls.arg, win_map=win_map, alloc_masks=alloc_masks)
        alloc_masks.pop()
        return res
    else:
        assert False, f"bad case: {type(ls)}"


def get_point_exprs(ls):
    all_bufs = dict()

    def _collect_buf(ls, win_map, alloc_masks):
        # default arg
        win_map = win_map or dict()
        alloc_masks = alloc_masks or []

        if isinstance(ls, LS.Empty):
            pass
        elif isinstance(ls, LS.Point):
            lspt = APoint(ls.name, ls.coords, ls.type)
            if ls.name in win_map:
                lspt = win_map[ls.name](lspt)

            if lspt.name not in alloc_masks:
                all_bufs[lspt.name] = (len(lspt.coords), ls.type)
        elif isinstance(ls, LS.WholeBuf):
            bufname = ls.name
            if bufname in win_map:
                bufname = win_map[bufname].name
            if bufname not in alloc_masks:
                all_bufs[bufname] = (ls.ndim, T.R)
        elif isinstance(ls, (LS.Union, LS.Isct, LS.Diff)):
            _collect_buf(ls.lhs, win_map, alloc_masks)
            _collect_buf(ls.rhs, win_map, alloc_masks)
        elif isinstance(ls, (LS.BigUnion, LS.Filter)):
            _collect_buf(ls.arg, win_map, alloc_masks)
        elif isinstance(ls, LS.LetEnv):
            win_map = ls.env.translate_win(win_map)
            _collect_buf(ls.arg, win_map, alloc_masks)
        elif isinstance(ls, LS.HideAlloc):
            alloc_masks.append(ls.name)
            _collect_buf(ls.arg, win_map, alloc_masks)
            alloc_masks.pop()

    _collect_buf(ls, dict(), [])

    points = [
        APoint(nm, [AInt(Sym(f"i{i}")) for i in range(ndim)], typ)
        for nm, (ndim, typ) in all_bufs.items()
    ]
    return points


def is_empty(ls):
    points = get_point_exprs(ls)

    terms = []
    for pt in points:
        term = A.Not(is_elem(pt, ls), T.bool, null_srcinfo())
        for iv in reversed(pt.coords):
            term = A.ForAll(iv.name, term, T.bool, null_srcinfo())
        terms.append(term)

    return AAnd(*terms)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Effects


E = ADT(
    """
module EffectsNew {
    eff  = Empty        ()
         | Guard        ( aexpr cond, eff* body )
         | Loop         ( sym name,   eff* body )
         --
         | BindEnv      ( aenv env )
         --
         | GlobalRead   ( sym name, type   type   )
         | GlobalWrite  ( sym name, type   type,  aexpr? rhs )
         | Read         ( sym name, aexpr* coords )
         | Write        ( sym name, aexpr* coords )
         | Reduce       ( sym name, aexpr* coords )
         --
         | Alloc        ( sym name, int ndim )
} """,
    {
        "sym": Sym,
        "aexpr": A.expr,
        "aenv": AEnv,
        "type": is_type_bound,
        # 'srcinfo': lambda x: isinstance(x, SrcInfo),
    },
)

# pretty printing
def _effstr(eff, tab=""):
    if isinstance(eff, E.Empty):
        return f"{tab}∅"
    elif isinstance(eff, (E.Guard, E.Loop)):
        lines = [_effstr(e, tab + "  ") for e in eff.body]
        if isinstance(eff, E.Guard):
            lines = [f"{tab}Guard({eff.cond})"] + lines
        else:
            lines = [f"{tab}Loop({eff.name})"] + lines
        return "\n".join(lines)
    elif isinstance(eff, E.BindEnv):
        return f"{tab}{eff.env}"
    elif isinstance(eff, (E.GlobalRead, E.GlobalWrite)):
        nm = "GlobalRead" if isinstance(eff, E.GlobalRead) else "GlobalWrite"
        rhs = "" if isinstance(eff, E.GlobalRead) or not eff.rhs else f",{eff.rhs}"
        return f"{tab}{nm}({eff.name},{eff.type}{rhs})"
    elif isinstance(eff, (E.Read, E.Write, E.Reduce)):
        nm = (
            "Read"
            if isinstance(eff, E.Read)
            else "Write"
            if isinstance(eff, E.Write)
            else "Reduce"
        )
        coords = ",".join([str(a) for a in eff.coords])
        return f"{tab}{nm}({eff.name},{coords})"
    elif isinstance(eff, E.Alloc):
        return f"{tab}Alloc({eff.name}|{eff.ndim})"
    else:
        assert False, f"bad case: {type(eff)}"


@extclass(E.eff)
def __str__(eff):
    return _effstr(eff)


# Here are codes for different location sets...
class ES(Enum):
    READ_G = 1
    WRITE_G = 2
    READ_H = 3
    WRITE_H = 4
    PRE_REDUCE = 5

    _DERIVED = 6

    READ_ALL = 7
    WRITE_ALL = 8
    REDUCE = 9
    ALL = 10
    ALL_H = 11
    MODIFY = 12
    MODIFY_H = 13
    READ_WRITE = 14

    ALLOC = 15


def get_basic_locsets(effs):
    RG = LS.Empty()
    WG = LS.Empty()
    RH = LS.Empty()
    WH = LS.Empty()
    Red = LS.Empty()
    Alc = LS.Empty()
    for eff in reversed(effs):
        if isinstance(eff, E.Empty):
            pass

        elif isinstance(eff, E.GlobalRead):
            ls1 = LS.Point(eff.name, [], eff.type)
            RG = LUnion(ls1, RG)
        elif isinstance(eff, E.GlobalWrite):
            ls1 = LS.Point(eff.name, [], eff.type)
            RG = LDiff(RG, ls1)
            WG = LUnion(ls1, WG)
        elif isinstance(eff, E.Read):
            ls1 = LS.Point(eff.name, eff.coords, None)
            RH = LUnion(ls1, RH)
        elif isinstance(eff, E.Write):
            ls1 = LS.Point(eff.name, eff.coords, None)
            RH = LDiff(RH, ls1)
            WH = LUnion(ls1, WH)
        elif isinstance(eff, E.Reduce):
            ls1 = LS.Point(eff.name, eff.coords, None)
            Red = LUnion(ls1, Red)

        elif isinstance(eff, E.Alloc):
            RG = LHideAlloc(eff.name, RG)
            WG = LHideAlloc(eff.name, WG)
            RH = LHideAlloc(eff.name, RH)
            WH = LHideAlloc(eff.name, WH)
            Red = LHideAlloc(eff.name, Red)
            Alc = LUnion(Alc, LS.WholeBuf(eff.name, eff.ndim))

        elif isinstance(eff, E.BindEnv):
            RG = LLetEnv(eff.env, RG)
            WG = LLetEnv(eff.env, WG)
            RH = LLetEnv(eff.env, RH)
            WH = LLetEnv(eff.env, WH)
            Red = LLetEnv(eff.env, Red)

        elif isinstance(eff, (E.Guard, E.Loop)):
            bodyLs = get_basic_locsets(eff.body)
            if isinstance(eff, E.Guard):
                bodyLs = tuple(LFilter(eff.cond, Ls) for Ls in bodyLs)
            else:
                bodyLs = tuple(LBigUnion(eff.name, Ls) for Ls in bodyLs)
            bRG, bWG, bRH, bWH, bRed, bAlc = bodyLs

            # now do the full interaction updates...
            RG = LUnion(bRG, LDiff(RG, bWG))
            WG = LUnion(bWG, WG)
            RH = LUnion(bRH, LDiff(RH, bWH))
            WH = LUnion(bWH, WH)
            Red = LUnion(bRed, Red)

        else:
            assert False, f"bad case: {type(eff)}"

    return (RG, WG, RH, WH, Red, Alc)


def getsets(codes, effs):
    RG, WG, RH, WH, preRed, Alc = get_basic_locsets(effs)
    RAll = LUnion(RG, RH)
    WAll = LUnion(WG, WH)
    Red = LDiff(preRed, WH)
    Mod = LUnion(WAll, preRed)
    ModH = LUnion(WH, preRed)
    RW = LUnion(RAll, WAll)
    All = LUnion(RW, preRed)
    AllH = LUnion(RH, ModH)

    def get_code(code):
        if code == ES.READ_G:
            return RG
        elif code == ES.WRITE_G:
            return WG
        elif code == ES.READ_H:
            return RH
        elif code == ES.WRITE_H:
            return WH
        elif code == ES.PRE_REDUCE:
            return preRed
        elif code == ES.READ_ALL:
            return RAll
        elif code == ES.WRITE_ALL:
            return WAll
        elif code == ES.REDUCE:
            return Red
        elif code == ES.ALL:
            return All
        elif code == ES.ALL_H:
            return AllH
        elif code == ES.MODIFY:
            return Mod
        elif code == ES.MODIFY_H:
            return ModH
        elif code == ES.READ_WRITE:
            return RW
        elif code == ES.ALLOC:
            return Alc
        else:
            assert False, f"bad case: {code}"

    return [get_code(c) for c in codes]


def get_changing_globset(env):
    """Computes a Location Set from an environment, corresponding
    to all globals / configuration-variables whose value is
    changed by that environment"""
    varmap, env = env.bind_to_copies()
    aenvs = [env]

    def change(x_old, x_new):
        return ANot(AEq(x_old, x_new))

    locs = LS.Empty()
    for name, newvar in varmap.items():
        oldvar = A.Var(name, newvar.type, newvar.srcinfo)
        cfgloc = LFilter(change(oldvar, newvar), LS.Point(name, [], newvar.type))
        locs = LUnion(locs, cfgloc)
    locs = LLetEnv(env, locs)
    return locs


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Extraction of Effects from programs


def window_effs(e):
    eff_access = []
    syms = {}
    for i, w in enumerate(e.idx):
        if isinstance(w, LoopIR.Interval):
            syms[i] = Sym(f"EXO_EFFECTS_WINDOW_TEMP_INDEX_{i}")
            eff_access.append(lift_e(syms[i]))
        else:
            eff_access.append(lift_e(w.pt))

    eff = [E.Read(e.name, [idx for idx in eff_access])]

    for i, w in enumerate(e.idx):
        if isinstance(w, LoopIR.Interval):
            sym = syms[i]
            bds = AAnd(lift_e(w.lo) <= AInt(sym), AInt(sym) < lift_e(w.hi))
            eff = E.Loop(syms[i][E.Guard(bds, eff)])

    return eff


def expr_effs(e):
    if isinstance(e, LoopIR.Read):
        if e.type.is_numeric():
            return [E.Read(e.name, lift_es(e.idx))]
        else:
            return []
    elif isinstance(e, LoopIR.Const):
        return []
    elif isinstance(e, LoopIR.USub):
        return expr_effs(e.arg)
    elif isinstance(e, LoopIR.BinOp):
        return expr_effs(e.lhs) + expr_effs(e.rhs)
    elif isinstance(e, LoopIR.BuiltIn):
        return list_expr_effs(e.args)
    elif isinstance(e, LoopIR.WindowExpr):

        def w_effs(w):
            if isinstance(w, LoopIR.Interval):
                return expr_effs(w.lo) + expr_effs(w.hi)
            else:
                return expr_effs(w.pt)

        return [eff for w in e.idx for eff in w_effs(w)]
    elif isinstance(e, LoopIR.StrideExpr):
        return []
    elif isinstance(e, LoopIR.ReadConfig):
        globname = e.config._INTERNAL_sym(e.field)
        return [E.GlobalRead(globname, e.config.lookup_type(e.field))]
    else:
        assert False, f"bad case: {type(e)}"


def list_expr_effs(es):
    return [eff for e in es for eff in expr_effs(e)]


def stmts_effs(stmts):
    effs = []
    for s in stmts:
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            EConstruct = E.Write if isinstance(s, LoopIR.Assign) else E.Reduce
            effs += list_expr_effs(s.idx)
            effs += expr_effs(s.rhs)
            effs.append(EConstruct(s.name, lift_es(s.idx)))
        elif isinstance(s, LoopIR.WriteConfig):
            effs += expr_effs(s.rhs)
            globname = s.config._INTERNAL_sym(s.field)
            effs.append(
                E.GlobalWrite(globname, s.config.lookup_type(s.field), lift_e(s.rhs))
            )
        elif isinstance(s, LoopIR.If):
            effs += expr_effs(s.cond)
            effs += [
                E.Guard(lift_e(s.cond), stmts_effs(s.body)),
                E.Guard(ANot(lift_e(s.cond)), stmts_effs(s.orelse)),
            ]
        elif isinstance(s, LoopIR.For):
            effs += expr_effs(s.lo)
            effs += expr_effs(s.hi)
            bds = AAnd(lift_e(s.lo) <= AInt(s.iter), AInt(s.iter) < lift_e(s.hi))
            # we must prefix the body with the loop-invariant dataflow
            # analysis of the loop, since that is the only precondition
            # we are sound in assuming for global values in the loop body
            body = [E.BindEnv(globenv([s]))] + stmts_effs(s.body)
            effs += [E.Loop(s.iter, [E.Guard(bds, body)])]
        elif isinstance(s, LoopIR.Call):
            # must filter out arguments that are simply
            # Read of a numeric buffer, since those arguments are
            # passed by reference, not by reading and passing a value.
            # Must also filter out numeric ReadConfigs, since those are
            # likewise being passed by reference, not being accessed
            for fa, a in zip(s.f.args, s.args):
                if fa.type.is_numeric() and isinstance(a, LoopIR.Read):
                    pass  # this is the case we want to skip
                elif fa.type.is_numeric() and isinstance(a, LoopIR.ReadConfig):
                    pass
                else:
                    effs += expr_effs(a)
            sub_proc = get_simple_proc(s.f)
            call_env = call_bindings(s.args, sub_proc.args)
            effs += [E.BindEnv(call_env)]
            effs += proc_effs(sub_proc)
        elif isinstance(s, LoopIR.Alloc):
            if isinstance(s.type, T.Tensor):
                effs += list_expr_effs(s.type.hi)
            effs += [E.Alloc(s.name, len(s.type.shape()))]
        elif isinstance(s, LoopIR.WindowStmt):
            effs += expr_effs(s.rhs)
        elif isinstance(s, (LoopIR.Free, LoopIR.Pass)):
            pass
        else:
            assert False, f"bad case: {type(s)}"

        # secondly, insert global value modifications into
        # the sequence of effects
        effs.append(E.BindEnv(globenv([s])))

    return effs


_proc_effs_cache = dict()


def proc_effs(proc):
    if proc not in _proc_effs_cache:
        _proc_effs_cache[proc] = stmts_effs(proc.body)
    return _proc_effs_cache[proc]
    raise NotImplementedError("TODO")


_proc_changeset_cache = dict()


def proc_changing_scalars(proc):
    if proc not in _proc_changeset_cache:
        _proc_changeset_cache[proc] = get_changing_scalars(proc.body)
    return _proc_changeset_cache[proc]


def get_changing_scalars(stmts, changeset=None, aliases=None):
    aliases = aliases or dict()
    changeset = changeset or set()

    def add_name(name):
        changeset.add(name)
        while name in aliases:
            name = aliases[name]
            changeset.add(name)

    for s in stmts:
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if len(s.idx) == 0:
                add_name(s.name)
        elif isinstance(s, LoopIR.If):
            get_changing_scalars(s.body, changeset, aliases)
            get_changing_scalars(s.orelse, changeset, aliases)
        elif isinstance(s, LoopIR.For):
            get_changing_scalars(s.body, changeset, aliases)
        elif isinstance(s, LoopIR.Call):
            for fa, a in zip(s.f.args, s.args):
                if fa.type.is_numeric():
                    if isinstance(a, (LoopIR.Read, LoopIR.WindowExpr)):
                        aliases[fa.name] = a.name
                    else:
                        assert isinstance(a, LoopIR.ReadConfig)
                        # ignore these; check that they don't matter elsewhere
            pchgs = proc_changing_scalars(s.f)
            for nm in pchgs:
                add_name(nm)
        elif isinstance(s, LoopIR.WindowStmt):
            aliases[s.name] = s.rhs.name
        else:
            pass

    return changeset


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Context Processing


class ContextExtraction:
    def __init__(self, proc, stmts):
        self.proc = proc
        self.stmts = stmts

    def get_control_predicate(self):
        assumed = AAnd(*[lift_e(p) for p in self.proc.preds])
        # collect assumptions that size arguments are positive
        pos_sizes = AAnd(
            *[AInt(a.name) > AInt(0) for a in self.proc.args if a.type == T.size]
        )
        ctrlp = self.ctrlp_stmts(self.proc.body)
        return AAnd(assumed, pos_sizes, ctrlp)

    def get_pre_globenv(self):
        return self.preenv_stmts(self.proc.body)

    def get_posteffs(self):
        a = self.posteff_stmts(self.proc.body)
        if len(self.proc.preds) > 0:
            assumed = AAnd(*[lift_e(p) for p in self.proc.preds])
            a = [E.Guard(assumed, a)]
        return a

    def ctrlp_stmts(self, stmts):
        for i, s in enumerate(stmts):
            if s is self.stmts[0]:
                return ABool(True)
            else:
                p = self.ctrlp_s(s)
                if p is not None:  # found the focused sub-tree
                    G = globenv(stmts[0:i])
                    return G(p)
        return None

    def ctrlp_s(self, s):
        if isinstance(s, LoopIR.If):
            p = self.ctrlp_stmts(s.body)
            if p is not None:
                return AAnd(lift_e(s.cond), p)
            p = self.ctrlp_stmts(s.orelse)
            if p is not None:
                return AAnd(ANot(lift_e(s.cond)), p)
            return None
        elif isinstance(s, LoopIR.For):
            p = self.ctrlp_stmts(s.body)
            if p is not None:
                G = self.loop_preenv(s)
                bds = AAnd(lift_e(s.lo) <= AInt(s.iter), AInt(s.iter) < lift_e(s.hi))
                return AAnd(bds, G(p))
            return None
        else:
            return None

    def preenv_stmts(self, stmts):
        for i, s in enumerate(stmts):
            if s is self.stmts[0]:
                preG = AEnv()
            else:
                preG = self.preenv_s(s)

            if preG is not None:  # found the focused sub-tree
                G = globenv(stmts[0:i])
                return G + preG
        return None

    def preenv_s(self, s):
        if isinstance(s, LoopIR.If):
            preG = self.preenv_stmts(s.body)
            if preG is not None:
                return preG
            preG = self.preenv_stmts(s.orelse)
            if preG is not None:
                return preG
            return None
        elif isinstance(s, LoopIR.For):
            preG = self.preenv_stmts(s.body)
            if preG is not None:
                G = self.loop_preenv(s)
                return G + preG
            return None
        else:
            return None

    def posteff_stmts(self, stmts):
        for i, s in enumerate(stmts):
            if s is self.stmts[0]:
                effs = [E.BindEnv(globenv(self.stmts))]
                post_stmts = stmts[i + len(self.stmts) :]
            else:
                effs = self.posteff_s(s)
                post_stmts = stmts[i + 1 :]

            if effs is not None:  # found the focused sub-tree
                preG = globenv(stmts[0:i])
                return [E.BindEnv(preG)] + effs + stmts_effs(post_stmts)
        return None

    def posteff_s(self, s):
        if isinstance(s, LoopIR.If):
            effs = self.posteff_stmts(s.body)
            if effs is not None:
                return [E.Guard(lift_e(s.cond), effs)]
            effs = self.posteff_stmts(s.orelse)
            if effs is not None:
                return [E.Guard(ANot(lift_e(s.cond)), effs)]
            return None
        elif isinstance(s, LoopIR.For):
            body = self.posteff_stmts(s.body)
            if body is None:
                return None
            else:
                orig_lo = lift_e(s.lo)
                lo_sym = Sym("lo_tmp")
                lo_env = AEnv(lo_sym, orig_lo)

                orig_hi = lift_e(s.hi)
                hi_sym = Sym("hi_tmp")
                hi_env = AEnv(hi_sym, orig_hi)

                bds = AAnd(AInt(lo_sym) <= AInt(s.iter), AInt(s.iter) < AInt(hi_sym))
                bds_sym = Sym("bds_tmp")
                bds_env = lo_env + hi_env + AEnv(bds_sym, bds)

                G = self.loop_preenv(s)

                guard_body = LoopIR.If(
                    LoopIR.Read(bds_sym, [], T.bool, s.srcinfo),
                    s.body,
                    [],
                    s.srcinfo,
                )
                G_body = globenv([guard_body])
                return [
                    E.BindEnv(bds_env),
                    E.BindEnv(G),
                    E.Guard(ABool(bds_sym), body),
                    E.BindEnv(G_body),
                ] + self.loop_posteff(s, LoopIR.Read(hi_sym, [], T.index, s.srcinfo))
        else:
            return None

    def loop_preenv(self, s):
        assert isinstance(s, LoopIR.For)
        old_i = LoopIR.Read(s.iter, [], T.index, s.srcinfo)
        new_i = LoopIR.Read(s.iter.copy(), [], T.index, s.srcinfo)
        pre_body = SubstArgs(s.body, {s.iter: new_i}).result()
        pre_loop = LoopIR.For(new_i.name, s.lo, old_i, pre_body, s.loop_mode, s.srcinfo)
        return globenv([pre_loop])

    def loop_posteff(self, s, hi):
        # want to generate a loop
        #   for x' in seq(x+1, hi): s
        # but instead generate
        #   for x' in seq(0, hi-(x+1)): [x' -> x' + (x+1)]s
        assert isinstance(s, LoopIR.For)
        old_i = LoopIR.Read(s.iter, [], T.index, s.srcinfo)
        new_i = LoopIR.Read(s.iter.copy(), [], T.index, s.srcinfo)
        old_plus1 = LoopIR.BinOp(
            "+", old_i, LoopIR.Const(1, T.int, s.srcinfo), T.index, s.srcinfo
        )
        post_body = SubstArgs(s.body, {s.iter: new_i}).result()
        post_loop = LoopIR.For(
            new_i.name, old_plus1, hi, post_body, s.loop_mode, s.srcinfo
        )
        return stmts_effs([post_loop])


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Common Predicates


def Disjoint_Memory(a1, a2):
    W1, Red1, All1 = getsets([ES.WRITE_ALL, ES.REDUCE, ES.ALL], a1)
    W2, Red2, All2 = getsets([ES.WRITE_ALL, ES.REDUCE, ES.ALL], a2)

    pred = AAnd(
        ADef(is_empty(LIsct(W1, All2))),
        ADef(is_empty(LIsct(W2, All1))),
        ADef(is_empty(LIsct(Red1, All2))),
        ADef(is_empty(LIsct(Red2, All1))),
    )

    return pred


def Commutes(a1, a2):
    W1, R1, Red1, All1 = getsets([ES.WRITE_ALL, ES.READ_ALL, ES.REDUCE, ES.ALL], a1)
    W2, R2, Red2, All2 = getsets([ES.WRITE_ALL, ES.READ_ALL, ES.REDUCE, ES.ALL], a2)

    pred = AAnd(
        ADef(is_empty(LIsct(W1, All2))),
        ADef(is_empty(LIsct(W2, All1))),
        ADef(is_empty(LIsct(Red1, R2))),
        ADef(is_empty(LIsct(Red2, R1))),
    )

    return pred


def Commutes_Fissioning(a1, a2, aenv1, aenv2, a1_no_loop_var=False):
    W1, R1, RG1, Red1, All1 = getsets(
        [ES.WRITE_H, ES.READ_H, ES.READ_G, ES.REDUCE, ES.ALL_H], a1
    )
    W2, R2, RG2, Red2, All2 = getsets(
        [ES.WRITE_H, ES.READ_H, ES.READ_G, ES.REDUCE, ES.ALL_H], a2
    )
    WG1 = get_changing_globset(aenv1)
    WG2 = get_changing_globset(aenv2)
    """
    print("CHANGING GLOB SET")
    print(WG1)
    print(WG2)
    print("W1 R1 RG1 Red1 All1")
    print(W1)
    print(R1)
    print(RG1)
    print(Red1)
    print(All1)
    print("W2 R2 RG2 Red2 All2")
    print(W2)
    print(R2)
    print(RG2)
    print(Red2)
    print(All2)
    """

    write_commute12 = ADef(is_empty(LIsct(W1, All2)))
    if a1_no_loop_var:
        # a1 does not vary syntactically between loop iterations,
        # so under the following conditions, we can assume that
        # a1 is idempotent
        #   - a1 contains no reductions
        a1_idempotent = AAnd(
            ADef(is_empty(Red1)),
            #   - a1 does not depend on (read) any modified values, namely
            #       + any values modified by a2 (accounted for below)
            #       + any heap value written by a1 (W1)
            ADef(is_empty(LIsct(W1, R1))),
            #       + any global value changed by a1 (WG1)
            ADef(is_empty(LIsct(WG1, RG1))),
        )
        # In this case, a1 being idempotent is sufficient as an
        # alternative to proving that a1 doesn't write anything
        # read by a2; since idempotency ensures that
        # a2 will always read the same values regardless of commuting
        write_commute12 = AOr(write_commute12, a1_idempotent)

    pred = AAnd(
        write_commute12,
        ADef(is_empty(LIsct(W2, All1))),
        ADef(is_empty(LIsct(Red1, R2))),
        ADef(is_empty(LIsct(Red2, R1))),
        ADef(is_empty(LIsct(WG1, RG2))),
        ADef(is_empty(LIsct(WG2, RG1))),
    )

    return pred


def AllocCommutes(a1, a2):

    Alc1, All1 = getsets([ES.ALLOC, ES.ALL], a1)
    Alc2, All2 = getsets([ES.ALLOC, ES.ALL], a2)
    pred = AAnd(ADef(is_empty(LIsct(Alc1, All2))), ADef(is_empty(LIsct(Alc2, All1))))
    return pred


def Shadows(a1, a2):
    Mod1 = getsets([ES.MODIFY], a1)[0]
    Rd2, Wr2, Red2 = getsets([ES.READ_ALL, ES.WRITE_ALL, ES.REDUCE], a2)

    # predicate via constituent conditions
    mod_is_unread = ADef(is_empty(LIsct(Mod1, LUnion(Rd2, Red2))))
    mod_is_shadowed = ADef(is_empty(LDiff(Mod1, Wr2)))

    pred = AAnd(mod_is_unread, mod_is_shadowed)
    return pred


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Scheduling Checks

import inspect
import textwrap
from .API_types import ProcedureBase


class SchedulingError(Exception):
    def __init__(self, message, **kwargs):
        ops = self._get_scheduling_ops()
        # TODO: include outer ops in message
        message = f"{ops[0]}: {message}"
        for name, blob in kwargs.items():
            message += self._format_named_blob(name.title(), blob)
        super().__init__(message)

    @staticmethod
    def _format_named_blob(name, blob):
        blob = str(blob).rstrip()
        n = len(name) + 2
        blob = textwrap.indent(blob, " " * n).strip()
        return f"\n{name}: " + blob

    @staticmethod
    def _get_scheduling_ops():
        ops = []
        for frame in inspect.stack():
            if obj := frame[0].f_locals.get("self"):
                fn = frame.function
                if isinstance(obj, ProcedureBase) and not fn.startswith("_"):
                    ops.append(fn)
        if not ops:
            ops = ["<<<unknown directive>>>"]
        return ops


def loop_globenv(i, lo_expr, hi_expr, body):
    assert isinstance(lo_expr, LoopIR.expr)
    assert isinstance(hi_expr, LoopIR.expr)

    loop = [LoopIR.For(i, lo_expr, hi_expr, body, LoopIR.Seq(), null_srcinfo())]
    return globenv(loop)


def Check_ReorderStmts(proc, s1, s2):
    ctxt = ContextExtraction(proc, [s1, s2])

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(p))

    a1 = stmts_effs([s1])
    a2 = stmts_effs([s2])

    pred = G(AAnd(Commutes(a1, a2), AllocCommutes(a1, a2)))
    is_ok = slv.verify(pred)
    slv.pop()
    if not is_ok:
        raise SchedulingError(
            f"Statements at {s1.srcinfo} and {s2.srcinfo} do not commute."
        )


def Check_ReorderLoops(proc, s):
    ctxt = ContextExtraction(proc, [s])

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(p))

    assert len(s.body) == 1
    assert isinstance(s.body[0], LoopIR.For)
    x_loop = s
    y_loop = s.body[0]
    body = y_loop.body
    x = x_loop.iter
    y = y_loop.iter
    x2 = x.copy()
    y2 = y.copy()
    subenv = {
        x: LoopIR.Read(x2, [], T.index, null_srcinfo()),
        y: LoopIR.Read(y2, [], T.index, null_srcinfo()),
    }
    body2 = SubstArgs(body, subenv).result()
    a_bd = (
        expr_effs(x_loop.lo)
        + expr_effs(x_loop.hi)
        + expr_effs(y_loop.lo)
        + expr_effs(y_loop.hi)
    )
    a = stmts_effs(body)
    a2 = stmts_effs(body2)

    def bds(x, lo, hi):
        return AAnd(lift_e(lo) <= AInt(x), AInt(x) < lift_e(hi))

    reorder_is_safe = AAnd(
        AForAll(
            [x, y],
            AImplies(
                AMay(AAnd(bds(x, x_loop.lo, x_loop.hi), bds(y, y_loop.lo, y_loop.hi))),
                Commutes(a_bd, a),
            ),
        ),
        AForAll(
            [x, y, x2, y2],
            AImplies(
                AMay(
                    AAnd(
                        bds(x, x_loop.lo, x_loop.hi),
                        bds(y, y_loop.lo, y_loop.hi),
                        bds(x2, x_loop.lo, x_loop.hi),
                        bds(y2, y_loop.lo, y_loop.hi),
                        AInt(x) < AInt(x2),
                        AInt(y2) < AInt(y),
                    )
                ),
                Commutes(a, a2),
            ),
        ),
    )

    pred = G(reorder_is_safe)
    is_ok = slv.verify(pred)
    slv.pop()
    if not is_ok:
        raise SchedulingError(f"Loops {x} and {y} at {s.srcinfo} cannot be reordered.")


# Formal Statement
#       for i in e: s1   -->  parallel_for i in e: s1
#
#   Let a1' = [i -> i']a1
#
#   (forall i. May(InBound(i,e)) ==> Commutes(ae, a1))
#   /\ ( forall i,i'. May(InBound(i,i',e) /\ i < i') => Commutes(a1', a1) )
#
def Check_ParallelizeLoop(proc, s):
    ctxt = ContextExtraction(proc, [s])

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(p))

    lo = s.lo
    hi = s.hi
    body = s.body
    i = s.iter

    i2 = i.copy()
    subenv = {i: LoopIR.Read(i2, [], T.index, null_srcinfo())}
    body2 = SubstArgs(body, subenv).result()

    a_bd = expr_effs(s.lo) + expr_effs(s.hi)
    a = stmts_effs(body)
    a2 = stmts_effs(body2)

    def bds(x, lo, hi):
        return AAnd(lift_e(lo) <= AInt(x), AInt(x) < lift_e(hi))

    no_bound_change = AForAll(
        [i],
        AImplies(AMay(bds(i, lo, hi)), Commutes(a_bd, a)),
    )
    bodies_commute = AForAll(
        [i, i2],
        AImplies(
            AMay(AAnd(bds(i, lo, hi), bds(i2, lo, hi), AInt(i) < AInt(i2))),
            Disjoint_Memory(a, a2),
        ),
    )

    pred = G(AAnd(no_bound_change, bodies_commute))
    is_ok = slv.verify(pred)
    slv.pop()
    if not is_ok:
        raise SchedulingError(f"Cannot parallelize loop over {i} at {s.srcinfo}")


# Formal Statement
#       for i in e: (s1 ; s2)  -->  (for i in e: s1); (for i in e: s2)
#
#   Let a1' = [i -> i']a1
#
#   (forall i. May(InBound(i,e)) ==> Commutes(ae, a1) /\ Commutes(ae, a2))
#   /\ ( forall i,i'. May(InBound(i,i',e) /\ i < i')  =>
#                     Commutes(a1', a2) /\ AllocCommutes(a1, a2) )
#
def Check_FissionLoop(proc, loop, stmts1, stmts2, no_loop_var_1=False):
    ctxt = ContextExtraction(proc, [loop])
    chgG = get_changing_scalars(proc.body)

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(p))

    assert isinstance(loop, LoopIR.For)
    i = loop.iter
    j = i.copy()
    lo = loop.lo
    hi = loop.hi
    subenv = {i: LoopIR.Read(j, [], T.index, null_srcinfo())}
    stmts1_j = SubstArgs(stmts1, subenv).result()

    Gloop = loop_globenv(
        i,
        LoopIR.Const(0, T.index, null_srcinfo()),
        LoopIR.Read(j, [], T.index, null_srcinfo()),
        stmts1,
    )
    # print("GLOOP")
    # print(Gloop)

    a_bd = expr_effs(lo) + expr_effs(hi)
    a1 = stmts_effs(stmts1)
    a1_j = stmts_effs(stmts1_j)
    a2 = stmts_effs(stmts2)

    def bds(x, lo, hi):
        return AAnd(lift_e(lo) <= AInt(x), AInt(x) < lift_e(hi))

    commute12 = Gloop(
        Commutes_Fissioning(
            a1_j, a2, globenv(stmts1_j), globenv(stmts2), a1_no_loop_var=no_loop_var_1
        )
    )

    no_bound_change = AForAll(
        [i],
        AImplies(AMay(bds(i, lo, hi)), AAnd(Commutes(a_bd, a1), Commutes(a_bd, a2))),
    )
    stmts_commute = AForAll(
        [i, j],
        AImplies(
            AMay(AAnd(bds(i, lo, hi), bds(j, lo, hi), AInt(i) < AInt(j))),
            AAnd(commute12, AllocCommutes(a1, a2)),
        ),
    )

    pred = filter_reals(G(AAnd(no_bound_change, stmts_commute)), chgG)
    # pred    = G(AAnd(no_bound_change, stmts_commute))
    is_ok = slv.verify(pred)
    slv.pop()
    if not is_ok:
        raise SchedulingError(f"Cannot fission loop over {i} at {loop.srcinfo}.")


def Check_DeleteConfigWrite(proc, stmts):
    assert len(stmts) > 0
    ctxt = ContextExtraction(proc, stmts)

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()
    ap = ctxt.get_posteffs()
    a = G(stmts_effs(stmts))
    stmtsG = globenv(stmts)

    slv = SMTSolver(verbose=False)
    slv.push()
    a = [E.Guard(AMay(p), a)]

    # extract effects
    WrG, Mod = getsets([ES.WRITE_G, ES.MODIFY], a)
    WrGp, RdGp = getsets([ES.WRITE_G, ES.READ_G], ap)

    # check that `stmts` does not modify any non-global data
    only_mod_glob = ADef(is_empty(LDiff(Mod, WrG)))
    is_ok = slv.verify(only_mod_glob)
    if not is_ok:
        slv.pop()
        raise SchedulingError(
            f"Cannot delete or insert statements at {stmts[0].srcinfo} "
            f"because they may modify non-configuration data"
        )

    # get the set of config variables potentially modified by
    # the statement block being focused on.  Filter out any
    # such configuration variables whose values are definitely unchanged
    def is_cfg_unmod_by_stmts(pt):
        pt_e = A.Var(pt.name, pt.typ, null_srcinfo())
        # cfg_unwritten = ADef( ANot(is_elem(pt, WrG)) )
        cfg_unchanged = ADef(G(AEq(pt_e, stmtsG(pt_e))))
        return slv.verify(cfg_unchanged)

    cfg_mod = {
        pt.name: pt for pt in get_point_exprs(WrG) if not is_cfg_unmod_by_stmts(pt)
    }

    # consider every global that might be modified
    cfg_mod_visible = set()
    for _, pt in cfg_mod.items():
        pt_e = A.Var(pt.name, pt.typ, null_srcinfo())
        is_written = is_elem(pt, WrG)
        is_unchanged = G(AEq(pt_e, stmtsG(pt_e)))
        is_read_post = is_elem(pt, RdGp)
        is_overwritten = is_elem(pt, WrGp)

        # if the value of the global might be read,
        # then it must not have been changed.
        safe_write = AImplies(AMay(is_read_post), ADef(is_unchanged))
        if not slv.verify(safe_write):
            slv.pop()
            raise SchedulingError(
                f"Cannot change configuration value of {pt.name} "
                f"at {stmts[0].srcinfo}; the new (and different) "
                f"values might be read later in this procedure"
            )
        # the write is invisible if its definitely unchanged or definitely
        # overwritten
        invisible = ADef(AOr(is_unchanged, is_overwritten))
        if not slv.verify(invisible):
            cfg_mod_visible.add(pt.name)

    slv.pop()
    return cfg_mod_visible


# This equivalence check assumes that we can
# externally verify that substituting stmts with something else
# is equivalent modulo the keys in `cfg_mod`, so
# the only thing we want to check is whether that can be
# extended, and if so, modulo what set of output globals?
def Check_ExtendEqv(proc, stmts0, stmts1, cfg_mod):
    assert len(stmts0) > 0
    assert len(stmts1) > 0
    ctxt = ContextExtraction(proc, stmts0)

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()
    ap = ctxt.get_posteffs()
    # a       = G(stmts_effs(stmts))
    sG0 = globenv(stmts0)
    sG1 = globenv(stmts1)

    slv = SMTSolver(verbose=False)
    slv.push()
    # slv.assume(AMay(p))

    # extract effects
    # WrG, Mod    = getsets([ES.WRITE_G, ES.MODIFY], a)
    WrGp, RdGp = getsets([ES.WRITE_G, ES.READ_G], ap)

    # check that none of the configuration variables which might have
    # changed are being observed.
    def make_point(key):
        cfg, fld = reverse_config_lookup(key)
        typ = cfg.lookup_type(fld)
        return APoint(key, [], typ)

    cfg_mod_pts = [make_point(key) for key in cfg_mod]
    cfg_mod_visible = set()
    for pt in cfg_mod_pts:
        pt_e = ABool(pt.name) if pt.typ == T.bool else AInt(pt.name)
        is_unchanged = AImplies(p, G(AEq(sG0(pt_e), sG1(pt_e))))
        is_read_post = is_elem(pt, RdGp)
        is_overwritten = is_elem(pt, WrGp)

        safe_write = AImplies(AMay(is_read_post), ADef(is_unchanged))
        if not slv.verify(safe_write):
            slv.pop()
            raise SchedulingError(
                f"Cannot rewrite at {stmts0[0].srcinfo} because the "
                f"configuration field {pt.name} might be read "
                f"subsequently"
            )

        shadowed = ADef(is_overwritten)
        if not slv.verify(shadowed):
            cfg_mod_visible.add(pt.name)

    slv.pop()
    return cfg_mod_visible


def Check_ExprEqvInContext(proc, expr0, stmts0, expr1, stmts1=None):
    assert len(stmts0) > 0
    stmts1 = stmts1 or stmts0
    ctxt0 = ContextExtraction(proc, stmts0)
    ctxt1 = ContextExtraction(proc, stmts1)

    p0 = ctxt0.get_control_predicate()
    G0 = ctxt0.get_pre_globenv()
    p1 = ctxt1.get_control_predicate()
    G1 = ctxt1.get_pre_globenv()

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(AAnd(p0, p1)))

    e0 = G0(lift_e(expr0))
    e1 = G1(lift_e(expr1))

    test = AEq(e0, e1)
    is_ok = slv.verify(test)
    slv.pop()
    if not is_ok:
        raise SchedulingError(f"Expressions are not equivalent:\n{expr0}\nvs.\n{expr1}")


def Check_BufferReduceOnly(proc, stmts, buf, ndim):
    assert len(stmts) > 0
    ctxt = ContextExtraction(proc, stmts)

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(p))

    wholebuf = LS.WholeBuf(buf, ndim)
    a = G(stmts_effs(stmts))
    RW = getsets([ES.READ_WRITE], a)[0]
    readwrite = LIsct(wholebuf, RW)

    no_rw = slv.verify(ADef(is_empty(readwrite)))
    slv.pop()
    if not no_rw:
        raise SchedulingError(
            f"The buffer {buf} is accessed in a way other than "
            f"simply reducing into it"
        )


# TODO: I think idxs should be passed as either a read, window, or write (assign/reduce)
def Check_Access_In_Window(proc, access_cursor, w_exprs, block_cursor):
    """
    Returns True if idxs always lies within w_exprs
    Returns False if idxs never lies within w_exprs
    Raises a SchedulingError otherwise

    block_cursor is the context in which to interpret the access in.
    """

    access = access_cursor._node
    block = [x._node for x in block_cursor]
    idxs = access.idx
    assert len(idxs) == len(w_exprs)

    ctxt = ContextExtraction(proc, block)
    p = ctxt.get_control_predicate()

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(p))

    # build a location set describing the allocated region of the buffer
    name = access.name

    def get_locset(w_idxs):
        coords = [Sym(f"i{i}") for i, _ in enumerate(w_idxs)]
        bounds = []
        for i, w_access in zip(coords, w_idxs):
            # Need four cases because accesses and w_exprs have different representations
            if isinstance(w_access, tuple):
                (w_lo, w_hi) = w_access
                bounds.append(AAnd(lift_e(w_lo) <= AInt(i), AInt(i) < lift_e(w_hi)))
            elif isinstance(w_access, LoopIR.Interval):
                w_lo = w_access.lo
                w_hi = w_access.hi
                bounds.append(AAnd(lift_e(w_lo) <= AInt(i), AInt(i) < lift_e(w_hi)))
            elif isinstance(w_access, LoopIR.Point):
                bounds.append(AEq(AInt(i), lift_e(w_access.pt)))
            else:
                bounds.append(AEq(AInt(i), lift_e(w_access)))
        bounds = AAnd(*bounds)

        pt = LS.Point(name, [AInt(i) for i in coords], T.index)
        locset = LFilter(bounds, pt)
        for i in reversed(coords):
            locset = LBigUnion(i, locset)
        return locset

    window_locset = get_locset(w_exprs)

    access_locset = get_locset(idxs)

    # Surround access_locset with the appropriate bounds based on the context block_cursor.
    cursor = access_cursor
    while cursor.depth() != block_cursor.depth():
        if isinstance(cursor._node, LoopIR.For):
            loop = cursor._node
            bounds = AAnd(
                lift_e(loop.lo) <= AInt(loop.iter), AInt(loop.iter) < lift_e(loop.hi)
            )
            access_locset = LFilter(bounds, access_locset)
        elif isinstance(cursor._node, LoopIR.If):
            cond = lift_e(cursor._node.cond)
            if cursor._path[-1][0] == "orelse":
                cond = ANot(lift_e(cursor._node.cond))
            access_locset = LFilter(cond, access_locset)

        cursor = cursor.parent()

    if slv.verify(ADef(is_empty(LIsct(access_locset, window_locset)))):
        # access_locset is disjoint from window_set
        slv.pop()
        return False

    if slv.verify(ADef(is_empty(LDiff(access_locset, window_locset)))):
        # access_locset is a subset of window_set
        slv.pop()
        return True

    raise SchedulingError(
        f"Buffer has accesses which are neither fully within nor disjoint from the window"
    )


def Check_Bounds(proc, alloc_stmt, block):
    if len(block) == 0:
        return
    ctxt = ContextExtraction(proc, block)

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(p))

    # build a location set describing
    # the allocated region of the buffer
    shape = alloc_stmt.type.shape()
    if len(shape) == 0:
        alloc_set = LS.Point(alloc_stmt.name, [], alloc_stmt.type.basetype())
    else:
        coords = [Sym(f"i{i}") for i, _ in enumerate(shape)]
        bounds = AAnd(
            *[
                AAnd(AInt(0) <= AInt(i), AInt(i) < lift_e(n))
                for i, n in zip(coords, shape)
            ]
        )
        pt = LS.Point(
            alloc_stmt.name, [AInt(i) for i in coords], alloc_stmt.type.basetype()
        )
        alloc_set = LFilter(bounds, pt)
        for i in reversed(coords):
            alloc_set = LBigUnion(i, alloc_set)

    a = G(stmts_effs(block))
    All = getsets([ES.ALL], a)[0]
    All_inbuf = LIsct(All, LS.WholeBuf(alloc_stmt.name, len(shape)))
    is_ok = slv.verify(ADef(is_empty(LDiff(All_inbuf, alloc_set))))
    slv.pop()
    if not is_ok:
        raise SchedulingError(f"The buffer {alloc_stmt.name} is accessed out-of-bounds")


def Check_IsDeadAfter(proc, stmts, bufname, ndim):
    assert len(stmts) > 0
    ctxt = ContextExtraction(proc, stmts)

    ap = ctxt.get_posteffs()

    slv = SMTSolver(verbose=False)
    slv.push()

    # extract effect location sets
    Allp = getsets([ES.ALL], ap)[0]

    wholebuf = LS.WholeBuf(bufname, ndim)
    is_dead = slv.verify(ADef(is_empty(LIsct(Allp, wholebuf))))
    slv.pop()
    if not is_dead:
        raise SchedulingError(
            f"The variable {bufname} can potentially be used after "
            + f"the statement at {stmts[0].srcinfo} executes."
        )


def Check_IsIdempotent(proc, stmts):
    assert len(stmts) > 0
    ctxt = ContextExtraction(proc, stmts)

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()
    ap = ctxt.get_posteffs()
    a = G(stmts_effs(stmts))

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(p))

    is_idempotent = slv.verify(ADef(Shadows(a, a)))
    slv.pop()
    if not is_idempotent:
        raise SchedulingError(f"The statement at {stmts[0].srcinfo} is not idempotent.")


def Check_ExprBound(proc, stmts, expr, op, value, exception=True):
    assert len(stmts) > 0

    ctxt = ContextExtraction(proc, stmts)

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()

    slv = SMTSolver(verbose=False)
    slv.push()
    slv.assume(AMay(p))

    e = G(lift_e(expr))

    if op == ">=":
        query = ADef(e >= AInt(value))
        err_msg = f"greater than or equal to {value}"
    elif op == ">":
        query = ADef(e > AInt(value))
        err_msg = f"greater than {value}"
    elif op == "<=":
        query = ADef(e <= AInt(value))
        err_msg = f"less than or equal to {value}"
    elif op == "<":
        query = ADef(e < AInt(value))
        err_msg = f"greater than {value}"
    elif op == "==":
        query = ADef(AEq(e, AInt(value)))
        err_msg = f"equal to {value}"
    else:
        assert False, "Bad case"

    success = slv.verify(query)
    slv.pop()

    if not exception:
        return success

    if not success:
        estr = str(expr)
        if estr[-1] == "\n":
            estr = estr[:-1]
        raise SchedulingError(
            f"The expression {estr} is not guaranteed to be {err_msg}."
        )


def Check_CodeIsDead(proc, stmts):
    assert len(stmts) > 0
    ctxt = ContextExtraction(proc, stmts)

    p = ctxt.get_control_predicate()
    G = ctxt.get_pre_globenv()
    ap = ctxt.get_posteffs()
    a = G(stmts_effs(stmts))

    # apply control predicate
    a = [E.Guard(AMay(p), a)]

    # The basic question for dead code is to ask:
    #       Is there any way that any memory modified by `stmts`
    #       could possibly affect any other code that runs later?
    #
    # Let X be some memory location modified by `stmts`.
    # If any code running after `stmts` reads/reduces/depends-on X,
    #   then `stmts` is not dead.
    # Otherwise if X is allocated local to `proc`, then `stmts`
    #   is dead w.r.t. X
    # Otherwise X is a global or an argument buffer;
    #   If X is not definitely overwritten before exiting `proc`,
    #   then `stmts` is not dead.
    #
    # The preceding analysis should never erroneously report code
    # as dead when it is not.

    Modp, WGp = getsets([ES.MODIFY, ES.WRITE_G], G(a))
    R_ap, Red_ap, W_ap = getsets([ES.READ_ALL, ES.REDUCE, ES.WRITE_ALL], ap)

    # get a set of globals and function arguments that
    # overapproximates the set of locations that might have been written
    # and are all memory locations visible after the lifetime of `proc`
    globs = {pt.name: pt.typ for pt in get_point_exprs(WGp)}
    args = {fa.name: len(fa.type.shape()) for fa in proc.args if fa.type.is_numeric()}
    # now we'll construct a location set out of these
    Outside = LS.Empty()
    for gnm, typ in globs.items():
        Outside = LUnion(Outside, LS.Point(gnm, [], typ))
    for nm, ndim in globs.items():
        Outside = LUnion(Outside, LS.WholeBuf(nm, ndim))

    # first condition
    mod_unread_in_proc = ADef(is_empty(LIsct(Modp, LUnion(R_ap, Red_ap))))
    # second condition
    mod_unread_outside = ADef(is_empty(LIsct(LDiff(Modp, W_ap), Outside)))

    slv = SMTSolver(verbose=False)
    slv.push()
    mod_unread_in_proc = slv.verify(mod_unread_in_proc)
    mod_unread_outside = slv.verify(mod_unread_outside)
    slv.pop()
    if not mod_unread_in_proc:
        raise SchedulingError(
            f"Code is not dead, because values modified might be "
            f"read later in this proc"
        )
    if not mod_unread_outside:
        raise SchedulingError(
            f"Code is not dead, because values modified might be "
            f"read later outside this proc"
        )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The following is a kludge to prevent aliasing problems
# and problems stemming from passing real_scalar configuration variables
# as data arguments to functions (related, but distinct)


class _OverApproxEffects(LoopIR_Do):
    """Computes all buffers and globals potentially accessed by a proc
    or any sub-procs transitively"""

    def __init__(self, proc):
        self._touched = set()
        self._aliases = dict()
        self._globals = set()

        super().__init__(proc)

        # filter the results down to arguments and globals
        args = {fa.name for fa in proc.args}
        self._touched = {
            nm for nm in self._touched if nm in self._globals or nm in args
        }

    def results(self):
        return self._touched

    def add_name(self, name):
        self._touched.add(name)
        while name in self._aliases:
            name = self._aliases[name]
            self._touched.add(name)

    def do_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            self.add_name(s.name)
        elif isinstance(s, LoopIR.WriteConfig):
            globname = s.config._INTERNAL_sym(s.field)
            self.add_name(globname)
            self._globals.add(globname)
        elif isinstance(s, LoopIR.Call):
            # build translation of the results of analysis from the sub-proc
            for fa, a in zip(s.f.args, s.args):
                # treat numeric arguments as aliases rather than
                # accesses to the corresponding expressions
                if fa.type.is_numeric():
                    if isinstance(a, (LoopIR.Read, LoopIR.WindowExpr)):
                        self._aliases[fa.name] = a.name
                        if isinstance(a, LoopIR.Read):
                            for i in a.idx:
                                self.do_e(i)
                        else:
                            for w in a.idx:
                                self.do_w_access(w)
                    else:
                        assert isinstance(a, LoopIR.ReadConfig)
                        globname = a.config._INTERNAL_sym(a.field)
                        self._aliases[fa.name] = globname
                else:
                    self.do_e(a)
            # now add all effect approximations from the sub-proc
            for name in overapprox_proc_effs(s.f):
                self.add_name(name)
            return  # don't call do_e on all of the arguments
        elif isinstance(s, LoopIR.WindowStmt):
            self._aliases[s.name] = s.rhs.name

        super().do_s(s)

    def do_e(self, e):
        if isinstance(e, LoopIR.Read):
            self.add_name(e.name)
        elif isinstance(e, LoopIR.ReadConfig):
            globname = e.config._INTERNAL_sym(e.field)
            self.add_name(globname)
            self._globals.add(globname)

        super().do_e(e)


_overapprox_proc_cache = dict()


def overapprox_proc_effs(proc):
    if proc not in _overapprox_proc_cache:
        _overapprox_proc_cache[proc] = _OverApproxEffects(proc).results()
    return _overapprox_proc_cache[proc]


class _Check_Aliasing_Helper(LoopIR_Do):
    def __init__(self, proc):
        self._aliases = dict()
        super().__init__(proc)

    def translate(self, name):
        if name in self._aliases:
            return self._aliases[name]
        else:
            return name

    def do_s(self, s):
        if isinstance(s, LoopIR.Call):
            # check for duplicate buffer argument names
            passed_buffers = set()
            argnames = {
                self.translate(a.name)
                for a in s.args
                if isinstance(a, (LoopIR.Read, LoopIR.WindowExpr))
            }
            for fa, a in zip(s.f.args, s.args):
                if fa.type.is_numeric():
                    if isinstance(a, (LoopIR.Read, LoopIR.WindowExpr)):
                        name = self.translate(a.name)
                        if name in passed_buffers:
                            raise SchedulingError(
                                f"Cannot Pass the same buffer '{name}' via "
                                f"multiple arguments in call to {s.f.name}, "
                                f"since doing so would introduce aliased "
                                f"arguments.  "
                                f"Please contact the Exo developers if you "
                                f"need support for aliased arguments."
                            )
                        passed_buffers.add(name)
                    elif isinstance(a, LoopIR.ReadConfig):
                        sub_effs = overapprox_proc_effs(s.f)
                        globname = a.config._INTERNAL_sym(a.field)
                        if globname in sub_effs:
                            raise SchedulingError(
                                f"Passing numeric-type (R, f32, i8, etc.) "
                                f"configuration variables is not currently "
                                f"supported in Exo due to internal "
                                f"complications and potential aliasing "
                                f"issues."
                            )
        elif isinstance(s, LoopIR.WindowStmt):
            name = s.rhs.name
            while name in self._aliases:
                name = self._aliases[name]
            self._aliases[s.name] = name
        else:
            super().do_s(s)


def Check_Aliasing(proc):
    helper = _Check_Aliasing_Helper(proc)
    # that's it
