from adt import ADT
from .LoopIR import LoopIR, Alpha_Rename
from .configs import Config
from .prelude import *

from collections    import ChainMap, OrderedDict
from itertools      import chain
from dataclasses    import dataclass
from typing         import Any
from enum           import Enum

from .LoopIR import T
from .LoopIR import SubstArgs

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
    name    : Sym
    coords  : list[A.expr]

@dataclass
class AWinCoord:
    is_pt   : bool
    val     : A.expr

@dataclass
class AWin:
    name    : Sym
    coords  : list[AWinCoord]
    strides : list[A.expr]

    def __str__(win):
        def coordstr(c):
            op = '=' if c.is_pt else '+'
            return f"{op}{c.val}"
        coords = ''
        if len(win.coords) > 0:
            coords = ','+(','.join([coordstr(c) for c in win.coords]))
        return f"({win.name}{coords})"

    def nslots(self):
        return sum([ not c.is_pt for c in self.coords ])

    # compose with another AWin
    def __add__(lhs, rhs):
        assert isinstance(rhs, AWin)
        # ignore rhs.name
        # check that the number of coordinates is compatible
        assert lhs.nslots() == len(rhs.coords)
        ri      = 0
        coords  = []
        for lc in lhs.coords:
            if lc.is_pt:
                coords.append( lc )
            else:
                rc = rhs.coords[ri]
                ri += 1
                coords.append( AWinCoord(rc.is_pt, rc.val + lc.val) )
        return AWin(lhs.name, coords, lhs.strides) 

    # apply to a point
    def __call__(self, pt):
        assert isinstance(pt, APoint)
        assert self.nslots() == len(pt.coords)
        pi      = 0
        coords  = []
        for wc in self.coords:
            if wc.is_pt:
                coords.append( wc.val )
            else:
                coords.append( wc.val + pt.coords[pi] )
                pi += 1
        return APoint(self.name, coords)

    # get a stride
    def get_stride(self, ndim):
        interval_strides = [ s for c,s in zip(self.coords, self.strides)
                               if not c.is_pt ]
        return interval_strides[ndim]

def AWinAlloc(name, sizes):
    assert all(isinstance(sz,LoopIR.expr) for sz in sizes)
    coords  = [ AWinCoord(is_pt=False, val=AInt(0)) for _ in sizes ]
    strides = [ A.Stride(name, i) for i in range(len(sizes)) ]

    # fill out constant strides where possible
    strides[-1] = AInt(1)
    sprod   = 1
    for i in reversed(range(len(sizes)-1)):
        sz = sizes[i+1]
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
    names   : list[Sym]
    rhs     : Any

@dataclass
class BindingList:
    names   : list[Sym]
    rhs     : list[Any]

@dataclass
class WinBind:
    name    : Sym
    rhs     : AWin

class AEnv:
    def __init__(self, name=None, expr=None, addnames=False):
        # bindings stored in order of definition
        # reverse order from application
        self.bindings   = []
        self.names      = set()
        if name is None:
            return
        elif isinstance(name, Sym):
            if isinstance(expr, A.expr):
                self.bindings = [BindingList([name],[expr])]
            else:
                assert isinstance(expr, AWin)
                self.bindings = [WinBind(name,expr)]

            if addnames:
                self.names.add(name)
        elif type(name) is list:
            assert isinstance(expr, A.expr)
            assert type(expr.type) is tuple and len(name) == len(expr.type)
            self.bindings = [TupleBinding(name,expr)]

            if addnames:
                self.names = self.names.union(name)
        else: assert False, "bad case"

    def __str__(self):
        def bstr(bd):
            if isinstance(bd, TupleBinding):
                nm = ','.join([str(n) for n in bd.names])
                return f"[{nm} ↦ {bd.rhs}]"
            elif isinstance(bd, WinBind):
                return f"[{bd.name} ↦ {bd.rhs}]"
            elif isinstance(BindingList):
                return ''.join([ f"[{x} ↦ {v}]"
                                 for x,v in zip(bd.names, bd.rhs) ])
            else: assert False, "bad case"
        return ''.join([ bstr(bd) for bd in self.bindings ])

    def __add__(lhs, rhs):
        assert isinstance(rhs, AEnv)
        result = AEnv()
        merged = False
        # compression optimization
        if len(lhs.bindings) > 0 and len(rhs.bindings) > 0:
            lb, rb = lhs.bindings[-1], rhs.bindings[0]
            if isinstance(lb,BindingList) and isinstance(rb, BindingList):
                mb = BindingList(lb.names + rb.names, lb.rhs + rb.rhs)
                result.bindings = lhs.bindings[:-1] + [mb] + rhs.bindings[1:]
                return result
        # otherwise
        result.bindings = lhs.bindings + rhs.bindings
        result.names    = lhs.names.union(rhs.names)
        return result

    def __call__(self, arg):
        if isinstance(arg, AExpr):
            for bd in reversed(self.bindings):
                if isinstance(bd, TupleBinding):
                    arg = A.LetTuple(bd.names, bd.rhs, arg, bd.rhs.srcinfo)
                elif isinstance(bd, WinBind):
                    # bind strides...
                    arg = ALetStride(bd.name, rhs.strides)
                    # TODO: Probably need to introduce let bindings
                    # for coordinates in a way to get the values of globals
                    # at the site of windowing correct
                    #raise NotImplementedError("TODO: how to handle?")
                elif isinstance(bd, BindingList):
                    arg = ALet(bd.names, bd.rhs, arg)
                else: assert False, "bad case"
            return arg
        else: assert False, f"Cannot apply AEnv to {type(arg)}"

    def translate_win(self, winmap):
        winmap = winmap.copy()
        for bd in self.bindings:
            if isinstance(bd, WinBind):
                win     = bd.rhs
                prewin  = winmap.get(win.name, None)
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
                for n,rt in zip(bd.names, bd.rhs.type):
                    if n in self.names:
                        assert n not in nmtyp or nmtyp[n] == rt
                        nmtyp[n] = rt
            elif isinstance(bd, WinBind):
                pass # ignore windows for this
            elif isinstance(bd, BindingList):
                for n,r in zip(bd.names, bd.rhs):
                    if n in self.names:
                        nmtyp[n] = r.type
        return nmtyp

    # we assume this packs up a scoping level
    # and therefore drop any WinBinds in the process
    def bind_to_copies(self):
        nmtyps      = self.names_types()
        orig_vars   = [ A.Var(nm, typ, null_srcinfo())
                        for nm,typ in nmtyps.items() ]
        copy_vars   = [ A.Var(nm.copy(), typ, null_srcinfo())
                        for nm,typ in nmtyps.items() ]
        varmap      = OrderedDict({ nm : copyv
                            for (nm,_),copyv in zip(nmtyps, copy_vars)})

        body        = A.Tuple(orig_vars, [e.type for e in orig_vars],
                                         null_srcinfo())
        # wrap the tuple in let-bindings representing this entire AEnv
        body        = self(body)
        # then construct an AEnv that maps the new variable symbols
        # to these values
        new_env     = AEnv([ v.name for v in copy_vars ], body, addnames=True)
        return varmap, new_env


def aenv_join(aenvs):
    aenv = AEnv()
    for a in aenvs:
        aenv = aenv + a
    return aenv

def AEnvPar(bind_dict, addnames=False):
    names, rhs, types = [], [], []
    for nm,e in bind_dict.items():
        names.append(nm)
        rhs.append(e)
        types.append(e.type)
    rhs_tuple = A.Tuple(rhs, types, rhs[0].srcinfo)
    return AEnv(names, rhs_tuple, addnames=addnames)

#@extclass(SMTSolver)
#def bindenv(smtslv, aenv):
#    for nm,rhs in aenv.bindings:
#        if type(nm) is list:
#            smtslv.bind_tuple(nm,rhs)
#        else:
#            smtslv.bind([nm],[rhs])

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Basic Global Dataflow Analysis


def lift_e(e):
    if isinstance(e, LoopIR.WindowExpr):
        def lift_w(w):
            if isinstance(w, LoopIR.Interval):
                return AWinCoord(is_pt=False,val=lift_e(w.lo))
            elif isinstance(w, LoopIR.Point):
                return AWinCoord(is_pt=True,val=lift_e(w.pt))
            else: assert False, f"bad w_access case"
        # default strides
        strides = [ A.Stride(e.name, i, T.stride, e.srcinfo)
                    for i in range(len(e.idx)) ]
        return AWin(e.name,[ lift_w(w) for w in e.idx ], strides)
    else:
        if (e.type.is_indexable() or e.type.is_stridable() or
            e.type == T.bool):
            if isinstance(e, LoopIR.Read):
                assert len(e.idx) == 0
                return A.Var(e.name, e.type, e.srcinfo)
            elif isinstance(e, LoopIR.Const):
                return A.Const(e.val, e.type, e.srcinfo)
            elif isinstance(e, LoopIR.BinOp):
                return A.BinOp(e.op, lift_e(e.lhs), lift_e(e.rhs),
                               e.type, e.srcinfo)
            elif isinstance(e, LoopIR.USub):
                return A.USub(lift_e(e.arg), e.type, e.srcinfo)
            elif isinstance(e, LoopIR.StrideExpr):
                return A.Stride( e.name, e.dim, e.type, e.srcinfo )
            elif isinstance(e, LoopIR.ReadConfig):
                globname = e.config._INTERNAL_sym(e.field)
                return A.Var( globname, e.type, e.srcinfo )
            else: f"bad case: {type(e)}"
        else:
            return A.Unk(T.err, e.srcinfo)

def lift_es(es):
    return [ lift_e(e) for e in es ]

def globenv(stmts):
    aenvs = []
    for s in stmts:
        if isinstance(s, LoopIR.WriteConfig):
            # don't bother tracking 
            if not s.rhs.type.is_numeric():
                globname    = s.config._INTERNAL_sym(s.field)
                rhs         = lift_e(s.rhs)
                aenv.append(AEnv(globname, rhs, addnames=True))
        elif isinstance(s, LoopIR.WindowStmt):
            win         = lift_e(s.rhs)
            aenv.append(AEnv(s.lhs, win))
        elif isinstance(s, LoopIR.Alloc):
            win         = AWinAlloc(s.name, s.type.shape())
            aenv.append(AEnv(s.name,win))
        elif isinstance(s, LoopIR.If):
            # extract environments for each side of the branch
            body_env    = globenv(s.body)
            else_env    = globenv(s.orelse)
            # get the map from old to new names, and binding env
            bvarmap, benv   = body_env.bind_to_copies()
            evarmap, eenv   = else_env.bind_to_copies()
            aenvs      += [benv, eenv]
            oldvars     = { nm : A.Var(nm, newv.type, s.srcinfo)
                            for nm, newv in chain(bvarmap.items(),
                                                  evarmap.items()) }

            # bind the condition so it isn't duplicated
            condsym     = Sym('if_cond')
            condvar     = A.Var(condsym, T.bool, s.cond.srcinfo)
            cond        = lift_e(s.cond)
            aenv.append(AEnv(condsym, cond))

            # We must now construct an environment that defines the
            # new value for variables `x` among the possibilities
            newbinds    = dict()
            for nm,oldv in oldvars:
                # default to old-value
                tcase   = bvarmap.get(nm,oldv)
                fcase   = evarmap.get(nm,oldv)
                val     = A.Select(condvar, tcase, fcase, typ, s.srcinfo)
                newbinds[nm] = val
            aenv.append(AEnvPar(newbinds,addnames=True))

        elif isinstance(s, (LoopIR.ForAll,LoopIR.Seq)):
            # extract environment for the body and bind its
            # results via copies of the variables
            body_env    = globenv(s.body)
            bvarmap, benv   = body_env.bind_to_copies()
            aenv.append(benv)

            # bind the bounds condition so it isn't duplicated
            #bdssym      = Sym('for_bounds')
            #bdsvar      = ABool(bdssym)
            bds         = AAnd(AInt(0) <= AInt(s.iter),
                               AInt(s.iter) < lift_e(s.hi))
            def fix(x,body_x):
                arg     = AImplies(bds, AEq(body_x,x))
                return A.ForAll(s.iter, arg, T.bool, s.srcinfo)

            # Now construct an environment that defines the new
            # value for variables `x` based on fixed-point conditions
            newbinds    = dict()
            for nm,bvar in bvarmap.items():
                oldvar  = A.Var(nm, bvar.type, s.srcinfo)
                val     = A.Select(fix(oldvar, bvar),
                                   oldvar,
                                   A.Unk(oldvar.type, s.srcinfo))
                newbinds[nm] = val
            aenv.append(AEnvPar(newbinds,addnames=True))

        elif isinstance(s, LoopIR.Call):
            sub_proc    = get_simple_proc(s.f)
            sub_env     = globenv_proc(sub_proc)
            call_env    = call_bindings(s.args, sub_proc.args)
            aenv       += [call_env, sub_env]

        else:
            pass

    return aenv_join(aenvs)


def call_bindings(call_args, sig_args):
    assert len(call_args) == len(sig_args)
    aenvs   = []
    for a,fa in zip(call_args, sig_args):
        aenvs.append( AEnv(fa.name, lift_e(a)) )
    return aenv_join(aenvs)


_globenv_proc_cache = dict()
def globenv_proc(proc):
    if proc not in _globenv_proc_cache:
        _globenv_proc_cache[proc] = globenv(proc.body)
    return _globenv_proc_cache[proc]





# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Location Sets

LS = ADT("""
module LocSet {
    locset  = Empty     ()
            | Point     ( sym name, aexpr* coords )
            | Union     ( locset lhs, locset rhs )
            | Isct      ( locset lhs, locset rhs )
            | Diff      ( locset lhs, locset rhs )
            | BigUnion  ( sym   name, locset arg )
            | Filter    ( aexpr cond, locset arg )
            | LetEnv    ( aenv   env, locset arg )
} """, {
    'sym':     lambda x: isinstance(x, Sym),
    'aexpr':   lambda x: isinstance(x, A.expr),
    #'srcinfo': lambda x: isinstance(x, SrcInfo),
})


ls_prec = {
    #
    "bigunion": 10,
    #
    "union":    20,
    "isct":     20,
    "diff":     20,
}

def LUnion(lhs,rhs):
    if isinstance(lhs, LS.Empty):
        return rhs
    elif isinstance(rhs, LS.Empty):
        return lhs
    else:
        return LS.Union(lhs,rhs)

def LIsct(lhs,rhs):
    if isinstance(lhs, LS.Empty) or isinstance(rhs, LS.Empty):
        return LS.Empty()
    else:
        return LS.Isct(lhs,rhs)

def LDiff(lhs,rhs):
    if isinstance(lhs, LS.Empty):
        return LS.Empty()
    elif isinstance(rhs, LS.Empty):
        return lhs
    else:
        return LS.Diff(lhs,rhs)

def LBigUnion(name, arg):
    if isinstance(arg, LS.Empty):
        return LS.Empty()
    else:
        return LS.BigUnion(name,arg)

def LFilter(cond, arg):
    if isinstance(arg, LS.Empty):
        return LS.Empty()
    else:
        return LS.Filter(cond,arg)

def LLetEnv(env, arg):
    if isinstance(arg, LS.Empty):
        return LS.Empty()
    else:
        return LS.LetEnv(env,arg)

# pretty printing
def _lsstr(ls, prec=0):
    if isinstance(ls, LS.Empty):
        return "∅"
    elif isinstance(ls, LS.Point):
        coords = ','.join([str(a) for a in ls.coords])
        return f"{{{ls.name},{coords}}}"
    elif isinstance(ls, (LS.Union,LS.Isct,LS.Diff)):
        if isinstance(ls, LS.Union):
            op, local_prec = '∪', ls_prec['union']
        elif isinstance(ls, LS.Union):
            op, local_prec = '∩', ls_prec['isct']
        else:
            op, local_prec = '-', ls_prec['diff']
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
    else: assert False, f"bad case: {type(ls)}"

@extclass(LS.locset)
def __str__(ls):
    return _lsstr(ls)


def is_elem(pt, ls, win_map=None):
    # default arg
    win_map = win_map or dict()

    if isinstance(ls, LS.Empty):
        return ABool(False)
    elif isinstance(ls, LS.Point):
        lspt        = APoint(ls.name, ls.coords)
        if ls.name in win_map:
            lspt    = win_map[ls.name](lspt)

        if pt.name != lspt.name:
            return ABool(False)
        assert len(pt.coords) == len(lspt.coords)
        eqs = [ AEq(q,p) for q,p in zip(pt.coords,lspt.coords) ]
        return AAnd(eqs)
    elif isinstance(ls, (LS.Union,LS.Isct,LS.Diff)):
        lhs = is_elem(pt, ls.lhs, win_map=win_map)
        rhs = is_elem(pt, ls.rhs, win_map=win_map)
        if isinstance(LS.Union):
            return AOr(lhs, rhs)
        elif isinstance(LS.Isct):
            return AAnd(lhs, rhs)
        elif isinstance(LS.Diff):
            return AAnd(lhs, ANot(rhs))
        else: assert False
    elif isinstance(ls, LS.BigUnion):
        arg = is_elem(pt, ls.arg, win_map=win_map)
        return A.Exists(ls.name, arg, T.bool, null_srcinfo())
    elif isinstance(ls, LS.Filter):
        arg = is_elem(pt, ls.arg, win_map=win_map)
        return AAnd(ls.cond, arg)
    elif isinstance(ls, LS.LetEnv):
        win_map = ls.env.translate_win(win_map)
        arg = is_elem(pt, ls.arg, win_map=win_map)
        # wrap binding around the expression
        return ls.env(arg)
    else: assert False, f"bad case: {type(ls)}"

def is_empty(ls):
    all_bufs = dict()
    def _collect_buf(ls):
        if isinstance(ls, LS.Empty):
            pass
        elif isinstance(ls, LS.Point):
            all_bufs[ls.name] = len(ls.coords)
        elif isinstance(ls, (LS.Union,LS.Isct,LS.Diff)):
            _collect_buf(ls.lhs)
            _collect_buf(ls.rhs)
        elif isinstance(ls, (LS.BigUnion, LS.Filter)):
            _collect_buf(ls.arg)
    _collect_buf(ls)

    terms = []
    for nm,ndim in all_bufs.items():
        coords  = [ AInt(Sym(f"i{i}")) for i in range(ndim) ]
        term    = A.Not(is_elem(APoint(nm,coords),ls))
        for iv in reversed(coords):
            term = A.ForAll(iv.name, term)
        terms.append(term)

    return AAnd(terms)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Effects


E = ADT("""
module Effects {
    eff  = Empty        ()
         | Guard        ( aexpr cond, eff* body )
         | Loop         ( sym name,   eff* body )
         --
         | BindEnv      ( aenv env )
         --
         | GlobalRead   ( sym name )
         | GlobalWrite  ( sym name )
         | Read         ( sym name, aexpr* coords )
         | Write        ( sym name, aexpr* coords )
         | Reduce       ( sym name, aexpr* coords )
} """, {
    'sym':      lambda x: isinstance(x, Sym),
    'aexpr':    lambda x: isinstance(x, A.expr),
    'aenv':     lambda x: isinstance(x, AEnv),
    #'srcinfo': lambda x: isinstance(x, SrcInfo),
})

# pretty printing
def _effstr(eff,tab=""):
    if isinstance(eff, E.Empty):
        return f"{tab}∅"
    elif isinstance(eff, (E.Guard,E.Loop)):
        lines   = [ _effstr(e,tab+'  ') for e in eff.body ]
        if isinstance(eff, E.Guard):
            lines = [ f"Guard({eff.cond})" ] + lines
        else:
            lines = [ f"Loop({eff.name})" ] + lines
        return '\n'.join(lines)
    elif isinstance(eff, E.BindEnv):
        return f"{tab}{eff.env}"
    elif isinstance(eff, (E.GlobalRead, E.GlobalWrite)):
        nm = 'GlobalRead' if isinstance(eff, E.GlobalRead) else 'GlobalWrite'
        return f"{tab}{nm}({eff.name})"
    elif isinstance(eff, (E.Read, E.Write, E.Reduce)):
        nm = ('Read'    if isinstance(eff, E.Read) else
              'Write'   if isinstance(eff, E.Write) else
              'Reduce')
        coords = ','.join([ str(a) for a in eff.coords ])
        return f"{tab}{nm}({eff.name},{coords})"
    else: assert False, f"bad case: {type(eff)}"

@extclass(E.eff)
def __str__(eff):
    return _effstr(eff)


# Here are codes for different location sets...
class ES(Enum):
    READ_G      = 1
    WRITE_G     = 2
    READ_H      = 3
    WRITE_H     = 4
    PRE_REDUCE  = 5

    DERIVED     = 6

    READ_ALL    = 7
    WRITE_ALL   = 8
    REDUCE      = 9
    ALL         = 10
    MODIFY      = 11
    READ_WRITE  = 12

def get_basic_locsets(effs):
    RG = LS.Empty()
    WG = LS.Empty()
    RH = LS.Empty()
    WH = LS.Empty()
    Red = LS.Empty()
    for eff in reversed(effs):
        if isinstance(eff, E.Empty):
            pass

        elif isinstance(eff, E.GlobalRead):
            ls1 = LS.Point(eff.name, [])
            RG  = LUnion(ls1, RG)
        elif isinstance(eff, E.GlobalWrite):
            ls1 = LS.Point(eff.name, [])
            RG  = LDiff(RG, ls1)
            WG  = LUnion(ls1, WG)
        elif isinstance(eff, E.Read):
            ls1 = LS.Point(eff.name, eff.coords)
            RH  = LUnion(ls1, RH)
        elif isinstance(eff, E.Write):
            ls1 = LS.Point(eff.name, eff.coords)
            RH  = LDiff(RH, ls1)
            WH  = LUnion(ls1, WH)
        elif isinstance(eff, E.Reduce):
            ls1 = LS.Point(eff.name, eff.coords)
            Red = LUnion(ls1, Red)

        elif isinstance(eff, E.BindEnv):
            RG  = LLetEnv(eff.env, RG)
            WG  = LLetEnv(eff.env, WG)
            RH  = LLetEnv(eff.env, RH)
            WH  = LLetEnv(eff.env, WH)
            Red = LLetEnv(eff.env, Red)

        elif isinstance(eff, (E.Guard,E.Loop)):
            bodyLs = get_basic_locsets(eff.body)
            if isinstance(eff, E.Guard):
                bodyLs = tuple( LFilter(eff.cond, Ls) for Ls in bodyLs )
            else:
                bodyLs = tuple( LBigUnion(eff.name, Ls) for Ls in bodyLs )
            bRG, bWG, bRH, bWH, bRed = bodyLs

            # now do the full interaction updates...
            RG  = LUnion(bRG, LDiff(RG, bWG))
            WG  = LUnion(bWG, WG)
            RH  = LUnion(bRH, LDiff(RH, bWH))
            WH  = LUnion(bWH, WH)
            Red = LUnion(bRed, Red)

        else: assert False, f"bad case: {type(eff)}"

    return (RG,WG,RH,WH,Red)


def getset(code, effs):
    if code >= ES.DERIVED:
        if code == ES.READ_ALL:
            return LUnion( getset(ES.READ_G, effs),
                           getset(ES.READ_H, effs) )
        elif code == ES.WRITE_ALL:
            return LUnion( getset(ES.WRITE_G, effs),
                           getset(ES.WRITE_H, effs) )
        elif code == ES.REDUCE:
            return LDiff( getset(ES.PRE_REDUCE, effs),
                          getset(ES.WRITE_H, effs) )
        elif code == ES.ALL:
            return LUnion( getset(ES.READ_WRITE, effs),
                           getset(ES.PRE_REDUCE, effs) )
        elif code == ES.MODIFY:
            return LUnion( getset(ES.WRITE_ALL, effs),
                           getset(ES.PRE_REDUCE, effs) )
        elif code == ES.READ_WRITE:
            return LUnion( getset(ES.READ_ALL, effs),
                           getset(ES.WRITE_ALL, effs) )
        else: assert False, f"bad case: {code}"
    else:
        RG, WG, RH, WH, Red = get_basic_locsets(effs)
        if code == ES.READ_G:
            return RG
        elif code == ES.WRITE_G:
            return WG
        elif code == ES.READ_H:
            return RH
        elif code == ES.WRITE_H:
            return WH
        elif code == ES.PRE_REDUCE:
            return Red



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Extraction of Effects from programs

def expr_effs(e):
    if isinstance(e, LoopIR.Read):
        if e.type.is_numeric():
            return [ E.Read(e.name, lift_es(e.idx)) ]
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
        return [ eff for w in e.idx for eff in w_effs(w) ]
    elif isinstance(e, LoopIR.StrideExpr):
        return []
    elif isinstance(e, LoopIR.ReadConfig):
        globname = e.config._INTERNAL_sym(e.field)
        return [ E.GlobalRead(globname) ]
    else: assert False, f"bad case: {type(e)}"

def list_expr_effs(es):
    return [ eff for e in es for eff in expr_effs(e) ]

def stmts_effs(e):
    effs = []
    for s in stmts:
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            EConstruct = E.Write if isinstance(s, LoopIR.Assign) else E.Reduce
            effs += list_expr_effs(s.idx)
            effs += expr_effs(s.rhs)
            effs.append( EConstruct(s.name, lift_es(s.idx)) )
        elif isinstance(s, LoopIR.WriteConfig):
            effs += expr_effs(s.rhs)
            globname = s.config._INTERNAL_sym(s.field)
            effs.append( E.Write(globname) )
        elif isinstance(s, LoopIR.If):
            effs += expr_effs(s.cond)
            effs += [ E.Guard( lift_e(s.cond), stmts_effs(s.body) ),
                      E.Guard( ANot(lift_e(s.cond)), stmts_effs(s.orelse) ) ]
        elif isinstance(s, (LoopIR.ForAll, LoopIR.Seq)):
            effs += expr_effs(s.hi)
            bds   = AAnd( AInt(0) <= AInt(s.iter),
                          AInt(s.iter) < lift_e(s.hi) )
            # we must prefix the body with the loop-invariant dataflow
            # analysis of the loop, since that is the only precondition
            # we are sound in assuming for global values in the loop body
            body  = ([ E.BindEnv(globenv([s])) ] + stmts_effs(s.body))
            effs += [ E.Loop(s.iter, E.Guard(bds, body)) ]
        elif isinstance(s, LoopIR.Call):
            sub_proc    = get_simple_proc(s.f)
            call_env    = call_bindings(s.args, sub_proc.args)
            effs       += [ E.BindEnv(call_env) ]
            effs       += proc_effs(sub_proc)
        elif isinstance(s, LoopIR.Alloc):
            if isinstance(s.type, T.Tensor):
                effs += list_expr_effs(s.type.hi)
        elif isinstance(s, LoopIR.WindowStmt):
            effs += expr_effs(s.rhs)
        elif isinstance(s, (LoopIR.Free,LoopIR.Pass)):
            pass
        else: assert False, f"bad case: {type(s)}"

        # secondly, insert global value modifications into
        # the sequence of effects
        effs.append( E.BindEnv(globenv([s])) )

    return effs

_proc_effs_cache = dict()
def proc_effs(proc):
    if proc not in _proc_effs_cache:
        _proc_effs_cache[proc] = stmts_effs(proc.body)
    return _proc_effs_cache[proc]
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Context Processing



class ContextExtraction:
    def __init__(self, proc, stmt):
        self.proc   = proc
        self.stmt   = stmt

    def get_control_predicate(self):
        assumed     = AAnd([ lift_e(p) for p in self.proc.preds ])
        return AAnd(assumed, self.ctrlp_stmts(self.proc.body))

    def get_pre_globenv(self):
        return self.preenv_stmts(self.proc.body)

    def get_posteffs(self):
        return self.posteff_stmts(self.proc.body)

    def ctrlp_stmts(self, stmts):
        for i,s in enumerate(stmts):
            p = self.ctrlp_s(s)
            if p is not None: # found the focused sub-tree
                G   = globenv(stmts[0:i])
                return G(p)
        assert False, "malformed context: Could not find the statement"

    def ctrlp_s(self, s):
        if s == self.stmt:
            return ABool(True)
        elif isinstance(s, LoopIR.If):
            p = self.ctrlp_stmts(s.body)
            if p is not None:
                return AAnd( lift_e(s.cond), p )
            p = self.ctrlp_stmts(s.orelse)
            if p is not None:
                return AAnd( ANot(lift_e(s.cond)), p )
            return None
        elif isinstance(s, (LoopIR.ForAll,LoopIR.Seq)):
            p = self.ctrlp_stmts(s.body)
            if p is not None:
                G   = self.loop_preenv(s)
                bds = AAnd( AInt(0) <= AInt(s.iter),
                            AInt(s.iter) < lift_e(s.hi) )
                return AAnd( bds, G(p) )
            return None
        else:
            return None

    def preenv_stmts(self, stmts):
        for i,s in enumerate(stmts):
            preG = self.preenv_s(s)
            if preG is not None: # found the focused sub-tree
                G   = globenv(stmts[0:i])
                return G + preG
        assert False, "malformed context: Could not find the statement"

    def preenv_s(self, s):
        if s == self.stmt:
            return AEnv()
        elif isinstance(s, LoopIR.If):
            preG = self.preenv_stmts(s.body)
            if preG is not None:
                return preG
            preG = self.preenv_stmts(s.orelse)
            if preG is not None:
                return preG
            return None
        elif isinstance(s, (LoopIR.ForAll,LoopIR.Seq)):
            preG = self.preenv_stmts(s.body)
            if preG is not None:
                G   = self.loop_preenv(s)
                return G + preG
            return None
        else:
            return None

    def posteffs_stmts(self, stmts):
        for i,s in enumerate(stmts):
            effs = self.ctrlp_s(s)
            if effs is not None: # found the focused sub-tree
                preG    = globenv(stmts[0:i])
                G       = globenv([s])
                return ([ E.BindEnv(preG) ] +
                        effs,
                        [ E.BindEnv(G) ] +
                        stmts_effs(stmts[i+1:]))
        assert False, "malformed context: Could not find the statement"

    def posteffs_s(self, s):
        if s == self.stmt:
            return []
        elif isinstance(s, LoopIR.If):
            effs = self.posteffs_stmts(s.body)
            if effs is not None:
                return [E.Guard( lift_e(s.cond), effs )]
            effs = self.posteffs_stmts(s.orelse)
            if effs is not None:
                return [E.Guard( ANot(lift_e(s.cond)), effs )]
            return None
        elif isinstance(s, (LoopIR.ForAll,LoopIR.Seq)):
            body = self.posteffs_stmts(s.body)
            if body is not None:
                orig_hi = lift_e(s.hi)
                hi_sym  = Sym('hi_tmp')
                hi_env  = AEnv(hi_sym, orig_hi)

                bds     = AAnd( AInt(0) <= AInt(s.iter),
                                AInt(s.iter) < AInt(hi_sym) )
                bds_sym = Sym('bds_tmp')
                hi_env  = hi_env + AEnv(bds_sym, bds)

                G       = self.loop_preenv(s)

                guard_body = LoopIR.If(LoopIR.Read(bds_sym, [],
                                                   T.bool, s.srcinfo),
                                       s.body, [], None, s.srcinfo)
                G_body  = globenv([guard_body])
                return ([ E.BindEnv(hi_env),
                          E.BindEnv(G),
                          E.Guard( ABool(bds_sym), body ),
                          E.BindEnv(G_body)
                        ] + self.loop_posteff(s,
                                LoopIR.Read(hi_sym, [],T.index, s.srcinfo)))
            return None
        else:
            return None

    def loop_preenv(self, s):
        assert isinstance(s, (LoopIR.ForAll, LoopIR.Seq))
        old_i    = LoopIR.Read(s.iter, [], T.index, s.srcinfo)
        new_i    = LoopIR.Read(s.iter.copy(), [], T.index, s.srcinfo)
        pre_body = SubstArgs(s.body, { s.iter : new_i }).result()
        pre_loop = LoopIR.Seq( new_i.name, old_i, pre_body,
                               None, s.srcinfo)
        return globenv([pre_loop])

    def loop_posteff(self, s, hi):
        # want to generate a loop
        #   for x' in seq(x+1, hi): s
        # but instead generate
        #   for x' in seq(0, hi-(x+1)): [x' -> x' + (x+1)]s
        assert isinstance(s, (LoopIR.ForAll, LoopIR.Seq))
        old_i       = LoopIR.Read(s.iter, [], T.index, s.srcinfo)
        new_i       = LoopIR.Read(s.iter.copy(), [], T.index, s.srcinfo)
        old_plus1   = LoopIR.BinOp('+',old_i,LoopIR.Const(1,T.int,s.srcinfo),
                                   T.index,s.srcinfo)
        upper       = LoopIR.BinOp('-', hi, old_plus1, T.index, s.srcinfo)
        shift_new   = LoopIR.BinOp('+', new_i, old_plus1, T.index, s.srcinfo)
        post_body   = SubstArgs(s.body, { s.iter : shift_new }).result()
        post_loop   = LoopIR.Seq( new_i.name, upper, post_body,
                                  None, s.srcinfo)
        return stmts_effs([post_loop])







