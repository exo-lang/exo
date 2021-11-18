from adt import ADT
from . import LoopIR
from .configs import Config
from .prelude import *

from collections    import ChainMap, OrderedDict
from itertools      import chain
from dataclasses    import dataclass
from typing         import Any

from .LoopIR import T

import pysmt
from pysmt import shortcuts as SMT


def _get_smt_solver():
    factory = pysmt.factory.Factory(pysmt.shortcuts.get_env())
    slvs    = factory.all_solvers()
    if len(slvs) == 0: raise OSError("Could not find any SMT solvers")
    return pysmt.shortcuts.Solver(name=next(iter(slvs)))

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Analysis Expr

front_ops = {
    "+":    True,
    "-":    True,
    "*":    True,
    "/":    True,
    "%":    True,
    #
    "<":    True,
    ">":    True,
    "<=":   True,
    ">=":   True,
    "==":   True,
    #
    "and":  True,
    "or":   True,
    "==>":  True,
}

A = ADT("""
module AExpr {
    expr    = Var( sym name )
            | Unk() -- unknown
            | Not( expr arg )
            | USub( expr arg )
            | Const( object val )
            | BinOp( binop op, expr lhs, expr rhs )
            | Stride( sym name, int dim )
            | LetStrides( sym name, expr* strides )
            | Select( expr cond, expr tcase, expr fcase )
            | ForAll( sym name, expr arg )
            | Exists( sym name, expr arg )
            | Definitely( expr arg )
            | Maybe( expr arg )
            | Tuple( expr* args )
            | LetTuple( sym* names, expr rhs, expr body )
            | Let( sym* names, expr* rhs, expr body )
            attributes( type type, srcinfo srcinfo )
} """, {
    'sym':     lambda x: isinstance(x, Sym),
    'type':    lambda x: type(x) is tuple or LoopIR.T.is_type(x),
    'binop':   lambda x: x in front_ops,
    'config':  lambda x: isinstance(x, Config),
    'srcinfo': lambda x: isinstance(x, SrcInfo),
})

# constructor helpers...
def AInt(x):
    if type(x) is int:
        return A.Const(x, T.int, null_srcinfo())
    elif isinstance(x, Sym):
        return A.Var(x, T.index, null_srcinfo())
    else: assert False, f"bad type {type(x)}"
def ABool(x):
    if type(x) is bool:
        return A.Const(x, T.bool, null_srcinfo())
    elif isinstance(x, Sym):
        return A.Var(x, T.bool, null_srcinfo())
    else: assert False, f"bad type {type(x)}"

def ALet(nm,rhs,body):
    assert isinstance(nm, Sym)
    names   = [nm]
    rhs     = [rhs]
    if isinstance(body, A.Let):
        names   = names + body.names
        rhs     = rhs   + body.rhs
        body    = body.body
    return A.Let(names, rhs, body, body.type, body.srcinfo)

def ALetStride(nm,strides,body):
    return A.LetStrides(nm, strides, body, body.type, body.srcinfo)

def ANot(x):
    return A.Not(x, T.bool, x.srcinfo)
def AAnd(*args):
    if len(args) == 0:
        return A.Const(True,T.bool,null_srcinfo())
    res = args[0]
    for a in args[1:]:
        res = A.And(res, a, T.bool, a.srcinfo)
    return res
def AOr(*args):
    if len(args) == 0:
        return A.Const(False,T.bool,null_srcinfo())
    res = args[0]
    for a in args[1:]:
        res = A.Or(res, a, T.bool, a.srcinfo)
    return res
def AImplies(lhs,rhs):
    return A.BinOp('==>', lhs, rhs, T.bool, lhs.srcinfo)
def AEq(lhs,rhs):
    return A.BinOp('==', lhs, rhs, T.bool, lhs.srcinfo)


@extclass(A.expr)
def __neg__(arg):
    return A.USub(arg, T.bool, arg.srcinfo)
# USub
# Binop
#   + - * / %  < > <= >= ==  and or
@extclass(A.expr)
def __add__(lhs,rhs):
    return A.BinOp('+', lhs, rhs, T.index, lhs.srcinfo)
@extclass(A.expr)
def __sub__(lhs,rhs):
    return A.BinOp('-', lhs, rhs, T.index, lhs.srcinfo)
@extclass(A.expr)
def __mul__(lhs,rhs):
    return A.BinOp('*', lhs, rhs, T.index, lhs.srcinfo)
@extclass(A.expr)
def __truediv__(lhs,rhs):
    return A.BinOp('/', lhs, rhs, T.index, lhs.srcinfo)
@extclass(A.expr)
def __mod__(lhs,rhs):
    return A.BinOp('%', lhs, rhs, T.index, lhs.srcinfo)

@extclass(A.expr)
def __lt__(lhs,rhs):
    return A.BinOp('<', lhs, rhs, T.bool, lhs.srcinfo)
@extclass(A.expr)
def __gt__(lhs,rhs):
    return A.BinOp('>', lhs, rhs, T.bool, lhs.srcinfo)
@extclass(A.expr)
def __le__(lhs,rhs):
    return A.BinOp('<=', lhs, rhs, T.bool, lhs.srcinfo)
@extclass(A.expr)
def __ge__(lhs,rhs):
    return A.BinOp('>=', lhs, rhs, T.bool, lhs.srcinfo)


op_prec = {
    "exists":   10,
    "forall":   10,
    "==>":      10,
    #
    "ternary":  20,
    #
    "or":       30,
    #
    "and":      40,
    #
    "<":        50,
    ">":        50,
    "<=":       50,
    ">=":       50,
    "==":       50,
    #
    "+":        60,
    "-":        60,
    #
    "*":        70,
    "/":        70,
    "%":        70,
    #
    "unary":    90,
}

binop_print = {
    "==>":      "⇒",
    "or":       "∨",
    "and":      "∧",
    "<":        "<",
    ">":        ">",
    "<=":       "≤",
    ">=":       "≥",
    "==":       "==",
    "+":        "+",
    "-":        "-",
    "*":        "*",
    "/":        "/",
    "%":        "%",
}

def _estr(e, prec=0, tab=""):
    if isinstance(e, A.Var):
        return str(e.name)
    elif isinstance(e, A.Unk):
        return "⊥"
    elif isinstance(e, A.Not):
        return f"¬{_estr(e.arg,op_prec['unary'],tab=tab)}"
    elif isinstance(e, A.USub):
        return f"-{_estr(e.arg,op_prec['unary'],tab=tab)}"
    elif isinstance(e, A.Const):
        return str(e.val)
    elif isinstance(e, A.BinOp):
        local_prec = op_prec[e.op]
        lhs = _estr(e.lhs, prec=local_prec,tab=tab)
        rhs = _estr(e.rhs, prec=local_prec + 1,tab=tab)
        if local_prec < prec:
            return f"({lhs} {binop_print[e.op]} {rhs})"
        else:
            return f"{lhs} {binop_print[e.op]} {rhs}"
    elif isinstance(e, A.Stride):
        return f"stride({e.name},{e.dim})"
    elif isinstance(e, A.LetStrides):
        strides = ','.join([ _estr(s,tab=tab+'  ') for s in e.strides ])
        bind    = f"{tab}{e.name} = ({strides})"
        body    = _estr(e.body,tab=tab+"  ")
        s = f"letStride {bind}\n{tab}in {body}"
        return f"({s}\n{tab})" if prec > 0 else s
    elif isinstance(e, A.Select):
        local_prec = op_prec["ternary"]
        cond = _estr(e.cond,tab=tab)
        tcase = _estr(e.tcase, prec=local_prec + 1,tab=tab)
        fcase = _estr(e.fcase, prec=local_prec + 1,tab=tab)
        if local_prec < prec:
            return f"(({cond})? {tcase} : {fcase})"
        else:
            return f"({cond})? {tcase} : {fcase}"
    elif isinstance(e, (A.ForAll, A.Exists)):
        op = "∀" if isinstance(e, A.ForAll) else "∃"
        return f"{op}{e.name},{_estr(e.arg,op_prec['forall'],tab=tab)}"
    elif isinstance(e, (A.Definitely, A.Maybe)):
        op = "D" if isinstance(e, A.Definitely) else "M"
        return f"{op}{_estr(e.arg,op_prec['unary'],tab=tab)}"
    elif isinstance(e, A.Let):
        # compress nested lets for printing
        if isinstance(e.body, A.Let):
            return _estr(A.Let(e.names + e.body.names,
                               e.rhs   + e.body.rhs,
                               e.body.body,
                               e.type, e.srcinfo), prec=prec,tab=tab)
        binds   = "\n".join([ f"{tab}{x} = {_estr(rhs,tab=tab+'  ')}"
                              for x,rhs in zip(e.names,e.rhs) ])
        body    = _estr(e.body,tab=tab+"  ")
        s = f"let\n{binds}\n{tab}in {body}"
        return f"({s}\n{tab})" if prec > 0 else s
    elif isinstance(e, A.Tuple):
        args    = ', '.join([ _estr(a,tab=tab) for a in e.args ])
        return f"({args})"
    elif isinstance(e, A.LetTuple):
        names   = ','.join(e.names)
        bind    = f"{tab}{names} = {_estr(e.rhs,tab=tab+'  ')}"
        body    = _estr(e.body,tab=tab+"  ")
        s = f"let_tuple {bind}\n{tab}in {body}"
        return f"({s}\n{tab})" if prec > 0 else s
    else:
        assert False, "bad case"

@extclass(A.expr)
def __str__(e):
    return _estr(e)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# SMT Solver wrapper; handles ternary logic etc.

class DebugSolverFrame:
    def __init__(self):
        self.commands = []

    def str_lines(self,show_smt=False):
        lines = []
        for c in self.commands:
            if c[0] == 'bind':
                cmd, names, rhs, smt = c
                cmd = "bind    "
                for nm,r in zip(names,rhs):
                    lines.append(f"{cmd}{nm} = {r}")
                    if show_smt:
                        lines.append(f"    smt {SMT.to_smtlib(smt)}")
                    cmd = "        "
            elif c[0] == 'tuplebind':
                cmd, names, rhs, smt = c
                nms = ','.join([str(n) for n in names])
                lines.append(f"bind    {nms} = {rhs}")
                assert type(smt) is tuple
                if show_smt:
                    for s in smt:
                        lines.append(f"    smt {SMT.to_smtlib(s)}")
            elif c[0] == 'assume':
                cmd, e, smt = c
                lines.append(f"assume  {e}")
                if show_smt:
                    lines.append(f"    smt {SMT.to_smtlib(smt)}")
            else: assert False, "bad case"
        return lines

    def add_bind(self, names,rhs,smt):
        self.commands.append(('bind',names,rhs,smt))

    def add_tuple_bind(self, names,rhs,smt):
        self.commands.append(('tuplebind',names,rhs,smt))

    def add_assumption(self, e, smt_e):
        self.commands.append(('assume',e,smt_e))

@dataclass
class TernVal:
    v : Any
    d : Any

    def __str__(self):
        return f"({self.v},{self.d})"

def is_ternary(x):
    return isinstance(x, TernVal)
def to_ternary(x):
    return x if is_ternary(x) else TernVal(x,SMT.Bool(True))

class SMTSolver:
    def __init__(self):
        self.env            = ChainMap()
        self.stride_sym     = ChainMap()
        self.solver         = _get_smt_solver()

        # used during lowering
        self.mod_div_tmps   = []

        # debug info
        self.frames         = [DebugSolverFrame()]

    def push(self):
        self.solver.push()
        self.internal_push()

    def pop(self):
        self.internal_pop()
        self.solver.pop()

    def internal_push(self):
        self.env = self.env.new_child()
        self.stride_sym = self.stride_sym.new_child()
        self.frames.append(DebugSolverFrame())

    def internal_pop(self):
        self.frames.pop()
        self.stride_sym = self.stride_sym.parents
        self.env = self.env.parents

    def debug_str(self,smt=False):
        lines = []
        for f in self.frames:
            lns     = f.str_lines(show_smt=smt)
            if len(lines) > 0 and len(lns) > 0:
                lines.append('')
            lines += lns
        return "\n".join(lines)

    # deprecated
    def defvar(self, names, rhs):
        """ bind will make sure the provided names are equal to
            the provided right-hand-sides for the remainder of the
            scope that we're in """
        # note that it's important that we
        # lower all right-hand-sides before calling _newvar
        # or else the name shadowing might be incorrect
        smt_rhs         = [ self._lower(e) for e in rhs ]
        self.frames[-1].add_bind(names, rhs, smt_rhs)
        self._bind_internal(names, smt_rhs, [ e.type for e in rhs ])

    #def bind_tuple(self, names, rhs):
    #    """ bind will make sure the provided names are equal to
    #        the provided right-hand-sides for the remainder of the
    #        scope that we're in.
    #        bind_tuple will simultaneously bind the whole list of
    #        names to a tuple-type right-hand-side """
    #    smt_rhs         = self._lower(rhs)
    #    self.frames[-1].add_tuple_bind(names, rhs, smt_rhs)
    #    self._bind_internal(e.names, smt_rhs, rhs.type)

    # deprecated
    def _def_internal(self, names, smt_rhs, typs):
        # we further must handle the cases where the variables
        # being defined are classical vs. ternary
        for x,smt_e,typ in zip(names,smt_rhs,typs):
            smt_sym     = self._newvar(x, typ, is_ternary(smt_e))
            EQ          = SMT.Iff if typ == T.bool else SMT.Equals
            if is_ternary(smt_e):
                self.solver.add_assertion(EQ(smt_sym.v, smt_e.v))
                self.solver.add_assertion(SMT.Iff(smt_sym.d, smt_e.d))
            else:
                self.solver.add_assertion(EQ(smt_sym, smt_e))

    def _bind(self, names, rhs):
        """ bind will make sure the provided names are equal to
            the provided right-hand-sides for the remainder of the
            scope that we're in """
        for x,e in zip(names, rhs):
            smt_e       = self._lower(e)
            self.env[x] = smt_e

    def _bind_tuple(self, names, rhs):
        """ bind will make sure the provided names are equal to
            the provided right-hand-sides for the remainder of the
            scope that we're in.
            bind_tuple will simultaneously bind the whole list of
            names to a tuple-type right-hand-side """
        smt_rhs         = self._lower(rhs)
        for x,e in zip(names, smt_rhs):
            self.env[x] = e

    def assume(self, e):
        assert e.type is T.bool
        smt_e       = self._lower(e)
        self.frames[-1].add_assumption(e, smt_e)
        assert not is_ternary(smt_e), "assumptions must be classical"
        self.solver.add_assertion(smt_e)

    def satisfy(self, e):
        assert e.type is T.bool
        smt_e       = self._lower(e)
        assert not is_ternary(smt_e), "formulas must be classical"
        is_sat      = self.solver.is_sat(smt_e)
        return is_sat

    def verify(self, e):
        assert e.type is T.bool
        smt_e       = self._lower(e)
        assert not is_ternary(smt_e), "formulas must be classical"
        is_valid    = self.solver.is_valid(smt_e)
        return is_valid

    def counter_example(self):
        def keep_sym(s):
            if type(s) is tuple:
                return True
            else:
                return (smt.get_type() == SMT.INT or
                        smt.get_type() == SMT.BOOL)
        env_syms = [ (sym,smt) for sym,smt in self.env.items()
                     if keep_sym(smt) ]
        smt_syms = []
        for _,smt in env_syms:
            if is_ternary(smt):
                smt_syms += [smt.v,smt.d]
            else:
                smt_syms.append(smt)
        val_map = self.solver.get_py_values(smt_syms)

        mapping = dict()
        for sym,smt in env_syms:
            if is_ternary(smt):
                x,d = val_map[smt.v], val_map[smt.d]
                mapping[sym] = 'unknown' if not d else x
            else:
                mapping[sym] = val_map[smt]
        return mapping

    def _get_stride_sym(self, name, dim):
        key     = (name, dim)
        if key not in self.stride_sym:
            self.stride_sym[key] = Sym(f"{e.name}_stride_{e.dim}")
        sym     = self.stride_sym[key]
        return sym

    def _getvar(self,sym, typ=T.index):
        if sym not in self.env:
            if typ.is_indexable() or typ.is_stridable():
                self.env[sym] = SMT.Symbol(repr(sym), SMT.INT)
            elif typ is T.bool:
                self.env[sym] = SMT.Symbol(repr(sym), SMT.BOOL)
        return self.env[sym]

    def _newvar(self,sym, typ=T.index, ternary=False):
        """ make sure that we have a new distinct copy of this name."""
        nm = repr(sym) if sym not in self.env else repr(sym.copy())
        smt_typ     = (SMT.INT if typ.is_indexable() or typ.is_stridable()
                            else SMT.BOOL)
        smt_sym     = SMT.Symbol(nm, smt_typ)
        if ternary:
            self.env[sym] = TernVal(smt_sym, SMT.Symbol(nm+"_def", SMT.BOOL))
        else:
            self.env[sym] = smt_sym
        return self.env[sym]

    def _add_mod_div_eq(self, new_sym, eq):
        self.mod_div_tmps.append((new_sym,eq))

    def _lower(self, e):
        smt_e = self._lower_body(e)
        # possibly empty the definition queue
        if len(self.mod_div_tmps) > 0 and e.type == T.bool:
            for sym,eq in self.mod_div_tmps:
                smt_e = SMT.ForAll([sym],SMT.Implies(eq,smt_e))
            self.mod_div_tmps = []
        # pass up the result
        return smt_e

    def _lower_body(self, e):
        if isinstance(e, A.Const):
            if e.type == T.bool:
                return SMT.Bool(e.val)
            elif e.type.is_indexable():
                return SMT.Int(e.val)
            else:
                assert False, f"unrecognized const type: {type(e.val)}"
        elif isinstance(e, A.Var):
            return self._getvar(e.name, e.type)
        elif isinstance(e, A.Unk):
            val     = SMT.Bool(False) if e.type == T.bool else SMT.Int(0)
            return TernVal(val, SMT.Bool(False))
        elif isinstance(e, A.Not):
            assert e.arg.type == T.bool
            a       = self._lower(e.arg)
            if is_ternary(a):
                return TernVal(SMT.Not(a.v),a.d)
            else:
                return SMT.Not(a)
        elif isinstance(e, A.USub):
            assert e.arg.type.is_indexable()
            a       = self._lower(e.arg)
            if is_ternary(a):
                return TernVal(SMT.Minus(SMT.Int(0), a.v),a.d)
            else:
                return SMT.Minus(SMT.Int(0), a)
        elif isinstance(e, A.Stride):
            return self._getvar(self._get_stride_sym(e.name, e.dim))
        elif isinstance(e, A.LetStrides):
            self.internal_push()
            for i,s in enumerate(e.strides):
                stridesym = self._get_stride_sym(e.name, i)
                self.env[stridesym] = self._lower(s)
            body = self._lower(e.body)
            self.internal_pop()
            return body
        elif isinstance(e, A.Select):
            assert e.cond.type == T.bool
            cond    = self._lower(e.cond)
            tcase   = self._lower(e.tcase)
            fcase   = self._lower(e.fcase)
            if is_ternary(cond) or is_ternary(tcase) or is_ternary(fcase):
                c   = to_ternary(cond)
                t   = to_ternary(tcase)
                f   = to_ternary(fcase)
                return (SMT.Ite(c.v,t.v,f.v),
                        SMT.And(c.d,SMT.Ite(c.v,t.d,f.d)))
            else:
                return SMT.Ite(cond, tcase, fcase)
        elif isinstance(e, (A.ForAll,A.Exists)):
            assert e.arg.type == T.bool
            self.internal_push()
            nm  = self._newvar(e.name)
            a   = self._lower(e.arg)
            self.internal_pop()
            OP  = SMT.ForAll if isinstance(e, A.ForAll) else SMT.Exists
            if is_ternary(a):
                # forall defined when (forall nm. d) \/ (exists nm. ¬a /\ d)
                # exists defined when (forall nm. d) \/ (exists nm. a /\ d)
                short_a = a.v if isinstance(e, A.Exists) else SMT.Not(a.v)
                return ( OP([nm],a.v),
                         SMT.Or( SMT.ForAll([nm], a.d),
                                 SMT.Exists([nm], SMT.And(short_a, a.d)) ) )
            else:
                return OP([nm],a)
        elif isinstance(e, (A.Definitely,A.Maybe)):
            assert a.arg.type == T.bool
            a       = self._lower(e.arg)
            if is_ternary(a):
                if isinstance(e, A.Definitely):
                    return SMT.And(a.v, a.d)
                else:
                    return SMT.Or(a.v, SMT.Not(a.d))
            else:
                return a
        elif isinstance(e, A.Let):
            self.internal_push()
            self._bind(e.names, e.rhs)
            body    = self._lower(e.body)
            self.internal_pop()
            return body
        elif isinstance(e, A.Tuple):
            return tuple( self._lower(a) for a in e.args )
        elif isinstance(e, A.LetTuple):
            assert type(e.rhs.type) is tuple
            assert len(e.names) == len(e.rhs.type)
            self.internal_push()
            self._bind_tuple(e.names, e.rhs)
            body    = self._lower(e.body)
            self.internal_pop()
            return body
        elif isinstance(e, A.BinOp):
            lhs     = self._lower(e.lhs)
            rhs     = self._lower(e.rhs)
            tern    = is_ternary(lhs) or is_ternary(rhs)
            if tern:
                lhs = to_ternary(lhs)
                rhs = to_ternary(rhs)
                lhs, dl = lhs.v, lhs.d
                rhs, dr = rhs.v, rhs.d
                dval    = SMT.And(dl, dr) # default for int sort

            if e.op == "+":
                val = SMT.Plus(lhs, rhs)
            elif e.op == "-":
                val = SMT.Minus(lhs, rhs)
            elif e.op == "*":
                val = SMT.Times(lhs, rhs)
            elif e.op == "/":
                assert isinstance(e.rhs, A.Const)
                assert e.rhs.val > 0
                # Introduce new Sym (z in formula below)
                div_tmp = self._getvar(Sym("div_tmp"))
                # rhs*z <= lhs < rhs*(z+1)
                rhs_eq  = SMT.LE(SMT.Times(rhs, div_tmp), lhs)
                lhs_eq  = SMT.LT(lhs,
                                 SMT.Times(rhs, SMT.Plus(div_tmp,
                                                         SMT.Int(1))))
                self._add_mod_div_eq(div_tmp, SMT.And(rhs_eq, lhs_eq))
                val     = div_tmp
            elif e.op == "%":
                assert isinstance(e.rhs, A.Const)
                assert e.rhs.val > 0
                # In the below, copy the logic above for division
                # to construct `mod_tmp` s.t.
                #   mod_tmp = floor(lhs / rhs)
                # Then,
                #   lhs % rhs = lhs - rhs * mod_tmp
                mod_tmp = self._getvar(Sym("mod_tmp"))
                rhs_eq  = SMT.LE(SMT.Times(rhs, mod_tmp), lhs)
                lhs_eq  = SMT.LT(lhs,
                                 SMT.Times(rhs, SMT.Plus(mod_tmp,
                                                         SMT.Int(1))))
                self._add_mod_div_eq(mod_tmp, SMT.And(rhs_eq, lhs_eq))
                val     = SMT.Minus(lhs, SMT.Times(rhs, mod_tmp))

            elif e.op == "<":
                val = SMT.LT(lhs, rhs)
            elif e.op == ">":
                val = SMT.GT(lhs, rhs)
            elif e.op == "<=":
                val = SMT.LE(lhs, rhs)
            elif e.op == ">=":
                val = SMT.GE(lhs, rhs)
            elif e.op == "==":
                if e.lhs.type == T.bool and e.rhs.type == T.bool:
                    val = SMT.Iff(lhs, rhs)
                elif (e.lhs.type.is_indexable() and
                      e.rhs.type.is_indexable()):
                    val = SMT.Equals(lhs, rhs)
                elif (e.lhs.type.is_stridable() and
                      e.rhs.type.is_stridable()):
                    val = SMT.Equals(lhs, rhs)
                else:
                    assert False, "bad case"
            elif e.op == "and":
                val = SMT.And(lhs, rhs)
                if tern:
                    dval    = SMT.Or(SMT.And(dl,dr),
                                     SMT.And(SMT.Not(lhs),dl),
                                     SMT.And(SMT.Not(rhs),dr))
            elif e.op == "or":
                val = SMT.Or(lhs, rhs)
                if tern:
                    dval    = SMT.Or(SMT.And(dl,dr),
                                     SMT.And(lhs,dl),
                                     SMT.And(rhs,dr))
            elif e.op == "==>":
                val = SMT.Implies(lhs, rhs)
                if tern:
                    dval    = SMT.Or(SMT.And(dl,dr),
                                     SMT.And(SMT.Not(lhs),dl),
                                     SMT.And(rhs,dr))

            else: assert False, f"bad op: {e.op}"

            return TernVal(val,dval) if tern else val

        else: assert False, f"bad case: {type(e)}"


