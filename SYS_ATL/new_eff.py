from adt import ADT
from . import LoopIR
from .configs import Config
from .prelude import *

from collections import ChainMap

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
            | Const( object val )
            | BinOp( binop op, expr lhs, expr rhs )
            | Stride( sym name, int dim )
            | Select( expr cond, expr tcase, expr fcase )
        --  | ConfigField( config config, string field )
            | Forall( sym name, expr arg )
            | Exists( sym name, expr arg )
            | Definitely( expr arg )
            | Maybe( expr arg )
            | Let( sym* names, expr* rhs, expr body )
            attributes( type type, srcinfo srcinfo )
} """, {
    'sym':     lambda x: isinstance(x, Sym),
    'type':    lambda x: LoopIR.T.is_type(x),
    'binop':   lambda x: x in front_ops,
    'config':  lambda x: isinstance(x, Config),
    'srcinfo': lambda x: isinstance(x, SrcInfo),
})

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
    "==":       "=",
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
        return f"¬{_estr(e.arg,op_prec["unary"],tab=tab)}"
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
    elif isinstance(e, A.Select):
        local_prec = op_prec["ternary"]
        cond = _estr(e.cond,tab=tab)
        tcase = _estr(e.tcase, prec=local_prec + 1,tab=tab)
        fcase = _estr(e.fcase, prec=local_prec + 1,tab=tab)
        if local_prec < prec:
            return f"(({cond})? {tcase} : {fcase})"
        else:
            return f"({cond})? {tcase} : {fcase}"
    #elif isinstance(e, A.ConfigField):
    #    return f"{e.config.name()}.{e.field}"
    elif isinstance(e, (A.Forall, A.Exists)):
        op = "∀" if isinstance(e, A.Forall) else "∃"
        return f"{op}{e.name},{_estr(e.arg,op_prec["forall"],tab=tab)}"
    elif isinstance(e, (A.Definitely, A.Maybe)):
        op = "D" if isinstance(e, A.Definitely) else "M"
        return f"{op}{_estr(e.arg,op_prec["unary"],tab=tab)}"
    elif isinstance(e, A.Let):
        binds   = [ f"\n{tab}{x} = {_estr(rhs,tab=tab+"  ")}"
                    for x,rhs in zip(e.names,e.rhs) ]
        body    = _estr(e.body,tab=tab+"  ")
        s = f"let{binds}\n{tab}in {body}\n{tab}"
        return f"({s})\n{tab}" if prec > 0 else s
    else:
        assert False, "bad case"

@extclass(A.expr)
def __str__(e):
    return _estr(e)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# SMT Solver wrapper; handles ternary logic etc.

class SMTSolver:
    def __init__(self):
        self.env        = ChainMap()
        self.stride_sym = dict()
        self.solver     = _get_smt_solver()

    def push(self):
        self.solver.push()
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents
        self.solver.pop()

    def bind(self, names, rhs):
        """ bind will make sure the provided names are equal to
            the provided right-hand-sides for the remainder of the
            scope that we're in """
        # note that it's important that we
        # lower all right-hand-sides before calling _newvar
        # or else the name shadowing might be incorrect
        smt_rhs     = [ self._lower(e) for e in rhs ]
        lhs         = [ self._newvar(sym, typ=e.type)
                        for sym,e in zip(names,rhs) ]
        for x,smt_e,e in zip(lhs,smt_rhs,rhs):
            EQ = SMT.Iff if e.type == T.bool else SMT.Equals
            self.solver.add_assertion(EQ(x, smt_e))

    def assume(self, e):
        assert e.type is T.bool
        smt_e       = self._lower(e)
        self.solver.add_assertion(smt_e)

    def satisfy(self, e):
        assert e.type is T.bool
        smt_e       = self._lower(e)
        is_sat      = self.solver.is_sat(smt_e)
        return is_sat

    def verify(self, e):
        assert e.type is T.bool
        smt_e       = self._lower(e)
        is_valid    = self.solver.is_valid(smt_e)
        return is_valid

    def counter_example(self):
        env_syms = [ (sym,smt) for sym,smt in self.env.items()
                     if (smt.get_type() == SMT.INT or
                         smt.get_type() == SMT.BOOL) ]
        smt_syms = [ smt for sym,smt in env_syms ]
        val_map = self.solver.get_py_values(smt_syms)

        mapping = { sym : val_map[smt] for sym,smt in env_syms }
        return mapping

    def _getvar(self,sym, typ=T.index):
        if sym not in self.env:
            if typ.is_indexable() or typ.is_stridable():
                self.env[sym] = SMT.Symbol(repr(sym), SMT.INT)
            elif typ is T.bool:
                self.env[sym] = SMT.Symbol(repr(sym), SMT.BOOL)
        return self.env[sym]

    def _newvar(self,sym, typ=T.index):
        """ make sure that we have a new distinct copy of this name."""
        if sym not in self.env:
            return self._getvar(sym,typ)
        else:
            if typ.is_indexable() or typ.is_stridable():
                self.env[sym] = SMT.Symbol(repr(sym.copy()), SMT.INT)
            elif typ is T.bool:
                self.env[sym] = SMT.Symbol(repr(sym.copy()), SMT.BOOL)
            return self.env[sym]

    def _lower(self, e):
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
            raise NotImplementedError("TODO: Unk")
        elif isinstance(e, A.Not):
            return SMT.Not( self._lower(e.arg) )
        elif isinstance(e, A.Stride):
            key     = (e.name, e.dim)
            if key not in self.stride_sym:
                self.stride_sym[key] = Sym(f"{e.name}_stride_{e.dim}")
            sym     = self.stride_sym[key]
            return self._getvar(sym)
        elif isinstance(e, A.Select):
            cond    = self._lower(e.cond)
            tcase   = self._lower(e.tcase)
            fcase   = self._lower(e.fcase)
            return SMT.Ite(cond, tcase, fcase)
        elif isinstance(e, (A.Forall,A.Exists)):
            
"""
    expr    = Var( sym name )
            | Unk() -- unknown
            | Not( expr arg )
            | Const( object val )
            | BinOp( binop op, expr lhs, expr rhs )
            | Stride( sym name, int dim )
            | Select( expr cond, expr tcase, expr fcase )
            | Forall( sym name, expr arg )
            | Exists( sym name, expr arg )
            | Definitely( expr arg )
            | Maybe( expr arg )
            | Let( sym* names, expr* rhs, expr body )
"""

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Effect grammar
