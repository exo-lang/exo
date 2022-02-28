from collections import ChainMap
from dataclasses import dataclass
from typing import Any, Union

import pysmt
import z3 as z3lib
from pysmt import logics
from pysmt import shortcuts as SMT

from asdl_adt import ADT, validators
from asdl_adt.validators import ValidationError
from .LoopIR import T, LoopIR
from .prelude import *

_first_run = True
def _get_smt_solver():
    factory = pysmt.factory.Factory(pysmt.shortcuts.get_env())
    slvs    = factory.all_solvers(logic=logics.LIA)
    if len(slvs) == 0: raise OSError("Could not find any SMT solvers")
    return pysmt.shortcuts.Solver(name=next(iter(slvs)))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Analysis Expr

def is_type_bound(val):
    if isinstance(val, (tuple, LoopIR.type)):
        return val
    raise ValidationError(Union[tuple, LoopIR.type], type(val))


class AOp(str):
    front_ops = {"+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "and",
                 "or", "==>"}

    def __new__(cls, op):
        op = str(op)
        if op in AOp.front_ops:
            return super().__new__(cls, op)
        raise ValueError(f'invalid operator: {op}')


A = ADT("""
module AExpr {
    expr    = Var( sym name )
            | Unk() -- unknown
            | Not( expr arg )
            | USub( expr arg )
            | Const( object val )
            | BinOp( binop op, expr lhs, expr rhs )
            | Stride( sym name, int dim )
            | LetStrides( sym name, expr* strides, expr body )
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
    'sym':     Sym,
    'type':    is_type_bound,
    'binop':   validators.instance_of(AOp, convert=True),
    'srcinfo': SrcInfo,
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

def ALet(names,rhs,body):
    assert isinstance(names, list) and isinstance(rhs, list)
    assert all(isinstance(n,Sym) for n in names)
    assert all(isinstance(r,A.expr) for r in rhs)
    if isinstance(body, A.Let):
        names   = names + body.names
        rhs     = rhs   + body.rhs
        body    = body.body
    if len(names) == 0:
        return body
    else:
        return A.Let(names, rhs, body, body.type, body.srcinfo)

def ALetStride(nm,strides,body):
    if len(strides) == 0:
        return body
    else:
        return A.LetStrides(nm, strides, body, body.type, body.srcinfo)

def ALetTuple(names,rhs,body):
    if len(names) == 0:
        return body
    else:
        return A.LetTuple(names, rhs, body, body.type, body.srcinfo)

def AForAll(names, body):
    for nm in reversed(names):
        body = A.ForAll(nm, body, T.bool, body.srcinfo)
    return body
def AExists(names, body):
    for nm in reversed(names):
        body = A.Exists(nm, body, T.bool, body.srcinfo)
    return body

def ANot(x):
    return A.Not(x, T.bool, x.srcinfo)
def AAnd(*args):
    if len(args) == 0:
        return A.Const(True,T.bool,null_srcinfo())
    res = args[0]
    for a in args[1:]:
        res = A.BinOp('and', res, a, T.bool, a.srcinfo)
    return res
def AOr(*args):
    if len(args) == 0:
        return A.Const(False,T.bool,null_srcinfo())
    res = args[0]
    for a in args[1:]:
        res = A.BinOp('or', res, a, T.bool, a.srcinfo)
    return res
def AImplies(lhs,rhs):
    return A.BinOp('==>', lhs, rhs, T.bool, lhs.srcinfo)
def AEq(lhs,rhs):
    return A.BinOp('==', lhs, rhs, T.bool, lhs.srcinfo)

def AMay(arg):
    return A.Maybe(arg, T.bool, arg.srcinfo)
def ADef(arg):
    return A.Definitely(arg, T.bool, arg.srcinfo)

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
        bind    = f"{e.name} = ({strides})"
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
        op          = "∀" if isinstance(e, A.ForAll) else "∃"
        local_prec  = op_prec['forall' if isinstance(e,A.ForAll) else 'exists']
        s = f"{op}{e.name},{_estr(e.arg,op_prec['forall'],tab=tab)}"
        if local_prec < prec:
            s = f"({s})"
        return s
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
        names   = ','.join([ str(n) for n in e.names])
        bind    = f"{names} = {_estr(e.rhs,tab=tab+'  ')}"
        body    = _estr(e.body,tab=tab+"  ")
        s = f"let_tuple {bind}\n{tab}in {body}"
        return f"({s}\n{tab})" if prec > 0 else s
    else:
        assert False, "bad case"

@extclass(A.expr)
def __str__(e):
    return _estr(e)



def aeFV(e,env=None):
    env     = env or ChainMap()
    def push():
        nonlocal env
        env = env.new_child()
    def pop():
        nonlocal env
        env = env.parents

    if isinstance(e, A.Var):
        if e.name not in env:
            return {e.name : e.type}
        else:
            return dict()
    elif isinstance(e, (A.Unk,A.Const)):
        return dict()
    elif isinstance(e, (A.Not,A.USub,A.Definitely, A.Maybe)):
        return aeFV(e.arg,env)
    elif isinstance(e, A.BinOp):
        return aeFV(e.lhs,env) | aeFV(e.rhs,env)
    elif isinstance(e, A.Stride):
        # stride symbol gets encoded as a tuple
        key = (e.name,e.dim)
        if key not in env:
            return {key : T.stride}
        else:
            return dict()
    elif isinstance(e, A.LetStrides):
        push()
        res = dict()
        for s in e.strides:
            res = res | aeFV(s,env)
        for i,_ in enumerate(e.strides):
            env[(e.name,i)] = True
        res = res | aeFV(e.body,env)
        pop()
        return res
    elif isinstance(e, A.Select):
        return aeFV(e.cond,env) | aeFV(e.tcase,env) | aeFV(e.fcase,env)
    elif isinstance(e, (A.ForAll, A.Exists)):
        push()
        env[e.name] = True
        res = aeFV(e.arg,env)
        pop()
        return res
    elif isinstance(e, A.Let):
        push()
        res = dict()
        for r in e.rhs:
            res = res | aeFV(r,env)
        for nm in e.names:
            env[nm] = True
        res = res | aeFV(e.body,env)
        pop()
        return res
    elif isinstance(e, A.Tuple):
        res = dict()
        for a in e.args:
            res = res | aeFV(a,env)
        return res
    elif isinstance(e, A.LetTuple):
        push()
        res = dict()
        res = aeFV(e.rhs,env)
        for nm in e.names:
            env[nm] = True
        res = res | aeFV(e.body,env)
        pop()
        return res
    else:
        assert False, "bad case"


def aeNegPos(e,pos,env=None,res=None):
    res     = res or dict()
    env     = env or dict()#ChainMap()
    def save_and_shadow(names):
        nonlocal env
        return { nm : env.pop(nm, None) for nm in names }
    def restore_shadowed(saved):
        nonlocal env
        for nm,save_val in saved.items():
            if save_val is not None:
                env[nm] = save_val
            elif nm in env:
                del env[nm]

    # set the result for this node independent of cases
    res[id(e)]  = pos

    if isinstance(e, A.Var):
        # backwards propagate through variable references
        if e.name not in env:
            env[e.name] = pos
        else:
            old_pos     = env[e.name]
            if pos != old_pos:
                env[e.name] = '0'
    elif isinstance(e, A.Stride):
        # stride symbol gets encoded as a tuple
        key = (e.name,e.dim)
        if key not in env:
            env[key] = pos
        else:
            old_pos     = env[e.name]
            if pos != old_pos:
                env[e.name] = '0'
    elif isinstance(e, (A.Unk,A.Const,A.Stride)):
        pass
    elif isinstance(e, A.Not):
        negpos = ('-' if pos == '+' else
                  '+' if pos == '-' else
                  '0')
        aeNegPos(e.arg, negpos, env, res)
    elif isinstance(e, (A.USub,A.Definitely, A.Maybe)):
        aeNegPos(e.arg, pos, env, res)
    elif isinstance(e, A.BinOp):
        aeNegPos(e.lhs, pos, env, res)
        aeNegPos(e.rhs, pos, env, res)
    elif isinstance(e, A.LetStrides):
        keys    = [ (e.name,i) for i,_ in enumerate(e.strides) ]
        # propagate negation-position flags through body
        saved   = save_and_shadow(keys)
        aeNegPos(e.body, pos, env, res)
        key_pos = { k : env.get(k, pos) for k in keys }
        restore_shadowed(saved)

        # propagate values pushed onto strides in the body back to
        # their definitions
        for k,s,kp in zip(keys, e.strides, key_pos):
            aeNegPos(s, kp, env, res)

    elif isinstance(e, A.Select):
        aeNegPos(e.cond,  pos, env, res)
        aeNegPos(e.tcase, pos, env, res)
        aeNegPos(e.fcase, pos, env, res)
    elif isinstance(e, (A.ForAll, A.Exists)):
        saved   = save_and_shadow([e.name])
        aeNegPos(e.arg, pos, env, res)
        restore_shadowed(saved)
    elif isinstance(e, A.Let):
        save_stack = [ save_and_shadow([nm]) for nm in e.names ]
        aeNegPos(e.body, pos, env, res)
        for nm,rhs,save in reversed(list(zip(e.names, e.rhs, save_stack))):
            nm_pos = env.get(nm, pos)
            restore_shadowed(save)
            aeNegPos(rhs, nm_pos, env, res)

    elif isinstance(e, A.Tuple):
        for a in e.args:
            aeNegPos(a, pos, env, res)

    elif isinstance(e, A.LetTuple):
        # propagate negation-position flags through body
        saved   = save_and_shadow(e.names)
        aeNegPos(e.body, pos, env, res)
        # merge negation position of the tuple variables
        nm_pos  = [ env[nm] for nm in e.names if nm in env ]
        if len(nm_pos) > 0:
            pos = nm_pos[0]
            for p in nm_pos[1:]:
                if p != pos:
                    pos = '0'
        # now back-propagate to the right-hand-side
        aeNegPos(e.rhs, pos, env, res)

    else:
        assert False, "bad case"

    return res


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
    def __init__(self, verbose=False):
        self.env            = ChainMap()
        self.stride_sym     = ChainMap()
        self.solver         = _get_smt_solver()
        self.verbose        = verbose
        self.z3             = Z3SubProc()

        # used during lowering
        self.mod_div_tmp_bins = []
        self.negative_pos   = None

        # debug info
        self.frames         = [DebugSolverFrame()]

    def push(self):
        self.solver.push()
        self.z3.push()
        self.internal_push()

    def pop(self):
        self.internal_pop()
        self.z3.pop()
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

    def _add_free_vars(self, e):
        fv = aeFV(e)
        for x,typ in fv.items():
            if type(x) is tuple:
                x = self._get_stride_sym(*x)
            if x in self.env:
                pass # already defined; no worries
            else:
                v = self._getvar(x,typ) # force adding to environment
                self.z3.add_var(v.symbol_name(),typ)

    def assume(self, e):
        assert e.type is T.bool
        self._add_free_vars(e)
        self.negative_pos = aeNegPos(e, '-')
        smt_e       = self._lower(e)
        assert not is_ternary(smt_e), "assumptions must be classical"
        self.frames[-1].add_assumption(e, smt_e)
        self.solver.add_assertion(smt_e)

    def satisfy(self, e):
        assert e.type is T.bool
        self.push()
        self._add_free_vars(e)
        self.negative_pos = aeNegPos(e, '-')
        smt_e       = self._lower(e)
        assert not is_ternary(smt_e), "formulas must be classical"
        self.z3.add_assertion(smt_e)
        is_sat      = self.z3.run_check_sat()
        #is_sat      = self.solver.is_sat(smt_e)
        self.pop()
        return is_sat

    def verify(self, e):
        assert e.type is T.bool
        self.push()
        self._add_free_vars(e)
        self.negative_pos = aeNegPos(e, '+')
        smt_e       = self._lower(e)
        assert not is_ternary(smt_e), "formulas must be classical"
        self.z3.add_assertion(SMT.Not(smt_e))
        if self.verbose:
            print('*******\n*******\n*******')
            print(self.debug_str(smt=False))
            print('to verify')
            print(e)
            print(SMT.to_smtlib(smt_e))

        is_valid    = not self.z3.run_check_sat()
        #is_valid    = self.solver.is_valid(smt_e)
        self.pop()
        return is_valid

    def counter_example(self):
        def keep_sym(s):
            if type(s) is tuple:
                return True
            else:
                return (s.get_type() == SMT.INT or
                        s.get_type() == SMT.BOOL)
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
            self.stride_sym[key] = Sym(f"{name}_stride_{dim}")
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
        self.mod_div_tmp_bins[-1].append((new_sym,eq))

    def _lower(self, e):
        if e.type == T.bool:
            self.mod_div_tmp_bins.append([])
        smt_e = self._lower_body(e)
        if e.type == T.bool:
            tmp_bin = self.mod_div_tmp_bins.pop()
            # possibly wrap some definitions of temporaries
            if len(tmp_bin) > 0:
                assert not is_ternary(smt_e), "TODO: handle ternary"
                all_syms    = [ sym for sym,eq in tmp_bin ]
                all_eq      = SMT.And(*[ eq  for sym,eq in tmp_bin])
                if self.negative_pos[id(e)] == '+':
                    smt_e   = SMT.ForAll(all_syms,SMT.Implies(all_eq,smt_e))
                else:
                    smt_e   = SMT.Exists(all_syms,SMT.And(all_eq,smt_e))
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
                return TernVal(SMT.Ite(c.v,t.v,f.v),
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
                is_def  = SMT.Or( SMT.ForAll([nm], a.d),
                                  SMT.Exists([nm], SMT.And(short_a, a.d)) )
                return TernVal( OP([nm],a.v), is_def )
            else:
                return OP([nm],a)
        elif isinstance(e, (A.Definitely,A.Maybe)):
            assert e.arg.type == T.bool
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





class Z3SubProc:
    def __init__(self):
        self.stack_lines = [ [] ] # list(i.e. stack) of lists
        pass

    def push(self):
        self.stack_lines.append([])

    def pop(self):
        self.stack_lines.pop()

    def _get_whole_str(self):
        return '\n'.join([ line for frame in self.stack_lines
                                for line in frame ])

    def add_var(self, varname, vartyp):
        sort = 'Bool' if vartyp == T.bool else 'Int'
        self.stack_lines[-1].append(f"(declare-const {varname} {sort})")

    def add_assertion(self, smt_formula):
        smtstr = SMT.to_smtlib(smt_formula)
        self.stack_lines[-1].append(f"(assert {smtstr})")

    def run_check_sat(self):
        slv = z3lib.Solver()
        slv.from_string(self._get_whole_str())
        result = slv.check()
        if result == z3lib.z3.sat:
            return True
        elif result == z3lib.z3.unsat:
            return False
        else:
            raise Error("unknown result from z3")
