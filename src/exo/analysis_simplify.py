from collections import OrderedDict, ChainMap

# from weakref import WeakKeyDictionary
# from enum import Enum
# from itertools import chain

# from .LoopIR import Alpha_Rename, SubstArgs, LoopIR_Do
# from .configs import reverse_config_lookup, Config
from .new_analysis_core import *

# from .proc_eqv import get_repr_proc

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Simplification Pass to Clean up Analysis Expressions
"""
_fv_cache = WeakKeyDictionary()

def _FV(a):
    fvs = _fv_cache.get(id(a))
    if fvs is not None:
        return fvs
    # otherwise...
    fvs = _A_Free_Vars(a)
    _fv_cache[id(a)] = fvs
    return fvs

def _union_FV(gen):
    fv = set()
    for e in gen:
        fv |= _FV(e)
    return fv

def _A_Free_Vars(a):
    if isinstance(a, A.Var):
        return { a.name }
    elif isinstance(a, (A.Unk,A.Const,A.ConstSym)):
        return set()
    elif isinstance(a, (A.Not,A.USub,A.Definitely,A.Maybe)):
        return _FV(a.arg)
    elif isinstance(a, A.BinOp):
        return _FV(a.lhs) | _FV(a.rhs)
    elif isinstance(a, A.Stride):
        return { (a.name,a.dim) }
    elif isinstance(a, A.LetStrides):
        bound_names = { (a.name,i) for i,_ in enumerate(a.strides) }
        return _union_FV(a.strides) | (_FV(a.body) - bound_names)
    elif isinstance(a, A.Select):
        return _FV(a.cond) | _FV(a.tcase) | _FV(a.fcase)
    elif isinstance(a, (A.ForAll,A.Exists)):
        return _FV(a.arg) - { a.name }
    elif isinstance(a, A.Tuple):
        return _union_FV(a.args)
    elif isinstance(a, A.LetTuple):
        bound_names = set(a.names)
        return _FV(a.rhs) | (_FV(a.body) - bound_names)
    elif isinstance(a, A.Let):
        bound_names = set(a.names)
        return _union_FV(a.rhs) | (_FV(a.body) - bound_names)
    else:
        assert False, f"Bad Case: {type(a)}"
"""


@extclass(A.expr)
def simplify(self):
    return ASimplify(self).result()


class ASimplify:
    def __init__(self, top_level_a):
        self._init_a = top_level_a

        self._const_prop_cache = dict()
        self._const_vals = ChainMap()

        self._fv_cache = dict()

        a = top_level_a
        # constant propagation
        a = self.cprop(a)
        # dead code elimination
        a = self.dcode(a)
        # constant propagation
        a = self.cprop(a)

        self._final_a = a

    def result(self):
        return self._final_a

    def push_cprop(self):
        self._const_vals = self._const_vals.new_child()

    def pop_cprop(self):
        self._const_vals = self._const_vals.parents

    def is_simple(self, a):
        return isinstance(a, (A.Var, A.Unk, A.Const, A.ConstSym, A.Stride))

    def is_simple_tuple(self, a):
        return isinstance(a, (A.Tuple)) and all(self.is_simple(e) for e in a.args)

    # constant propagation
    def cprop(self, a):
        # if a not in self._const_prop_cache:
        #    self._const_prop_cache[a] = self.cprop_helper(a)
        # return self._const_prop_cache[a]
        return self.cprop_helper(a)

    def cprop_helper(self, a):
        if isinstance(a, A.Var):
            val = self._const_vals.get(a.name)
            return val if val is not None else a

        elif isinstance(a, (A.Unk, A.Const, A.ConstSym)):
            return a

        elif isinstance(a, A.Not):
            arg = self.cprop(a.arg)
            if isinstance(arg, A.Const):
                return ABool(not arg.val)
            elif isinstance(arg, A.Unk):
                return arg
            else:
                return type(a)(arg, a.type, a.srcinfo)

        elif isinstance(a, A.USub):
            arg = self.cprop(a.arg)
            if isinstance(arg, A.Const):
                return A.Const(-arg.val, arg.type, arg.srcinfo)
            elif isinstance(arg, A.Unk):
                return arg
            else:
                return type(a)(arg, a.type, a.srcinfo)

        elif isinstance(a, (A.Definitely, A.Maybe)):
            arg = self.cprop(a.arg)
            if isinstance(arg, A.Const):
                return arg
            elif isinstance(arg, A.Unk):
                return ABool(False if isinstance(a, A.Definitely) else True)
            else:
                return type(a)(arg, a.type, a.srcinfo)

        elif isinstance(a, A.BinOp):
            lhs = self.cprop(a.lhs)
            rhs = self.cprop(a.rhs)
            lunk, runk = isinstance(lhs, A.Unk), isinstance(rhs, A.Unk)
            lconst, rconst = isinstance(lhs, A.Const), isinstance(rhs, A.Const)

            # first, operators resulting in a boolean
            if a.type == T.bool:
                if a.op == "and":
                    if lconst:
                        return rhs if lhs.val else lhs
                    elif rconst:
                        return lhs if rhs.val else rhs
                    # fall-through
                elif a.op == "or":
                    if lconst:
                        return lhs if lhs.val else rhs
                    elif rconst:
                        return rhs if rhs.val else lhs
                    # fall-through
                elif a.op == "==>":
                    if lconst:
                        return rhs if lhs.val else ABool(True)
                    elif rconst:
                        return rhs if rhs.val else ANot(lhs)
                    # fall-through
                elif a.op in ("<", ">", "<=", ">=", "=="):
                    if lconst and rconst:
                        return ABool(
                            (lhs.val < rhs.val)
                            if a.op == "<"
                            else (lhs.val > rhs.val)
                            if a.op == ">"
                            else (lhs.val <= rhs.val)
                            if a.op == "<="
                            else (lhs.val >= rhs.val)
                            if a.op == ">="
                            else (lhs.val == rhs.val)
                        )
                    elif lunk or runk:
                        return A.Unk(T.bool, a.srcinfo)
                    # fall-through
                else:
                    assert False, f"bad case: {a.op}"

            else:
                if lconst and rconst:
                    assert a.type.is_indexable()
                    if a.op == "/":
                        pass
                    else:
                        return AInt(
                            (lhs.val + rhs.val)
                            if a.op == "+"
                            else (lhs.val - rhs.val)
                            if a.op == "-"
                            else (lhs.val * rhs.val)
                            if a.op == "*"
                            else (lhs.val % rhs.val)
                        )
                elif lconst:
                    if a.op == "+" and lhs.val == 0:
                        return rhs
                    elif a.op == "-" and lhs.val == 0:
                        return -rhs
                    elif a.op in ("*", "/", "%") and lhs.val == 0:
                        return lhs
                    elif a.op == "*" and lhs.val == 1:
                        return rhs
                    elif a.op == "%" and lhs.val == 1:
                        return lhs
                    # fall through
                elif rconst:
                    if a.op in ("+", "-") and rhs.val == 0:
                        return lhs
                    elif a.op == "*" and rhs.val == 0:
                        return rhs
                    elif a.op in ("*", "/") and rhs.val == 1:
                        return lhs
                    # fall through
                elif lunk or runk:
                    return A.Unk(a.type, a.srcinfo)
                # fall through

            # finally check for a number of other equality tests
            # that we can collapse out
            if a.op == "==" and type(a.lhs) == type(a.rhs):
                if isinstance(a.lhs, (A.Var, A.ConstSym)):
                    if a.lhs.name == a.rhs.name:
                        return ABool(True)
                elif isinstance(a.lhs, A.Stride):
                    if a.lhs.name == a.rhs.name and a.lhs.dim == a.rhs.dim:
                        return ABool(True)

            # catch the fall-throughs
            return A.BinOp(a.op, lhs, rhs, a.type, a.srcinfo)

        elif isinstance(a, A.Stride):
            key = (a.name, a.dim)
            val = self._const_vals.get(key)
            return val if val is not None else a

        elif isinstance(a, A.LetStrides):
            strides = [self.cprop(s) for s in a.strides]
            self.push_cprop()
            if all(self.is_simple(s) for s in strides):
                for i, s in enumerate(strides):
                    self._const_vals[(a.name, i)] = s
                body = self.cprop(a.body)
            else:
                # must mask out to indicate non-propagation here
                for i, _ in enumerate(strides):
                    self._const_vals[(a.name, i)] = None
                body = self.cprop(a.body)
                body = A.LetStrides(a.name, strides, body, a.type, a.srcinfo)
            self.pop_cprop()
            return body

        elif isinstance(a, A.Select):
            cond = self.cprop(a.cond)
            tcase = self.cprop(a.tcase)
            fcase = self.cprop(a.fcase)
            if isinstance(cond, A.Const):
                return tcase if cond.val else fcase
            else:
                return A.Select(cond, tcase, fcase, a.type, a.srcinfo)

        elif isinstance(a, (A.ForAll, A.Exists)):
            self.push_cprop()
            self._const_vals[a.name] = None
            arg = self.cprop(a.arg)
            self.pop_cprop()
            if isinstance(arg, (A.Const, A.Unk)):
                return arg
            else:
                return type(a)(a.name, arg, a.type, a.srcinfo)

        elif isinstance(a, A.Tuple):
            args = [self.cprop(e) for e in a.args]
            return A.Tuple(args, a.type, a.srcinfo)

        elif isinstance(a, A.LetTuple):
            rhs = self.cprop(a.rhs)
            self.push_cprop()
            if self.is_simple_tuple(rhs):
                for nm, e in zip(a.names, rhs.args):
                    self._const_vals[nm] = e
                body = self.cprop(a.body)
            else:
                for nm in a.names:
                    self._const_vals[nm] = None
                body = self.cprop(a.body)
                body = A.LetTuple(a.names, rhs, body, a.type, a.srcinfo)
            self.pop_cprop()
            return body

        elif isinstance(a, A.Let):
            self.push_cprop()
            name_list = []
            rhs_list = []
            for nm, e in zip(a.names, a.rhs):
                rhs = self.cprop(e)
                if self.is_simple(rhs):
                    self._const_vals[nm] = rhs
                else:
                    name_list.append(nm)
                    rhs_list.append(rhs)
                    self._const_vals[nm] = None
            body = self.cprop(a.body)
            self.pop_cprop()
            if len(name_list) == 0:
                return body
            else:
                body = A.Let(name_list, rhs_list, body, a.type, a.srcinfo)
                return body

        else:
            assert False, f"Bad Case: {type(a)}"

    # dead code elimination
    def dcode(self, a):
        return self.dcode_helper(a)

    def dcode_helper(self, a):
        if isinstance(a, (A.Var, A.Unk, A.Const, A.ConstSym, A.Stride)):
            return a
        elif isinstance(a, (A.Not, A.USub, A.Definitely, A.Maybe)):
            arg = self.dcode(a.arg)
            return type(a)(arg, a.type, a.srcinfo)
        elif isinstance(a, A.BinOp):
            lhs = self.dcode(a.lhs)
            rhs = self.dcode(a.rhs)
            return A.BinOp(a.op, lhs, rhs, a.type, a.srcinfo)
        elif isinstance(a, A.LetStrides):
            names = {(a.name, i) for i, _ in enumerate(a.strides)}
            body = self.dcode(a.body)
            bodyFV = self._FV(body)
            if all(nm not in bodyFV for nm in names):
                return body
            else:
                strides = [self.dcode(s) for s in a.strides]
                return A.LetStrides(a.name, strides, body, a.type, a.srcinfo)
        elif isinstance(a, A.Select):
            cond = self.dcode(a.cond)
            tcase = self.dcode(a.tcase)
            fcase = self.dcode(a.fcase)
            return A.Select(cond, tcase, fcase, a.type, a.srcinfo)
        elif isinstance(a, (A.ForAll, A.Exists)):
            arg = self.dcode(a.arg)
            return type(a)(a.name, arg, a.type, a.srcinfo)
        elif isinstance(a, A.Tuple):
            args = [self.dcode(e) for e in a.args]
            return A.Tuple(args, a.type, a.srcinfo)
        elif isinstance(a, A.LetTuple):
            body = self.dcode(a.body)
            bodyFV = self._FV(body)
            if all(nm not in bodyFV for nm in a.names):
                return body
            else:
                rhs = self.dcode(a.rhs)
                return A.LetTuple(a.names, rhs, body, a.type, a.srcinfo)
        elif isinstance(a, A.Let):
            body = self.dcode(a.body)
            FV = self._FV(body)
            name_list = []
            rhs_list = []
            for nm, rhs in reversed(list(zip(a.names, a.rhs))):
                if nm not in FV:
                    pass
                else:
                    rhs = self.dcode(rhs)
                    FV = (FV - {nm}) | self._FV(rhs)
                    name_list.append(nm)
                    rhs_list.append(rhs)

            if len(name_list) == 0:
                return body
            else:
                return A.Let(name_list, rhs_list, body, a.type, a.srcinfo)
        else:
            assert False, f"Bad Case: {type(a)}"

    def _FV(self, a):
        fvs = self._fv_cache.get(id(a))
        if fvs is not None:
            return fvs
        # otherwise...
        fvs = self._A_Free_Vars(a)
        self._fv_cache[id(a)] = fvs
        return fvs

    def _union_FV(self, gen):
        fv = set()
        for e in gen:
            fv |= self._FV(e)
        return fv

    def _A_Free_Vars(self, a):
        if isinstance(a, A.Var):
            return {a.name}
        elif isinstance(a, (A.Unk, A.Const, A.ConstSym)):
            return set()
        elif isinstance(a, (A.Not, A.USub, A.Definitely, A.Maybe)):
            return self._FV(a.arg)
        elif isinstance(a, A.BinOp):
            return self._FV(a.lhs) | self._FV(a.rhs)
        elif isinstance(a, A.Stride):
            return {(a.name, a.dim)}
        elif isinstance(a, A.LetStrides):
            bound_names = {(a.name, i) for i, _ in enumerate(a.strides)}
            return self._union_FV(a.strides) | (self._FV(a.body) - bound_names)
        elif isinstance(a, A.Select):
            return self._FV(a.cond) | self._FV(a.tcase) | self._FV(a.fcase)
        elif isinstance(a, (A.ForAll, A.Exists)):
            return self._FV(a.arg) - {a.name}
        elif isinstance(a, A.Tuple):
            return self._union_FV(a.args)
        elif isinstance(a, A.LetTuple):
            bound_names = set(a.names)
            return self._FV(a.rhs) | (self._FV(a.body) - bound_names)
        elif isinstance(a, A.Let):
            bound_names = set(a.names)
            return self._union_FV(a.rhs) | (self._FV(a.body) - bound_names)
        else:
            assert False, f"Bad Case: {type(a)}"
