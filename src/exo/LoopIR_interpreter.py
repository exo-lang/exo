from collections import ChainMap

import numpy as np

from .LoopIR import LoopIR
from .LoopIR import T
from .prelude import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Interpreter


# method copied from Python ChainMap docs https://docs.python.org/3/library/collections.html#collections.ChainMap
# to delete items from parent maps
@extclass(ChainMap)
def __delitem__(self, key):
    for mapping in self.maps:
        if key in mapping:
            del mapping[key]
            return
    raise KeyError(key)


def _eshape(typ, env):
    return tuple(r if is_pos_int(r) else env[r] for r in typ.shape())


def run_interpreter(proc, kwargs):
    Interpreter(proc, kwargs)


class Interpreter:
    def __init__(self, proc, kwargs, use_randomization=False):
        assert isinstance(proc, LoopIR.proc)

        self.proc = proc
        self.env = ChainMap()
        # dependency analysis not relevant?
        # context struct (configs) not relevant?
        # range_env not relevant?
        # what are strides?
        self.use_randomization = use_randomization

        # type check args
        for a in proc.args:
            if not str(a.name) in kwargs:
                raise TypeError(f"expected argument '{a.name}' to be supplied")

            if a.type is T.size:
                if not is_pos_int(kwargs[str(a.name)]):
                    raise TypeError(
                        f"expected size '{a.name}' to have positive integer value"
                    )
                self.env[a.name] = kwargs[str(a.name)]
            elif a.type is T.index:
                if type(kwargs[str(a.name)]) is not int:
                    raise TypeError(
                        f"expected index variable '{a.name}' to be an integer"
                    )
                self.env[a.name] = kwargs[str(a.name)]
            elif a.type is T.bool:
                if type(kwargs[str(a.name)]) is not bool:
                    raise TypeError(f"expected bool variable '{a.name}' to be a bool")
                self.env[a.name] = kwargs[str(a.name)]
            elif a.type is T.stride:
                if type(kwargs[str(a.name)]) is not int:
                    raise TypeError(
                        f"expected stride variable '{a.name}' to be an integer"
                    )
                self.env[a.name] = kwargs[str(a.name)]
            else:
                assert a.type.is_numeric()
                assert a.type.basetype() != T.R  # R => real
                self.simple_typecheck_buffer(a, kwargs)
                self.env[a.name] = kwargs[str(a.name)]

        # evaluate preconditions
        for pred in proc.preds:
            if isinstance(pred, LoopIR.Const):
                continue
            else:
                assert self.eval_e(pred.lhs) == self.eval_e(pred.rhs)

        # eval statements
        self.env = self.env.new_child()
        self.eval_stmts(proc.body)
        self.env = self.env.parents

    def _new_scope(self):
        self.env = self.env.new_child()

    def _del_scope(self):
        self.env = self.env.parents

    # input buffers should be numpy arrays with floating-point values
    def simple_typecheck_buffer(self, fnarg, kwargs):
        typ = fnarg.type
        buf = kwargs[str(fnarg.name)]
        nm = fnarg.name

        # check data type
        pre = f"bad argument '{nm}'"
        if not isinstance(buf, np.ndarray):
            raise TypeError(f"{pre}: expected numpy.ndarray")
        elif buf.dtype != float and buf.dtype != np.float32 and buf.dtype != np.float16:
            raise TypeError(
                f"{pre}: expected buffer of floating-point values; "
                f"had '{buf.dtype}' values"
            )

        # check shape
        if typ.is_real_scalar():
            if tuple(buf.shape) != (1,):
                raise TypeError(
                    f"{pre}: expected buffer of shape (1,), "
                    f"but got shape {tuple(buf.shape)}"
                )
        else:
            shape = self.eval_shape(typ)
            if shape != tuple(buf.shape):
                raise TypeError(
                    f"{pre}: expected buffer of shape {shape}, "
                    f"but got shape {tuple(buf.shape)}"
                )

    def eval_stmts(self, stmts):
        for s in stmts:
            self.eval_s(s)

    def eval_s(self, s):
        if isinstance(s, LoopIR.Pass):
            pass
        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            lbuf = self.env[s.name]
            if len(s.idx) == 0:
                # lbuf = rhs
                idx = (0,)
            else:
                # lbuf[a0,a1,...] = rhs
                idx = tuple(self.eval_e(a) for a in s.idx)
            rhs = self.eval_e(s.rhs)
            if isinstance(s, LoopIR.Assign):
                lbuf[idx] = rhs
            else:
                lbuf[idx] += rhs

        elif isinstance(s, LoopIR.WriteConfig):
            assert False, "TODO: impl LoopIR.WriteConfig"

        elif isinstance(s, LoopIR.WindowStmt):
            assert False, "TODO: impl LoopIR.WindowStmt"

        elif isinstance(s, LoopIR.If):
            cond = self.eval_e(s.cond)
            if cond:
                self._new_scope()
                self.eval_stmts(s.body)
                self._del_scope()
            if s.orelse and not cond:
                self._new_scope()
                self.eval_stmts(s.orelse)
                self._del_scope()

        elif isinstance(s, LoopIR.For):
            lo = self.eval_e(s.lo)
            hi = self.eval_e(s.hi)
            assert self.use_randomization is False, "TODO: Implement Rand"
            self._new_scope()
            for itr in range(lo, hi):
                self.env[s.iter] = itr
                self.eval_stmts(s.body)
            self._del_scope()

        elif isinstance(s, LoopIR.Alloc):
            if s.type.is_real_scalar():
                self.env[s.name] = np.empty([1])
            else:
                size = self.eval_shape(s.type)
                # TODO: Maybe randomize?
                self.env[s.name] = np.empty(size)

        # TODO (andrew) figure out a way to test this, no explicit frees that I can find
        elif isinstance(s, LoopIR.Free):
            # use extension to chain map from python docs
            del self.env[s.name]

        elif isinstance(s, LoopIR.Call):
            argvals = [self.eval_e(a, call_arg=True) for a in s.args]
            argnames = [str(a.name) for a in s.f.args]
            kwargs = {nm: val for nm, val in zip(argnames, argvals)}
            Interpreter(s.f, kwargs, use_randomization=self.use_randomization)

        else:
            assert False, "bad case"

    def eval_e(self, e, call_arg=False):

        if isinstance(e, LoopIR.Read):
            buf = self.env[e.name]
            if call_arg or isinstance(buf, (int, bool)):
                # read without indices
                return buf
            else:
                idx = (0,) if len(e.idx) == 0 else tuple(self.eval_e(a) for a in e.idx)
                return buf[idx]

        elif isinstance(e, LoopIR.WindowExpr):
            buf = self.env[e.name]

            def case_eval(e):
                if isinstance(e, LoopIR.Interval):
                    return f"{e.lo}:{e.hi}"
                else:
                    self.eval_e(e)

            # hack to handle interval indexes: LoopIR.Interval returns a string representing the interval
            idx = ("0",) if len(e.idx) == 0 else tuple(str(case_eval(a)) for a in e.idx)
            res = eval(f"buf[{','.join(idx)}]")
            return res

        elif isinstance(e, LoopIR.Const):
            return e.val

        elif isinstance(e, LoopIR.BinOp):
            lhs, rhs = self.eval_e(e.lhs), self.eval_e(e.rhs)
            if e.op == "+":
                return lhs + rhs
            elif e.op == "-":
                return lhs - rhs
            elif e.op == "*":
                return lhs * rhs
            elif e.op == "/":  # is this right?
                if isinstance(lhs, int):
                    return (lhs + rhs - 1) // rhs
                else:
                    return lhs / rhs
            elif e.op == "%":
                return lhs % rhs
            elif e.op == "==":
                return lhs == rhs
            elif e.op == "<":
                return lhs < rhs
            elif e.op == ">":
                return lhs > rhs
            elif e.op == "<=":
                return lhs <= rhs
            elif e.op == ">=":
                return lhs >= rhs
            elif e.op == "and":
                return lhs and rhs
            elif e.op == "or":
                return lhs or rhs

        elif isinstance(e, LoopIR.USub):
            return -self.eval_e(e.arg)

        elif isinstance(e, LoopIR.BuiltIn):
            args = [self.eval_e(a) for a in e.args]
            return e.f.interpret(args)

        elif isinstance(e, LoopIR.StrideExpr):
            buf = self.env[e.name]
            assert e.dim < len(buf.strides), "invalid dim in stride expression"
            # grammar guarantees int (not an expression)
            return int(buf.strides[e.dim] / buf.dtype.itemsize)

        elif isinstance(e, LoopIR.ReadConfig):
            assert False, "TODO: impl LoopIR.ReadConfig"

        else:
            assert False, "bad case"

    def eval_shape(self, typ):
        return tuple(self.eval_e(s) for s in typ.shape())
