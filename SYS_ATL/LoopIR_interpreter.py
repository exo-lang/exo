from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Interpreter

def _eshape(typ, env):
    return tuple(r if is_pos_int(r) else env[r]
                 for r in typ.shape())


def _simple_typecheck_buffer(typ, buf, env):
    if type(buf) is not np.ndarray:
        return False
    elif buf.dtype != float and buf.dtype != np.float32:
        return False

    if typ is T.R:
        if tuple(buf.shape) != (1,):
            return False
    else:
        shape = _eshape(typ, env)
        if shape != tuple(buf.shape):
            return False

    return True

def run_interpreter(proc, kwargs):
    Interpreter(proc, kwargs)

class Interpreter:
    def __init__(self, proc, kwargs, use_randomization=False):
        assert type(proc) is LoopIR.proc

        self.proc = proc
        self.env = Environment()
        self.use_randomization = use_randomization

        # must bind all size arguments first
        for a in proc.args:
            if a.type is T.size:
                if not is_pos_int(kwargs[str(a.name)]):
                    raise TypeError(f"expected size '{a.name}' to "
                                    f"have positive integer value")
                self.env[a.name] = kwargs[str(a.name)]

        # setup, buffer argument binding
        for a in proc.args:
            if not str(a.name) in kwargs:
                raise TypeError(f"expected argument '{a.name}' "
                                f"to be supplied")
            if a.type is T.size:
                continue # already bound these
            else:
                if not _simple_typecheck_buffer(a.type, kwargs[str(a.name)],
                                                self.env):
                    raise TypeError(f"type of argument '{a.name}' "
                                    f"value mismatches")
            self.env[a.name] = kwargs[str(a.name)]

        self.env.push()
        self.eval_stmts(proc.body)
        self.env.pop()

    def eval_stmts(self, stmts):
        for s in stmts:
            self.eval_s(s)

    def eval_s(self, s):
        styp = type(s)

        if styp is LoopIR.Pass:
            pass
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            # lbuf[a0,a1,...] = rhs
            lbuf = self.env[s.name]
            if len(s.idx) == 0:
                idx = (0,)
            else:
                idx = tuple(self.eval_e(a) for a in s.idx)
            rhs = self.eval_e(s.rhs)
            if styp is LoopIR.Assign:
                lbuf[idx] = rhs
            else:
                lbuf[idx] += rhs
        elif styp is LoopIR.If:
            cond = self.eval_e(s.cond)
            if cond:
                self.env.push()
                self.eval_stmts(s.body)
                self.env.pop()
            if s.orelse and not cond:
                self.env.push()
                self.eval_stmts(s.orelse)
                self.env.pop()
        elif styp is LoopIR.ForAll:
            hi = self.eval_e(s.hi)
            assert self.use_randomization is False, "TODO: Implement Rand"
            self.env.push()
            for itr in range(0, hi):
                self.env[s.iter] = itr
                self.eval_stmts(s.body)
            self.env.pop()
        elif styp is LoopIR.Alloc:
            if s.type is T.R:
                self.env[s.name] = np.empty([1])
            else:
                size = _eshape(s.type, self.env)
                # TODO: Maybe randomize?
                self.env[s.name] = np.empty(size)
        elif styp is LoopIR.Instr:
            self.eval_s(s.body)
        elif styp is LoopIR.Call:
            argvals     = [ self.eval_e(a, call_arg=True) for a in s.args ]
            argnames    = [ str(a.name) for a in s.f.args ]
            kwargs      = { nm : val for nm,val in zip(argnames,argvals) }
            Interpreter(s.f, kwargs,
                        use_randomization=self.use_randomization)
        else:
            assert False, "bad case"

    def eval_e(self, e, call_arg=False):
        etyp = type(e)

        if etyp is LoopIR.Read:
            buf = self.env[e.name]
            if type(buf) is int:
                return buf
            elif type(buf) is bool:
                return buf
            if call_arg:
                return buf
            else:
                idx = ((0,) if len(e.idx) == 0
                       else tuple(self.eval_e(a) for a in e.idx))
                return buf[idx]
        elif etyp is LoopIR.Const:
            return e.val
        elif etyp is LoopIR.BinOp:
            lhs, rhs = self.eval_e(e.lhs), self.eval_e(e.rhs)
            if e.op == "+":
                return lhs + rhs
            elif e.op == "-":
                return lhs - rhs
            elif e.op == "*":
                return lhs * rhs
            elif e.op == "/": # is this right?
                if type(lhs) is int:
                    return lhs // rhs
                else:
                    return lhs / rhs
            elif e.op == "%":
                return lhs % rhs
            elif e.op == "==":
                return (lhs == rhs)
            elif e.op == "<":
                return (lhs < rhs)
            elif e.op == ">":
                return (lhs > rhs)
            elif e.op == "<=":
                return (lhs <= rhs)
            elif e.op == ">=":
                return (lhs >= rhs)
            elif e.op == "and":
                return (lhs and rhs)
            elif e.op == "or":
                return (lhs or rhs)
        else:
            assert False, "bad case"
