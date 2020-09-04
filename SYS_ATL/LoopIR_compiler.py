from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler

# top level compiler function called by tests!
def compile(proc_list,c_file,h_file):
    # take proc_list
    # for each p in proc_list:
    #   run Compiler() pass to get (decl, def)
    #
    # check for name conflicts between procs
    #
    # write out c_file and h_file

def _eshape(typ,env):
    return tuple( r if is_pos_int(r) else env[r]
                  for r in typ.shape() )

def _simple_typecheck_buffer(typ, buf, env):
    if type(buf) is not np.ndarray:
        return False
    elif buf.dtype != float and buf.dtype != np.float64:
        return False

    if typ is T.R:
        if tuple(buf.shape) != (1,):
            return False
    else:
        shape = _eshape(typ,env)
        if shape != tuple(buf.shape):
            return False

    return True

class Compiler:
    def __init__(self, proc, use_randomization=False, **kwargs):
        assert type(proc) is LoopIR.proc

        self.proc   = proc
        self.env    = Environment()
        self.use_randomization = use_randomization

        # setup, size argument binding
        for sz in proc.sizes:
            if not str(sz) in kwargs:
                raise TypeError(f"expected size '{sz}' "
                                f"to be supplied")
            if not is_pos_int(kwargs[str(sz)]):
                raise TypeError(f"expected size '{sz}' to "
                                f"have positive integer value")
            self.env[sz] = kwargs[str(sz)]

        # setup, buffer argument binding
        for a in proc.args:
            if not str(a.name) in kwargs:
                raise TypeError(f"expected argument '{a.name}' "
                                f"to be supplied")
            if not _simple_typecheck_buffer(a.type, kwargs[str(a.name)],
                                            self.env):
                raise TypeError(f"type of argument '{a.name}' "
                                f"value mismatches")
            self.env[a.name] = kwargs[str(a.name)]

        self.env.push()
        self.eval_s(proc.body)
        self.env.pop()

        proc_def = (f"void {proc.name}(buffer_pointers)\n"+
                    f"\n"+
                    f"\n"+)
        proc_decl = ()

        self.c_def = proc_def
        self.c_decl = proc_decl

    def get_str(self):
        return self.c_decl, self.c_def

    def comp_s(self, s):
        styp    = type(s)

        if styp is LoopIR.Seq:
            self.eval_s(s.s0)
            self.eval_s(s.s1)
        elif styp is LoopIR.Pass:
            pass
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            # lbuf[a0,a1,...] = rhs
            lbuf = self.env[s.name]
            if len(s.idx) == 0:
                idx = (0,)
            else:
                idx  = tuple( self.eval_a(a) for a in s.idx )
            rhs  = self.eval_e(s.rhs)
            if styp is LoopIR.Assign:
                lbuf[idx] = rhs
            else:
                lbuf[idx] += rhs
        elif styp is LoopIR.If:
            cond = self.eval_p(s.cond)
            self.env.push()
            if cond:
                self.eval_s(s.body)
            self.env.pop()
        elif styp is LoopIR.ForAll:
            hi      = self.env[s.hi] # this should be a string
            iter    = self.new_varname(s.iter) # allocate a new string?
            body    = self.comp_s(s.body)
            return (f"for (int {iter}=0; {iter} < {hi}; {iter}++) {{\n"+
                    f"{body}\n"+
                    f"}}")
        #elif styp is LoopIR.ForAllWhere:
        #    for itr in
        elif styp is LoopIR.Alloc:
            if s.type is T.R:
                self.env[s.name] = np.empty([1])
            else:
                size = _eshape(s.type, self.env)
                #TODO: Maybe randomize?
                self.env[s.name] = np.empty(size)
        else: assert False, "bad case"

    def eval_e(self, e):
        etyp    = type(e)

        if etyp is LoopIR.Read:
            buf = self.env[e.name]
            idx = ( (0,) if len(e.idx) == 0
                         else tuple( self.eval_a(a) for a in e.idx ))
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
            elif e.op == "/":
                return lhs / rhs
        elif etyp is LoopIR.Select:
            cond    = self.eval_p(e.cond)
            return self.eval_e(e.body) if cond else 0.0
        else: assert False, "bad case"

    def eval_a(self, a):
        atyp    = type(a)

        if atyp is LoopIR.AVar or atyp is LoopIR.ASize:
            return self.env[a.name]
        elif atyp is LoopIR.AConst:
            return a.val
        elif atyp is LoopIR.AScale:
            return a.coeff * self.eval_a(a.rhs)
        elif atyp is LoopIR.AAdd:
            return self.eval_a(a.lhs) + self.eval_a(a.rhs)
        elif atyp is LoopIR.ASub:
            return self.eval_a(a.lhs) - self.eval_a(a.rhs)
        else: assert False, "bad case"

    def eval_p(self, p):
        ptyp = type(p)

        if ptyp is LoopIR.BConst:
            return p.val
        elif ptyp is LoopIR.Cmp:
            lhs, rhs = self.eval_a(p.lhs), self.eval_a(p.rhs)
            if p.op == "==":
                return (lhs == rhs)
            elif p.op == "<":
                return (lhs < rhs)
            elif p.op == ">":
                return (lhs > rhs)
            elif p.op == "<=":
                return (lhs <= rhs)
            elif p.op == ">=":
                return (lhs >= rhs)
            else: assert False, "bad case"
        elif ptyp is LoopIR.And or ptyp is LoopIR.Or:
            lhs, rhs = self.eval_p(p.lhs), self.eval_p(p.rhs)
            if ptyp is LoopIR.And:
                return (lhs and rhs)
            elif ptyp is LoopIR.Or:
                return (lhs or rhs)
            else: assert False, "bad case"
        else: assert False, "bad case"
