from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory Analysis Pass

class MemoryAnalysis:
    def __init__(self, proc, **kwargs):
        assert type(proc) is LoopIR.proc

        self.proc   = proc
        #self.env    = Environment()
        self.tofree = []

    def result(self):
        #return proc_decl, proc_def
        return proc_decl, proc_def

    def push_frame(self):
        self.tofree.append(set())

    def pop_frame(self, body):
        for nm in self.tofree.pop():
            body = LoopIR.Seq(body,
                              LoopIR.Free(nm,body.srcinfo),
                              body.srcinfo)
        return body

    def mem_s(self, s):
        styp    = type(s)

        if styp is LoopIR.Seq:
            s0 = self.mem_s(s.s0)
            s1 = self.mem_s(s.s1)
            return LoopIR.Seq(s0, s1, s.srcinfo)
        elif styp is LoopIR.Pass or is LoopIR.Assign or styp is LoopIR.Reduce:
            return s
        elif styp is LoopIR.If:
            self.push_frame()
            body = self.mem_s(s.body)
            body = self.pop_frame(body)
            return LoopIR.If(s.cond, body, s.srcinfo)
        elif styp is LoopIR.ForAll:
            hi      = self.env[s.hi] # this should be a string
            itr     = self.new_varname(s.iter) # allocate a new string
            body    = self.comp_s(s.body)
            return (f"for (int {itr}=0; {itr} < {hi}; {itr}++) {{\n"+
                    f"{body}\n"+
                    f"}}")
        elif styp is LoopIR.Alloc:
            if s.type is T.R:
                name = self.env[s.name]
                empty = np.empty([1])
                return (f"{name} = {empty}")
            else:
                size = _eshape(s.type, self.env)
                #TODO: Maybe randomize?
                name = self.env[s.name]
                empty = np.empty(size)
                return (f"{name} = {empty}")
        elif styp is LoopIR.Free:
            assert False, ("There should not be frees inserted "+
                           "before mem analysis")
        else: assert False, "bad case"
