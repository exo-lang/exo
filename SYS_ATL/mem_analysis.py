from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory Analysis Pass


class MemoryAnalysis:
    def __init__(self, proc):
        assert type(proc) is LoopIR.proc

        self.proc = proc
        self.tofree = []

    def result(self):
        body = self.mem_stmts(self.proc.body)

        return LoopIR.proc(
            self.proc.name,
            self.proc.args,
            body,
            self.proc.srcinfo)

    def push_frame(self):
        self.tofree.append(set())

    def add_malloc(self, sym, typ, mem):
        self.tofree[-1].add((sym, typ, mem))

    def pop_frame(self, srcinfo):
        suffix = [ LoopIR.Free(nm, typ, mem, srcinfo)
                   for (nm, typ, mem) in self.tofree.pop() ]

        return suffix

    def mem_stmts(self, stmts):
        if len(stmts) == 0:
            return stmts

        body = []
        self.push_frame()
        for b in stmts:
            body.append(self.mem_s(b))
        body += self.pop_frame(body[0].srcinfo)

        return body

    def mem_s(self, s):
        styp = type(s)

        if (styp is LoopIR.Pass or styp is LoopIR.Assign or
              styp is LoopIR.Reduce or styp is LoopIR.Call):
            return s
        elif styp is LoopIR.If:
            body    = self.mem_stmts(s.body)
            ebody   = self.mem_stmts(s.orelse)
            return LoopIR.If(s.cond, body, ebody, s.srcinfo)
        elif styp is LoopIR.ForAll:
            body = self.mem_stmts(s.body)
            return LoopIR.ForAll(s.iter, s.hi, body, s.srcinfo)
        elif styp is LoopIR.Alloc:
            self.add_malloc(s.name, s.type, s.mem)
            return s
        elif styp is LoopIR.Free:
            assert False, ("There should not be frees inserted " +
                           "before mem analysis")
        else:
            assert False, "bad case"
