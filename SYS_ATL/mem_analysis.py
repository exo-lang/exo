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
        self.push_frame()
        body = self.mem_s(self.proc.body)
        body = self.pop_frame(body)
        return LoopIR.proc(
            self.proc.name,
            self.proc.sizes,
            self.proc.args,
            body,
            self.proc.srcinfo)

    def push_frame(self):
        self.tofree.append(set())

    def add_malloc(self, sym, typ, mem):
        self.tofree[-1].add((sym, typ, mem))

    def pop_frame(self, body):
        for (nm, typ, mem) in self.tofree.pop():
            body = LoopIR.Seq(body,
                              LoopIR.Free(nm, typ, mem, body.srcinfo),
                              body.srcinfo)
        return body

    def mem_s(self, s):
        styp = type(s)

        if styp is LoopIR.Seq:
            s0 = self.mem_s(s.s0)
            s1 = self.mem_s(s.s1)
            return LoopIR.Seq(s0, s1, s.srcinfo)
        elif (styp is LoopIR.Pass or styp is LoopIR.Assign or
              styp is LoopIR.Reduce):
            return s
        elif styp is LoopIR.If:
            self.push_frame()
            body = self.mem_s(s.body)
            body = self.pop_frame(body)
            ebody = None
            if s.orelse:
                self.push_frame()
                ebody = self.mem_s(s.orelse)
                ebody = self.pop_frame(ebody)
            return LoopIR.If(s.cond, body, ebody, s.srcinfo)
        elif styp is LoopIR.ForAll:
            self.push_frame()
            body = self.mem_s(s.body)
            body = self.pop_frame(body)
            return LoopIR.ForAll(s.iter, s.hi, body, s.srcinfo)
        elif styp is LoopIR.Instr:
            body = self.mem_s(s.body)
            return LoopIR.Instr(s.op, body, s.srcinfo)
        elif styp is LoopIR.Alloc:
            self.add_malloc(s.name, s.type, s.mem)
            return s
        elif styp is LoopIR.Free:
            assert False, ("There should not be frees inserted " +
                           "before mem analysis")
        else:
            assert False, "bad case"
