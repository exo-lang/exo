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

    def pop_frame(self, body):
        new_body = [body]
        for (nm, typ, mem) in self.tofree.pop():
            new_body.append(LoopIR.Free(nm, typ, mem, body.srcinfo))

        return new_body
    
    def mem_stmts(self, stmts):
        body = []
        for b in stmts:
            self.push_frame()
            mem   = self.mem_s(b)
            body += self.pop_frame(mem)

        return body

    def mem_s(self, s):
        styp = type(s)

        if (styp is LoopIR.Pass or styp is LoopIR.Assign or
              styp is LoopIR.Reduce):
            return s
        elif styp is LoopIR.If:
            body = self.mem_stmts(s.body)
            ebody = []
            if s.orelse:
                ebody = self.mem_stmts(s.orelse)
            return LoopIR.If(s.cond, body, ebody, s.srcinfo)
        elif styp is LoopIR.ForAll:
            body = self.mem_stmts(s.body)
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
