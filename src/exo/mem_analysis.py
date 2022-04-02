from .LoopIR import LoopIR

from .memory import DRAM


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory Analysis Pass

class MemoryAnalysis:
    def __init__(self, proc):
        assert isinstance(proc, LoopIR.proc)

        self.mem_env = {}

        self.proc = proc
        self.tofree = []

        for a in proc.args:
            if a.type.is_numeric():
                mem = a.mem if a.mem else DRAM
                self.mem_env[a.name] = mem

    def result(self):
        body = self.mem_stmts(self.proc.body)
        assert (len(self.tofree) == 0)

        return LoopIR.proc(
            self.proc.name,
            self.proc.args,
            self.proc.preds,
            body,
            self.proc.instr,
            self.proc.eff,
            self.proc.srcinfo)

    def push_frame(self):
        self.tofree.append([])

    def add_malloc(self, sym, typ, mem):
        assert isinstance(self.tofree[-1], list)
        assert isinstance((sym, typ, mem), tuple)
        self.tofree[-1].append((sym, typ, mem))

    def pop_frame(self, srcinfo):
        suffix = [ LoopIR.Free(nm, typ, mem, None, srcinfo)
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

    def get_e_mem(self, e):
        if isinstance(e, (LoopIR.WindowExpr, LoopIR.Read)):
            return self.mem_env[e.name]
        else: assert False

    def mem_s(self, s):
        styp = type(s)

        if (styp is LoopIR.Pass or styp is LoopIR.Assign or
              styp is LoopIR.Reduce or styp is LoopIR.WriteConfig):
            return s

        elif styp is LoopIR.WindowStmt:
            mem = self.get_e_mem(s.rhs)
            self.mem_env[s.lhs] = mem
            return s

        elif styp is LoopIR.Call:
            # check memory consistency at call boundaries
            for ca, sa in zip(s.args, s.f.args):
                if sa.type.is_numeric():
                    smem = sa.mem if sa.mem else DRAM
                    cmem = self.get_e_mem(ca)
                    if not issubclass(cmem, smem):
                        raise TypeError(f"{ca.srcinfo}: expected "
                                        f"argument in {smem.name()} but got an "
                                        f"argument in {cmem.name()}")

            return s

        elif styp is LoopIR.If:
            body    = self.mem_stmts(s.body)
            ebody   = self.mem_stmts(s.orelse)
            return LoopIR.If(s.cond, body, ebody, None, s.srcinfo)
        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            body = self.mem_stmts(s.body)
            return styp(s.iter, s.hi, body, None, s.srcinfo)
        elif styp is LoopIR.Alloc:
            mem = s.mem if s.mem else DRAM
            self.mem_env[s.name] = mem
            self.add_malloc(s.name, s.type, s.mem)
            return s
        elif styp is LoopIR.Free:
            assert False, ("There should not be frees inserted before mem "
                           "analysis")
        else:
            assert False, f"bad case {styp}"
