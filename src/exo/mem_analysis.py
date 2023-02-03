from collections import ChainMap
from .LoopIR import LoopIR

from .memory import DRAM


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory Analysis Pass


class Check_CallsMemoryTypes:
    def __init__(self):
        self.mem_env = ChainMap()
        self.call_stmts = None
        self.reason = ""

    def check(self, proc, call_stmts=None):
        assert isinstance(proc, LoopIR.proc)

        self.mem_env = ChainMap()
        self.call_stmts = call_stmts
        self.check_passed = True
        self.reason = ""

        for a in proc.args:
            if a.type.is_numeric():
                mem = a.mem if a.mem else DRAM
                self.mem_env[a.name] = mem

        self.push()
        self.check_stmts(proc.body)
        self.pop()

        return self.check_passed, self.reason

    def push(self):
        self.mem_env = self.mem_env.new_child()

    def pop(self):
        self.mem_env = self.mem_env.parents

    def get_e_mem(self, e):
        if isinstance(e, (LoopIR.WindowExpr, LoopIR.Read)):
            return self.mem_env[e.name]
        else:
            assert False

    def check_stmts(self, stmts):
        for stmt in stmts:
            self.check_s(stmt)

    def check_s(self, s):
        styp = type(s)

        if styp is LoopIR.WindowStmt:
            mem = self.get_e_mem(s.rhs)
            self.mem_env[s.lhs] = mem

        elif styp is LoopIR.Call:
            if self.call_stmts is None or s in self.call_stmts:
                # check memory consistency at call boundaries
                for ca, sa in zip(s.args, s.f.args):
                    if sa.type.is_numeric():
                        smem = sa.mem if sa.mem else DRAM
                        cmem = self.get_e_mem(ca)
                        if not issubclass(cmem, smem):
                            self.check_passed = False
                            self.reason = (
                                f"{ca.srcinfo}: expected "
                                f"argument in {smem.name()} but got an "
                                f"argument in {cmem.name()}"
                            )

        elif styp is LoopIR.If:
            self.push()
            self.check_stmts(s.body)
            self.pop()
            self.push()
            self.check_stmts(s.orelse)
            self.pop()
        elif styp is LoopIR.Seq:
            self.push()
            self.check_stmts(s.body)
            self.pop()
        elif styp is LoopIR.Alloc:
            mem = s.mem if s.mem else DRAM
            self.mem_env[s.name] = mem


class MemoryAnalysis:
    def __init__(self):
        self.tofree = []

    def run(self, proc):
        assert isinstance(proc, LoopIR.proc)

        check, reason = Check_CallsMemoryTypes().check(proc)
        if not check:
            raise TypeError(reason)

        self.tofree = []

        self.push()
        body = self.mem_stmts(proc.body)
        self.pop()
        assert len(self.tofree) == 0

        return LoopIR.proc(
            proc.name,
            proc.args,
            proc.preds,
            body,
            proc.instr,
            proc.eff,
            proc.srcinfo,
        )

    def push(self):
        self.tofree.append([])

    def pop(self):
        assert len(self.tofree[-1]) == 0
        self.tofree.pop()

    def add_malloc(self, sym, typ, mem):
        assert isinstance(self.tofree[-1], list)
        assert isinstance((sym, typ, mem), tuple)
        self.tofree[-1].append((sym, typ, mem))

    def mem_stmts(self, stmts):
        if len(stmts) == 0:
            return stmts

        def used_e(e):
            res = []
            if isinstance(e, LoopIR.Read):
                res += [e.name]
                for ei in e.idx:
                    res += used_e(ei)
            elif isinstance(e, LoopIR.USub):
                res += used_e(e.arg)
            elif isinstance(e, LoopIR.BinOp):
                res += used_e(e.lhs)
                res += used_e(e.rhs)
            elif isinstance(e, LoopIR.BuiltIn):
                for ei in e.args:
                    res += used_e(ei)
            elif isinstance(e, (LoopIR.WindowExpr, LoopIR.StrideExpr)):
                res += [e.name]
            return res

        def used_s(s):
            res = []
            if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
                res += [s.name]
                res += used_e(s.rhs)
            elif isinstance(s, LoopIR.WriteConfig):
                res += used_e(s.rhs)
            elif isinstance(s, LoopIR.If):
                res += used_e(s.cond)
                for b in s.body:
                    res += used_s(b)
                for b in s.orelse:
                    res += used_s(b)
            elif isinstance(s, LoopIR.Seq):
                for b in s.body:
                    res += used_s(b)
            elif isinstance(s, LoopIR.Alloc):
                res += [s.name]
            elif isinstance(s, LoopIR.Call):
                for e in s.args:
                    res += used_e(e)
            elif isinstance(s, LoopIR.WindowStmt):
                res += used_e(s.rhs)
            return res

        body = []
        for b in reversed([self.mem_s(b) for b in stmts]):
            used = used_s(b)
            rm = []
            for (nm, typ, mem) in self.tofree[-1]:
                if nm in used:
                    rm += [(nm, typ, mem)]
            for (nm, typ, mem) in rm:
                body += [LoopIR.Free(nm, typ, mem, None, b.srcinfo)]
                self.tofree[-1].remove((nm, typ, mem))
            body += [b]

        return list(reversed(body))

    def mem_s(self, s):
        styp = type(s)

        if (
            styp is LoopIR.Pass
            or styp is LoopIR.Assign
            or styp is LoopIR.Reduce
            or styp is LoopIR.WriteConfig
            or styp is LoopIR.WindowStmt
            or styp is LoopIR.Call
        ):
            return s

        elif styp is LoopIR.If:
            self.push()
            body = self.mem_stmts(s.body)
            self.pop()
            self.push()
            ebody = self.mem_stmts(s.orelse)
            self.pop()
            return LoopIR.If(s.cond, body, ebody, None, s.srcinfo)
        elif styp is LoopIR.Seq:
            self.push()
            body = self.mem_stmts(s.body)
            self.pop()
            return styp(s.iter, s.hi, body, None, s.srcinfo)
        elif styp is LoopIR.Alloc:
            self.add_malloc(s.name, s.type, s.mem)
            return s
        elif styp is LoopIR.Free:
            assert False, "There should not be frees inserted before mem " "analysis"
        else:
            assert False, f"bad case {styp}"
