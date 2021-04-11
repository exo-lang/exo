from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from .LoopIR import T
from .LoopIR import LoopIR

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Precision Analysis Pass

class PrecisionAnalysis:
    def __init__(self, proc):
        assert type(proc) is LoopIR.proc
        self.errors = []
        self.proc = proc

    def result(self):
        body = self.map_stmts(self.proc.body)

        if len(self.errors) > 0:
            raise TypeError("Errors occurred during precision checking:\n" +
                            "\n".join(self.errors))

        return LoopIR.proc(
            self.proc.name,
            self.proc.args,
            self.proc.preds,
            body,
            self.proc.instr,
            self.proc.eff,
            self.proc.srcinfo)

    def err(self, node, msg):
        self.errors.append(f"{node.srcinfo}: {msg}")

    def map_stmts(self, stmts):
        if len(stmts) == 0:
            return stmts

        body = []
        for b in stmts:
            body.append(self.map_s(b))

        return body

    def map_s(self, s):
        styp = type(s)

        # Don't do anything for call for now.
        if (styp is LoopIR.Pass or styp is LoopIR.Alloc or styp is LoopIR.Free):
            return s
        # Check precision of caller and callee
        elif styp is LoopIR.Call:
            args = [ self.map_e(a) for a in s.args ]
            for call_a,sig_a in zip(args, s.f.args):
                if call_a.type.basetype() != sig_a.type.basetype():
                    self.err(s, "cannot call a subprocedure"+
                                " with a different precision")
            return s

        # Allow implicit casting for assign and reduce
        # If binop an operation in different precision type,
        # raise error
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            rhs = self.map_e(s.rhs)
            cast = None
            if (s.type != T.err and rhs.type != T.err and
                s.type != s.rhs.type):
                cast = s.type.basetype().ctype()
            return styp ( s.name, s.type, cast, s.idx, rhs, s.eff, s.srcinfo)

        elif styp is LoopIR.If:
            body    = self.map_stmts(s.body)
            ebody   = self.map_stmts(s.orelse)
            return LoopIR.If(s.cond, body, ebody, s.eff, s.srcinfo)

        elif styp is LoopIR.ForAll:
            body = self.map_stmts(s.body)
            return LoopIR.ForAll(s.iter, s.hi, body, s.eff, s.srcinfo)
        else:
            assert False, "bad case"

    def map_e(self, e):
        typ = type(e)
        if typ is LoopIR.Read or typ is LoopIR.Const:
            return e
        elif typ is LoopIR.BinOp:
            lhs = self.map_e(e.lhs)
            rhs = self.map_e(e.rhs)
            if lhs.type != rhs.type:
                # Typeerror if precision types are different
                self.err(e, "cannot compute different precision types")
                typ = T.err
            else:
                typ = lhs.type
            return LoopIR.BinOp(e.op, e.lhs, e.rhs, typ, e.srcinfo)
        else:
            assert False, "not a LoopIR in check_e"
