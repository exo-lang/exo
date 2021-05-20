from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from .LoopIR import LoopIR, T, LoopIR_Rewrite

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory Analysis Pass

class WindowAnalysis(LoopIR_Rewrite):
    def __init__(self, proc):
        assert type(proc) is LoopIR.proc
        super().__init__(proc)

    def err(self, node, msg):
        raise TypeError(f"{node.srcinfo}: {msg}")

    def map_s(self, s):
        if type(s) is LoopIR.Call:
            args = s.args
            def promote_tensor(a,sa):
                assert sa.type.is_win()
                assert not a.type.is_win()
                assert type(a) is LoopIR.Read and len(a.idx) == 0
                shape = a.type.shape()
                idx = [ LoopIR.Interval(LoopIR.Const(0,T.int,N.srcinfo),
                                        N, N.srcinfo)
                        for N in shape ]
                win_e = LoopIR.WindowExpr(a.name, idx, T.err, a.srcinfo)
                win_e.type = T.Window(a.type,
                                      T.Tensor(shape, True,
                                               a.type.basetype()),
                                      win_e)
                return win_e

            def promote_arg(a,sa):
                if sa.type.is_win() and not a.type.is_win():
                    return promote_tensor(a,sa)
                elif (type(sa.type) is T.Tensor and
                      not sa.type.is_win() and
                      a.type.is_win()):
                    self.err(a, "expected a non-window tensor")

                return a

            args = [ promote_arg(a,sa) for a,sa in zip(args, s.f.args) ] 

            return [LoopIR.Call( s.f, args, s.eff, s.srcinfo )]

        return super().map_s(s)

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self,e):
        return e
    def map_t(self,t):
        return t
    def map_eff(self,eff):
        return eff



