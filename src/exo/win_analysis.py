from .LoopIR import LoopIR, T, LoopIR_Rewrite

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory Analysis Pass

class WindowAnalysis(LoopIR_Rewrite):
    def __init__(self, proc):
        assert isinstance(proc, LoopIR.proc)
        super().__init__(proc)

    def err(self, node, msg):
        raise TypeError(f"{node.srcinfo}: {msg}")

    def map_s(self, s):
        if isinstance(s, LoopIR.Call):
            args = s.args

            def promote_tensor(a, sa):
                assert sa.type.is_win()
                assert not a.type.is_win()
                assert isinstance(a, LoopIR.Read) and len(a.idx) == 0
                shape = a.type.shape()
                assert len(shape) > 0
                idx = [LoopIR.Interval(LoopIR.Const(0, T.int, N.srcinfo),
                                       N, N.srcinfo)
                       for N in shape]
                as_tens = T.Tensor(shape, True, a.type.basetype())
                win_typ = T.Window(a.type, as_tens, a.name, idx)
                return LoopIR.WindowExpr(a.name, idx, win_typ, a.srcinfo)

            def promote_arg(a,sa):
                if sa.type.is_win() and not a.type.is_win():
                    return promote_tensor(a,sa)
                elif (isinstance(sa.type, T.Tensor) and
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
