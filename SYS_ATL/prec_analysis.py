from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from .LoopIR import T
from .LoopIR import LoopIR, LoopIR_Rewrite



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Default Precision Management

_default_prec = T.f32
def set_default_prec(name):
    global _default_prec
    vals = {
        'f32'   : T.f32,
        'f64'   : T.f64,
        'i8'    : T.i8,
        'i32'   : T.i32,
    }
    if name not in vals:
        raise TypeError(f"Got {name}, but "+
                        "expected one of the following precision types: "+
                        ','.join([ k for k in vals ]))
    _default_prec = vals[name]

def get_default_prec():
    return _default_prec


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Precision Analysis Pass


class PrecisionAnalysis(LoopIR_Rewrite):
    def __init__(self, proc):
        assert type(proc) is LoopIR.proc
        self._errors = []

        super().__init__(proc)

        if len(self._errors) > 0:
            raise TypeError("Errors occurred during precision checking:\n" +
                            "\n".join(self._errors))

    def err(self, node, msg):
        self._errors.append(f"{node.srcinfo}: {msg}")

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Call:
            args = [ self.map_e(a) for a in s.args ]
            for call_a, sig_a in zip(args, s.f.args):
                if sig_a.type.is_numeric():
                    if call_a.type.basetype() != sig_a.type.basetype():
                        self.err(call_a,
                            f"expected precision {sig_a.type.basetype()}, "+
                            f"but got {call_a.type.basetype()}")
            return [LoopIR.Call( s.f, [ self.map_e(a) for a in s.args ],
                                 self.map_eff(s.eff), s.srcinfo )]

        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            rhs = self.map_e(s.rhs)
            cast = None
            if (s.type != T.err and rhs.type != T.err and
                s.type != s.rhs.type):
                cast = s.type.basetype().ctype()
            return [styp(s.name, s.type, cast, s.idx, rhs, s.eff, s.srcinfo)]

        # fall-through
        return super().map_s(s)

    def map_e(self, e):
        if type(e) is LoopIR.BinOp:
            lhs = self.map_e(e.lhs)
            rhs = self.map_e(e.rhs)
            if (lhs.type.is_numeric() and rhs.type.is_numeric() and 
                lhs.type != rhs.type):
                # Typeerror if precision types are different
                self.err(e, f"cannot compute operation '{e.op}' between "+
                            f"an {lhs.type} and {rhs.type} value")
                typ = T.err
            else:
                typ = lhs.type
            return LoopIR.BinOp(e.op, e.lhs, e.rhs, typ, e.srcinfo)

        return super().map_e(e)

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_t(self,t):
        return t
    def map_eff(self,eff):
        return eff
