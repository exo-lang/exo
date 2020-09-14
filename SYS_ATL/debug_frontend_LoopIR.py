
#from .prelude import *

from . import shared_types as _T
from . import prelude as _prelude

from .LoopIR import LoopIR as IR

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

Sym = _prelude.Sym

class TypeWrap:
    def __init__(self,typ):
        self.typ = typ

    def __getitem__(self,keys):
        if type(keys) is not tuple:
            keys = (keys,)
        typ = self.typ
        for k in reversed(keys):
            assert type(k) is _prelude.Sym or _prelude.is_pos_int(k)
            typ = _T.Tensor(k, typ)
        return TypeWrap(typ)

R = TypeWrap(_T.R)

#  typ = R[N,M]

def _as_stmt_list(stmts,srcinfo):
    assert type(stmts) is list
    if len(stmts) == 0:
        return IR.Pass()
    # otherwise
    s = stmts[0]
    for s2 in stmts[1:]:
        s = IR.Seq(s, s2, srcinfo)
    return s

def Proc(name, sizes, args, body):
    srcinfo = _prelude.get_srcinfo()
    def eff_convert(e):
        if e == 'IN':
            return _T.In
        elif e == 'OUT':
            return _T.Out
        elif e == 'INOUT':
            return _T.InOut
        else: assert False, f"bad effect: {e}"
    args = [ IR.fnarg(name, t.typ, eff_convert(e), srcinfo)
             for (name, t, e) in args ]
    return IR.proc(name, sizes, args, _as_stmt_list(body,srcinfo), srcinfo)
