from .prelude import Sym, SrcInfo, Operator, extclass

from asdl_adt import ADT, validators

# --------------------------------------------------------------------------- #
# C Codegen AST
# --------------------------------------------------------------------------- #

CIR = ADT(
    """
module CIR {

    expr    = Read    ( sym name, bool is_non_neg )
            | Const   ( object val )
            | BinOp   ( op op, expr lhs, expr rhs, bool is_non_neg )
            | USub    ( expr arg, bool is_non_neg )
            | AddressOf(expr arg )
            | Indexed ( expr ptr, expr idx )  -- ptr[idx]
            | GetAttr ( expr arg, str attr ) -- arg.attr
            -- format.format(**kwargs) if kwargs else format
            | Custom  ( str format, dict kwargs, srcinfo srcinfo )
} """,
    ext_types={
        "bool": bool,
        "sym": Sym,
        "op": validators.instance_of(Operator, convert=True),
        "str": str,
        "dict": dict,
        "srcinfo": SrcInfo,
    },
)


@extclass(CIR.expr)
def exo_get_cir(self):
    return self


class CIR_Wrapper:
    __slots__ = ["_ir"]
    _ir: CIR.expr

    def __init__(self, arg):
        if isinstance(arg, Sym):
            # is_non_neg maybe shouldn't be always False?
            self._ir = CIR.Read(arg, False)
        elif isinstance(arg, int):
            self._ir = CIR.Const(arg)
        else:
            self._ir = arg.exo_get_cir()

    def __repr__(self):
        return f"CIR_Wrapper({self._ir})"

    def __getattr__(self, attr):
        assert not attr.startswith(
            "exo_"
        ), f"{attr} is reserved (or typo of member function)"
        return CIR_Wrapper(CIR.GetAttr(self._ir, attr))

    def __getitem__(self, index):
        result = CIR_Wrapper(index)
        result._ir = CIR.Indexed(self._ir, result._ir)
        return result

    def __add__(self, val):
        return self.exo_bin_op(val, "+")

    def __radd__(self, val):
        return self.exo_bin_op(val, "+")

    def __sub__(self, val):
        return self.exo_bin_op(val, "-")

    def __mul__(self, val):
        return self.exo_bin_op(val, "*")

    def __rmul__(self, val):
        return self.exo_bin_op(val, "*")

    def __floordiv__(self, val):
        return self.exo_bin_op(val, "/")

    def __truediv__(self, val):
        return self.exo_bin_op(val, "/")

    def __mod__(self, val):
        return self.exo_bin_op(val, "*")

    def exo_get_cir(self):
        return self._ir

    def exo_address_of(self):
        return CIR_Wrapper(CIR.AddressOf(self._ir))

    def exo_bin_op(self, rhs, op):
        result = CIR_Wrapper(rhs)
        # is_non_neg maybe shouldn't be always False?
        result._ir = CIR.BinOp(op, self._ir, result._ir, False)
        return result
