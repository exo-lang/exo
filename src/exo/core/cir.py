from .prelude import Sym, SrcInfo, Operator, extclass

from asdl_adt import ADT, validators
from dataclasses import dataclass
from typing import Optional

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


@dataclass(slots=True)
class CIR_Wrapper:
    _ir: CIR.expr
    _compiler: "Compiler"
    _origin_story: str

    def __init__(self, ir, compiler, origin_story):
        assert isinstance(ir, CIR.expr)
        self._ir = ir
        assert hasattr(compiler, "comp_cir")
        self._compiler = compiler
        assert isinstance(origin_story, str)
        self._origin_story = origin_story

    def __repr__(self):
        return f"CIR_Wrapper({self._ir}, {self._compiler}, {self._origin_story})"

    def __str__(self):
        return self._compiler.comp_cir(self._ir, 0)

    def __getattr__(self, attr):
        assert not attr.startswith(
            "exo_"
        ), f"{attr} is reserved (or typo of member function)"
        return CIR_Wrapper(
            CIR.GetAttr(self._ir, attr), self._compiler, self._origin_story
        )

    def __getitem__(self, index):
        if isinstance(index, int):
            index = CIR.Const(index)
        else:
            index = index.exo_get_cir()
        return CIR_Wrapper(
            CIR.Indexed(self._ir, index), self._compiler, self._origin_story
        )

    def __add__(self, val):
        return self.exo_bin_op(val, "+")

    def __radd__(self, val):
        return self.exo_bin_op(val, "+", True)

    def __sub__(self, val):
        return self.exo_bin_op(val, "-")

    def __rsub__(self, val):
        return self.exo_bin_op(val, "-", True)

    def __mul__(self, val):
        return self.exo_bin_op(val, "*")

    def __rmul__(self, val):
        return self.exo_bin_op(val, "*", True)

    def __floordiv__(self, val):
        return self.exo_bin_op(val, "/")

    def __truediv__(self, val):
        return self.exo_bin_op(val, "/")

    def __mod__(self, val):
        return self.exo_bin_op(val, "%")

    def exo_get_cir(self):
        return self._ir

    def exo_address_of(self):
        return CIR_Wrapper(CIR.AddressOf(self._ir), self._compiler, self._origin_story)

    def exo_bin_op(self, rhs, op, r=False):
        if isinstance(rhs, int):
            rhs = CIR.Const(rhs)
        elif isinstance(rhs, Sym):
            rhs = CIR.Read(rhs, False)  # is_non_neg?
        else:
            rhs = rhs.exo_get_cir()
        lhs = self._ir
        if r:
            lhs, rhs = rhs, lhs
        # is_non_neg maybe shouldn't be always False?
        bin_op = CIR.BinOp(op, lhs, rhs, False)
        return CIR_Wrapper(bin_op, self._compiler, self._origin_story)

    def __int__(self):
        assert self._origin_story is not None
        ir = self._ir
        if isinstance(ir, CIR.Const):
            return int(ir.val)
        else:
            raise ValueError(f"{self._origin_story}: needs to be int, not `{self}`")
