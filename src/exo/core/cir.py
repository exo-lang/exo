from .prelude import Sym, Operator, extclass

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
            | ReadSeparateDataptr ( sym name )
            | Custom  ( object callback, expr* args )  -- callback(str(args)...) -> str
} """,
    ext_types={
        "bool": bool,
        "sym": Sym,
        "op": validators.instance_of(Operator, convert=True),
        "str": str,
    },
)


@extclass(CIR.expr)
def exo_get_cir(self):
    return self


_operations = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "%": lambda x, y: x % y,
}


def simplify_cir(e):
    if isinstance(e, (CIR.Read, CIR.Const)):
        return e

    elif isinstance(e, CIR.BinOp):
        lhs = simplify_cir(e.lhs)
        rhs = simplify_cir(e.rhs)

        if isinstance(lhs, CIR.Const) and isinstance(rhs, CIR.Const):
            return CIR.Const(_operations[e.op](lhs.val, rhs.val))

        if isinstance(lhs, CIR.Const) and lhs.val == 0:
            if e.op == "+":
                return rhs
            elif e.op == "*" or e.op == "/":
                return CIR.Const(0)
            elif e.op == "-":
                pass  # cannot simplify
            else:
                assert False

        if isinstance(rhs, CIR.Const) and rhs.val == 0:
            if e.op == "+" or e.op == "-":
                return lhs
            elif e.op == "*":
                return CIR.Const(0)
            elif e.op == "/":
                assert False, "division by zero??"
            else:
                assert False, "bad case"

        if isinstance(lhs, CIR.Const) and lhs.val == 1 and e.op == "*":
            return rhs

        if isinstance(rhs, CIR.Const) and rhs.val == 1 and (e.op == "*" or e.op == "/"):
            return lhs

        return CIR.BinOp(e.op, lhs, rhs, e.is_non_neg)
    elif isinstance(e, CIR.USub):
        arg = simplify_cir(e.arg)
        if isinstance(arg, CIR.USub):
            return arg.arg
        if isinstance(arg, CIR.Const):
            return arg.update(val=-(arg.val))
        return e.update(arg=arg)
    elif isinstance(e, CIR.AddressOf):
        return e.update(arg=simplify_cir(e.arg))
    elif isinstance(e, CIR.Indexed):
        return e.update(ptr=simplify_cir(e.ptr), idx=simplify_cir(e.idx))
    elif isinstance(e, CIR.GetAttr):
        return e.update(arg=simplify_cir(e.arg))
    elif isinstance(e, CIR.Custom):
        args = [simplify_cir(value) for value in e.args]
        return e.update(args=args)
    else:
        assert False, f"bad case: {type(e)}"


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
        return self._compiler.comp_cir(simplify_cir(self._ir), 0)

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
