from inspect import currentframe as _curr_frame, getframeinfo as _get_frame_info
from re import compile as _re_compile
from dataclasses import dataclass as _dataclass, replace as _replace
from typing import Optional as _Optional, List as _List
from warnings import warn


from ..API_types import ExoType as _ExoType


def is_pos_int(obj):
    return isinstance(obj, int) and obj >= 1


_valid_pattern = _re_compile(r"^[a-zA-Z_]\w*$")
valid_name_pattern = r"[a-zA-Z_]\w*"


def is_valid_name(obj):
    return (
        isinstance(obj, str)
        and obj != "_"
        and (_valid_pattern.match(obj) is not None)  # prohibit the name '_' universally
    )


class Sym:
    _unq_count = 1

    def __init__(self, nm):
        if not is_valid_name(nm):
            raise TypeError(f"expected an alphanumeric name string, but got '{nm}'")
        self._nm = nm
        self._id = Sym._unq_count
        Sym._unq_count += 1

    def __str__(self):
        return self._nm

    def __repr__(self):
        return f"{self._nm}_{self._id}"

    def __hash__(self):
        return id(self)

    def __lt__(self, rhs):
        assert isinstance(rhs, Sym)
        return (self._nm, self._id) < (rhs._nm, rhs._id)

    def __eq__(self, rhs):
        if not isinstance(rhs, Sym):
            return False
        return self._nm == rhs._nm and self._id == rhs._id

    def __ne__(self, rhs):
        return not (self == rhs)

    def name(self):
        return self._nm

    def copy(self):
        return Sym(self._nm)

    def id_number(self):
        return self._id


# from a github gist by victorlei
def extclass(cls):
    return lambda f: (setattr(cls, f.__name__, f) or f)


@_dataclass(slots=True)
class SrcInfo:
    filename: str
    lineno: int
    col_offset: _Optional[int] = None
    end_lineno: _Optional[int] = None
    end_col_offset: _Optional[int] = None
    function: _Optional[object] = None
    stmt_id: _Optional[int] = None
    expr_id: _Optional[int] = None

    def __str__(self):
        colstr = "" if self.col_offset is None else f":{self.col_offset}"
        s_str = "" if self.stmt_id is None else f":(s{self.stmt_id})"
        e_str = "" if self.expr_id is None else f":(e{self.expr_id})"
        return f"{self.filename}:{self.lineno}{colstr}{s_str}{e_str}"

    def update(self, **kwargs):
        return _replace(self, **kwargs)


SrcInfo.stmt_id_pattern = r":\(s(\d+)\)"
SrcInfo.expr_id_pattern = r":\(e(\d+)\)"


def get_srcinfo(depth=1):
    f = _curr_frame()
    for k in range(0, depth):
        f = f.f_back
    finfo = _get_frame_info(f)
    filename, lineno, function = finfo.filename, finfo.lineno, finfo.function
    del f, finfo
    return SrcInfo(filename, lineno, function)


_null_srcinfo_obj = SrcInfo("unknown", 0)


def null_srcinfo():
    return _null_srcinfo_obj


# --------------------------------------------------------------------------- #
# Validated string subtypes
# --------------------------------------------------------------------------- #


class Identifier(str):
    _valid_re = _re_compile(r"^(?:_\w|[a-zA-Z])\w*$")

    def __new__(cls, name):
        name = str(name)
        if Identifier._valid_re.match(name):
            return super().__new__(cls, name)
        raise ValueError(f"invalid identifier: {name}")


class IdentifierOrHole(str):
    _valid_re = _re_compile(r"^[a-zA-Z_]\w*$")

    def __new__(cls, name):
        name = str(name)
        if IdentifierOrHole._valid_re.match(name):
            return super().__new__(cls, name)
        raise ValueError(f"invalid identifier: {name}")


comparison_ops = {"<", ">", "<=", ">=", "=="}
arithmetic_ops = {"+", "-", "*", "/", "%"}
logical_ops = {"and", "or"}

front_ops = comparison_ops | arithmetic_ops | logical_ops


class Operator(str):
    def __new__(cls, op):
        op = str(op)
        if op in front_ops:
            return super().__new__(cls, op)
        raise ValueError(f"invalid operator: {op}")


# --------------------------------------------------------------------------- #
# Scalar Type Info
# --------------------------------------------------------------------------- #


_scalar_info_dict = {}


@_dataclass(slots=True, init=False)
class ScalarInfo:
    shorthand: str  # Exo name, e.g. f64
    ctype: str  # C name, e.g. double
    bits: int  # bit width, e.g. 64
    uast: "UAST.type"
    loopir: "LoopIR.type"
    exotype: _ExoType

    def __new__(cls, arg):
        """From ScalarInfo (no-op), Exo name, C name, or UAST/LoopIR/ExoType"""
        if isinstance(arg, ScalarInfo):
            return arg
        if isinstance(arg, (str, _ExoType)):
            return _scalar_info_dict[arg]
        if isinstance(arg, type):
            return _scalar_info_dict[arg]

        from .LoopIR import LoopIR, UAST

        if isinstance(arg, (LoopIR.type, UAST.type)):
            return _scalar_info_dict[type(arg)]

        raise TypeError("Expect str, ScalarInfo, or ExoType")
        # or internal LoopIR or UAST

    def __repr__(self):
        return self.shorthand

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def extclass(uast, t, exotype, shorthand, ctype, bits):
        from .LoopIR import (
            LoopIR,
            UAST,
            uast_prim_types,
            loopir_from_uast_metatype_table,
            uast_concrete_scalar_metatypes,
            loopir_concrete_scalar_metatypes,
        )

        assert shorthand != "R" and ctype != "R"
        assert isinstance(uast, UAST.type)
        assert isinstance(t, LoopIR.type)
        assert isinstance(exotype, _ExoType)
        info = object.__new__(ScalarInfo)
        info.shorthand = shorthand
        info.ctype = ctype
        info.bits = bits
        info.uast = uast
        info.loopir = t
        info.exotype = exotype
        loopir_metatype = type(t)
        uast_metatype = type(uast)
        _scalar_info_dict[shorthand] = info
        _scalar_info_dict[ctype] = info
        _scalar_info_dict[loopir_metatype] = info
        _scalar_info_dict[uast_metatype] = info
        _scalar_info_dict[exotype] = info
        _ExoType.numerics_set.add(exotype)
        uast_prim_types[shorthand] = uast
        loopir_from_uast_metatype_table[uast_metatype] = t
        uast_concrete_scalar_metatypes.append(uast_metatype)
        loopir_concrete_scalar_metatypes.append(loopir_metatype)

        if shorthand == "f16":
            _scalar_info_dict["_Float16"] = info

        @extclass(type(t))
        def scalar_info(t):
            return info

        return info

    @staticmethod
    def get_scalar_names() -> _List[str]:
        return sorted(s for s in _scalar_info_dict if isinstance(s, str))

    def get_scale_bytes_suffix(self):
        bits = self.bits
        if bits == 4:
            return " / 2"
        elif bits == 8:
            return ""
        else:
            assert bits % 8 == 0
            return " * " + str(bits // 8)

    class same:
        __slots__ = []

        def __contains__(self, tup):
            t0 = tup[0]
            assert isinstance(t0, ScalarInfo)
            return all(t == t0 for t in tup)

        def __str__(self):
            return "tuple of identical types"
