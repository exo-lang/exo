from inspect import currentframe as _curr_frame, getframeinfo as _get_frame_info
from re import compile as _re_compile
from dataclasses import dataclass as _dataclass


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


class SrcInfo:
    def __init__(
        self,
        filename,
        lineno,
        col_offset=None,
        end_lineno=None,
        end_col_offset=None,
        function=None,
    ):
        self.filename = filename
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset
        self.function = function

    def __str__(self):
        colstr = "" if self.col_offset is None else f":{self.col_offset}"
        return f"{self.filename}:{self.lineno}{colstr}"


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

    def __new__(cls, arg):
        """From ScalarInfo (no-op), Exo name, C name, or (internal) LoopIR type"""
        if isinstance(arg, ScalarInfo):
            return arg
        if isinstance(arg, str):
            return _scalar_info_dict[arg]
        from .LoopIR import LoopIR, UAST

        if isinstance(arg, (LoopIR.type, UAST.type)):
            return _scalar_info_dict[type(arg)]
        if isinstance(arg, type):
            return _scalar_info_dict[arg]
        assert 0, "Expect str, ScalarInfo, or LoopIR.type"

    def extclass(uast, t, shorthand, ctype, bits):
        from .LoopIR import LoopIR, UAST, uast_prim_types, loopir_from_uast_type_table

        assert isinstance(uast, UAST.type)
        assert isinstance(t, LoopIR.type)
        info = object.__new__(ScalarInfo)
        info.shorthand = shorthand
        info.ctype = ctype
        info.bits = bits
        _scalar_info_dict[shorthand] = info
        _scalar_info_dict[ctype] = info
        _scalar_info_dict[type(t)] = info
        _scalar_info_dict[type(uast)] = info
        uast_prim_types[shorthand] = uast
        loopir_from_uast_type_table[type(uast)] = t

        @extclass(type(t))
        def scalar_info(t):
            return info
