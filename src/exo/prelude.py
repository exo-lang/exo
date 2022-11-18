from __future__ import annotations

import inspect
import re

__all__ = [
    "is_pos_int",
    "is_valid_name",
    "Sym",
    "extclass",
    "SrcInfo",
    "get_srcinfo",
    "null_srcinfo",
]


def is_pos_int(obj):
    return isinstance(obj, int) and obj >= 1


_valid_pattern = re.compile(r"^[a-zA-Z_]\w*$")


def is_valid_name(obj):
    # prohibit the name '_' universally
    return (
        isinstance(obj, str) and obj != "_" and (_valid_pattern.match(obj) is not None)
    )


class Sym:
    _unq_count = 1

    def __init__(self, nm):
        if not is_valid_name(nm):
            raise TypeError(f"expected an alphanumeric name string, " f"but got '{nm}'")
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

    def name(self):
        return self._nm

    def copy(self):
        return Sym(self._nm)


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
    f = inspect.currentframe()
    for k in range(0, depth):
        f = f.f_back
    finfo = inspect.getframeinfo(f)
    filename, lineno, function = finfo.filename, finfo.lineno, finfo.function
    del f, finfo
    return SrcInfo(filename, lineno, function)


_null_srcinfo_obj = SrcInfo("unknown", 0)


def null_srcinfo():
    return _null_srcinfo_obj
