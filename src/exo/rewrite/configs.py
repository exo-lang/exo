from . import LoopIR

from weakref import WeakKeyDictionary
from .prelude import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Config Objects


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


# Configuration objects should work like structs
# for the time being, we will skip over implementing a
# nice front-end syntax for these using pyparser-style hijacking
# Instead, we will specify a creation/factory function here

# Because of the recursive inclusion, we cannot use ctype in LoopIR here..
def ctyp(typ):
    if isinstance(typ, LoopIR.T.F16):
        return "_Float16"
    elif isinstance(typ, LoopIR.T.F32):
        return "float"
    elif isinstance(typ, LoopIR.T.F64):
        return "double"
    elif isinstance(typ, LoopIR.T.INT8):
        return "int8_t"
    elif isinstance(typ, LoopIR.T.INT32):
        return "int32_t"
    elif isinstance(typ, LoopIR.T.Bool):
        return "bool"
    elif (
        isinstance(typ, LoopIR.T.Index)
        or isinstance(typ, LoopIR.T.Size)
        or isinstance(typ, LoopIR.T.Stride)
    ):
        return "int_fast32_t"
    else:
        assert False, f"bad case! {typ}"


_reverse_symbol_lookup = WeakKeyDictionary()


def reverse_config_lookup(sym):
    return _reverse_symbol_lookup[sym]


class Config:
    def __init__(self, name, fields, disable_rw):
        self._name = name
        self._fields = fields
        self._rw_ok = not disable_rw

        uast_to_type = {
            LoopIR.UAST.Size(): LoopIR.T.size,
            LoopIR.UAST.Bool(): LoopIR.T.bool,
            LoopIR.UAST.Index(): LoopIR.T.index,
            LoopIR.UAST.Stride(): LoopIR.T.stride,
            LoopIR.UAST.F16(): LoopIR.T.f16,
            LoopIR.UAST.F32(): LoopIR.T.f32,
            LoopIR.UAST.F64(): LoopIR.T.f64,
            LoopIR.UAST.INT8(): LoopIR.T.i8,
            LoopIR.UAST.INT32(): LoopIR.T.i32,
        }

        for nm, typ in fields:
            assert typ in uast_to_type

        self._lookup = {nm: uast_to_type[typ] for nm, typ in fields}

        self._field_syms = {nm: Sym(f"{name}_{nm}") for nm, typ in fields}
        for fname, sym in self._field_syms.items():
            _reverse_symbol_lookup[sym] = (self, fname)

    def name(self):
        return self._name

    def fields(self):
        return self._fields

    def has_field(self, fname):
        return fname in self._lookup

    def lookup_type(self, fname):
        return self._lookup[fname]

    def _INTERNAL_sym(self, fname):
        return self._field_syms[fname]

    def is_allow_rw(self):
        return self._rw_ok

    def c_struct_def(self):
        lines = []
        lines += [f"struct {self._name} {{"]
        for f in self._fields:
            ltyp = self.lookup_type(f[0])
            lines += [f"    {ctyp(ltyp)} {f[0]};"]
        lines += [f"}} {self._name};"]
        return lines
