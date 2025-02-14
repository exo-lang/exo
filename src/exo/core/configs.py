from weakref import WeakKeyDictionary
from .prelude import *

# CAUTION: cannot import LoopIR here due to circular inclusion.

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


_reverse_symbol_lookup = WeakKeyDictionary()


def reverse_config_lookup(sym):
    return _reverse_symbol_lookup[sym]


class Config:
    def __init__(self, name, fields, disable_rw):
        self._name = name
        self._fields = fields
        self._rw_ok = not disable_rw

        from .LoopIR import loopir_from_uast_metatype_table as table

        self._lookup = {nm: table[type(typ)] for nm, typ in fields}

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
            lines += [f"    {ltyp.ctype()} {f[0]};"]
        lines += [f"}} {self._name};"]
        return lines
