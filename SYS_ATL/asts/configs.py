from .conversions import uast_to_type
from ..prelude import Sym


################################################################################
# Config Objects

class Config:
    def __init__(self, name, fields, disable_rw):
        self._name = name
        self._fields = fields
        self._rw_ok = not disable_rw

        self._lookup = {nm: (i, uast_to_type(typ))
                        for i, (nm, typ) in enumerate(fields)}

        self._field_syms = {nm: Sym(f"{name}_{nm}")
                            for nm, typ in fields}

    def name(self):
        return self._name

    def fields(self):
        return self._fields

    def has_field(self, fname):
        return fname in self._lookup

    def lookup(self, fname):
        return self._lookup[fname]

    def _INTERNAL_sym(self, fname):
        return self._field_syms[fname]

    def is_allow_rw(self):
        return self._rw_ok

    def c_struct_def(self):
        lines = []
        lines += [f"struct {self._name} {{"]
        for f in self._fields:
            ltyp = self.lookup(f[0])[1]
            lines += [f"    {ltyp.ctype()} {f[0]};"]
        lines += [f"}} {self._name};"]
        return lines
