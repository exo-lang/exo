
# don't import T from LoopIR in order to break file inclusion circularity
#from .LoopIR import T
from . import LoopIR

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Config Objects


class ConfigError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

"""
# Configuration objects should work like structs
# for the time being, we will skip over implementing a
# nice front-end syntax for these using pyparser-style hijacking
# Instead, we will specify a creation/factory function here
def new_config(name, fields, disable_rw=False):
    str_to_type = {
        'size'      : LoopIR.T.size,
        'bool'      : LoopIR.T.bool,
        'index'     : LoopIR.T.index,
        'stride'    : LoopIR.T.stride,
        'f32'       : LoopIR.T.f32,
        'f64'       : LoopIR.T.f64,
        'i8'        : LoopIR.T.i8,
        'i32'       : LoopIR.T.i32,
    }
    good_args = (type(name) is str and
                 isinstance(fields, list) and
                 all( type(f) is tuple for f in fields ) and
                 all( type(f[0]) is str for f in fields ) and
                 all( type(f[1]) is str for f in fields ) and
                 all( f[1] in str_to_type for f in fields ))
    if not good_args:
        raise TypeError("Expected call to new_config to have the form:\n"+
                        "new_config('config_name',[\n"+
                        "  ('field1_name', 'field1_type'),\n"+
                        "  ('field2_name', 'field2_type'),\n"+
                        "  ..."+
                        "])\n"+
                        "where types are either: "+
                        "'size', 'index', 'bool', 'stride', or some "+
                        "real scalar type with a given precision")

    type_fields = [ (nm, str_to_type[t]) for nm,t in fields ]
    return Config(name, type_fields, disable_rw)
"""

class Config:
    def __init__(self, name, fields, disable_rw):
        self._name      = name
        self._fields    = fields
        self._rw_ok     = not disable_rw

        uast_to_type = {
            LoopIR.UAST.Size()      : LoopIR.T.size,
            LoopIR.UAST.Bool()      : LoopIR.T.bool,
            LoopIR.UAST.Index()     : LoopIR.T.index,
            LoopIR.UAST.Stride()    : LoopIR.T.stride,
            LoopIR.UAST.F32()       : LoopIR.T.f32,
            LoopIR.UAST.F64()       : LoopIR.T.f64,
            LoopIR.UAST.INT8()      : LoopIR.T.i8,
            LoopIR.UAST.INT32()     : LoopIR.T.i32,
        }

        self._lookup    = { nm : (i,uast_to_type[typ])
                            for i,(nm,typ) in enumerate(fields) }

    def name(self):
        return self._name

    def fields(self):
        return self._fields

    def has_field(self, fname):
        return fname in self._lookup

    def lookup(self, fname):
        return self._lookup[fname]

    def is_allow_rw(self):
        return self._rw_ok

    def c_struct_def(self):
        lines = []
        lines += [f"struct {self._name} {{"]
        for f in self._fields:
            lines += [f"    {f[1].ctype()} {f[0]};"]
        lines += [f"}} {self._name};"]
        return lines
