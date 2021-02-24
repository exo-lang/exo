from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Types

_Types = ADT("""
module Types {
    type    = Num   ()
            | Bool  ()
            | Int   ()
            | Index ()
            | Size  ()
            | Error ()
            | Tensor( range hi, type type )

    effect  = IN    ()
            | OUT   ()
            | INOUT ()
} """, {
    'range': lambda x: is_pos_int(x) or type(x) is Sym,
})
ADTmemo(_Types, ['Num', 'Bool', 'Int', 'Index', 'Size', 'Error',
                 'Tensor', 'IN', 'OUT', 'INOUT'], {
    'range': lambda x: x,
})

Num     = _Types.Num
Bool    = _Types.Bool
Int     = _Types.Int
Index   = _Types.Index
Size    = _Types.Size
Error   = _Types.Error
Tensor  = _Types.Tensor
R       = Num()
bool    = Bool()    # note: accessed as T.bool outside this module
int     = Int()
index   = Index()
size    = Size()
err     = Error()

IN = _Types.IN
OUT = _Types.OUT
INOUT = _Types.INOUT
In = IN()
Out = OUT()
InOut = INOUT()

# --------------------------------------------------------------------------- #
# type helper functions


def is_type(obj):
    return isinstance(obj, _Types.type)


def is_effect(obj):
    return isinstance(obj, _Types.effect)

@extclass(Tensor)
@extclass(Num)
def shape(t):
    shp = []
    while type(t) is Tensor:
        shp.append(t.hi)
        t = t.type
    assert t is R
    return shp
del shape

@extclass(_Types.type)
def is_numeric(t):
    return type(t) is Num or type(t) is Tensor
del is_numeric

@extclass(_Types.type)
def is_indexable(t):
    return type(t) is Int or type(t) is Index or type(t) is Size
del is_indexable

@extclass(_Types.type)
def is_sizeable(t):
    return type(t) is Int or type(t) is Size
del is_sizeable

@extclass(Tensor)
@extclass(Num)
def base(t):
    return R
del base

@extclass(_Types.type)
def subst(t, lookup):
    if type(t) is Tensor:
        typ     = t.type.subst(lookup)
        hi      = t.hi if is_pos_int(t.hi) else lookup[t.hi]
        return Tensor(hi, typ)
    else:
        return t
del subst

# --------------------------------------------------------------------------- #
# string representation of types...

@extclass(_Types.type)
def __str__(t):
    if not hasattr(t, '_str_cached'):
        if type(t) is Num:
            t._str_cached = "R"
        elif type(t) is Bool:
            t._str_cached = "bool"
        elif type(t) is Int:
            t._str_cached = "int"
        elif type(t) is Index:
            t._str_cached = "index"
        elif type(t) is Size:
            t._str_cached = "size"
        elif type(t) is Error:
            t._str_cached = "err"
        elif type(t) is Tensor:
            rngs = ",".join([str(r) for r in t.shape()])
            t._str_cached = f"{t.base()}[{rngs}]"
        else:
            assert False, "impossible type case"
    return t._str_cached


@extclass(IN)
def __str__(self):
    return "IN"


@extclass(OUT)
def __str__(self):
    return "OUT"


@extclass(INOUT)
def __str__(self):
    return "INOUT"


del __str__
