from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Types

_Types = ADT("""
module Types {
    type    = Num   ()
            | Error ()
            | Tensor( range hi, type type )
} """, {
    'range': lambda x: is_pos_int(x) or type(x) is Sym,
})
ADTmemo(_Types,['Num','Error','Tensor'],{
    'range': lambda x: x,
})

Num     = _Types.Num
Error   = _Types.Error
Tensor  = _Types.Tensor
R       = Num()
err     = Error()

# --------------------------------------------------------------------------- #
# type helper functions

def is_type(obj):
  return isinstance(obj,_Types.type)

@extclass(Tensor)
@extclass(Num)
def shape(t):
  shp = []
  while type(t) is Tensor:
    shp.append(t.range)
    t = t.type
  assert t is R
  return shp
del shape

@extclass(Tensor)
@extclass(Num)
def base(t):
  return R
del base


# --------------------------------------------------------------------------- #
# string representation of types...

@extclass(_Types.type)
def __str__(t):
  if not hasattr(t,'_str_cached'):
    if   type(t) is Num:
      t._str_cached = "R"
    elif type(t) is Error:
      t._str_cached = "err"
    elif type(t) is Tensor:
      rngs = ",".join([ str(r) for r in t.shape() ])
      t._str_cached = f"[{rngs}]{t.base()}"
    else: assert False, "impossible type case"
  return t._str_cached
del __str__
