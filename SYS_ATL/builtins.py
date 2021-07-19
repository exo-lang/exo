
from .prelude import *

from . import LoopIR

import math


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# BuiltIn superclass

class BuiltIn_Typecheck_Error(Exception):
  def __init__(self,msg):
    self._builtin_err_msg   = str(msg)
  def __str__(self):
    return self._builtin_err_msg

_BErr = BuiltIn_Typecheck_Error


class BuiltIn:
  def __init__(self,name):
    self._name  = name

  def name(self):
    return self._name

  def typecheck(self, *args):
    raise NotImplementedError()

  def interpret(self, *args):
    raise NotImplementedError()

  def compile(self, *args):
    raise NotImplementedError()

class _Sin(BuiltIn):
  def __init__(self):
    super().__init__('sin')

  def typecheck(self, *args):
    if len(args) != 1:
      raise _BErr(f"expected 1 argument, got {len(args)}")
    atyp    = args[0].type
    if not atyp.is_real_scalar():
      raise _BErr(f"expected argument 1 to be a real scalar value, but "+
                  f"got type {atyp}")
    return atyp

  def interpret(self, *args):
    raise NotImplementedError()

  def compile(self, *args):
    raise NotImplementedError()

sin = _Sin()












