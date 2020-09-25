from .prelude import *
from .LoopIR  import UAST
from . import shared_types as T

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Procedure Objects

class Procedure:
  def __init__(self, uast, _testing=None):
    assert isinstance(uast, UAST.proc)

    self._uast_proc   = uast
    if _testing != "UAST":
      pass # continue with rest of function

  def _TESTING_UAST(self):
    return self._uast_proc
