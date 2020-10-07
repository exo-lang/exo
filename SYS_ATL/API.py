from .prelude import *
from .LoopIR import UAST
from . import shared_types as T
from .typecheck import TypeChecker
from .LoopIR_compiler import Compiler, run_compile

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Procedure Objects


class Procedure:
    def __init__(self, uast, _testing=None):
        assert isinstance(uast, UAST.proc)

        self._uast_proc = uast
        if _testing != "UAST":
            pass  # continue with rest of function

    def compile_c(self, directory, filename):
        # Call TypeChecker to typecheck and to lower UAST to LoopIR
        proc = TypeChecker(self._uast_proc).get_loopir()
        run_compile([proc], directory, (filename + ".c"), (filename + ".h"))
        return proc
