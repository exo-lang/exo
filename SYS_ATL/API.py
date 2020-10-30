from .prelude import *
from .LoopIR import UAST, LoopIR
from . import shared_types as T
from .typecheck import TypeChecker
from .LoopIR_compiler import Compiler, run_compile
from .LoopIR_scheduling import Schedules, name_str_2_symbols, name_str_2_pairs

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Procedure Objects


class Procedure:
    def __init__(self, proc, _testing=None):
        if isinstance(proc, LoopIR.proc):
            self._loopir_proc = proc
        else:
            assert isinstance(proc, UAST.proc)

            self._uast_proc = proc
            if _testing != "UAST":
                self._loopir_proc = TypeChecker(proc).get_loopir()

    def __str__(self):
        if hasattr(self,'_loopir_proc'):
            return str(self._loopir_proc)
        else:
            return str(self._uast_proc)

    def compile_c(self, directory, filename):
        run_compile([self._loopir_proc], directory,
                    (filename + ".c"), (filename + ".h"))

    # scheduling operations
    def split(self, split_var, split_const, out_vars):
        if type(split_var) is not str:
            raise TypeError("expected first arg to be a string")
        elif not is_pos_int(split_const):
            raise TypeError("expected second arg to be a positive integer")
        elif split_const == 1:
            raise TypeError("why are you trying to split by 1?")
        elif not isinstance(out_vars,list) and not isinstance(out_vars, tuple):
            raise TypeError("expected third arg to be a list or tuple")
        elif len(out_vars) != 2:
            raise TypeError("expected third arg list/tuple to have length 2")
        elif type(out_vars[0]) != str or type(out_vars[1]) != str:
            raise TypeError("expected third arg to be a list/tuple of two "+
                            "strings")

        split_names = name_str_2_symbols(self._loopir_proc, split_var)
        if len(split_names) == 0:
            raise TypeError(f"failed to find any symbols described by "+
                            f"'{split_var}'")
        loopir      = self._loopir_proc
        for nm in split_names:
            loopir  = Schedules.DoSplit(loopir, nm, quot=split_const,
                              hi=out_vars[0], lo=out_vars[1]).result()
        return Procedure(loopir)

    def reorder(self, out_var, in_var):
        if type(out_var) is not str:
            raise TypeError("expected first arg to be a string")
        elif type(in_var) is not str:
            raise TypeError("expected second arg to be a string")

        reorder_pairs = name_str_2_pairs(self._loopir_proc, out_var, in_var)
        if len(reorder_pairs) == 0:
            raise TypeError(f"failed to find nested symbol pairs described "+
                            f"by '{out_var}' outside of '{in_var}'")
        loopir      = self._loopir_proc
        for out_v, in_v in reorder_pairs:
            loopir  = Schedules.DoReorder(loopir, out_v, in_v).result()
        return Procedure(loopir)

    def unroll(self, unroll_var):
        if type(unroll_var) is not str:
            raise TypeError("expected first arg to be a string")

        unroll_names = name_str_2_symbols(self._loopir_proc, unroll_var)
        if len(unroll_names) == 0:
            raise TypeError(f"failed to find any symbols described by "+
                            f"'{unroll_var}'")
        loopir      = self._loopir_proc
        for nm in unroll_names:
            loopir  = Schedules.DoUnroll(loopir, nm).result()
        return Procedure(loopir)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Compilation of sets of Procedures

def compile(procs, dir, c_file, h_file):
    for p in procs:
        if type(p) is not Procedure:
            raise TypeError("compile() expects a list of Procedures "+
                            "as its first argument")
    run_compile([ p._loopir_proc for p in procs ], dir, c_file, h_file)
