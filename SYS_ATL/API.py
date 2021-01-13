from .prelude import *
from .LoopIR import UAST, LoopIR
from . import shared_types as T
from .typecheck import TypeChecker
from .LoopIR_compiler import Compiler, run_compile, compile_to_strings
from .LoopIR_interpreter import Interpreter, run_interpreter
from .LoopIR_scheduling import Schedules, name_str_2_symbols, name_str_2_pairs

from .pattern_match import match_pattern

from weakref import WeakKeyDictionary

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   proc provenance tracking

# every LoopIR.proc is either a root (not in this dictionary)
# or there is some other LoopIR.proc which is its root
_proc_root = WeakKeyDictionary()

def _proc_prov_eq(lhs, rhs):
    """ test whether two procs have the same provenance """
    lhs = lhs if lhs not in _proc_root else _proc_root[lhs]
    rhs = rhs if rhs not in _proc_root else _proc_root[rhs]
    assert lhs not in _proc_root and rhs not in _proc_root
    return lhs is rhs

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   iPython Display Object

class MarkDownBlob:
    def __init__(self, mkdwn_str):
        self.mstr = mkdwn_str

    def _repr_markdown_(self):
        return self.mstr

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Procedure Objects
class Procedure:
    def __init__(self, proc, _testing=None, _provenance_eq_Procedure=None):
        if isinstance(proc, LoopIR.proc):
            self._loopir_proc = proc
        else:
            assert isinstance(proc, UAST.proc)

            self._uast_proc = proc
            if _testing != "UAST":
                self._loopir_proc = TypeChecker(proc).get_loopir()

        # find the root provenance
        parent = _provenance_eq_Procedure
        if parent is None:
            pass # this is a new root; done
        else:
            parent = parent._loopir_proc
            # if the provenance Procedure is not a root, find its root
            if parent in _proc_root:
                parent = _proc_root[parent]
            assert parent not in _proc_root
            # and then set this new proc's root
            _proc_root[self._loopir_proc] = parent


    def __str__(self):
        if hasattr(self,'_loopir_proc'):
            return str(self._loopir_proc)
        else:
            return str(self._uast_proc)

    def _repr_markdown_(self):
        return ("```python\n"+self.__str__()+"\n```")

    def name(self):
        return self._loopir_proc.name

    def rename(self, name):
        if not is_valid_name(name):
            raise TypeError(f"'{name}' is not a valid name")
        p = self._loopir_proc
        p = LoopIR.proc( name, p.args, p.body, p.srcinfo )
        return Procedure(p, _provenance_eq_Procedure=self)

    def INTERNAL_proc(self):
        return self._loopir_proc

    def c_code_mkdwn(self):
        return MarkDownBlob("```c\n"+self.c_code_str()+"\n```")

    def c_code_str(self):
        decls, defns = compile_to_strings([self._loopir_proc])
        return defns

    def compile_c(self, directory, filename, malloc=False):
        run_compile([self._loopir_proc], directory,
                    (filename + ".c"), (filename + ".h"), malloc)

    def interpret(self, **kwargs):
        run_interpreter(self._loopir_proc, kwargs)

    # ------------------------------- #
    #     scheduling operations
    # ------------------------------- #

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
        return Procedure(loopir, _provenance_eq_Procedure=self)

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
        return Procedure(loopir, _provenance_eq_Procedure=self)

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
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def abstract(self, subproc, pattern):
        pass

    def _find_callsite(self, call_site_pattern):
        body        = self._loopir_proc.body
        stmt_lists  = match_pattern(body, call_site_pattern,
                                    call_depth=2, default_match_no=0)
        if len(stmt_lists) == 0 or len(stmt_lists[0]) == 0:
            raise TypeError("failed to find call site")
        else:
            call_stmt = stmt_lists[0][0]
            if type(call_stmt) is not LoopIR.Call:
                raise TypeError("pattern did not describe a call-site")

        return call_stmt

    def inline(self, call_site_pattern):
        call_stmt   = self._find_callsite(call_site_pattern)

        loopir      = self._loopir_proc
        loopir      = Schedules.DoInline(loopir, call_stmt).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def call_eqv(self, other_Procedure, call_site_pattern):
        call_stmt   = self._find_callsite(call_site_pattern)

        old_proc    = call_stmt.f
        new_proc    = other_Procedure._loopir_proc
        if not _proc_prov_eq(old_proc, new_proc):
            raise TypeError("the procedures were not equivalent")

        loopir      = self._loopir_proc
        loopir      = Schedules.DoCallSwap(loopir, call_stmt,
                                                   new_proc).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)
