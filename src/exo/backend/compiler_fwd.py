from dataclasses import dataclass
from typing import List, Dict, Type, Optional

from ..core.memory import MemWin
from ..core.prelude import Sym
from ..core.LoopIR import LoopIR, T, BaseCompilerDebugLog

from ..spork.barrier_usage import BarrierUsage, BarrierUsageAnalysis, SyncInfo
from ..spork.coll_analysis import CollAnalysis


@dataclass(slots=True)
class BackendChecks:
    debug_log: BaseCompilerDebugLog
    original: LoopIR.proc
    after_mem_analysis: LoopIR.proc
    analyzed: LoopIR.proc
    proc_uses_cuda: bool
    mem_env: Dict[Sym, Type[MemWin]]
    barrier_uses: Optional[Dict[Sym, BarrierUsage]]
    coll_analysis: Optional[CollAnalysis]
    lazy_camspork_program: object = None  # for sync_check's usage
    lazy_sync_syms: object = None  # for sync_check's usage


def dataptr_name(wname):
    """C variable name used to store the separate dataptr of a window

    We prepend the (reserved) exo_data_ prefix, but this is
    complicated by the fact that sometimes C variables are stored in
    structs (e.g. exo_deviceArgs) and we need to avoid modifying the
    struct name.

    """
    fragments = wname.split(".")
    fragments[-1] = "exo_data_" + fragments[-1]
    return ".".join(fragments)


@dataclass(slots=True)
class SporkLoweringCtx(object):
    """Communication object between main LoopIR compiler and Spork backend.

    The task of the spork backend is to transform a subtree of LoopIR
    to a new subtree of LoopIR that the main compiler is able to
    understand.  Usually, the backend will return a tree rooted with
    an ExtWithContext to redirect the generated subtree C-like code to
    separate files for accelerator code (e.g. .cuh or .cu code for
    cuda).

    """

    _lib_name: str
    _proc_name: str
    _kernel_index: int
    _compiler: "Compiler"
    debug_log: BaseCompilerDebugLog

    def lib_name(self):
        return self._lib_name

    def proc_name(self):
        return self._proc_name

    def kernel_index(self):
        return self._kernel_index

    def sym_c_name(self, sym: Sym):
        assert isinstance(sym, Sym)
        return self._compiler.env[sym]

    def sym_type(self, sym: Sym, overrides: Dict[Sym, LoopIR.type]):
        assert isinstance(sym, Sym)
        return overrides.get(sym) or self._compiler.envtyp[sym]

    def sym_mem(self, sym: Sym):
        assert isinstance(sym, Sym)
        return self._compiler.mems[sym]

    def sym_is_scalar_ref(self, sym: Sym):
        assert isinstance(sym, Sym)
        return sym in self._compiler._scalar_refs

    def is_const(self, sym: Sym):
        assert isinstance(sym, Sym)
        return self._compiler.is_const(sym)

    def append_fnarg_decl(
        self,
        a: LoopIR.fnarg,
        name_arg: str,
        arg_strs: List[str],
        typ_comments: List[str],
        *,
        force_pass_by_value=False,
    ):
        return self._compiler.append_fnarg_decl(
            a, name_arg, arg_strs, typ_comments, force_pass_by_value=force_pass_by_value
        )

    def fnarg_values(self, e, is_const, force_pass_by_value):
        mem = self._compiler.mems[e.name]
        return self._compiler.comp_fnarg_impl(
            e, mem, is_const, force_pass_by_value
        ).to_arg_strs()

    def get_barrier_usage(self, name: Sym) -> BarrierUsage:
        return self._compiler.barrier_uses[name]

    def coll_analysis(self) -> CollAnalysis:
        analysis = self._compiler._coll_analysis
        assert isinstance(analysis, CollAnalysis)
        return analysis
