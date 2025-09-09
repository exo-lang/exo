from typing import Callable, Dict, Optional, Type, List, Set

from ..core.memory import MemWin, BarrierType
from ..core.prelude import Sym
from ..core.instr_info import InstrInfo
from ..core.LoopIR import T, LoopIR, LoopIR_Do, BaseCompilerDebugLog

from .async_config import CudaDeviceFunction
from .base_with_context import is_if_holding_with
from .coll_analysis import CollAnalysis
from .camspork import camspork
from .distributed_memory import ThreadIter
from .loop_modes import Seq, CudaTasks, _CodegenPar
from .timelines import Instr_tl, Qual_tl, Sync_tl


class CamsporkDo(LoopIR_Do):
    __slots__ = [
        "_builder",
        "_coll_analysis",
        "_value_syms",
        "_sync_syms",
        "_envtyp",
        "_mem_env",
    ]

    _builder: camspork.ProgramBuilder
    _coll_analysis: CollAnalysis
    _value_syms: Set[
        Sym
    ]  # Syms of data vars simulated in value env; control vars always simulated.
    _sync_syms: Set[Sym]
    _envtyp: Dict[Sym, LoopIR.type]
    _mem_env: Dict[Sym, Type[MemWin]]

    def __init__(
        self,
        builder: camspork.ProgramBuilder,
        coll_analysis: CollAnalysis,
        p: LoopIR.proc,
        value_syms: Set[Sym],
        sync_syms: Set[Sym],
    ):
        self.proc = p
        self._builder = builder
        self._coll_analysis = coll_analysis
        self._value_syms = value_syms
        self._sync_syms = sync_syms
        self._envtyp = {}
        self._mem_env = {}

        b = self._builder

        explicit_syms = set(nm for _set in (value_syms, sync_syms) for nm in _set)
        for nm in explicit_syms:
            b.add_variable(nm)

        for a in self.proc.args:
            nm = a.name
            if nm not in explicit_syms:
                b.add_variable(nm)
            self._envtyp[nm] = a.type
            self._mem_env[nm] = a.mem

        self.do_stmts(self.proc.body)

    def do_s(self, s: LoopIR.stmt):
        b = self._builder
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            super().do_s(s)
            want_value = s.name in self._value_syms
            want_sync = s.name in self._sync_syms
            am_rhs = self.comp_e(s.rhs, want_value)
            if want_sync or want_value:
                am_dst = self.comp_index_expr(s.name, s.idx)
            if want_sync:
                q_bit = 1  # XXX TODO TODO TODO correct qual-tl
                b.SyncEnvAccess(am_dst, q_bit, q_bit, is_mutate=True, is_ooo=False)
            if want_value:
                op = "=" if isinstance(s, LoopIR.Assign) else "+"
                b.MutateValue(am_dst, op, am_rhs)
        elif isinstance(s, LoopIR.SyncStmt):
            # TODO
            pass
        elif is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, CudaDeviceFunction):
                # TODO TODO TODO correct qual-tl
                L1_qual_bits = 1
                L2_qual_bits = 1
                b.Fence(True, L1_qual_bits, L2_qual_bits, L2_qual_bits)
                clusterDim = ctx.clusterDim
                blockDim = ctx.blockDim
                coords = (clusterDim, blockDim) if clusterDim != 1 else (blockDim,)
                with b.ParallelBlock(*coords):
                    self.do_stmts(s.body)
            else:
                self.do_stmts(s.body)
        # Must be after is_if_holding_with
        elif isinstance(s, LoopIR.If):
            am_cond = self.comp_e(s.cond, True)
            with b.If(am_cond):
                self.do_stmts(s.body)
                if s.orelse:
                    b.begin_orelse()
                    self.do_stmts(s.orelse)
        elif isinstance(s, LoopIR.For):
            if s.iter not in self._value_syms:
                b.add_variable(s.iter)
            am_iter = b.get_varname(s.iter)
            am_lo = self.comp_e(s.lo, True)
            am_hi = self.comp_e(s.hi, True)
            loop_mode = s.loop_mode
            if isinstance(loop_mode, Seq):
                am_ctx = b.SeqFor(am_iter, am_lo, am_hi)
            elif isinstance(loop_mode, CudaTasks):
                am_ctx = b.TasksFor(am_iter, am_lo, am_hi)
            elif isinstance(loop_mode, _CodegenPar):
                thread_iter: ThreadIter = self._coll_analysis.thread_iters[s.iter]
                coll_tiling = thread_iter.coll_tiling
                # assert not coll_tiling.split_idx_factors, "TODO DomainSplit"
                dim_idx = coll_tiling.dim_idx
                if dim_idx is None:
                    am_ctx = None
                else:
                    offset = coll_tiling.offset_from_parent  # NOT offset[dim_idx]
                    box = coll_tiling.box[dim_idx]
                    am_ctx = b.ThreadsFor(am_iter, am_lo, am_hi, dim_idx, offset, box)
            else:
                assert not isinstance(loop_mode, CudaThreads), "Need CollAnalysis"
                raise TypeError(f"Unsupported loop mode {loop_mode}")

            if am_ctx is None:
                self.do_stmts(s.body)
            else:
                with am_ctx:
                    self.do_stmts(s.body)
        elif isinstance(s, LoopIR.Alloc):
            self._envtyp[s.name] = s.type
            self._mem_env[s.name] = s.mem
            want_barrier = issubclass(s.mem, BarrierType)
            want_sync = s.name in self._sync_syms
            want_value = s.name in self._value_syms
            if want_barrier and not want_sync and not want_value:
                b.add_variable(s.name)
            if want_barrier or want_sync or want_value:
                am_array = self.comp_index_expr(s.name, s.type.shape())
            if want_barrier:
                b.BarrierEnvAlloc(am_array)
            if want_sync:
                b.SyncEnvAlloc(am_array)
            if want_value:
                b.ValueEnvAlloc(am_array)
        else:
            super().do_s(s)

    # We emit SyncEnvRead for all reads found (filtered by sync_syms)
    # and translate the LoopIR expr to a camspork.BuilderExpr.
    # We completely ignore do_e; we don't want unexpected SyncEnvRead
    # from LoopIR.Read in "static" places like tensor types (do_t).
    def comp_e(
        self, e: LoopIR.expr, want_value: bool
    ) -> Optional[camspork.BuilderExpr]:
        b = self._builder
        if isinstance(e, LoopIR.Read):
            want_sync = e.name in self._sync_syms
            if want_value or want_sync:
                am_src = self.comp_index_expr(e.name, e.idx)
            if want_sync:
                q_bit = 1  # XXX TODO TODO TODO correct qual-tl
                b.SyncEnvAccess(am_src, q_bit, q_bit, is_mutate=False, is_ooo=False)
            if want_value:
                return am_src
        elif isinstance(e, LoopIR.Const):
            return camspork.BuilderConst(e.val) if want_value else None
        elif isinstance(e, LoopIR.USub):
            return -self.comp_e(e, True) if want_value else None
        elif isinstance(e, LoopIR.BinOp):
            # NB we create am_lhs, am_rhs unconditionally to emit the
            # SyncEnvRead effects even when !want_value.
            am_lhs = self.comp_e(e.lhs, want_value)
            am_rhs = self.comp_e(e.rhs, want_value)
            if want_value:
                return camspork.BuilderBinOp(e.op, am_lhs, am_rhs)
        elif isinstance(e, LoopIR.Extern):
            if want_value:
                raise ValueError(f"Cannot sync-check {e}, unsupported Extern")
            for e in e.args:
                self.comp_e(e, want_value)
        else:
            raise TypeError(f"Unexpected case {e}")

    def comp_index_expr(
        self, name: Sym, idx: List[LoopIR.expr]
    ) -> camspork.BuilderExpr:
        am_name = self._builder.get_varname(name)
        return am_name[tuple(self.comp_e(e, True) for e in idx)]


def coll_analysis_to_camspork(
    coll_analysis: CollAnalysis,
    p: LoopIR.proc,
    value_syms: Set[Sym],
    sync_syms: Set[Sym],
):
    builder = camspork.ProgramBuilder()
    CamsporkDo(builder, coll_analysis, p, value_syms, sync_syms)
    builder.finish()
    return builder
