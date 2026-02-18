import re
from typing import Callable, Dict, Optional, Type, List

from ..core.prelude import Sym, SrcInfo
from ..core.instr_info import InstrInfo
from ..core.LoopIR import T, LoopIR, LoopIR_Rewrite, BaseCompilerDebugLog

from .async_config import CudaDeviceFunction
from .barrier_usage import BarrierUsageAnalysis, BarrierUsage
from .base_with_context import is_if_holding_with
from .distributed_memory import ThreadIter, DistributedIdxFsm, DistributedAllocState
from .coll_algebra import (
    CollParam,
    CollUnit,
    clusterDim_param,
    blockDim_param,
    CollIndexExpr,
    CollTiling,
    CollTilingError,
    cuda_thread,
    cuda_warp,
    cuda_cta_in_cluster,
    cuda_agnostic_sub_cta,
    cuda_agnostic_intact_cta,
)
from .cuda_memory import CudaBasicDeviceVisible
from .cuda_warp_config import WarpLayoutInfo
from .loop_modes import CudaThreads, _CodegenPar
from .timelines import Qual_tl, Sync_tl
from .with_cuda_warps import CudaWarps


# No BarrierExpr here; handled specially as part of SyncStmt.
coll_idx_e_types = (LoopIR.Read, LoopIR.WindowExpr)
coll_idx_s_types = (LoopIR.Assign, LoopIR.Reduce)


def wrap_codegen_par(codegen_par, body, srcinfo, iter_sym=None):
    assert isinstance(codegen_par, _CodegenPar)
    if iter_sym is None:
        iter_sym = Sym("tmp")
    return LoopIR.For(
        iter_sym,
        LoopIR.Const(0, T.int, srcinfo),
        LoopIR.Const(1, T.int, srcinfo),
        body,
        codegen_par,
        srcinfo,
    )


class CollAnalysis(LoopIR_Rewrite):
    __slots__ = [
        "distributed_alloc_states",
        "thread_iters",
        "_stmt_stack",
        "_coll_env",
        "_coll_tiling",
        "_current_warp_name",
        "_envtyp",
        "_cuda_device_function",
        "_barrier_uses",
        "_debug_log",
        "_proc_name",
        "_qual_tl_thread_alignments",
        "_qual_tl_fallback_thread_alignment",
    ]

    # Public variables
    distributed_alloc_states: Dict[Sym, DistributedAllocState]
    thread_iters: Dict[Sym, ThreadIter]  # Info on iterators of cuda_threads loops

    _stmt_stack: List[LoopIR.stmt]
    _coll_env: Dict[CollParam, int]
    _coll_tiling: Optional[CollTiling]
    _current_warp_name: Optional[str]
    _envtyp: Dict[Sym, LoopIR.type]
    _cuda_device_function: Optional[CudaDeviceFunction]
    _barrier_uses: Dict[Sym, BarrierUsage]
    _debug_log: BaseCompilerDebugLog
    _proc_name: str
    _qual_tl_thread_alignments: Dict[Qual_tl, int]
    _qual_tl_fallback_thread_aligment: int
    # Update __slots__ above if you add more.

    # TODO barrier_usage_analysis only needed to check barrier guarding.
    # Consider making this an optional feature.
    def __init__(
        self,
        barrier_usage_analysis: BarrierUsageAnalysis,
        debug_log: BaseCompilerDebugLog = BaseCompilerDebugLog(),
    ):
        self.distributed_alloc_states = {}
        self.thread_iters = {}
        self._stmt_stack = []
        self._coll_env = None
        self._coll_tiling = None
        self._current_warp_name = None
        self._envtyp = {}
        self._cuda_device_function = None
        self._barrier_uses = barrier_usage_analysis.uses
        self._debug_log = debug_log
        self._qual_tl_thread_alignments = {}
        self._qual_tl_fallback_thread_alignment = 1 << 31

    def run(self, proc):
        self._proc_name = proc.name
        return super().apply_proc(proc)

    def in_cuda(self):
        return self._cuda_device_function is not None

    def map_fnarg(self, a):
        self._envtyp[a.name] = a.type
        return None

    def get_qual_tl_thread_alignment(self, qual_tl: Qual_tl) -> int:
        """For out-of-order non-convergent abstract machine optimization"""
        try:
            return self._qual_tl_thread_alignments[qual_tl]
        except KeyError:
            assert isinstance(qual_tl, Qual_tl)
            return self._qual_tl_fallback_thread_alignment

    def map_s(self, s):
        # Save state
        old_cuda_device_function = self._cuda_device_function
        old_coll_tiling = self._coll_tiling
        old_warp_name = self._current_warp_name
        self._stmt_stack.append(s)

        try:
            stmts = self.map_s_impl(s)
        except AssertionError:
            raise
        except Exception as exc:
            # Re-raise all errors, but if the error doesn't seem to contain srcinfo
            # then we wrap the error message with a srcinfo.
            exc_str = str(exc)
            if not re.findall(SrcInfo.stmt_id_pattern, exc_str):
                raise ValueError(f"{s.srcinfo}: {exc_str}") from exc
            raise

        # Restore state
        self._stmt_stack.pop()
        self._current_warp_name = old_warp_name
        self._coll_tiling = old_coll_tiling
        self._cuda_device_function = old_cuda_device_function
        return stmts

    def map_s_impl(self, s):
        if isinstance(s, LoopIR.Alloc):
            self._envtyp[s.name] = s.type
        elif isinstance(s, LoopIR.WindowStmt):
            self._envtyp[s.name] = s.rhs.type

        if self.in_cuda():
            thread_iter = self.cuda_inspect_s(s)

        if isinstance(s, LoopIR.For) and isinstance(s.loop_mode, CudaThreads):
            if self.in_cuda():
                stmts = self.map_cuda_threads_loop(s, thread_iter)
            else:
                raise TypeError(
                    f"{s.srcinfo}: cannot have cuda_threads loop outside CUDA device function"
                )
        elif is_if_holding_with(s, LoopIR):
            if isinstance(cuda_warps := s.cond.val, CudaWarps):
                if self.in_cuda():
                    stmts = self.map_with_cuda_warps(s, thread_iter)
                else:
                    raise TypeError(
                        f"{s.srcinfo}: cannot have with CudaWarps outside CUDA device function"
                    )
            elif isinstance(cuda_device_function := s.cond.val, CudaDeviceFunction):
                assert not self.in_cuda()
                self.apply_cuda_device_function(cuda_device_function)
                self.remark_coll_tiling_in_body(s, self._coll_tiling)
                self._qual_tl_fallback_thread_alignment = min(
                    self._qual_tl_fallback_thread_alignment,
                    self._coll_tiling.get_pow2_thread_alignment(),
                )
                stmts = super().map_s(s)
            else:
                stmts = super().map_s(s)
        elif isinstance(s, LoopIR.Call) and self.in_cuda():
            stmts = self.cuda_map_call_stmt(s)
            # cuda_map_call_stmt cannot use super().map_s(s) due to window handling.
        else:
            stmts = super().map_s(s)
        return stmts

    def cuda_inspect_s(self, s) -> Optional[ThreadIter]:
        thread_iter = None
        if isinstance(s, coll_idx_s_types):
            self.cuda_inspect_idx(s, s, ())
        elif not isinstance(s, (LoopIR.WindowStmt, LoopIR.Alloc, LoopIR.Free)):
            assert not hasattr(s, "name"), "Add handling for array indexing"

        if is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, CudaWarps):
                thread_iter = self.apply_with_cuda_warps(s)
        elif isinstance(s, LoopIR.For):
            if isinstance(s.loop_mode, CudaThreads):
                thread_iter = self.apply_cuda_threads_loop(s)
        elif isinstance(s, LoopIR.WindowStmt):
            # Unlike for Calls, the WindowExpr here do not allow intervals for
            # any distributed dimensions ... this would be very hard to support.
            # Basically the dimensionality of the WindowStmt will never change!
            # See WindowExpr case for remove_distributed_idx.
            pass
        elif isinstance(s, LoopIR.Alloc):
            if s.type.is_barrier():
                native_unit = None
                separate_await = s.mem.traits().different_arrive_await_threads
            else:
                assert issubclass(s.mem, CudaBasicDeviceVisible)
                native_unit = s.mem.native_unit()
                separate_await = False
            self.distributed_alloc_states[s.name] = DistributedAllocState(
                s,
                self._coll_tiling,
                native_unit,
                self._coll_env,
                separate_await,
            )
        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if (n_threads := self._coll_tiling.get_box_num_threads()) != 1:
                raise ValueError(
                    f"{s.srcinfo}: write must be executed by one "
                    f"thread only (current: {n_threads} threads)\n"
                    f"stmt: {s}"
                )
        elif isinstance(s, LoopIR.SyncStmt):
            # Update per-Qual_tl thread alignment for
            # out-of-order non-convergent abstract machine optimization
            if (L1 := s.sync_type.first_sync_tl) is not None:
                align = self._coll_tiling.get_pow2_thread_alignment()
                for qual_tl in L1.get_full_timeline_set():
                    _dict = self._qual_tl_thread_alignments
                    _dict[qual_tl] = min(align, _dict.get(qual_tl, 1 << 31))

            # Distributed memory analysis and CollTiling for Fence/Arrive/Await
            if s.sync_type.is_split():
                assert len(s.barriers) >= 1
                name = s.barriers[0].name
                usage: BarrierUsage = self._barrier_uses[name]
                state = self.distributed_alloc_states.get(name)
                assert isinstance(state, DistributedAllocState)

                fsm = DistributedIdxFsm(
                    s.home_barrier_expr(),
                    s,
                    state,
                    "cuda_threads",
                    self.thread_iters,  # May be modified
                    self._coll_env,
                    self._coll_tiling,
                    (),
                )
                # There is no native_unit; we parse all indices as distributed
                assert state.optional_native_unit is None
                e0 = s.barriers[0]
                for i in range(len(e0.idx)):
                    fsm.consume_SyncStmt_idx(
                        state, self._stmt_stack, s, self._envtyp[e0.name], i
                    )

                # We now have the distributed indices in distributed_iters.
                # Store in DistributedAllocState if this is the first use, or check
                # consistency (index equality) with prior uses.
                if remark_state := fsm.check_store_state(state):
                    self.remark_distributed_alloc_state(remark_state, s)
                fsm.inspect_arrive_await(
                    s,
                    self._coll_tiling,
                    self.thread_iters,
                    lambda nm: self._barrier_uses[nm],
                    lambda nm: self.distributed_alloc_states.get(nm),
                )
            else:
                assert len(s.barriers) == 1
                e = s.barriers[0]
                assert isinstance(e, LoopIR.BarrierExpr)
                assert e.name not in self.distributed_alloc_states
                state = DistributedAllocState.from_fence(s, self._coll_tiling)
                self.distributed_alloc_states[e.name] = state
        return thread_iter

    def map_e(self, e, distributed_coll_units=()):
        if self.in_cuda():
            self.cuda_inspect_e(e, distributed_coll_units)
        return super().map_e(e)

    def cuda_inspect_e(self, e, distributed_coll_units):
        if isinstance(e, coll_idx_e_types):
            # BarrierExpr not handled here; part of SyncStmt handling.
            self.cuda_inspect_idx(e, self._stmt_stack[-1], distributed_coll_units)
        elif not isinstance(e, (LoopIR.BarrierExpr, LoopIR.StrideExpr)):
            assert not hasattr(e, "name"), "Add handling for array indexing"

    def map_cuda_threads_loop(self, s: LoopIR.For, thread_iter: ThreadIter):
        stmts = super().map_s(s)
        if stmts is not None:
            assert len(stmts) == 1
            s = stmts[0]
        return [s.update(loop_mode=thread_iter.codegen_par)]

    def map_with_cuda_warps(self, s: LoopIR.stmt, thread_iter: ThreadIter):
        assert not s.orelse
        stmts = self.map_stmts(s.body)
        if stmts is None:
            stmts = s.body
        s2 = wrap_codegen_par(
            thread_iter.codegen_par, stmts, s.srcinfo, iter_sym=thread_iter.iter
        )
        self.thread_iters[s2.iter] = thread_iter
        return [s2]

    def apply_cuda_device_function(self, cuda_device_function: CudaDeviceFunction):
        self._coll_tiling = cuda_device_function.top_level_coll_tiling()
        self._coll_env = cuda_device_function.coll_env()
        named_warps = cuda_device_function.named_warps
        if len(named_warps) == 1:
            # Special case required for apply_with_cuda_warps.
            self._current_warp_name = tuple(named_warps.keys())[0]
        else:
            self._current_warp_name = None
        self._cuda_device_function = cuda_device_function

    def cuda_inspect_idx(self, node, context_stmt, distributed_coll_units):
        """Consistent distributed memory analysis"""
        assert isinstance(context_stmt, LoopIR.stmt)
        state: DistributedAllocState
        state = self.distributed_alloc_states.get(node.name)
        if state is None:
            return  # Allocated outside CUDA, or not numeric

        assert state.optional_native_unit is not None

        fsm = DistributedIdxFsm(
            node,
            context_stmt,
            state,
            "cuda_threads",
            self.thread_iters,  # May be modified
            self._coll_env,
            self._coll_tiling,
            distributed_coll_units,
        )
        for i in range(len(node.idx)):
            if fsm.is_done():
                break
            fsm.consume_idx(state, i)

        # We now have the distributed indices in distributed_iters.
        # Store in DistributedAllocState if this is the first use, or check
        # consistency (CollTiling equivalence) with prior uses.
        if remark_state := fsm.check_store_state(state):
            self.remark_distributed_alloc_state(remark_state, context_stmt)

    def cuda_map_call_stmt(self, s: LoopIR.Call):
        # Check collective unit.
        callee = s.f
        instr_info: InstrInfo = callee.instr
        assert isinstance(instr_info, InstrInfo), "Unimplemented: CUDA function calls"
        needed = callee.proc_coll_unit()
        if msg := self._coll_tiling.unit_mismatch(needed, self._coll_env):
            raise TypeError(
                f"{s.srcinfo}: wrong collective unit (need {needed}) for {callee.name}(): {msg}"
            )

        # Inspect distributed indices of arguments (safer after above check)
        assert len(callee.args) == len(s.args)
        for decl, e in zip(callee.args, s.args):
            arg_name_str = str(decl.name)
            coll_units = ()
            if e.type.is_tensor_or_window():
                access_info: AccessInfo = instr_info.access_info[arg_name_str]
                coll_units = access_info.distributed_coll_units
            self.cuda_inspect_e(e, coll_units)

        # Inspect trailing barrier expression
        if bar_e := s.trailing_barrier_expr:
            name = bar_e.name
            state = self.distributed_alloc_states.get(name)
            barrier_loopir_type = self._envtyp[name]
            assert barrier_loopir_type.is_barrier()
            assert isinstance(state, DistributedAllocState)

            # Inspect intervals (as opposed to points) in BarrierExpr
            coll_units = instr_info.barrier_coll_units
            interval_count = 0
            for coord in bar_e.idx:
                interval_count += isinstance(coord, LoopIR.Interval)
            if interval_count != len(coll_units):
                raise ValueError(
                    f"{s.srcinfo}: {callee.name} #intervals in barrier {bar_e} wrong; "
                    f"have {interval_count}, need {len(coll_units)}, "
                    f"for barrier_coll_units={coll_units}"
                )

            # Distributed memory deduction
            fsm = DistributedIdxFsm(
                bar_e,
                s,
                state,
                "cuda_threads",
                self.thread_iters,  # May be modified
                self._coll_env,
                self._coll_tiling,
                coll_units,
            )
            # There is no native_unit; we parse all indices as distributed
            assert state.optional_native_unit is None
            for i in range(len(bar_e.idx)):
                fsm.consume_idx(state, i)

            # We now have the distributed indices in distributed_iters.
            # Store in DistributedAllocState if this is the first use, or check
            # consistency (CollTiling equivalence) with prior uses.
            if remark_state := fsm.check_store_state(state):
                self.remark_distributed_alloc_state(remark_state, s)

        # Cannot use super().map_s(s) due to window handling
        return None

    def apply_cuda_threads_loop(self, s: LoopIR.For) -> ThreadIter:
        def get_const(e, name):
            expected = "literal int value"
            if isinstance(e, LoopIR.Const):
                if e.type.is_indexable():
                    v = int(e.val)
                    if v != 0 and name == "lo":
                        expected = "0"
                    else:
                        return v
            raise ValueError(
                f"{e.srcinfo}: expected {expected} for {name} of {s.iter} loop (rewrite with simplify(...) if needed)"
            )

        lo_int = get_const(s.lo, "lo")
        hi_int = get_const(s.hi, "hi")
        assert lo_int == 0

        # Update stored CollTiling
        try:
            new_tiling = self._coll_tiling.tiled(
                s.iter, s.loop_mode.unit, hi_int, self._coll_env
            )
        except AssertionError:
            raise
        except Exception as e:
            loop_str = f"for {s.iter} in {s.loop_mode.format_loop_cond(s.lo, s.hi)}"
            raise ValueError(f"{s.srcinfo}: Failed to compile {loop_str}: {e}") from e
        self._coll_tiling = new_tiling

        # We will advise replacing the loop mode with _CodegenPar
        assert s.iter not in self.thread_iters, f"{s.srcinfo}"
        thread_iter = ThreadIter(
            self._coll_tiling,
            s.loop_mode.format_loop_cond(lo_int, hi_int),
            self._current_warp_name,
            mangle=True,
        )

        log_lhs = thread_iter.cname(s.iter)
        log_rhs = thread_iter.codegen_par.c_index
        self._debug_log.remark(
            self._proc_name,
            f"thread_pitch={thread_iter.thread_pitch}; {log_lhs} = {log_rhs} @ {s.srcinfo}",
        )
        self.remark_coll_tiling_in_body(s, new_tiling)

        self.thread_iters[s.iter] = thread_iter
        return thread_iter

    def apply_with_cuda_warps(self, s) -> ThreadIter:
        assert self.in_cuda()
        ctx: CudaWarps = s.cond.val
        assert isinstance(ctx, CudaWarps)
        coll_tiling = self._coll_tiling
        is_top_level = self._current_warp_name is None
        named_warps = self._cuda_device_function.named_warps

        top_am_dim_idx = None
        top_am_offset = 0
        top_am_box = None

        # Top-level CudaWarps: adjust CollTiling to account for offset of named warps.
        # We ignore the codegen here ... because of how the deviceTask is specialized
        # per named-warp set, we already can assume the physical code is executed
        # only by the subset of warps that are part of the named warp set.
        #
        # NB it's important that this is skipped when the user doesn't
        # use named warps (fallback len-1 case) because the (***)
        # restriction must not be enforced.
        if is_top_level:
            assert len(named_warps) > 1
            name = "" if ctx.name is None else ctx.name
            if (info := named_warps.get(name)) is None:
                known_names = sorted(named_warps)
                raise ValueError(
                    f"{s.srcinfo}: top-level CudaWarps must provide valid warp name, not {ctx.name!r}; your CudaDeviceFunction defines: {known_names}"
                )

            # (***) Named warps won't work if the CTA has already been
            # subdivided by a cuda_threads loop.
            if detail := self._coll_tiling.unit_mismatch(
                cuda_agnostic_intact_cta, self._coll_env
            ):
                raise ValueError(
                    f"{s.srcinfo}: named {ctx} requires CTA not to be subdivided by parent cuda_threads loop (detail: {detail})"
                )

            # Extract lo/hi offsets (with defaulted values allowed).
            # This gets handled towards the end of the function.
            warps_lo = 0 if ctx.lo is None else ctx.lo
            warps_hi = info.count if ctx.hi is None else ctx.hi
            if warps_hi > info.count:
                raise ValueError(
                    f"{s.srcinfo}: CudaWarps.hi={warps_hi} out-of-range for {name!r}-named warps (only have {info.count})"
                )

            # (1/2) adjust CollTiling for named warps offset.
            # Codegen for CUDA C++ is discarded, since we handle testing the membership
            # of threads in named warps in the deviceMainLoop. However, we will have
            # to handle changing values for the abstract machine (am_*).
            coll_tiling = coll_tiling.specialized(
                Sym(f"CudaWarps_{name}"),
                cuda_warp,
                info.offset,
                (info.offset + info.count),
                self._coll_env,
            )
            codegen = coll_tiling.get_codegen()
            top_am_dim_idx = codegen.dim_idx
            top_am_offset = codegen.offset
            top_am_box = codegen.box

        # Nested CudaWarps: interpret lo/hi literally as the higher-level
        # CudaWarps will have already handled the named warp offset adjustment.
        # Can't request different named warps now.
        else:
            name = self._current_warp_name if ctx.name is None else ctx.name
            if name != self._current_warp_name:
                raise ValueError(
                    f"{s.srcinfo}: nested CudaWarps cannot change warp name from {self._current_warp_name!r} to {name!r}"
                )
            warps_lo = ctx.lo
            warps_hi = ctx.hi
            if warps_lo is None or warps_hi is None:
                raise ValueError(
                    f"{s.srcinfo}: nested CudaWarps must define lo and hi explicitly"
                )

        self._current_warp_name = name

        # (2/2) Adjust CollTiling for lo/hi offset.
        try:
            _iter = Sym(f"CudaWarps_{ctx.lo}_{ctx.hi}_{name or ''}")
            coll_tiling = coll_tiling.specialized(
                _iter, cuda_warp, warps_lo, warps_hi, self._coll_env
            )
        except AssertionError:
            raise
        except Exception as e:
            raise ValueError(f"{s.srcinfo}: failed to compile {ctx}: {e}") from e

        self._coll_tiling = coll_tiling
        self.remark_coll_tiling_in_body(s, coll_tiling)
        return ThreadIter(
            coll_tiling,
            str(ctx),
            name,
            mangle=False,
            prior_am_dim_idx=top_am_dim_idx,
            prior_am_offset=top_am_offset,
            prior_am_box=top_am_box,
        )

    def remark_distributed_alloc_state(
        self, state: DistributedAllocState, context_stmt: LoopIR.stmt
    ):
        thread_iters = self.thread_iters
        distributed_iters = state.first_distributed_iters
        tup = tuple(thread_iters[it].thread_pitch for it in distributed_iters)
        s = state.alloc_stmt
        _from = ""
        if isinstance(context_stmt, LoopIR.SyncStmt):
            _from = (
                "from Arrive " if context_stmt.sync_type.is_arrive() else "from Await "
            )
        self._debug_log.remark(
            self._proc_name,
            f"distributed dims: {len(tup)}, thread pitch tuple {tup} {_from}@ {s.srcinfo}",
        )

    def remark_coll_tiling_in_body(self, s: LoopIR.stmt, coll_tiling: CollTiling):
        assert hasattr(s, "body")
        assert isinstance(coll_tiling, CollTiling)
        if s.body:

            def fmt_tup(t):
                return "[" + ",".join("%4i" % n for n in t) + "]"

            D = fmt_tup(coll_tiling.get_domain())
            B = fmt_tup(coll_tiling.get_box())
            a = coll_tiling.get_pow2_thread_alignment()
            remark = (
                f"Domain (ω.D) = {D}\n"
                f"   Box (ω.B) = {B}\n"
                f"  pow2 align = {a} @ {s.body[0].srcinfo}"
            )
            self._debug_log.remark(self._proc_name, remark)
