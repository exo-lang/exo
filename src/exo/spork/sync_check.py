# Don't import this module until the camspork JIT is initialized.
# exocc and the Exo pytest tests should handle this early during init.

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Type, List, Set, Tuple
from warnings import warn

from ..backend.LoopIR_compiler import run_backend_checks, BackendChecks

from ..core.memory import MemWin, BarrierMechanism, SpecialWindow
from ..core.prelude import Sym
from ..core.instr_info import AccessInfo, InstrInfo
from ..core.LoopIR import (
    T,
    LoopIR,
    LoopIR_Do,
    BaseCompilerDebugLog,
    chain_window_idx,
    SubstArgs,
    get_writes_of_stmts,
)

from .async_config import BaseAsyncConfig, CudaDeviceFunction
from .base_with_context import is_if_holding_with
from .coll_algebra import CollTiling, CollDim, CollDimOp, CollDimExpectation, CollParam
from .coll_analysis import CollAnalysis
from .distributed_memory import ThreadIter
from .loop_modes import Seq, CudaTasks, cuda_tasks, _CodegenPar, CudaThreads
from .sync_types import SyncType
from .timelines import DeviceScope, Instr_tl, Qual_tl, Sync_tl
from . import timelines

from .camspork import camspork

"""
*******************************************************************************
Problems as of 2025-12-19
*******************************************************************************
SyncEnvFreeShard is not actually used to check-free shards, only the entire tensor.
We keep this around because a more fine-grained approach will almost certainly
be needed at some future time.

access_by_owner_only = True is basically abandonware, and not documented much.

The paper describes the abstract machine as an alternative to Exo value
semantics, where we have a sync env instead of a value environment.
In reality, I view this as an augmentation of value semantics.
libcamspork supports a value environment, which by default is never used
other than for control variables.
If we were to support Chexo (no more strict data vs control variable split),
then enabling the value environment for data will be useful, but the
code paths for this (expanding the _value_syms set) is incomplete and not tested.
In particular, value semantics for functions (param substitution) aren't
implemented, and there may be tricky cases regarding reads to non-sync-exempt
memory being required to calculate index values.

"""


class CamsporkDo(LoopIR_Do):
    """coll_analysis_to_camspork implementation."""

    __slots__ = [
        "_builder",
        "_coll_analysis",
        "_value_syms",
        "_sync_syms",
        "_envtyp",
        "_mem_env",
        "_device",
        "_default_instr_tl",
        "_tmp_call_args",
        "_domain",
        "_saw_alloc",
        "_saw_free",
        "_coll_tiling",
        "_coll_env",
    ]

    _builder: camspork.ProgramBuilder
    _coll_analysis: CollAnalysis
    # Syms of data vars simulated in value env; control vars always simulated.
    _value_syms: Set[Sym]
    _sync_syms: Set[Sym]
    _envtyp: Dict[Sym, LoopIR.type]
    _mem_env: Dict[Sym, Type[MemWin]]
    _device: DeviceScope
    _default_instr_tl: Instr_tl
    _tmp_call_args: Dict[Sym, LoopIR.expr]
    _domain: Tuple[int]
    _saw_alloc: bool
    _saw_free: bool
    _coll_tiling: Optional[CollTiling]
    _coll_env: Dict[CollParam, int]

    def __init__(
        self,
        builder: camspork.ProgramBuilder,
        mem_env: Dict[Sym, Type[LoopIR.type]],
        coll_analysis: CollAnalysis,
        p: LoopIR.proc,
        value_syms: Set[Sym],
        sync_syms: Set[Sym],
    ):
        instr_tl = timelines.cpu_basic_device.get_default_instr_tl()

        self.proc = p
        self._builder = builder
        self._coll_analysis = coll_analysis
        self._value_syms = value_syms
        self._sync_syms = sync_syms
        self._envtyp = {}
        self._mem_env = mem_env
        self._device = timelines.cpu_basic_device
        self._default_instr_tl = instr_tl
        self._tmp_call_args = {}
        self._domain = ()
        self._saw_alloc = False
        self._saw_free = False
        self._coll_tiling = None
        self._coll_env = None

        b = self._builder

        explicit_syms = set(nm for _set in (value_syms, sync_syms) for nm in _set)
        for nm in sorted(explicit_syms):
            b.add_variable(nm)

        for a in self.proc.args:
            nm = a.name
            if nm not in explicit_syms:
                b.add_variable(nm)
            self._envtyp[nm] = a.type
            if a.type.is_numeric():
                assert self._mem_env[nm] == a.mem
            if nm in sync_syms:
                am_array = self.comp_index_expr(nm, a.type.shape(), instr_tl)
                b.ExpectSyncEnvAlloc(am_array, srcinfo=a.srcinfo)

        self.do_stmts(self.proc.body)
        assert self._saw_free or not self._saw_alloc, "Need MemAnalysis before"

    def comp_qual_tl(self, node: LoopIR.expr | LoopIR.stmt, instr_tl: Instr_tl):
        """Get initial Qual_tl, initial Qual_tl as bit, ext Qual_tl as bits.

        Deduces the variable being accessed from node.name.
        Computes the Qual_tl info as a function of the variable's memory
        and the instr_tl of the instruction used to access the variable.

        """
        nm = node.name
        mem = self._mem_env[nm]
        assert not issubclass(mem, SpecialWindow)
        try:
            q = mem.qual_tl_dict[instr_tl]
        except KeyError:
            raise ValueError(
                f"{node.srcinfo}: implementation limitation: no qual-tl for "
                f"({nm} @ {mem.name()}) given instr-tl {instr_tl}"
            )
        if isinstance(q, Qual_tl):
            initial_qual_tl = q
        else:
            initial_qual_tl = q[0]
        return initial_qual_tl, Qual_tl.make_bits(initial_qual_tl), Qual_tl.make_bits(q)

    def do_s(self, s: LoopIR.stmt):
        b = self._builder
        instr_tl = self._default_instr_tl

        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            super().do_s(s)
            want_value = s.name in self._value_syms
            want_sync = s.name in self._sync_syms
            am_rhs = self.comp_e(s.rhs, want_value, instr_tl)
            if want_sync or want_value:
                am_dst = self.comp_index_expr(s.name, s.idx, instr_tl)
            if want_sync:
                _, initial_q, ext_q = self.comp_qual_tl(s, instr_tl)
                flags = b.mutate_flag | b.convergent_flag
                if isinstance(s, LoopIR.Reduce):
                    flags |= b.write_only_flag
                b.SyncEnvAccess(
                    am_dst,
                    initial_q,
                    ext_q,
                    flags=flags,
                    srcinfo=s.srcinfo,
                )
            if want_value:
                op = "=" if isinstance(s, LoopIR.Assign) else "+"
                b.MutateValue(am_dst, op, am_rhs)
        elif isinstance(s, LoopIR.SyncStmt):
            sync_type: SyncType = s.sync_type
            L1 = sync_type.first_sync_tl
            L2 = sync_type.second_sync_tl
            if L1 is not None:
                L1_bits = L1.get_full_timeline_set_bits()
            if L2 is not None:
                L2_full_bits = L2.get_full_timeline_set_bits()
                L2_temporal_bits = L2.get_temporal_timeline_set_bits()
            if sync_type.is_arrive():
                home = s.home_barrier_expr()
                multicasts = s.multicasts()
                am_home_barrier = self.comp_index_expr(home.name, home.idx, instr_tl)
                if home.name in self._sync_syms:
                    # Model as "read" since concurrent access is allowed
                    mem = self._mem_env[home.name]
                    q_bit = mem.arrive_qual_tl(L1).as_bit()
                    b.SyncEnvAccess(
                        am_home_barrier,
                        q_bit,
                        q_bit,
                        flags=b.convergent_flag,
                        access_multicasts=multicasts,
                        srcinfo=s.srcinfo,
                        # TODO wouldn't it make more sense to use barrier
                        # and barrier_multicasts so the paired await is
                        # what makes the prior arrive "visible"?
                    )
                b.Arrive(L1_bits, am_home_barrier, multicasts, srcinfo=s.srcinfo)
            elif sync_type.is_await():
                home = s.home_barrier_expr()
                am_home_barrier = self.comp_index_expr(home.name, home.idx, instr_tl)
                if home.name in self._sync_syms:
                    mem = self._mem_env[home.name]
                    q_bit = mem.await_qual_tl(L2).as_bit()
                    # Model as "read" since concurrent access is allowed
                    b.SyncEnvAccess(
                        am_home_barrier,
                        q_bit,
                        q_bit,
                        flags=b.convergent_flag,
                        srcinfo=s.srcinfo,
                    )
                b.Await(
                    am_home_barrier,
                    L2_full_bits,
                    L2_temporal_bits,
                    N=sync_type.N,
                    srcinfo=s.srcinfo,
                )
            else:
                b.Fence(
                    L1_bits,
                    L2_full_bits,
                    L2_temporal_bits,
                    srcinfo=s.srcinfo,
                )

        elif is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            old_device = self._device
            old_default_instr_tl = self._default_instr_tl
            if isinstance(ctx, BaseAsyncConfig):
                self._device = ctx.get_child_device()
                self._default_instr_tl = self._device.get_default_instr_tl()
            # CudaDeviceFunction handled as BaseAsyncConfig, and below.
            if isinstance(ctx, CudaDeviceFunction):
                self._coll_tiling = ctx.top_level_coll_tiling()
                self._coll_env = ctx.coll_env()
                cuda_bits = timelines.cuda_stream_sync.get_full_timeline_set_bits()
                cpu_bit = timelines.cpu_in_order_qual.as_bit()
                clusterDim = ctx.clusterDim
                blockDim = ctx.blockDim
                domain = (clusterDim, blockDim) if clusterDim != 1 else (blockDim,)
                # Implicit (cpu, cuda_stream) -> cuda_stream sync before;
                # End CudaDeviceFunction with JoinThreads.
                # implicit cuda_stream -> cuda_stream sync after.
                b.Fence(cpu_bit | cuda_bits, cuda_bits, cuda_bits, srcinfo=s.srcinfo)
                with b.ParallelBlock(*domain, srcinfo=s.srcinfo):
                    old_domain = self._domain
                    self._domain = domain
                    self.do_stmts(s.body)
                    self._domain = old_domain
                # TODO: the join and second fence will have remarks logged above the
                # CudaDeviceFunction, which is misleading as it's after
                # the device function launch.
                b.JoinThreads(srcinfo=s.srcinfo)
                b.Fence(cuda_bits, cuda_bits, cuda_bits, srcinfo=s.srcinfo)
            else:
                self.do_stmts(s.body)
            self._device = old_device
            self._default_instr_tl = old_default_instr_tl
        # Must be after is_if_holding_with
        elif isinstance(s, LoopIR.If):
            am_cond = self.comp_e(s.cond, True, instr_tl)
            with b.If(am_cond, srcinfo=s.srcinfo):
                self.do_stmts(s.body)
                if s.orelse:
                    b.begin_orelse()
                    self.do_stmts(s.orelse)
        elif isinstance(s, LoopIR.For):
            if s.iter not in self._value_syms:
                b.add_variable(s.iter)
            am_iter = b[s.iter]
            am_lo = self.comp_e(s.lo, True, instr_tl)
            am_hi = self.comp_e(s.hi, True, instr_tl)
            loop_mode = s.loop_mode
            if isinstance(loop_mode, Seq):
                with b.SeqFor(am_iter, am_lo, am_hi, srcinfo=s.srcinfo):
                    self.do_stmts(s.body)
            elif isinstance(loop_mode, CudaTasks):
                with b.TasksFor(am_iter, am_lo, am_hi, srcinfo=s.srcinfo):
                    self.do_stmts(s.body)

                    # End device task with JoinThreads
                    is_device_task = cuda_tasks.validate_loop(s)
                    if is_device_task:
                        # TODO put remark at end of task loop?
                        b.JoinThreads(srcinfo=s.srcinfo)
            elif isinstance(loop_mode, _CodegenPar):
                old_coll_tiling = self._coll_tiling
                self.do_codegen_par(s, am_iter, am_lo, am_hi)
                self._coll_tiling = old_coll_tiling
            else:
                assert not isinstance(loop_mode, CudaThreads), "Need CollAnalysis"
                raise TypeError(
                    f"{s.srcinfo}: sync_check doesn't support loop mode {loop_mode.loop_mode_name()}"
                )
        elif isinstance(s, LoopIR.WindowStmt):
            self._envtyp[s.name] = s.rhs.type
            self._mem_env[s.name] = s.special_window or self._mem_env[s.rhs.name]

        elif isinstance(s, LoopIR.Alloc):
            self._saw_alloc = True
            self._envtyp[s.name] = s.type
            assert self._mem_env[s.name] == s.mem
            want_barrier = issubclass(s.mem, BarrierMechanism)
            want_sync = s.name in self._sync_syms
            want_value = s.name in self._value_syms
            if want_barrier and not want_sync and not want_value:
                b.add_variable(s.name)
            if want_barrier or want_sync or want_value:
                am_array = self.comp_index_expr(s.name, s.type.shape(), instr_tl)
            if want_barrier:
                b.BarrierEnvAlloc(am_array, srcinfo=s.srcinfo)
            if want_sync:
                b.SyncEnvAlloc(am_array, srcinfo=s.srcinfo)
            if want_value:
                b.ValueEnvAlloc(am_array, srcinfo=s.srcinfo)

        elif isinstance(s, LoopIR.Free):
            self._saw_free = True
            want_barrier = issubclass(s.mem, BarrierMechanism)
            want_sync = s.name in self._sync_syms
            want_free_shards = False

            if want_sync and s.mem.is_cuda_smem():
                # fmt: off
                assert isinstance(self._coll_tiling, CollTiling), "SMEM outside CUDA scope?"
                # fmt: on
                box = self._coll_tiling.get_box()
                domain = self._coll_tiling.get_domain()
                if box != domain:
                    # This over-approximation is because when we "free" SMEM in a CTA,
                    # it goes into a free pool that could be used for future allocs,
                    # and those could be the target of a multicast. So all threads
                    # in the CLUSTER must have visibility to the SMEM, not just CTA.
                    raise ValueError(
                        f"{s.srcinfo}: Sorry, sync-check for {s.name} @ {s.mem.name()} "
                        f"(SMEM) allocated outside cluster scope not implemented; "
                        f"currently have box={box} of {domain} threads active in cluster."
                    )
                b.SyncEnvFreeShard(
                    b[s.name],
                    timelines.cuda_in_order_ram_qual.as_bit(),
                    srcinfo=s.srcinfo,
                )

            if want_free_shards:
                # XXX dead code for now, but could be important later.
                # Look up distributed memory information for the variable.
                state = self._coll_analysis.distributed_alloc_states[s.name]
                distributed_iters = state.first_distributed_iters
                alloc_coll_tiling = state.alloc_coll_tiling
                target_coll_tiling: CollTiling = state.first_usage_coll_tiling
                target_domain = target_coll_tiling.get_domain()

                # We will prepare a parallel-for loop nest for accessing
                # each shard of the array. Also, possibly reshape domain.
                loop_nest = []
                if target_domain != self._domain:
                    loop_nest.append(b.DomainReshape(*target_domain, srcinfo=s.srcinfo))
                for dim_idx, dim in enumerate(target_coll_tiling.get_dims()):
                    dim: CollDim
                    # Generate minimal loops for this dimension needed
                    # to recover the distributed_iters. Avoids inappropriately
                    # specializing on dimensions not relevant to the sharding
                    # (e.g. checking only subset of CTA threads for SMEM free).
                    num_ops = 0
                    for op_i, op in enumerate(dim.dim_ops):
                        if op.iter in distributed_iters:
                            num_ops = op_i + 1
                    for op in dim.dim_ops[:num_ops]:
                        # Only add loops for levels that correspond to code subtree
                        # rooted under the scope that the alloc is done.
                        if op.tree_depth <= alloc_coll_tiling.get_tree_depth():
                            continue
                        # KeyError hack: I'm unsure if the variable already exists
                        # in the camspork program due to the horribly confusing
                        # CALLEE_DISTRIBUTED "synthetic iterators" (I'm sorry).
                        try:
                            am_iter = b[op.iter]
                        except KeyError:
                            am_iter = b.add_variable(op.iter)
                        loop_nest.append(
                            b.ThreadsFor(
                                am_iter,
                                0,
                                op.tile_count,
                                dim_idx,
                                op.offset,
                                op.box,
                                srcinfo=s.srcinfo,
                            )
                        )
                # Emit the generated loop nest (begin/end instead of with).
                # Use 0 for iterators known to have tile_count=1, i.e. always 0.
                # This is actually needed to avoid crashing for "trivial iterators"
                # that won't appear in the CollTiling, and hence the loop nest.
                for ctx in loop_nest:
                    ctx.begin()
                am_idx = tuple(
                    0 if self._coll_analysis.thread_iters[it].tile_count == 1 else b[it]
                    for it in distributed_iters
                )
                b.SyncEnvFreeShard(
                    b[s.name][am_idx],
                    Qual_tl.make_bits(free_qual_tl),
                    srcinfo=s.srcinfo,
                )
                for ctx in reversed(loop_nest):
                    ctx.end()

            # Must be after SyncEnvFreeShards,
            # since this deletes the SyncEnv data for the variable.
            if want_barrier:
                b.BarrierFree(s.name, srcinfo=s.srcinfo)
            elif want_sync:
                b.DataFree(s.name, srcinfo=s.srcinfo)

        elif isinstance(s, LoopIR.Call):
            callee = s.f
            instr = callee.instr
            if instr is None:
                # Would have to do CollAnalysis etc. on s.f and introspect it.
                raise ValueError(
                    f"{s.srcinfo}: sync-check doesn't support call to non-instr {callee.name}"
                )
            instr_tl = instr.instr_tl  # replaces _default_instr_tl
            assert len(s.args) == len(callee.args)
            # _tmp_call_args is used in comp_index_expr in order to evaluate
            # parameterized types, e.g. f(M: size, a: [f32][M]) M -> caller arg
            self._tmp_call_args = {
                callee_a.name: caller_a
                for caller_a, callee_a in zip(s.args, callee.args)
            }
            barrier, barrier_multicasts = self.comp_trailing_barrier_expr(s, instr_tl)
            for caller_a, callee_a in zip(s.args, callee.args):
                fnarg_type = callee_a.type
                if not fnarg_type.is_numeric():
                    # Avoids caller_a.name AttributeError for BinOp etc.
                    continue
                if caller_a.name not in self._sync_syms:
                    continue
                arg_info: AccessInfo = instr.access_info[str(callee_a.name)]
                # TODO value environment
                dst_lo, extent, loop_nest = self.comp_fnarg(
                    fnarg_type, caller_a, arg_info, instr_tl
                )
                for ctx in loop_nest:
                    ctx.begin()
                qual_tl, initial_qual_bits, ext_qual_bits = self.comp_qual_tl(
                    caller_a, instr_tl
                )
                flags = 0
                thread_access_granularity = 1
                if not arg_info.const:
                    flags |= b.mutate_flag
                    if arg_info.write_only:
                        flags |= b.write_only_flag
                if arg_info.out_of_order:
                    flags |= b.ooo_flag
                    # out-of-order non-convergent abstract machine optimization
                    thread_access_granularity = (
                        self._coll_analysis.get_qual_tl_thread_alignment(qual_tl)
                    )
                if qual_tl.get_default_convergent_access():
                    flags |= b.convergent_flag

                if (atomicity := arg_info.atomicity) is None:
                    atomic_qual_bits = 0
                else:
                    atomic_qual_bits = Qual_tl.make_bits(atomicity.qual_tl_list)
                    # fmt: off
                    assert atomic_qual_bits != 0, s.f.name
                    assert (flags & b.mutate_flag), f"unimplemented, in atomic read {s.f.name}"
                    # fmt: on

                b.SyncEnvAccess(
                    dst_lo,
                    initial_qual_bits,
                    ext_qual_bits,
                    flags=flags,
                    extent=extent,
                    barrier=barrier,
                    barrier_multicasts=barrier_multicasts,
                    atomic_qual_bits=atomic_qual_bits,
                    thread_access_granularity=thread_access_granularity,
                    srcinfo=s.srcinfo,
                )
                for ctx in loop_nest:
                    ctx.end()
            if barrier and s.trailing_barrier_expr.name in self._sync_syms:
                # Sync-check the trailing barrier itself.
                _, initial_qual_bits, _ = self.comp_qual_tl(
                    s.trailing_barrier_expr, instr_tl
                )
                b.SyncEnvAccess(
                    barrier,
                    initial_qual_bits,
                    initial_qual_bits,
                    # model as in-order read since concurrent access is allowed
                    flags=b.convergent_flag,
                    access_multicasts=barrier_multicasts,
                    barrier=barrier,
                    barrier_multicasts=barrier_multicasts,
                    srcinfo=s.srcinfo,
                )
        else:
            super().do_s(s)

    def comp_trailing_barrier_expr(self, s: LoopIR.Call, instr_tl: Instr_tl):
        b = self._builder
        bar_e = s.trailing_barrier_expr
        if bar_e is None:
            return None, ()
        idx = list(bar_e.idx)
        multicast_flags = tuple(isinstance(e, LoopIR.Interval) for e in bar_e.idx)
        for i, idx_e in enumerate(idx):
            if isinstance(idx_e, LoopIR.Interval):
                idx[i] = LoopIR.Const(0, T.int, idx_e.srcinfo)
        return self.comp_index_expr(bar_e.name, idx, instr_tl), (multicast_flags,)

    def do_codegen_par(self, s: LoopIR.For, am_iter, am_lo, am_hi):
        b = self._builder
        loop_mode = s.loop_mode
        self._coll_tiling = self._coll_analysis.thread_iters[s.iter].coll_tiling
        if None is (dim_idx := loop_mode.am_dim_idx):
            # Do-nothing parallel-for loop.
            with b.SeqFor(am_iter, am_lo, am_hi, srcinfo=s.srcinfo):
                self.do_stmts(s.body)
        else:
            loops_ctx = b.ThreadsFor(
                am_iter,
                am_lo,
                am_hi,
                dim_idx,
                loop_mode.am_offset,
                loop_mode.am_box,
                srcinfo=s.srcinfo,
            )
            new_domain = loop_mode.domain
            if new_domain == self._domain:
                with loops_ctx:
                    self.do_stmts(s.body)
            else:
                old_domain = self._domain
                self._domain = new_domain
                with b.DomainReshape(*new_domain, srcinfo=s.srcinfo):
                    with loops_ctx:
                        self.do_stmts(s.body)
                self._domain = old_domain

    def is_single_threaded(self):
        return self._coll_tiling is None or self._coll_tiling.get_box_num_threads() == 1

    # We emit SyncEnvRead for all reads found (filtered by sync_syms)
    # and translate the LoopIR expr to a camspork.BuilderExpr.
    # We completely ignore do_e; we don't want unexpected SyncEnvRead
    # from LoopIR.Read in "static" places like tensor types (do_t).
    def comp_e(
        self, e: LoopIR.expr | LoopIR.Point, want_value: bool, instr_tl: Instr_tl
    ) -> Optional[camspork.BuilderExpr]:
        if not isinstance(e, LoopIR.expr):
            assert isinstance(e, LoopIR.Point), type(e)
            e = e.pt
        b = self._builder
        if isinstance(e, LoopIR.Read):
            want_sync = e.name in self._sync_syms
            if want_value or want_sync:
                am_src = self.comp_index_expr(e.name, e.idx, instr_tl)
            if want_sync:
                _, initial_q, ext_q = self.comp_qual_tl(e, instr_tl)
                if self.is_single_threaded():
                    # convergent access makes no functional difference
                    # when the thread count is 1, but I suspect the
                    # implementation is faster if we set this flag.
                    flags = b.convergent_flag
                else:
                    flags = 0
                b.SyncEnvAccess(
                    am_src,
                    initial_q,
                    ext_q,
                    flags=flags,
                    srcinfo=e.srcinfo,
                )
            if want_value:
                return am_src
        elif isinstance(e, LoopIR.Const):
            return camspork.BuilderConst(e.val) if want_value else None
        elif isinstance(e, LoopIR.USub):
            return -self.comp_e(e, True, instr_tl) if want_value else None
        elif isinstance(e, LoopIR.BinOp):
            # NB we create am_lhs, am_rhs unconditionally to emit the
            # SyncEnvRead effects even when !want_value.
            am_lhs = self.comp_e(e.lhs, want_value, instr_tl)
            am_rhs = self.comp_e(e.rhs, want_value, instr_tl)
            if want_value:
                return camspork.BuilderBinOp(e.op, am_lhs, am_rhs)
        elif isinstance(e, LoopIR.Extern):
            if want_value:
                raise ValueError(f"Cannot sync-check {e}, unsupported Extern")
            for e in e.args:
                self.comp_e(e, want_value, instr_tl)
        else:
            raise TypeError(f"Unexpected case {e}")

    def comp_index_expr(
        self, name: Sym, idx: List[LoopIR.expr | LoopIR.Point], instr_tl: Instr_tl
    ) -> camspork.BuilderExpr:
        """Translate name + LoopIR indices to BuilderExpr

        SyncEnvRead effects will be generated for any of the indices
        (which isn't possible unless data/control separation goes away),
        but none are generated for the data referenced by name[idx] itself.

        """
        if caller_a := self._tmp_call_args.get(name):
            # If we work on Chexo, we may have to think about the unwanted
            # effects generated by comp_e here.
            assert not idx
            return self.comp_e(caller_a, True, instr_tl)
        if isinstance(typ := self._envtyp.get(name), LoopIR.WindowType):
            name = typ.src_buf
            idx = chain_window_idx(typ.idx, idx)
        am_name = self._builder.get_varname(name)
        return am_name[tuple(self.comp_e(e, True, instr_tl) for e in idx)]

    def comp_fnarg(
        self,
        fnarg_type: LoopIR.type,
        e: LoopIR.expr,
        arg_info: AccessInfo,
        instr_tl: Instr_tl,
    ):
        """Compile Read or WindowExpr to BuilderExpr + optional extent + loop nest.

        Similar policy on SyncEnvRead effects as comp_index_expr.
        The loop nest is needed for the access_by_owner_only=True case,
        where we have different input shards accessed by different threads
        (hence must communicate this to the abstract machine with a ThreadsFor).

        """
        b = self._builder
        if not isinstance(e, LoopIR.WindowExpr):
            return self.comp_e(e, True, instr_tl), None, ()
        shape = fnarg_type.shape()

        if arg_info.access_by_owner_only:
            tiling = self._coll_tiling
            assert isinstance(tiling, CollTiling)
            loop_nest = []
            coll_unit_stack = arg_info.distributed_coll_units[::-1]
        else:
            loop_nest = ()
            coll_unit_stack = ()

        idx_lo = []
        extent = []
        shape_i = 0
        unit_i = 0
        for w_idx, w in enumerate(e.idx):
            if isinstance(w, LoopIR.Interval):
                shape_coord = shape[shape_i]
                if coll_unit_stack:
                    # Distributed dimension with access_by_owner_only.
                    # We program a parallel loop over the collective unit specifed by
                    # the instruction, with each iteration accessing one slice.
                    #
                    # This is a much simplified version of what
                    # distributed_memory.py (in CollAnalysis) is doing,
                    # since at this point we assume the code is correct.
                    unit = coll_unit_stack.pop()
                    assert isinstance(shape_coord, LoopIR.Const)
                    tmp_iter = Sym(f"_{unit_i}_CALLEE_DISTRIBUTED")
                    tiling = tiling.tiled(
                        tmp_iter, unit, shape_coord.val, self._coll_env
                    )
                    am_iter = b.add_variable(tmp_iter)
                    codegen = tiling.get_codegen()
                    idx_lo.append(LoopIR.Read(tmp_iter, [], T.index, w.srcinfo))
                    extent.append(1)
                    # Possible optimization: redundant DomainReshape may be removed.
                    loop_nest.append(b.DomainReshape(*tiling.get_domain()))
                    loop_nest.append(
                        b.ThreadsFor(
                            am_iter,
                            0,
                            shape_coord.val,
                            codegen.dim_idx,
                            codegen.offset,
                            codegen.box,
                            srcinfo=w.srcinfo,
                        )
                    )
                    unit_i += 1
                else:
                    # Non-distributed dimension, or !access_by_owner_only.
                    # So we program the access to span the entire extent of the dimension.
                    idx_lo.append(w.lo)
                    extent.append(self.comp_e(shape_coord, True, instr_tl))
                shape_i += 1
            else:
                assert isinstance(w, LoopIR.Point)
                idx_lo.append(w.pt)
                extent.append(1)
        assert shape_i == len(shape)

        if all(x == 1 for x in extent):
            extent = None

        # If this is a read of a SpecialWindow, comp_index_expr takes
        # care of additional offset from the WindowStmt.
        return (self.comp_index_expr(e.name, idx_lo, instr_tl), extent, loop_nest)


def coll_analysis_to_camspork(
    mem_env: Dict[Sym, Type[LoopIR.type]],
    coll_analysis: CollAnalysis,
    p: LoopIR.proc,
    value_syms: Set[Sym],
    sync_syms: Set[Sym],
) -> camspork.ProgramBuilder:
    """Convert LoopIR.proc to a finished camspork program.

    We require a pre-computed dict of memory types and that CollAnalysis
    was applied to the proc (with the CollAnalysis object supplied).

    We only add value env + sync env stmts for variables that are named in
    the value_syms & sync_syms sets, respectively; except, for loop iterators
    and proc params always have their value environment enabled.
    The barrier environment is mandatory for barrier-type objects.

    The value environment really only exists in case we want to extend
    this for programs that don't have data-control type separation.

    TODO, in a sense, it would be good to have an extra step, where
    we convert to another LoopIR.proc holding "instrs" that are
    camspork.program stmts. So that we can do program analysis
    on the abstract machine program in a familiar environment.

    """
    builder = camspork.ProgramBuilder()
    CamsporkDo(builder, mem_env, coll_analysis, p, value_syms, sync_syms)
    builder.finish()
    return builder


def proc_size_tuple(
    p: LoopIR.proc, args_dict: Dict[str, int]
) -> Tuple[Optional[int], ...]:
    """Convert dictionary of argument values to tuple of Optional[int].

    i-th proc arg has value result[i]; None if the user provided nothing.
    We require values for all control values. For "future proofing" for
    Chexo we optionally accept values for data as well.

    """

    def generator():
        for a in p.args:
            try:
                strnm = str(a.name)
                value = args_dict[strnm]
                assert isinstance(value, int)
                assert value >= 0
                yield value
            except KeyError:
                if a.type.is_indexable():
                    raise KeyError(
                        f"{p.name}.sync_check: missing keyword argument {strnm}"
                    )
                else:
                    yield None

    return tuple(generator())


def make_buffer_sizes(
    p: LoopIR.proc, size_tuple: Tuple[int, ...]
) -> Dict[Sym, Tuple[int, ...]]:
    """Use arg substitute to convert proc_size_tuple (above) to buffer sizes."""
    return result


def top_level_check(backend, args_dict: Dict[str, int]):
    backend: BackendChecks
    p = backend.analyzed
    debug_log = backend.debug_log

    # Kills perf when true
    debug_on_exit = False
    debug_always = False

    # Compile and log abstract machine program once
    camspork_program = backend.lazy_camspork_program
    if camspork_program is None:
        # Exclude from sync checking any variables that either
        #   * are in sync-exempt memory, or
        #   * are non-barrier types and are only read from
        mutable_syms = set(nm for (nm, typ) in get_writes_of_stmts(p.body))
        sync_check_syms = set()
        for nm, mem in backend.mem_env.items():
            if nm in mutable_syms or issubclass(mem, BarrierMechanism):
                # SpecialWindow doesn't implement this; has to be inner if.
                if not mem.sync_exempt():
                    sync_check_syms.add(nm)
        backend.lazy_sync_syms = sync_check_syms
        camspork_program = coll_analysis_to_camspork(
            backend.mem_env,
            backend.coll_analysis,
            p,
            (),
            backend.lazy_sync_syms,
        )
        backend.lazy_camspork_program = camspork_program
        debug_log.log(p.name, "camspork", str(camspork_program))
    sync_syms = backend.lazy_sync_syms

    # Evaluate buffer sizes for variables that are subject to sync-check.
    size_tuple = proc_size_tuple(p, args_dict)
    proc_name_with_sizes = "-".join(
        [str(p.name)] + [str(sz) for sz in size_tuple if sz is not None]
    )
    binding = {}
    for i, a in enumerate(p.args):
        value = size_tuple[i]
        if value is not None:
            binding[a.name] = LoopIR.Const(size_tuple[i], a.type, a.srcinfo)
    buffer_sizes = {}
    for a in p.args:
        if not a.name in sync_syms:
            continue
        functor = SubstArgs(a.type.shape(), binding)
        concrete_shape = []
        for node in functor.result():
            assert isinstance(node, LoopIR.Const)
            concrete_shape.append(node.val)
        buffer_sizes[a.name] = tuple(concrete_shape)

    # Create environment with sync-env buffers initialized to expected sizes
    # and value environment initialized.
    def make_env(single_sync_var):
        env = camspork.ProgramEnv(camspork_program)
        for i, loopir_arg in enumerate(p.args):
            nm = loopir_arg.name
            if single_sync_var is None or single_sync_var == nm:
                if nm in sync_syms:
                    extent = buffer_sizes[nm]
                    env.alloc_sync(nm, *extent)
            if (val := size_tuple[i]) is not None:
                env.alloc_scalar_value(nm, val)
        for bit_index, qual_tl in enumerate(Qual_tl.get_all()):
            env.set_qual_tl_name(bit_index, str(qual_tl))
        env.set_debug_validation_enable(debug_always)
        return env

    # Run validation up to 2 times.
    # First is with all checking enabled.
    # If a syncv error is detected that is attributed to a specific var[idx],
    # we run checking again on that var[idx] only with history logging.
    # We debug log all remarks generated if an error occurs.
    error_remarks = ()
    single_sync_var = None
    single_sync_idx = ()
    try:
        for i in range(2):
            if i > 0 and single_sync_var is None:
                break
            env = make_env(single_sync_var)
            try:
                if single_sync_var is not None:
                    env.set_history_enable(True)
                env.exec(filter_name=single_sync_var, filter_idx=single_sync_idx)
                env.set_debug_validation_enable(debug_on_exit | debug_always)
            except Exception:
                if i == 0:
                    # Insert remarks for the main debug log, prior to adding
                    # more detailed remarks.
                    for camspork_stmt, text in env.get_remarks():
                        srcinfo = camspork_program.get_stmt_srcinfo(camspork_stmt)
                        debug_log.remark(p.name, f"{srcinfo}:\n{text}")
                env.add_error_history_remarks()
                error_remarks = env.get_remarks()
                single_sync_var = env.get_syncv_fail_var()
                if i == 0 and single_sync_var is not None:
                    # Do the second round of sync checking
                    single_sync_idx = env.get_syncv_fail_idx()
                else:
                    debug_log.log(
                        proc_name_with_sizes, "camspork", env.program_with_remarks()
                    )
                    raise
    except Exception:
        # We want to show something similar to the user's proc, but
        # after mem_analysis so that free is visible. Insert detailed
        # info to a separately-logged debug output proc.
        debug_log.log(
            proc_name_with_sizes,
            "sync-error",
            backend.after_mem_analysis,
            preferred=True,
        )
        for camspork_stmt, text in error_remarks:
            srcinfo = camspork_program.get_stmt_srcinfo(camspork_stmt)
            debug_log.remark(proc_name_with_sizes, f"{srcinfo}:\n{text}")
        raise
