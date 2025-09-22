from typing import Callable, Dict, Optional, Type, List, Set, Tuple

from ..core.memory import MemWin, BarrierType
from ..core.prelude import Sym
from ..core.instr_info import AccessInfo, InstrInfo
from ..core.LoopIR import T, LoopIR, LoopIR_Do, BaseCompilerDebugLog, chain_window_idx

from .async_config import BaseAsyncConfig, CudaDeviceFunction
from .base_with_context import is_if_holding_with
from .coll_algebra import CollTiling, CollDim, CollDimOp, CollDimExpectation
from .coll_analysis import CollAnalysis
from . import camspork
from .distributed_memory import ThreadIter
from .loop_modes import Seq, CudaTasks, _CodegenPar, CudaThreads
from .sync_types import SyncType
from .timelines import DeviceScope, Instr_tl, Qual_tl, Sync_tl
from . import timelines


class CamsporkDo(LoopIR_Do):
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
        self._device = timelines.cpu_basic_device
        self._default_instr_tl = timelines.cpu_basic_device.get_default_instr_tl()
        self._tmp_call_args = {}
        self._domain = ()
        self._saw_alloc = False
        self._saw_free = False

        b = self._builder

        explicit_syms = set(nm for _set in (value_syms, sync_syms) for nm in _set)
        for nm in sorted(explicit_syms):
            b.add_variable(nm)

        for a in self.proc.args:
            nm = a.name
            if nm not in explicit_syms:
                b.add_variable(nm)
            self._envtyp[nm] = a.type
            self._mem_env[nm] = a.mem

        self.do_stmts(self.proc.body)
        assert self._saw_free or not self._saw_alloc, "Need MemAnalysis before"

    def get_qual_bits(self, node: LoopIR.expr | LoopIR.stmt, instr_tl: Instr_tl):
        nm = node.name
        mem = self._mem_env[nm]
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
        return Qual_tl.make_bits(initial_qual_tl), Qual_tl.make_bits(q)

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
                initial_q, ext_q = self.get_qual_bits(s, instr_tl)
                b.SyncEnvAccess(am_dst, initial_q, ext_q, is_mutate=True, is_ooo=False)
            if want_value:
                op = "=" if isinstance(s, LoopIR.Assign) else "+"
                b.MutateValue(am_dst, op, am_rhs)
        elif isinstance(s, LoopIR.SyncStmt):
            sync_type: SyncType = s.sync_type
            L1 = sync_type.first_sync_tl
            L2 = sync_type.second_sync_tl
            if L1 is not None:
                transitive = L1.is_V1_transitive()
                L1_bits = L1.get_full_timeline_set_bits()
            if L2 is not None:
                L2_full_bits = L2.get_full_timeline_set_bits()
                L2_temporal_bits = L2.get_temporal_timeline_set_bits()
            if sync_type.is_arrive():
                # TODO add read effects
                home = s.home_barrier_expr()
                multicasts = s.multicasts()
                am_home_barrier = self.comp_index_expr(home.name, home.idx, instr_tl)
                b.Arrive(transitive, L1_bits, am_home_barrier, multicasts)
            elif sync_type.is_await():
                # TODO add read effects
                home = s.home_barrier_expr()
                am_home_barrier = self.comp_index_expr(home.name, home.idx, instr_tl)
                b.Await(am_home_barrier, L2_full_bits, L2_temporal_bits, N=sync_type.N)
            else:
                b.Fence(transitive, L1_bits, L2_full_bits, L2_temporal_bits)

        elif is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            old_device = self._device
            old_default_instr_tl = self._default_instr_tl
            if isinstance(ctx, BaseAsyncConfig):
                self._device = ctx.get_child_device()
                self._default_instr_tl = self._device.get_default_instr_tl()
            # CudaDeviceFunction handled as BaseAsyncConfig, and below.
            if isinstance(ctx, CudaDeviceFunction):
                cuda_bits = timelines.cuda_stream_sync.get_full_timeline_set_bits()
                cpu_bit = timelines.cpu_in_order_qual.as_bit()
                clusterDim = ctx.clusterDim
                blockDim = ctx.blockDim
                domain = (clusterDim, blockDim) if clusterDim != 1 else (blockDim,)
                # Implicit (cpu, cuda_stream) -> cuda_stream sync before;
                # implicit cuda_stream -> cuda_stream sync after.
                b.Fence(True, cpu_bit | cuda_bits, cuda_bits, cuda_bits)
                with b.ParallelBlock(*domain):
                    old_domain = self._domain
                    self._domain = domain
                    self.do_stmts(s.body)
                    self._domain = old_domain
                b.Fence(True, cuda_bits, cuda_bits, cuda_bits)
            else:
                self.do_stmts(s.body)
            self._device = old_device
            self._default_instr_tl = old_default_instr_tl
        # Must be after is_if_holding_with
        elif isinstance(s, LoopIR.If):
            am_cond = self.comp_e(s.cond, True, instr_tl)
            with b.If(am_cond):
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
                with b.SeqFor(am_iter, am_lo, am_hi):
                    self.do_stmts(s.body)
            elif isinstance(loop_mode, CudaTasks):
                with b.TasksFor(am_iter, am_lo, am_hi):
                    self.do_stmts(s.body)
            elif isinstance(loop_mode, _CodegenPar):
                self.do_codegen_par(s, am_iter, am_lo, am_hi)
            else:
                assert not isinstance(loop_mode, CudaThreads), "Need CollAnalysis"
                raise TypeError(
                    f"{s.srcinfo}: unexpected loop mode {loop_mode.loop_mode_name()}"
                )
        elif isinstance(s, LoopIR.WindowStmt):
            self._envtyp[s.name] = s.rhs.type
            self._mem_env[s.name] = s.special_window or self._mem_env[s.rhs.name]

        elif isinstance(s, LoopIR.Alloc):
            self._saw_alloc = True
            self._envtyp[s.name] = s.type
            self._mem_env[s.name] = s.mem
            want_barrier = issubclass(s.mem, BarrierType)
            want_sync = s.name in self._sync_syms
            want_value = s.name in self._value_syms
            if want_barrier and not want_sync and not want_value:
                b.add_variable(s.name)
            if want_barrier or want_sync or want_value:
                am_array = self.comp_index_expr(s.name, s.type.shape(), instr_tl)
            if want_barrier:
                b.BarrierEnvAlloc(am_array)
            if want_sync:
                b.SyncEnvAlloc(am_array)
            if want_value:
                b.ValueEnvAlloc(am_array)

        elif isinstance(s, LoopIR.Free):
            self._saw_free = True
            want_barrier = issubclass(s.mem, BarrierType)
            want_sync = s.name in self._sync_syms
            want_free_shards = False
            if want_sync:
                free_qual_tl = s.mem.free_qual_tl()
                want_free_shards = not (free_qual_tl is None)

            if want_barrier:
                b.BarrierEnvFree(s.name)
            if want_free_shards:
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
                    loop_nest.append(b.DomainReshape(*target_domain))
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
                                am_iter, 0, op.tile_count, dim_idx, op.offset, op.box
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
                b.SyncEnvFreeShard(b[s.name][am_idx], Qual_tl.make_bits(free_qual_tl))
                for ctx in reversed(loop_nest):
                    ctx.end()
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
            for caller_a, callee_a in zip(s.args, callee.args):
                fnarg_type = callee_a.type
                if fnarg_type.is_indexable():
                    continue
                if caller_a.name not in self._sync_syms:
                    continue
                arg_info: AccessInfo = instr.access_info[str(callee_a.name)]
                assert (
                    not arg_info.access_by_owner_only
                ), "TODO generate ThreadsFor loop for shards of input"
                # TODO atomics
                dst_lo, extent = self.comp_fnarg(fnarg_type, caller_a, instr_tl)
                if dst_lo is not None:
                    initial_q, ext_q = self.get_qual_bits(caller_a, instr_tl)
                    barrier, barrier_multicasts = self.comp_trailing_barrier_expr(
                        s, instr_tl
                    )
                    b.SyncEnvAccess(
                        dst_lo,
                        initial_q,
                        ext_q,
                        is_mutate=not arg_info.const,
                        is_ooo=arg_info.out_of_order,
                        extent=extent,
                        barrier=barrier,
                        barrier_multicasts=barrier_multicasts,
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
        if None is (dim_idx := loop_mode.am_dim_idx):
            # Do-nothing parallel-for loop.
            with b.SeqFor(am_iter, am_lo, am_hi):
                self.do_stmts(s.body)
        else:
            loops_ctx = b.ThreadsFor(
                am_iter, am_lo, am_hi, dim_idx, loop_mode.am_offset, loop_mode.am_box
            )
            new_domain = loop_mode.domain
            if new_domain == self._domain:
                with loops_ctx:
                    self.do_stmts(s.body)
            else:
                old_domain = self._domain
                self._domain = new_domain
                with b.DomainReshape(*new_domain):
                    with loops_ctx:
                        self.do_stmts(s.body)
                self._domain = old_domain

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
                initial_q, ext_q = self.get_qual_bits(e, instr_tl)
                b.SyncEnvAccess(am_src, initial_q, ext_q, is_mutate=False, is_ooo=False)
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

    def comp_fnarg(self, fnarg_type: LoopIR.type, e: LoopIR.expr, instr_tl: Instr_tl):
        """Compile Read or WindowExpr to BuilderExpr + optional extent

        Similar policy on SyncEnvRead effects as comp_index_expr.

        """
        if not isinstance(e, LoopIR.WindowExpr):
            return self.comp_e(e, True, instr_tl), None
        shape = fnarg_type.shape()
        idx_lo = []
        extent = []
        shape_i = 0
        for w in e.idx:
            if isinstance(w, LoopIR.Interval):
                idx_lo.append(w.lo)
                extent.append(self.comp_e(shape[shape_i], True, instr_tl))
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
        return (self.comp_index_expr(e.name, idx_lo, instr_tl), extent)


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
