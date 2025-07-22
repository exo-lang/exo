# Compiler: Generate exo_SyncState for CUDA C++ device functions, and
# lowered Arrive/Await/Fence statements.

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Optional, Type, List, Tuple

from ..core.prelude import Sym, SrcInfo
from ..core.LoopIR import LoopIR

from .barrier_usage import BarrierUsage, SyncInfo
from .coll_algebra import (
    CollParam,
    CollUnit,
    clusterDim_param,
    blockDim_param,
    CollIndexExpr,
    CollTiling,
    cuda_thread,
    cuda_warp,
    cuda_warpgroup,
    cuda_cta_in_cluster,
    cuda_cluster,
    cuda_agnostic_sub_cta,
)
from .cuda_memory import (
    CudaCommitGroup,
    CudaMbarrier,
    CudaClusterSync,
)
from .distributed_memory import DistributedAllocState, ThreadIter
from .excut import InlinePtxGen, simple_ptx_c_lines
from .lowered_barrier import LoweredBarrierType, LoweredBarrier
from .sync_types import SyncType
from . import timelines
from .timelines import Instr_tl, Sync_tl


@dataclass(slots=True)
class SyncStateBuilder:
    _coll_env: Dict[CollParam, int]

    # LoweredBarrier for each barrier lowered, indexed by name
    lowered: Dict[Sym, LoweredBarrier] = field(default_factory=dict)

    # tuples (mbarrier_count, arrive_count)
    # to initialize in SMEM, e.g. (8, 64), (2, 384) means initialize an
    # array of 10 mbarriers in SMEM with the first 8 having
    # arrive_count=64, last 2 arrive_count=384
    _mbarrier_pairs: List[Tuple[int, int]] = field(default_factory=list)
    mbarrier_count: int = 0

    # C++ lines to join into exo_SyncState struct
    SyncState_lines: List[str] = field(default_factory=list)

    # Need to assign a name unique-ifying suffix for each barrier
    # This is different than what the main LoopIR->C compiler does because the
    # name needs to be unique throughout the full device function, i.e. it's not
    # enough to be unique just within the barrier's scope in Exo object code.
    _sym_counters: Dict[Sym, int] = field(default_factory=dict)

    _uses_async_proxy: bool = False

    def add_barrier(
        self,
        name: Sym,
        usage: BarrierUsage,  # Not to be confused with Usage_tl (sorry)
        coll_tilings: DistributedAllocState,
        thread_iters: Dict[Sym, ThreadIter],
    ):
        for info in usage.sync_info:
            if info is not None:
                if not timelines.cuda_async_proxy.disjoint_full_timeline_set(
                    info.sync_tl
                ):
                    self._uses_async_proxy = True

        srcinfo = usage.decl_stmt.srcinfo
        barrier_type = usage.barrier_type
        suffix = self._assign_suffix(name)
        if usage.is_fence():
            if usage.get_front_arrive().sync_tl == timelines.wgmma_fence_1:
                self.add_wgmma_fence(name, usage, coll_tilings, thread_iters, suffix)
            else:
                self.add_garden_variety_or_cluster_sync(
                    name, usage, coll_tilings, thread_iters, suffix, False
                )
        elif issubclass(barrier_type, CudaMbarrier):
            self.add_mbarrier(name, usage, coll_tilings, thread_iters, suffix)
        elif issubclass(barrier_type, CudaCommitGroup):
            self.add_commit_group(name, usage, coll_tilings, thread_iters, suffix)
        elif issubclass(barrier_type, CudaClusterSync):
            self.add_garden_variety_or_cluster_sync(
                name, usage, coll_tilings, thread_iters, suffix, True
            )
        else:
            raise TypeError(
                f"{srcinfo}: {barrier_type.name()} "
                f"not supported in CUDA device function"
            )

    def add_wgmma_fence(
        self,
        name: Sym,
        usage: BarrierUsage,
        coll_tilings: DistributedAllocState,
        thread_iters: Dict[Sym, ThreadIter],
        suffix: str,
    ):
        Arrive = usage.get_front_arrive()
        Await = usage.get_front_await()
        L1 = Arrive.sync_tl
        L2 = Await.sync_tl
        srcinfo = Arrive.get_srcinfo()
        assert L1 == timelines.wgmma_fence_1
        if L2 != timelines.wgmma_fence_2:
            raise ValueError(
                f"{srcinfo}: wgmma fence needs second sync-tl wgmma_fence_2"
            )

        coll_tiling = coll_tilings.get_front_arrive()
        # Should be the case for a Fence
        assert coll_tiling is coll_tilings.get_front_await()

        if msg := coll_tiling.unit_mismatch(cuda_warpgroup, self._coll_env):
            raise ValueError(
                f"{srcinfo}: wgmma fence must be executed by a warpgroup: {msg}"
            )

        lowered = LoweredBarrier(False, LoweredBarrierType.wgmma_fence)
        lowered.codegen_sync_stmt = lambda _: simple_ptx_c_lines(
            "wgmma.fence.sync.aligned"
        )
        self.lowered[name] = lowered

    def add_garden_variety_or_cluster_sync(
        self,
        name: Sym,
        usage: BarrierUsage,
        coll_tilings: DistributedAllocState,
        thread_iters: Dict[Sym, ThreadIter],
        suffix: str,
        force_cluster_sync: bool,
    ):
        """Do up to 3 things

        - wait_all if first sync-tl includes Sm80_cp_async
        - barrier arrive/await if more than 1 thread, or special exception (*)
        - fence.proxy.async if second sync-tl includes any async proxy

        (*) special exception, if thread collective is a warpgroup and
        the second sync-tl only includes wgmma_async_smem, we can elide
        the barrier. This relies on wgmma_async_smem not being V1-transitive.

        """

        Arrive = usage.get_front_arrive()
        Await = usage.get_front_await()
        L1 = Arrive.sync_tl
        L2 = Await.sync_tl
        srcinfo = usage.get_srcinfo()
        clusterDim = self._clusterDim()
        coll_tiling = coll_tilings.get_front_arrive()
        await_coll_tiling = coll_tilings.get_front_await()
        assert not usage.has_back_array()

        mismatch_messages = []

        def match_unit(unit):
            if msg := coll_tiling.unit_mismatch(unit, self._coll_env):
                mismatch_messages.append(msg)
                return False
            return True

        if force_cluster_sync:
            is_cluster_sync = True
            if msg := coll_tiling.unit_mismatch(cuda_cluster, self._coll_env):
                raise ValueError(
                    f"{srcinfo}: Arrive for {name} must be by full cluster ({msg})"
                )
            # If the Arrive passed, then so should the Await, since the
            # pairing requirement enforces identical coll units.
            assert CudaClusterSync.traits().requires_pairing
            assert not await_coll_tiling.unit_mismatch(cuda_cluster, self._coll_env)
        else:
            is_cluster_sync = self._clusterDim() > 1 and match_unit(cuda_cluster)
            assert (
                coll_tiling is await_coll_tiling
            ), "Expected Fence to have identical Arrive/Await tiling"
        if is_cluster_sync:
            # solitary=True as there's only one built-in cluster sync per cluster
            lowered = LoweredBarrier(True, LoweredBarrierType.cluster_sync)
        else:
            lowered = LoweredBarrier(False, LoweredBarrierType.cluster_sync)
        arrive_lines = []
        await_lines = []

        # Insert wait for sm_80 cp.async if needed.
        if timelines.cuda_in_order.implements_first(L1):
            pass
        elif timelines.Sm80_generic.implements_first(L1):
            arrive_lines += simple_ptx_c_lines("cp.async.wait_all")
        else:
            raise ValueError(
                f"{srcinfo}: Fence first sync-tl "
                f"{L1} not supported (we allow Sm80_generic)"
            )

        # Insert cross-thread sync if needed
        assert not timelines.wgmma_async_smem.is_V1_transitive()
        wgmma_special_case = timelines.wgmma_async_smem.implements_second(
            L1
        ) and match_unit(cuda_warpgroup)

        if not wgmma_special_case:
            mismatch_messages.append("warpgroup with second sync-tl=wgmma_async_smem")

        if is_cluster_sync:
            arrive_lines.extend(simple_ptx_c_lines("barrier.cluster.arrive.aligned"))
            await_lines.extend(simple_ptx_c_lines("barrier.cluster.wait.aligned"))
        elif wgmma_special_case or match_unit(cuda_thread):
            pass
        elif match_unit(cuda_warp):
            arrive_lines.append("__syncwarp();")
        elif match_unit(cuda_cta_in_cluster):
            # We need to use barrier.cta.sync, not bar or syncthreads
            # due to divergent control flow in "full CTA" code
            # if there's [named] warp specialization.
            arrive_lines.extend(simple_ptx_c_lines("barrier.cta.sync", 0))
        else:
            raise ValueError(
                "\n".join(
                    [f"{srcinfo}: Fence collective unit matched no known case"]
                    + mismatch_messages
                )
            )

        # Insert fence.proxy.async if needed
        if timelines.Sm80_generic.implements_second(L2):
            pass
        elif timelines.cuda_generic_and_async_proxy.implements_second(L2):
            await_lines.extend(simple_ptx_c_lines("fence.proxy.async"))
        else:
            raise ValueError(
                f"{srcinfo}: Fence second sync-tl {L2} not "
                f"supported (at most CUDA generic+async proxy)"
            )

        def codegen(sync_stmt: LoopIR.SyncStmt):
            sync_type = sync_stmt.sync_type
            if sync_type.is_arrive():
                return arrive_lines
            elif sync_type.is_await():
                assert sync_type.N == 0, "should have been flagged earlier"
                return await_lines
            else:
                return arrive_lines + await_lines

        lowered.codegen_sync_stmt = codegen
        self.lowered[name] = lowered

    def add_mbarrier(
        self,
        name: Sym,
        usage: BarrierUsage,
        coll_tilings: DistributedAllocState,
        thread_iters: Dict[Sym, ThreadIter],
        suffix: str,
    ):
        # Each queue barrier object (equiv, mbarrier ring buffer) must be
        # resident in 1 CTA only. NB any Arrive/Await will do here.
        if msg := coll_tilings.get_front_arrive().unit_mismatch(
            cuda_agnostic_sub_cta, self._coll_env
        ):
            raise ValueError(
                f"{usage.get_srcinfo()}: {name} must be distributed so each mbarrier is resident in 1 CTA only ({msg})"
            )

        # Reserve C name and space for mbarriers in SMEM
        lowered = LoweredBarrier(False, LoweredBarrierType.mbarrier)
        mbarrier_offset = self.mbarrier_count
        nm_suffix = f"{suffix}_{name}"

        # Translate N to number of trivial Awaits (skip mbarrier wait)
        def n_skips(info: SyncInfo):
            assert info.min_N == info.max_N
            return ~info.min_N

        # Calculate the size of the ring buffer (number of mbarriers)
        # and CTA indices to XOR with (cluster feature)
        front_skips = n_skips(usage.get_front_await())
        front_cta_xor_list = coll_tilings.cta_xor_list(
            self._blockDim(), thread_iters, usage.get_front_arrive()
        )
        if usage.has_back_array():
            back_skips = n_skips(usage.get_back_await())
            ring = front_skips + back_skips
            back_cta_xor_list = coll_tilings.cta_xor_list(
                self._blockDim(), thread_iters, usage.get_back_arrive()
            )
        else:
            ring = front_skips + 1
        if ring == 0:
            raise ValueError(
                f"{usage.get_srcinfo()}: {name} must have some await with nonzero skips (e.g. set N = ~1)"
            )

        # Number of physical mbarriers is slice_count * ring, where
        # slice_count is the number of logical Exo queue barrier objects per CTA
        # (usually 1) and ring is the depth of the ring buffer.
        slice_count = coll_tilings.codegen_slices_to_root(
            self._blockDim(), thread_iters
        )

        # Need to be able to store values 0 through (ring-1)
        ring_bits = (ring - 1).bit_length()
        # Need to be able to count 0 to ring (inclusive) skips.
        # This value will not be used if skipping is not actually enabled.
        skip_bits = ring.bit_length()

        # black formatting will ruin the readability of the generated C++ code below
        # fmt: off
        def mbarrier_to_u32(lines, is_back, ringidx):
            byte_offset = 8 * (mbarrier_offset + (ring * slice_count) if is_back else mbarrier_offset)
            idx = f"(slice * {ring} + {ringidx})"
            lines.append(f"  const auto mbarrier_u32 = exo_smemU32(exo_smem + {byte_offset} + 8*{idx});")

        def generate_arrive(is_back):
            b = "Back" if is_back else "Front"
            info = usage.get_back_arrive() if is_back else usage.get_front_arrive()
            cta_xor_list = back_cta_xor_list if is_back else front_cta_xor_list
            sync_tl = info.sync_tl

            if timelines.Sm80_cp_async.implements_first(sync_tl):
                is_Sm80_cp_async = True
            elif timelines.cuda_in_order.implements_first(sync_tl):
                is_Sm80_cp_async = False
            elif timelines.tma_to_smem_async.implements_first(sync_tl):
                raise ValueError(
                    f"{info.get_srcinfo()}: mbarrier Arrive sync-tl {sync_tl} "
                    f"not supported: use cuda_temporal, and add trailing barriers to TMA instrs")
            else:
                raise ValueError(
                    f"{info.get_srcinfo()}: mbarrier Arrive sync-tl {sync_tl} "
                    f"not supported: need cuda_in_order or Sm80_cp_async")

            lines = self.SyncState_lines
            idx = f"{b}ArriveIdx{nm_suffix}"
            if ring_bits > 0:
                lines.append(f"unsigned {idx} : {ring_bits} = 0;")
            else:
                lines.append(f"static constexpr unsigned {idx} = 0;  // Trivial size-1 ring buffer")
            lines.append(f"EXO_CUDA_INLINE uint32_t {b}Arrive{nm_suffix}(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, bool enable) {{")
            mbarrier_to_u32(lines, is_back, idx);
            lines.append(f"  if (enable) {{")

            # Optional broadcast to other CTAs in cluster.
            if len(cta_xor_list) > 1:
                multicast = True
                cta_or_cluster = "cluster"
                lines.append(f"    const unsigned cta_rank = blockIdx.x % {self._clusterDim()};")
            else:
                multicast = False
                assert cta_xor_list[0] == 0
                cta_or_cluster = "cta"

            # Issue arrives to each CTA (special code for 1-CTA case, so we
            # don't break sm_80, and don't waste time translating addresses)
            for cta_xor in cta_xor_list:
                ptx_format = f"// {b}Arrive{nm_suffix}\n"
                if multicast:
                    ptx_format += f"// cta_xor={cta_xor}\n"
                if is_Sm80_cp_async:
                    if cta_or_cluster != "cta":
                        bad_stmt = usage.get_front_arrive().stmts[0]
                        raise ValueError(f"{bad_stmt.srcinfo}: Sm80_cp_async mbarrier must be within 1 CTA (in {bad_stmt})")
                    ptx_format += f"cp.async.mbarrier.arrive.noinc.shared::cta.b64 #0#;"
                else:
                    ptx_format += f"mbarrier.arrive.shared::{cta_or_cluster}.b64 _, #0#;"
                ptx = InlinePtxGen(ptx_format, volatile=True)
                if multicast:
                    ptx.add_arg(f"exo_mapa_shared_cluster(mbarrier_u32, cta_rank ^ {cta_xor})",
                                constraint="r", log_as="bits", brackets=True)
                else:
                    ptx.add_arg("mbarrier_u32", constraint="r", log_as="bits", brackets=True)
                lines.extend(ptx.as_c_lines(py_format=False, tab="    "))
            if ring_bits > 0:
                lines.append(f"    // Advance ring buffer state")
                lines.append(f"    {idx} = {idx} == {ring - 1} ? 0 : {idx} + 1;")
            lines.append(f"  }}")
            lines.append(f"  return mbarrier_u32;")
            lines.append(f"}}")

        def generate_await(is_back, L1):
            b = "Back" if is_back else "Front"
            info = usage.get_back_await() if is_back else usage.get_front_await()
            L2 = info.sync_tl

            if timelines.cuda_temporal.implements_first(L1):
                # No values from the first full visibility set are being made
                # visible so no proxy fence regardless of second sync timeline.
                proxy_fence = False
            elif timelines.Sm80_generic.implements_second(L2):
                proxy_fence = False
            elif timelines.cuda_generic_and_async_proxy.implements_second(L2):
                proxy_fence = True
            else:
                if L2 == timelines.wgmma_async:
                    remark = "consider wgmma_async_smem"
                else:
                    remark = "at most CUDA generic+async proxy"
                raise ValueError(
                    f"{info.get_srcinfo()}: mbarrier Await sync-tl {L2} "
                    f"not supported ({remark})")

            lines = self.SyncState_lines
            # If we have a back queue barrier array, the mbarriers for them
            # are allocated right after those for the front queue barrier array.
            offset = mbarrier_offset + ring if is_back else mbarrier_offset
            idx = f"{b}AwaitIdx{nm_suffix}"
            skips = f"{b}Skips{nm_suffix}"
            parity_bits = f"{b}Parity{nm_suffix}"
            n_skips = back_skips if is_back else front_skips
            enable_skips = n_skips != 0

            # Define (register) exo_SyncState member variables: ring buffer
            # index, parity bitfield, and, if needed, counter for inital skips.
            if ring_bits > 0:
                lines.append(f"unsigned {idx} : {ring_bits} = 0;")
            else:
                lines.append(f"static constexpr unsigned {idx} = 0;  // Trivial size-1 ring buffer")
            lines.append(f"unsigned {parity_bits} : {ring} = 0;")
            if enable_skips:
                lines.append(f"unsigned {skips} : {skip_bits} = 0;")

            # Define Await member function
            # The initial_skips parameter is included iff skipping is enabled,
            # as a last line of defense against future Exo compiler bugs.
            if enable_skips:
                lines.append(f"EXO_CUDA_INLINE void {b}Await{nm_suffix}(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, int initial_skips = 0) {{")
                mbarrier_to_u32(lines, is_back, idx)
                lines.append(f"  const bool enable = {skips} >= initial_skips;")
            else:
                lines.append(f"EXO_CUDA_INLINE void {b}Await{nm_suffix}(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice) {{")
                mbarrier_to_u32(lines, is_back, idx)
                lines.append(f"  const bool enable = true;")
            comment = f"// {b}Await{nm_suffix}"
            lines.append(f"  if (enable) {{")
            # sm_90 needed for try_wait; condition on __CUDA_ARCH__
            def add_inline_ptx(try_or_test):
                ptx_format = """{
                    %s
                    .reg.pred P1;
                    EXO_BEFORE_WAIT:
                    mbarrier.%s_wait.parity.acquire.cta.shared::cta.b64 P1, #0#;
                    @P1 bra.uni EXO_WAIT_DONE;
                    bra.uni EXO_BEFORE_WAIT;
                    EXO_WAIT_DONE:
                    }""" % (comment, try_or_test)
                ptx = InlinePtxGen(ptx_format, volatile=True)
                ptx.add_arg("mbarrier_u32", constraint="r", log_as="bits", brackets=True)
                ptx.add_arg(f"1u & {parity_bits} >> {idx}", constraint="r", log_as="bits")
                lines.extend(ptx.as_c_lines(py_format=False, tab="    "))
            lines.append("#if __CUDA_ARCH__ < 900")
            add_inline_ptx("test")
            lines.append("#else")
            add_inline_ptx("try")
            lines.append("#endif")
            lines.append(f"    // Flip parity")
            lines.append(f"    {parity_bits} ^= 1u << {idx};")
            if ring_bits > 0:
                lines.append(f"    // Advance ring buffer state")
                lines.append(f"    {idx} = {idx} == {ring - 1} ? 0 : {idx} + 1;")
            if proxy_fence:
                lines.append(f'    // Needed for first sync-tl {L1}; second sync-tl {L2}')
                lines.extend(simple_ptx_c_lines("fence.proxy.async", tab="    "))
            lines.append(f"  }}")
            if enable_skips:
                lines.append(f"  else {{")
                lines.append(f"    // {b}Await({name}) returns without waiting for mbarrier first <initial_skips> times")
                lines.append(f"    {skips}++;")
                lines.append(f"  }}")
            lines.append(f"}}")

        # mbarrier allocator: record mbarriers to initialize, first the front queue barrier
        # array, then the optional back queue barrier array.
        RS = ring * slice_count
        lines = self.SyncState_lines
        arrive_count = coll_tilings.get_front_arrive().box_num_threads() * len(front_cta_xor_list)
        self._mbarrier_pairs.append((RS, arrive_count))
        self.mbarrier_count += RS
        lines.append(f"// {name}: barrier @ CudaMbarrier, ring={ring}, slice_count={slice_count}")
        lines.append(f"// front mbarriers [{mbarrier_offset}, {mbarrier_offset + RS}]; "
                     f"arrive_count={arrive_count}")

        if usage.has_back_array():
            arrive_count = coll_tilings.get_back_arrive().box_num_threads() * len(back_cta_xor_list)
            self._mbarrier_pairs.append((RS, arrive_count))
            self.mbarrier_count += RS
            lines.append(f"// back mbarriers [{mbarrier_offset + RS}, {mbarrier_offset + RS * 2}]; "
                         f"arrive_count={arrive_count}")

        # Generate Arrive and Await syntax
        # Awaits must be aware with the sync-tl
        # of the matched Arrive
        generate_arrive(False)
        generate_await(False, usage.get_front_arrive().sync_tl)
        if usage.has_back_array():
            generate_arrive(True)
            generate_await(True, usage.get_back_arrive().sync_tl)

        # Arrive/Await lowers to call to generated exo_syncState member function.
        Arrive_txt = f"Arrive{nm_suffix}(exo_smem, exo_excutLog"
        def codegen(node: LoopIR.SyncStmt | LoopIR.BarrierExpr):
            # Unpack BarrierExpr from SyncStmt
            if isinstance(node, LoopIR.SyncStmt):
                # Generating stateful Arrive/Await call.
                e = node.barriers[0]
                sync_type = node.sync_type
                is_arg = False
            else:
                # Generating expression to pass as arg to TMA instr.
                e = node
                is_arg = True
            assert isinstance(e, LoopIR.BarrierExpr)

            # The purpose of this is to generate an expression of threadIdx.x
            # to select between mbarriers in the same CTA
            # (niche usage ... I hope we test this).
            # DICEY: in the chosen BarrierExpr, intervals lo:hi become None (ignored).
            # These are distributed dims, which correspond to CTA-in-cluster dimensions,
            # which codegen_slices_to_root will ignore anyway due to
            # hi_thread_pitch=blockDim.
            iter_syms = [None if multicast else idx.pt.name for multicast, idx in zip(e.multicast_flags(), e.idx)]
            slice = coll_tilings.codegen_slices_to_root(self._blockDim(), thread_iters, iter_syms)
            b = "Back" if e.back else "Front"

            if is_arg:
                return f"exo_syncState.{b}{Arrive_txt}, {slice}, 0)"
            elif sync_type.is_arrive():
                assert sync_type.N == 1
                lines = []
                for e in node.barriers:
                    cta_mask = coll_tilings.codegen_cta_mask(self._blockDim(), thread_iters, e)
                    lines.append(f"// cta_mask: {cta_mask}")
                lines.append(f"exo_syncState.{b}{Arrive_txt}, {slice}, 1);")
                return lines
            else:
                assert sync_type.is_await()
                skips_arg = ""
                if skips := ~sync_type.N:
                    assert skips > 0, "should have been caught by BarrierUsageAnalysis"
                    skips_arg = f", {skips}"
                return [f"exo_syncState.{b}Await{nm_suffix}(exo_smem, exo_excutLog, {slice}{skips_arg});"]
        lowered.codegen_sync_stmt = codegen
        lowered.codegen_barrier_arg = codegen
        lowered.codegen_cta_mask = lambda e: coll_tilings.codegen_cta_mask(self._blockDim(), thread_iters, e)
        self.lowered[name] = lowered
        # fmt: on

    def add_commit_group(
        self,
        name: Sym,
        usage: BarrierUsage,
        coll_tilings: DistributedAllocState,
        thread_iters: Dict[Sym, ThreadIter],
        suffix: str,
    ):
        # Commit groups
        #
        # Sm80_cp_async -> Sm80_generic; 1 thread
        # tma_to_gmem_async -> cuda_generic_and_async_proxy; 1 thread
        # wgmma_async -> cuda_generic_and_async_proxy; 128 threads
        #
        # Can fail due to
        #   * unsupported first sync-tl
        #   * incorrect second sync-tl given supported first sync-tl
        #   * incorrect collective unit given supported first sync-tl
        assert not usage.has_back_array()

        solitary = True
        L1 = usage.get_front_arrive().sync_tl
        L2 = usage.get_front_await().sync_tl

        def check_coll_unit(coll_tiling, action_name, coll_unit):
            if msg := coll_tiling.unit_mismatch(coll_unit, self._coll_env):
                raise TypeError(  # XXX srcinfo should be of location
                    f"{usage.get_srcinfo()}: {action_name} of CudaCommitGroup "
                    f"{name} with Arrive({L1}) "
                    f"expects collective unit {coll_unit}: {msg}"
                )

        def check_L2_coll_unit(expect_L2, coll_unit):
            if not expect_L2.implements_second(L2):
                raise TypeError(
                    f"{usage.get_srcinfo()}: commit group "
                    f"{name} with Arrive({L1}) "
                    f"expects Await({expect_L2}), "
                    f"not {L2} (wrong second sync-tl)"
                )
            check_coll_unit(coll_tilings.get_front_arrive(), "Arrive", coll_unit)
            check_coll_unit(coll_tilings.get_front_await(), "Await", coll_unit)

        if timelines.Sm80_cp_async.implements_first(L1):
            # sm_80 non-bulk cp.async
            check_L2_coll_unit(timelines.Sm80_generic, cuda_thread)
            lowered = LoweredBarrier(solitary, LoweredBarrierType.Sm80_commit_group)
            arrive_instr = "cp.async.commit_group"
            await_instr = "cp.async.wait_group"
        elif timelines.tma_to_gmem_async.implements_first(L1):
            # sm_90a bulk cp.async SMEM->GMEM
            check_L2_coll_unit(timelines.cuda_generic_and_async_proxy, cuda_thread)
            lowered = LoweredBarrier(
                solitary, LoweredBarrierType.tma_to_gmem_commit_group
            )
            arrive_instr = "cp.async.bulk.commit_group"
            await_instr = "cp.async.bulk.wait_group"
        elif timelines.wgmma_async.implements_first(L1):
            # sm_90a wgmma; note unit is now warpgroup and not a single thread.
            check_L2_coll_unit(timelines.cuda_generic_and_async_proxy, cuda_warpgroup)
            lowered = LoweredBarrier(solitary, LoweredBarrierType.wgmma_commit_group)
            arrive_instr = "wgmma.commit_group.sync.aligned"
            await_instr = "wgmma.wait_group.sync.aligned"
        else:
            raise TypeError(
                f"{usage.get_srcinfo()}: {name} @ CudaCommitGroup "
                f"does not support Arrive({L1}) (wrong first sync-tl)"
            )

        def codegen(s: LoopIR.SyncStmt):
            sync_type = s.sync_type
            assert sync_type.is_split()
            if sync_type.is_arrive():
                assert (
                    sync_type.N == 1
                ), "should have been flagged by BarrierUsageAnalysis"
                lowered_arrive = simple_ptx_c_lines(arrive_instr)
                return lowered_arrive
            else:
                assert (
                    sync_type.N >= 0
                ), "should have been flagged by BarrierUsageAnalysis"
                return simple_ptx_c_lines(await_instr, sync_type.N)

        lowered.codegen_sync_stmt = codegen
        self.lowered[name] = lowered

    def generate_SyncState_body(self):
        lines = []
        for line in self.SyncState_lines:
            if line:
                lines.append("    " + line)
        return "\n".join(lines)

    def generate_device_setup(self):
        """Generate body of exo_deviceSetup function.

        Return (body text, SMEM bytes needed)

        """
        # fmt: off
        lines = []
        offset = 0
        if self._mbarrier_pairs:
            lines.append("    if (threadIdx.x == 0) {")
            for mbarrier_count, arrive_count in self._mbarrier_pairs:
                lines.append(f"      for (int i = 0; i < {mbarrier_count}; ++i) {{")
                ptx = InlinePtxGen("mbarrier.init.shared::cta.b64 #0#;", volatile=True)
                ptx.add_arg(f"exo_smem + {8*offset} + 8*i", constraint="smem", log_as="bits")
                ptx.add_arg(arrive_count, constraint="n", log_as="bits")
                lines.extend(ptx.as_c_lines(py_format=False, tab="          "))
                lines.append(f"      }}")
                offset += mbarrier_count
            if self._uses_async_proxy:
                lines.extend(simple_ptx_c_lines("fence.proxy.async", tab="      "))
            lines.append("    }")
            if self._clusterDim() > 1:
                lines.extend(simple_ptx_c_lines("barrier.cluster.arrive.aligned", tab="    "))
                lines.extend(simple_ptx_c_lines("barrier.cluster.wait.aligned", tab="    "))
            else:
                lines.extend(simple_ptx_c_lines("barrier.cta.sync", 0, tab="    "))
        # HACK: align mbarriers to 128 bytes for now
        assert offset == self.mbarrier_count
        smem_bytes = 128 * ((offset + 15) // 16)
        if lines:
            return "\n".join(lines), smem_bytes
        else:
            return "  // No mbarriers used", 0
        # fmt: on

    def _assign_suffix(self, barrier_name):
        assert isinstance(barrier_name, Sym)
        count = self._sym_counters.get(barrier_name, 0)
        self._sym_counters[barrier_name] = count + 1
        suffix = str(count)
        return suffix

    def _blockDim(self):
        return self._coll_env[blockDim_param]

    def _clusterDim(self):
        return self._coll_env[clusterDim_param]
