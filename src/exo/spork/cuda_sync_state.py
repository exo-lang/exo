# Compiler: Generate exo_SyncState for CUDA C++ device functions, and
# lowered Arrive/Await/Fence statements.

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Optional, Type, List, Tuple

from ..core.prelude import Sym, SrcInfo
from ..core.LoopIR import LoopIR

from . import actor_kinds
from .actor_kinds import ActorKind
from .barrier_usage import BarrierUsage, SyncInfo
from .coll_algebra import (
    CollParam,
    CollUnit,
    clusterDim_param,
    blockDim_param,
    CollIndexExpr,
    CollTiling,
    CollLoweringAdvice,
    cuda_thread,
    cuda_warp,
    cuda_warpgroup,
    cuda_cta_in_cluster,
)
from .cuda_memory import (
    CudaCommitGroup,
    CudaMbarrier,
)
from .distributed_memory import DistributedAllocState, ThreadIter
from .excut import InlinePtxGen, simple_ptx_c_lines
from .sync_types import SyncType


class LoweredBarrierType(Enum):
    garden_variety_fence = auto()
    wgmma_fence = auto()
    mbarrier = auto()
    Sm80_commit_group = auto()
    tma_to_gmem_commit_group = auto()
    wgmma_commit_group = auto()


@dataclass(slots=True)
class LoweredPrologueSync:
    actor_kind: ActorKind
    lines: List[str]


@dataclass(slots=True)
class LoweredEpilogueSync:
    actor_kind: ActorKind
    lines: List[str]


@dataclass(slots=True)
class CudaLoweredBarrier:
    # If set, two barrier objects of the same type_enum (in Exo code)
    # cannot be live at the same time.
    solitary: bool

    # More specific than the BarrierType (specialized by actor kind).
    # Also applies to Fence(...), which has no associated barrier object.
    type_enum: LoweredBarrierType

    # Lower SyncStmt to lines of C++ code (List[str])
    # (you may assume the SyncStmt uses this lowered barrier)
    # If the sync must appear as the prologue/epilogue sync
    # of a CudaAsync(A) block, warp the lines with
    # LoweredPrologueSync(A, lines) or LoweredEpilogueSync(A, lines)
    codegen: Callable[[LoopIR.SyncStmt], object] = None


@dataclass(slots=True)
class SyncStateBuilder:
    _coll_env: Dict[CollParam, int]

    # CudaLoweredBarrier for each barrier lowered, indexed by name
    lowered: Dict[Sym, CudaLoweredBarrier] = field(default_factory=dict)

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
        usage: BarrierUsage,
        coll_tilings: DistributedAllocState,
        thread_iters: Dict[Sym, ThreadIter],
    ):
        for info in (
            usage.Arrive,
            usage.Await,
            usage.ReverseArrive,
            usage.ReverseAwait,
        ):
            if info is not None:
                if not actor_kinds.cuda_async_proxy.full_signatures.isdisjoint(
                    info.actor_kind.full_signatures
                ):
                    self._uses_async_proxy = True

        srcinfo = usage.decl_stmt.srcinfo
        barrier_type = usage.barrier_type
        suffix = self._assign_suffix(name)
        if usage.is_fence():
            if usage.Arrive.actor_kind == actor_kinds.wgmma_fence_1:
                self.add_wgmma_fence(name, usage, coll_tilings, thread_iters, suffix)
            else:
                self.add_garden_variety_fence(
                    name, usage, coll_tilings, thread_iters, suffix
                )
        elif issubclass(barrier_type, CudaMbarrier):
            self.add_mbarrier(name, usage, coll_tilings, thread_iters, suffix)
        elif issubclass(barrier_type, CudaCommitGroup):
            self.add_commit_group(name, usage, coll_tilings, thread_iters, suffix)
        else:
            raise TypeError(
                f"{srcinfo}: {barrier_type.__name__} "
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
        Arrive = usage.Arrive
        Await = usage.Await
        A1 = Arrive.actor_kind
        A2 = Await.actor_kind
        srcinfo = Arrive.get_srcinfo()
        assert A1 == actor_kinds.wgmma_fence_1
        if A2 != actor_kinds.wgmma_fence_2:
            raise ValueError(
                f"{srcinfo}: wgmma fence needs second actor kind wgmma_fence_2"
            )

        coll_tiling = coll_tilings.Arrive
        # Should be the case for a Fence
        assert coll_tiling is coll_tilings.Await

        if msg := coll_tiling.unit_mismatch(cuda_warpgroup, self._coll_env):
            raise ValueError(
                f"{srcinfo}: wgmma fence must be executed by a warpgroup: {msg}"
            )

        lowered = CudaLoweredBarrier(False, LoweredBarrierType.wgmma_fence)
        lowered.codegen = lambda _: LoweredPrologueSync(
            actor_kinds.wgmma_async, simple_ptx_c_lines("wgmma.fence.sync.aligned")
        )
        self.lowered[name] = lowered

    def add_garden_variety_fence(
        self,
        name: Sym,
        usage: BarrierUsage,
        coll_tilings: DistributedAllocState,
        thread_iters: Dict[Sym, ThreadIter],
        suffix: str,
    ):
        """Do up to 3 things

        - wait_all if first actor kind includes Sm80_cp_async
        - barrier arrive/await if more than 1 thread, or special exception (*)
        - fence.proxy.async if second actor kinds includes any async proxy

        (*) special exception, if thread collective is a warpgroup and
        the second actor kind only includes wgmma_async, we can elide
        the barrier. This relies on wgmma_async not being V1-transitive.

        """

        Arrive = usage.Arrive
        Await = usage.Await
        A1 = Arrive.actor_kind
        A2 = Await.actor_kind
        srcinfo = usage.get_srcinfo()
        clusterDim = self._clusterDim()

        lowered = CudaLoweredBarrier(False, LoweredBarrierType.garden_variety_fence)
        lines = []

        # Insert wait for sm_80 cp.async if needed.
        if actor_kinds.cuda_classic.implements_first(A1):
            pass
        elif actor_kinds.Sm80_generic.implements_first(A1):
            lines += simple_ptx_c_lines("cp.async.wait_all")
        else:
            raise ValueError(
                f"{srcinfo}: Fence first actor kind "
                f"{A1} not supported (we allow Sm80_generic)"
            )

        coll_tiling = coll_tilings.Arrive
        # Should be the case for a Fence
        assert coll_tiling is coll_tilings.Await

        cta_count = self._get_cta_count(coll_tiling, srcinfo)
        threads = coll_tiling.box_num_threads()
        n_warps = threads // 32
        if threads != 1:
            if threads % 32 != 0:
                raise ValueError(
                    f"{srcinfo}: Fence expects convergent warps "
                    f"(threads = {threads} is not divisible by 32)"
                )
            unit = n_warps * cuda_warp
            if msg := coll_tiling.unit_mismatch(unit, self._coll_env):
                raise ValueError(f"{srcinfo}: Fence expects convergent warps: {msg}")

        # Insert cross-thread sync if needed
        assert not actor_kinds.wgmma_async.V1_transitive
        wgmma_special_case = actor_kinds.wgmma_async.implements_second(
            A2
        ) and not coll_tiling.unit_mismatch(cuda_warpgroup, self._coll_env)
        if n_warps > 0 and not wgmma_special_case:
            if cta_count == 1:
                if not coll_tiling.unit_mismatch(cuda_cta_in_cluster, self._coll_env):
                    # We need to use barrier.cta.sync, not bar or syncthreads
                    # due to divergent control flow in "full CTA" code
                    # if there's [named] warp specialization.
                    lines.extend(simple_ptx_c_lines("barrier.cta.sync", 0))
                elif n_warps == 1:
                    lines.append("__syncwarp();")
                else:
                    raise NotImplementedError(
                        "TODO Fence lowering other than warp/CTA/cluster"
                    )
            elif cta_count == clusterDim:
                if msg := coll_tiling.unit_mismatch(
                    cta_count * cuda_cta_in_cluster, self._coll_env
                ):
                    raise ValueError(
                        f"{srcinfo}: expected full cluster " f"or only 1 CTA: {msg}"
                    )
                else:
                    lines.extend(simple_ptx_c_lines("barrier.cluster.arrive.aligned"))
                    lines.extend(simple_ptx_c_lines("barrier.cluster.wait.aligned"))
            else:
                raise ValueError(
                    f"{srcinfo}: {cta_count}/{clusterDim} CTAs in cluster active for thread collective for Fence; must have 1 or all"
                )

        # Insert fence.proxy.async if needed
        if actor_kinds.Sm80_generic.implements_second(A2):
            pass
        elif actor_kinds.cuda_generic_and_async_proxy.implements_second(A2):
            lines.extend(simple_ptx_c_lines("fence.proxy.async"))
        else:
            raise ValueError(
                f"{srcinfo}: Fence second actor kind {A2} not "
                f"supported (at most CUDA generic+async proxy)"
            )
        lowered.codegen = lambda _: lines
        self.lowered[name] = lowered

    def add_mbarrier(
        self,
        name: Sym,
        usage: BarrierUsage,
        coll_tilings: DistributedAllocState,
        thread_iters: Dict[Sym, ThreadIter],
        suffix: str,
    ):
        lowered = CudaLoweredBarrier(False, LoweredBarrierType.mbarrier)
        mbarrier_offset = self.mbarrier_count
        nm_suffix = f"{suffix}_{name}"

        # Calculate the size of the ring buffer (number of mbarriers)
        def n_skips(info: SyncInfo):
            assert info.min_N == info.max_N
            return ~info.min_N

        forward_skips = n_skips(usage.Await)
        if usage.has_reverse():
            reverse_skips = n_skips(usage.ReverseAwait)
            ring = forward_skips + reverse_skips
        else:
            ring = forward_skips + 1
        if ring == 0:
            raise ValueError(
                f"{usage.get_srcinfo()}: {name} must have some await with nonzero skips (e.g. set N = ~1)"
            )

        # Number of physical mbarriers is slice_count * ring, where
        # slice_count is the number of logical Exo barrier objects per CTA
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
        def mbarrier_to_u32(lines, is_reverse, ringidx):
            byte_offset = 8 * (mbarrier_offset + (ring * slice_count) if is_reverse else mbarrier_offset)
            idx = f"(slice * {ring} + {ringidx})"
            lines.append(f"  const auto mbarrier_u32 = exo_smemU32(exo_smem + {byte_offset} + 8*{idx});")

        def generate_arrive(is_reverse):
            r = "Reverse" if is_reverse else ""
            info = usage.ReverseArrive if is_reverse else usage.Arrive
            actor_kind = info.actor_kind

            if actor_kinds.Sm80_cp_async.implements_first(actor_kind):
                is_Sm80_cp_async = True
                is_tma = False
            elif actor_kinds.cuda_classic.implements_first(actor_kind):
                is_Sm80_cp_async = False
                is_tma = False
            elif actor_kinds.tma_to_smem_async.implements_first(actor_kind):
                is_Sm80_cp_async = False
                is_tma = True
            else:
                raise ValueError(
                    f"{info.get_srcinfo()}: mbarrier Arrive actor kind {actor_kind} "
                    f"not supported: need cuda_classic, Sm80_cp_async, or tma_to_smem_async")

            lines = self.SyncState_lines
            idx = f"{r}ArriveIdx{nm_suffix}"
            if ring_bits > 0:
                lines.append(f"unsigned {idx} : {ring_bits} = 0;")
            else:
                lines.append(f"static constexpr unsigned {idx} = 0;  // Trivial size-1 ring buffer")
            lines.append(f"EXO_CUDA_INLINE uint32_t {r}Arrive{nm_suffix}(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, bool enable) {{")
            mbarrier_to_u32(lines, is_reverse, idx);
            lines.append(f"  if (enable) {{")
            # TODO cluster broadcast if needed
            ptx_format = f"// {r}Arrive{nm_suffix}\n"
            if is_Sm80_cp_async:
                ptx_format += "cp.async.mbarrier.arrive.noinc.shared::cta.b64 #0#;"
            else:
                ptx_format += "mbarrier.arrive.shared::cta.b64 _, #0#;"
            ptx = InlinePtxGen(ptx_format, volatile=True)
            ptx.add_arg("mbarrier_u32", constraint="r", log_as="bits", brackets=True)
            lines.extend(ptx.as_c_lines(py_format=False, tab="    "))
            if ring_bits > 0:
                lines.append(f"    // Advance ring buffer state")
                lines.append(f"    {idx} = {idx} == {ring - 1} ? 0 : {idx} + 1;")
            lines.append(f"  }}")
            lines.append(f"  return mbarrier_u32;")
            lines.append(f"}}")
            return is_tma

        def generate_await(is_reverse, A1):
            r = "Reverse" if is_reverse else ""
            info = usage.ReverseAwait if is_reverse else usage.Await
            A2 = info.actor_kind

            if actor_kinds.cuda_async_proxy_wgmma.implements_first(A1):
                # proxy fence always elided if first actor kind includes only
                # async proxy and wgmma register access.
                proxy_fence = False
            elif actor_kinds.Sm80_generic.implements_second(A2):
                proxy_fence = False
            elif actor_kinds.cuda_generic_and_async_proxy.implements_second(A2):
                proxy_fence = True
            else:
                if A2 == actor_kinds.wgmma_async:
                    remark = "consider wgmma_async_smem"
                else:
                    remark = "at most CUDA generic+async proxy"
                raise ValueError(
                    f"{info.get_srcinfo()}: mbarrier Await actor kind {A2} "
                    f"not supported ({remark})")

            lines = self.SyncState_lines
            # If we have ReverseAwait/ReverseArrive, the mbarriers for them
            # are allocated right after those for Arrive/Await
            offset = mbarrier_offset + ring if is_reverse else mbarrier_offset
            idx = f"{r}AwaitIdx{nm_suffix}"
            skips = f"{r}Skips{nm_suffix}"
            parity_bits = f"{r}Parity{nm_suffix}"
            n_skips = reverse_skips if is_reverse else forward_skips
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
                lines.append(f"EXO_CUDA_INLINE void {r}Await{nm_suffix}(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, int initial_skips = 0) {{")
                mbarrier_to_u32(lines, is_reverse, idx)
                lines.append(f"  const bool enable = {skips} >= initial_skips;")
            else:
                lines.append(f"EXO_CUDA_INLINE void {r}Await{nm_suffix}(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice) {{")
                mbarrier_to_u32(lines, is_reverse, idx)
                lines.append(f"  const bool enable = true;")
            comment = f"// {r}Await{nm_suffix}"
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
                lines.append(f'    // Needed for first actor kind {A1}; second actor kind {A2}')
                lines.extend(simple_ptx_c_lines("fence.proxy.async", tab="    "))
            lines.append(f"  }}")
            if enable_skips:
                lines.append(f"  else {{")
                lines.append(f"    // {r}Await({name}) returns without waiting for mbarrier first <initial_skips> times")
                lines.append(f"    {skips}++;")
                lines.append(f"  }}")
            lines.append(f"}}")

        # Reserve mbarriers in mbarrier allocator
        RS = ring * slice_count
        lines = self.SyncState_lines
        arrive_count = coll_tilings.Arrive.box_num_threads()
        self._mbarrier_pairs.append((RS, arrive_count))
        self.mbarrier_count += RS
        lines.append(f"// {name}: barrier @ CudaMbarrier, ring={ring}, slice_count={slice_count}")
        lines.append(f"// (forward) mbarriers [{mbarrier_offset}, {mbarrier_offset + RS}]; "
                     f"arrive_count={arrive_count}")

        if usage.has_reverse():
            arrive_count = coll_tilings.ReverseArrive.box_num_threads()
            self._mbarrier_pairs.append((RS, arrive_count))
            self.mbarrier_count += RS
            lines.append(f"// (reverse) mbarriers [{mbarrier_offset + RS}, {mbarrier_offset + RS * 2}]; "
                         f"arrive_count={arrive_count}")

        # Generate Arrive and Await syntax
        # {Reverse}Awaits must be aware with the actor kind
        # of the matched {Reverse}Arrive
        Arrive_is_tma = generate_arrive(False)
        generate_await(False, usage.Arrive.actor_kind)
        if usage.has_reverse():
            ReverseArrive_is_tma = generate_arrive(True)
            generate_await(True, usage.ReverseArrive.actor_kind)

        # Arrive/Await lowers to call to generated exo_syncState member function.
        # We also record mbarriers to initialize, first those for Arrive/Await,
        # then those for ReverseArrive/ReverseAwait.
        # Finally, for arrives with actor kind tma_to_smem_async, we need to enforce
        # that it is an epilogue sync for tx-count to work correctly.
        Arrive_txt = f"Arrive{nm_suffix}(exo_smem, exo_excutLog"
        def codegen(s: LoopIR.SyncStmt):
            sync_type = s.sync_type
            slice = coll_tilings.codegen_slices_to_root(self._blockDim(), thread_iters, [e.name for e in s.idx])
            r = "Reverse" if sync_type.is_reversed else ""
            assert sync_type.is_split()
            if sync_type.is_arrive():
                assert sync_type.N <= 1, "TODO implement me"
                result = [f"exo_syncState.{r}{Arrive_txt}, {slice}, {sync_type.N});"]
                is_tma = ReverseArrive_is_tma if sync_type.is_reversed else Arrive_is_tma
                if is_tma:
                    # See also codegen_exo_tma_mbarrier if you change this!
                    result = LoweredEpilogueSync(actor_kinds.tma_to_smem_async, result)
                return result
            else:
                skips_arg = ""
                if skips := ~sync_type.N:
                    assert skips > 0, "should have been caught earlier"
                    skips_arg = f", {skips}"
                return [f"exo_syncState.{r}Await{nm_suffix}(exo_smem, exo_excutLog, {slice}{skips_arg});"]
        lowered.codegen = codegen
        self.lowered[name] = lowered
        # fmt: on

    def codegen_exo_tma_mbarrier(self, _arrive: LoopIR.SyncStmt):
        """Special-case code for generating the C++ def for exo_tma_mbarrier

        For the mbarrier Arrive at the end of a CudaAsync(tma_to_smem_async)
        block, we need to make the mbarrier used for the upcoming arrive
        available in the entire block as `uint32_t exo_tma_mbarrier`.
        This is needed to make expect-tx work.

        Given the LoopIR node for said Arrive, return C++ syntax for declaring
        exo_tma_smem. The outer compiler (cuda_backend) should paste this
        at the beginning of the CudaAsync(tma_to_smem_async) block.

        """
        assert isinstance(_arrive, LoopIR.SyncStmt)
        sync_type = _arrive.sync_type
        assert sync_type.is_arrive()
        assert sync_type.first_actor_kind == actor_kinds.tma_to_smem_async
        lowered_mbarrier = self.lowered[_arrive.name]
        # Read-only access to the mbarrier by lowering an arrive with N=0
        # (arrive-count 0). This is pretty hacky; if we have more need for
        # such custom barrier lowering, we may want a "proper" interface.
        arrive0_sync_type = _arrive.sync_type.copy()
        arrive0_sync_type.N = 0
        arrive0 = _arrive.update(sync_type=arrive0_sync_type)
        mbarrier_txt = lowered_mbarrier.codegen(arrive0).lines[0]
        assert mbarrier_txt[-1] == ";"
        c_alias = f"const uint32_t exo_tma_mbarrier = {mbarrier_txt}"
        return c_alias

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
        #   * unsupported first actor kind
        #   * incorrect second actor kind given supported first actor kind
        #   * incorrect collective unit given supported first actor kind
        assert usage.ReverseAwait is None
        assert usage.ReverseArrive is None

        solitary = True
        A1 = usage.Arrive.actor_kind
        A2 = usage.Await.actor_kind
        is_wgmma = False

        def check_coll_unit(coll_tiling, action_name, coll_unit):
            if msg := coll_tiling.unit_mismatch(coll_unit, self._coll_env):
                raise TypeError(  # XXX srcinfo should be of location
                    f"{usage.get_srcinfo()}: {action_name} of commit group "
                    f"{name} with Arrive({A1}) "
                    f"expects collective unit {coll_unit}: {msg}"
                )

        def check_A2_coll_unit(expect_A2, coll_unit):
            if not expect_A2.implements_second(A2):
                raise TypeError(
                    f"{usage.get_srcinfo()}: commit group "
                    f"{name} with Arrive({A1}) "
                    f"expects Await({expect_A2}), "
                    f"not {A2} (wrong second actor kind)"
                )
            check_coll_unit(coll_tilings.Arrive, "Arrive", coll_unit)
            check_coll_unit(coll_tilings.Await, "Await", coll_unit)

        if actor_kinds.Sm80_cp_async.implements_first(A1):
            # sm_80 non-bulk cp.async
            check_A2_coll_unit(actor_kinds.Sm80_generic, cuda_thread)
            lowered = CudaLoweredBarrier(solitary, LoweredBarrierType.Sm80_commit_group)
            arrive_instr = "cp.async.commit_group"
            await_instr = "cp.async.wait_group"
        elif actor_kinds.tma_to_gmem_async.implements_first(A1):
            # sm_90a bulk cp.async SMEM->GMEM
            check_A2_coll_unit(actor_kinds.cuda_generic_and_async_proxy, cuda_thread)
            lowered = CudaLoweredBarrier(
                solitary, LoweredBarrierType.tma_to_gmem_commit_group
            )
            arrive_instr = "cp.async.bulk.commit_group"
            await_instr = "cp.async.bulk.wait_group"
        elif actor_kinds.wgmma_async.implements_first(A1):
            # sm_90a wgmma; note unit is now warpgroup and not a single thread;
            # also enforce that this is an epilogue sync of CudaAsync(wgmma_async).
            check_A2_coll_unit(actor_kinds.cuda_generic_and_async_proxy, cuda_warpgroup)
            lowered = CudaLoweredBarrier(
                solitary, LoweredBarrierType.wgmma_commit_group
            )
            arrive_instr = "wgmma.commit_group.sync.aligned"
            await_instr = "wgmma.wait_group.sync.aligned"
            is_wgmma = True
        else:
            raise TypeError(
                f"{usage.get_srcinfo()}: {name} : "
                f"cuda_commit_group does not support "
                f"Arrive({A1}) (wrong first actor kind)"
            )

        def codegen(s: LoopIR.SyncStmt):
            sync_type = s.sync_type
            assert sync_type.is_split()
            assert not sync_type.is_reversed
            if sync_type.is_arrive():
                if sync_type.N != 1:
                    raise ValueError("Expect N=1 for Arrive of commit group")
                lowered_arrive = simple_ptx_c_lines(arrive_instr)
                if is_wgmma:
                    lowered_arrive = LoweredEpilogueSync(
                        actor_kinds.wgmma_async, lowered_arrive
                    )
                return lowered_arrive
            else:
                if sync_type.N < 0:
                    raise ValueError("Expect N>=0 for Await of commit group")
                return simple_ptx_c_lines(await_instr, sync_type.N)

        lowered.codegen = codegen
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

    def _get_cta_count(self, coll_tiling: CollTiling, srcinfo: SrcInfo):
        assert isinstance(srcinfo, SrcInfo)
        clusterDim = self._clusterDim()
        if clusterDim == 1:
            return 1
        else:
            # Only if the clusterDim is not 1 can we rely on the left-most
            # dimension of the domain to correspond to the CTA-in-cluster axis.
            domain = coll_tiling.full_domain
            box = coll_tiling.box
            assert len(domain) == len(box)
            if domain[0] != clusterDim:
                # Unlikely error, only occurs of the user defines their own
                # unit splitting the cluster dimension of the coll tiling.
                raise TypeError(f"{srcinfo}: could not deduce cluster count")
            return box[0]
