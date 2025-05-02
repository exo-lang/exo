# Compiler: Generate exo_SyncState for CUDA C++ device functions, and
# lowered Arrive/Await/Fence statements.

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Optional, Type, List, Tuple

from ..core.prelude import Sym, SrcInfo

from . import actor_kinds
from .actor_kinds import ActorKind
from .barrier_usage import BarrierUsage
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
from .distributed_memory import DistributedAllocState
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
    codegen: Callable[[SyncType], object] = None

    # If applicable, syntax for getting the mbarrier used for
    # an Arrive/ReverseArrive
    c_Arrive_mbarrier: str = None
    c_ReverseArrive_mbarrier: str = None


@dataclass(slots=True)
class SyncStateBuilder:
    _coll_env: Dict[CollParam, int]

    # CudaLoweredBarrier for each barrier lowered, indexed by name
    lowered: Dict[Sym, CudaLoweredBarrier] = field(default_factory=dict)

    # tuples (mbarrier_count, arrive_count)
    # to initialize in SMEM, e.g. (8, 64), (2, 384) means initialize an
    # array of 10 mbarriers in SMEM with the first 8 having
    # arrive_count=64, last 2 arrive_count=384
    mbarrier_pairs: List[Tuple[int, int]] = field(default_factory=list)

    # C++ lines to join into exo_SyncState struct
    SyncState_lines: List[str] = field(default_factory=list)

    # Need to assign a name unique-ifying suffix for each barrier
    # This is different than what the main LoopIR->C compiler does because the
    # name needs to be unique throughout the full device function, i.e. it's not
    # enough to be unique just within the barrier's scope in Exo object code.
    _sym_counters: Dict[Sym, int] = field(default_factory=dict)

    _uses_async_proxy: bool = False

    def add_barrier(
        self, name: Sym, usage: BarrierUsage, coll_tilings: DistributedAllocState
    ):
        srcinfo = usage.decl_stmt.srcinfo
        barrier_type = usage.barrier_type
        suffix = self._assign_suffix(name)
        if usage.is_fence():
            if usage.Arrive.actor_kind == actor_kinds.wgmma_fence_1:
                self.add_wgmma_fence(name, usage, coll_tilings, suffix)
            else:
                self.add_garden_variety_fence(name, usage, coll_tilings, suffix)
        elif issubclass(barrier_type, CudaMbarrier):
            self.add_mbarrier(name, usage, coll_tilings, suffix)
        elif issubclass(barrier_type, CudaCommitGroup):
            self.add_commit_group(name, usage, coll_tilings, suffix)
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
            actor_kinds.wgmma_async, ['asm("wgmma.fence.sync.aligned;");']
        )
        self.lowered[name] = lowered

    def add_garden_variety_fence(
        self,
        name: Sym,
        usage: BarrierUsage,
        coll_tilings: DistributedAllocState,
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
            lines.append('asm volatile("cp.async.wait_all;" ::);')
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
                    lines.append("__syncthreads();")
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
                    lines.append('asm("barrier.cluster.arrive.aligned;"::);')
                    lines.append('asm("barrier.cluster.wait.aligned;"::);')
            else:
                raise ValueError(
                    f"{srcinfo}: {cta_count}/{clusterDim} CTAs in cluster active for thread collective for Fence; must have 1 or all"
                )

        # Insert fence.proxy.async if needed
        if actor_kinds.Sm80_generic.implements_second(A2):
            pass
        elif actor_kinds.cuda_generic_and_async_proxy.implements_second(A2):
            lines.append('asm("fence.proxy.async;");')
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
        suffix: str,
    ):
        lowered = CudaLoweredBarrier(False, LoweredBarrierType.mbarrier)
        mbarrier_offset = sum(c for c, _ in self.mbarrier_pairs)  # O(n^2)
        nm_suffix = f"{suffix}_{name}"

        # Calculate the size of the ring buffer (number of mbarriers)
        def max_skips(sync_stmts):
            return max(~s.sync_type.N for s in sync_stmts)

        max_await_skips = max_skips(usage.Await.stmts)
        if usage.has_reverse():
            max_reverse_await_skips = max_skips(usage.ReverseAwait.stmts)
            ring = max_await_skips + max_reverse_await_skips
        else:
            ring = max_await_skips + 1
        if ring == 0:
            raise ValueError(
                f"{usage.get_srcinfo()}: {name} must have some await with nonzero skips (e.g. set N = ~1)"
            )

        # Need to be able to store values 0 through (ring-1)
        ring_bits = (ring - 1).bit_length()
        # Need to be able to count 0 to ring (inclusive) skips.
        # This value will not be used if skipping is not actually enabled.
        skip_bits = ring.bit_length()

        # black formatting will ruin the readability of the generated C++ code below
        # fmt: off
        def mbarrier_to_u32(lines, is_reverse, idx):
            byte_offset = 8 * (mbarrier_offset + ring if is_reverse else mbarrier_offset)
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
            lines.append(f"__device__ __forceinline__ uint32_t {r}Arrive{nm_suffix}(char* exo_smem, bool enable) {{")
            mbarrier_to_u32(lines, is_reverse, idx);
            lines.append(f"  if (enable) {{")
            # TODO cluster broadcast if needed
            comment = f"// {r}Arrive{nm_suffix}\\n\\t"
            if is_Sm80_cp_async:
                lines.append(f'    asm("{comment}cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(mbarrier_u32));');
            else:
                lines.append(f'    asm("{comment}mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mbarrier_u32));');
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
            max_skips = max_reverse_await_skips if is_reverse else max_await_skips
            enable_skips = max_skips != 0

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
                lines.append(f"__device__ __forceinline__ void {r}Await{nm_suffix}(char* exo_smem, int initial_skips = 0) {{")
                mbarrier_to_u32(lines, is_reverse, idx)
                lines.append(f"  const bool enable = {skips} >= initial_skips;")
            else:
                lines.append(f"__device__ __forceinline__ void {r}Await{nm_suffix}(char* exo_smem) {{")
                mbarrier_to_u32(lines, is_reverse, idx)
                lines.append(f"  const bool enable = true;")
            comment = f"// {r}Await{nm_suffix}\\n\\t"
            lines.append(f"  if (enable) {{")
            # sm_90 needed for try_wait
            test_or_try = "try" if self._uses_async_proxy else "test"  # XXX
            lines.append(f"    // Wait for mbarrier ... PTX loop needed for this")
            lines.append(f'    asm volatile("{comment}{{.reg.pred P1; EXO_BEFORE_WAIT: mbarrier.{test_or_try}_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1; @P1 bra.uni EXO_WAIT_DONE; bra.uni EXO_BEFORE_WAIT; EXO_WAIT_DONE: }}"::')
            lines.append(f'        "r"(mbarrier_u32), "r"(1u & {parity_bits} >> {idx}));')
            lines.append(f"    // Flip parity")
            lines.append(f"    {parity_bits} ^= 1u << {idx};")
            if ring_bits > 0:
                lines.append(f"    // Advance ring buffer state")
                lines.append(f"    {idx} = {idx} == {ring - 1} ? 0 : {idx} + 1;")
            if proxy_fence:
                lines.append(f'    // Needed for first actor kind {A1}; second actor kind {A2}')
                lines.append(f'    asm("fence.proxy.async;");')
            lines.append(f"  }}")
            if enable_skips:
                lines.append(f"  else {{")
                lines.append(f"    // {r}Await({name}) returns without waiting for mbarrier first <initial_skips> times")
                lines.append(f"    {skips}++;")
                lines.append(f"  }}")
            lines.append(f"}}")

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
        Arrive_txt = f"Arrive{nm_suffix}(exo_smem, "
        def codegen(sync_type: SyncType):
            r = "Reverse" if sync_type.is_reversed else ""
            assert sync_type.is_split()
            if sync_type.is_arrive():
                assert sync_type.N == 1, "TODO implement me"
                result = [f"exo_syncState.{r}{Arrive_txt}{sync_type.N});"]
                is_tma = ReverseArrive_is_tma if sync_type.is_reversed else Arrive_is_tma
                if is_tma:
                    result = LoweredEpilogueSync(actor_kinds.tma_to_smem_async, result)
                return result
            else:
                skips_arg = ""
                if skips := ~sync_type.N:
                    assert skips > 0, "should have been caught earlier"
                    skips_arg = f", {skips}"
                return [f"exo_syncState.{r}Await{nm_suffix}(exo_smem{skips_arg});"]
        lowered.codegen = codegen
        lowered.c_Arrive_mbarrier = f"exo_syncState.{Arrive_txt}0)"
        arrive_count = coll_tilings.Arrive.box_num_threads()
        self.mbarrier_pairs.append((ring, arrive_count))

        if usage.has_reverse():
            lowered.c_ReverseArrive_mbarrier = f"exo_syncState.Reverse{Arrive_txt}0)"
            arrive_count = coll_tilings.ReverseArrive.box_num_threads()
            self.mbarrier_pairs.append((ring, arrive_count))
        self.lowered[name] = lowered
        # fmt: on

    def add_commit_group(
        self,
        name: Sym,
        usage: BarrierUsage,
        coll_tilings: DistributedAllocState,
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
            lowered_arrive = ['asm("cp.async.commit_group;");']
            await_fmt = 'asm("cp.async.wait_group {N};");'
        elif actor_kinds.tma_to_gmem_async.implements_first(A1):
            # sm_90a bulk cp.async SMEM->GMEM
            check_A2_coll_unit(actor_kinds.cuda_generic_and_async_proxy, cuda_thread)
            lowered = CudaLoweredBarrier(
                solitary, LoweredBarrierType.tma_to_gmem_commit_group
            )
            lowered_arrive = ['asm("cp.async.bulk.commit_group;");']
            await_fmt = 'asm("cp.async.bulk.wait_group {N};");'
        elif actor_kinds.wgmma_async.implements_first(A1):
            # sm_90a wgmma; note unit is now warpgroup and not a single thread;
            # also enforce that this is an epilogue sync of CudaAsync(wgmma_async).
            check_A2_coll_unit(actor_kinds.cuda_generic_and_async_proxy, cuda_warpgroup)
            lowered = CudaLoweredBarrier(
                solitary, LoweredBarrierType.wgmma_commit_group
            )
            lowered_arrive = LoweredEpilogueSync(
                actor_kinds.wgmma_async, ['asm("wgmma.commit_group.sync.aligned;");']
            )
            await_fmt = 'asm("wgmma.wait_group.sync.aligned {N};");'
            is_wgmma = True
        else:
            raise TypeError(
                f"{usage.get_srcinfo()}: {name} : "
                f"cuda_commit_group does not support "
                f"Arrive({A1}) (wrong first actor kind)"
            )

        def codegen(sync_type):
            assert sync_type.is_split()
            assert not sync_type.is_reversed
            if sync_type.is_arrive():
                if sync_type.N != 1:
                    raise ValueError("Expect N=1 for Arrive of commit group")
                return lowered_arrive
            else:
                if sync_type.N < 0:
                    raise ValueError("Expect N>=0 for Await of commit group")
                return [await_fmt.format(N=sync_type.N)]

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
        lines.append("    if (threadIdx.x == 0) {")
        lines.append("      const auto mbarrier_u32 = exo_smemU32(exo_smem);")
        offset = 0
        for mbarrier_count, arrive_count in self.mbarrier_pairs:
            lines.append(f"      for (int i = 0; i < {mbarrier_count}; ++i) {{")
            lines.append(f'        asm volatile("mbarrier.init.shared::cta.b64 [%0], {arrive_count};"::')
            lines.append(f'          "r"(mbarrier_u32 + {8*offset} + 8*i));')
            lines.append(f"      }}")
            offset += mbarrier_count
        if self._uses_async_proxy:
            lines.append('      asm("fence.proxy.async;");')
        lines.append("    }")
        if self._clusterDim() > 1:
            lines.append('    asm("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;\n"::);')
        else:
            lines.append('    __syncthreads();')
        # HACK: align mbarriers to 128 bytes for now
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
