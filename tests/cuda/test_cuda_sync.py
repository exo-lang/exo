from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np
import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *

from exo.spork import excut

from exo.core.LoopIR import get_global_debug_log


def get_remarks_golden(compiler, proc_name):
    # Fragile test utility we rarely use to get the contents
    # of the debug logs that exocc would have written out.
    # Anytime we add more logging, the golden will change.
    # We just want to force some logging code coverage.
    debug_log = get_global_debug_log()
    debug_log.write_all_impl()
    with open(str(compiler.workdir / "debug" / f"{proc_name}-analysis.py")) as f:
        return f.read()


def mkproc_wgmma_fence(
    lo=4,
    hi=12,
    unit=cuda_warpgroup,
    first_sync_tl=wgmma_fence_1,
    second_sync_tl=wgmma_fence_2,
    have_fence=True,
):
    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=384):
            for task in cuda_tasks(0, 4):
                with CudaWarps(lo, hi):
                    for wg in cuda_threads(0, 1, unit=unit):
                        Fence(first_sync_tl, second_sync_tl)

    return simplify(test_proc)


def test_wgmma_fence_positive(compiler, golden):
    compiler.cuda_cpu_test(mkproc_wgmma_fence, golden, sm="90a")


def test_wgmma_fence_wrong_coll_unit_size(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_wgmma_fence, unit=cuda_warp)
    assert "warpgroup" in str(exc.value)


def test_wgmma_fence_wrong_coll_unit_align(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_wgmma_fence, lo=5)
    assert "alignment" in str(exc.value)


def test_wgmma_fence_wrong_second_sync_tl(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_wgmma_fence, second_sync_tl=cuda_temporal)
    assert "wgmma_fence_2" in str(exc.value)


def mkproc_mixed_syncs(
    unit_a: CollUnit,
    unit_b: CollUnit,
    barrier_type_a,
    barrier_type_b,
    first_sync_tl_a,
    first_sync_tl_b,
    blockDim,
    clusterDim=1,
    fence_first_sync_tl=None,
    fence_second_sync_tl=cuda_in_order,
    second_sync_tl_a=cuda_in_order,
    second_sync_tl_b=cuda_in_order,
    alt_first_sync_tl_a=None,
    alt_second_sync_tl_a=None,
):
    have_fence = bool(fence_first_sync_tl)
    fence_first_sync_tl = fence_first_sync_tl or cuda_in_order
    alt_first_sync_tl_a = alt_first_sync_tl_a or first_sync_tl_a
    alt_second_sync_tl_a = alt_second_sync_tl_a or second_sync_tl_a

    @proc
    def test_proc():
        with CudaDeviceFunction(clusterDim=clusterDim, blockDim=blockDim):
            for task in cuda_tasks(0, 1):
                barrier_a: barrier[1] @ barrier_type_a
                barrier_b: barrier[1] @ barrier_type_b
                for a in cuda_threads(0, 1, unit=unit_a):
                    Arrive(first_sync_tl_a, 1) >> barrier_a[a]
                for b in cuda_threads(0, 1, unit=unit_b):
                    Arrive(first_sync_tl_b, 1) >> barrier_b[b]
                if have_fence:
                    Fence(fence_first_sync_tl, fence_second_sync_tl)
                for a in cuda_threads(0, 1, unit=unit_a):
                    Await(barrier_a[a], second_sync_tl_a, 0)
                    Arrive(alt_first_sync_tl_a, 1) >> barrier_a[a]
                    Await(barrier_a[a], alt_second_sync_tl_a, 0)
                for b in cuda_threads(0, 1, unit=unit_b):
                    Await(barrier_b[b], second_sync_tl_b, 0)

    return simplify(test_proc)


def test_mixed_syncs_baseline(compiler):
    compiler.cuda_cpu_test(
        mkproc_mixed_syncs,
        clusterDim=4,
        unit_a=cuda_cluster,
        unit_b=cuda_cta_in_cluster,
        blockDim=128,
        first_sync_tl_a=cuda_in_order,
        first_sync_tl_b=wgmma_async,
        barrier_type_a=CudaClusterSync,
        barrier_type_b=CudaCommitGroup,
        sm="90a",
    )


def test_mixed_syncs_solitary_cluster_sync(compiler):
    # Two CudaClusterSync in scope not allowed
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mixed_syncs,
            clusterDim=4,
            unit_a=cuda_cluster,
            unit_b=cuda_cluster,
            blockDim=128,
            first_sync_tl_a=cuda_in_order,
            first_sync_tl_b=cuda_in_order,
            barrier_type_a=CudaClusterSync,
            barrier_type_b=CudaClusterSync,
        )
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "barrier_b" in msg
    assert "CudaClusterSync" in msg


def test_mixed_syncs_solitary_wgmma_commit_group(compiler):
    # Two wgmma CudaCommitGroup in scope not allowed
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mixed_syncs,
            clusterDim=1,
            unit_a=cuda_cluster,
            unit_b=cuda_cluster,
            blockDim=128,
            first_sync_tl_a=wgmma_async,
            first_sync_tl_b=wgmma_async,
            barrier_type_a=CudaCommitGroup,
            barrier_type_b=CudaCommitGroup,
        )
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "barrier_b" in msg
    assert "CudaCommitGroup" in msg


def test_mixed_syncs_mixed_commit_group(compiler):
    # Mixed commit group of Sm80 cp.async and wgmma, should be allowed, along with cluster fence
    compiler.cuda_cpu_test(
        mkproc_mixed_syncs,
        fence_first_sync_tl=cuda_in_order,
        clusterDim=1,
        unit_a=cuda_warpgroup,
        unit_b=cuda_thread,
        blockDim=128,
        first_sync_tl_a=wgmma_async,
        first_sync_tl_b=Sm80_cp_async,
        barrier_type_a=CudaCommitGroup,
        barrier_type_b=CudaCommitGroup,
        sm="90a",
    )


def test_mixed_syncs_solitary_Sm80_commit_group(compiler):
    # Two cp.async CudaCommitGroup in scope not allowed
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mixed_syncs,
            clusterDim=1,
            unit_a=cuda_thread,
            unit_b=cuda_thread,
            blockDim=128,
            first_sync_tl_a=Sm80_cp_async,
            first_sync_tl_b=Sm80_cp_async,
            barrier_type_a=CudaCommitGroup,
            barrier_type_b=CudaCommitGroup,
        )
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "barrier_b" in msg
    assert "CudaCommitGroup" in msg


def test_mixed_cluster_sync_fence_positive(compiler):
    # Mixed CudaClusterSync and CTA fence, allowed
    compiler.cuda_cpu_test(
        mkproc_mixed_syncs,
        fence_first_sync_tl=cuda_in_order,
        clusterDim=1,
        unit_a=cuda_cluster,
        unit_b=cuda_cta_in_cluster,
        blockDim=128,
        first_sync_tl_a=cuda_in_order,
        first_sync_tl_b=wgmma_async,
        barrier_type_a=CudaClusterSync,
        barrier_type_b=CudaCommitGroup,
        sm="90a",
    )


def test_mixed_cluster_sync_fence_negative(compiler):
    # Mixed CudaClusterSync and cluster fence, not allowed
    # The only difference from above is we have clusterDim > 1 now.
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mixed_syncs,
            fence_first_sync_tl=cuda_in_order,
            clusterDim=2,
            unit_a=cuda_cluster,
            unit_b=cuda_cta_in_cluster,
            blockDim=128,
            first_sync_tl_a=cuda_in_order,
            first_sync_tl_b=wgmma_async,
            barrier_type_a=CudaClusterSync,
            barrier_type_b=CudaCommitGroup,
        )
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "Fence" in msg


def test_mixed_syncs_wgmma_commit_group_unit(compiler):
    # wgmma commit group requires execution by 128 threads, not 64
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mixed_syncs,
            clusterDim=4,
            unit_a=cuda_cluster,
            unit_b=cuda_cta_in_cluster,
            blockDim=64,
            first_sync_tl_a=cuda_in_order,
            first_sync_tl_b=wgmma_async,
            barrier_type_a=CudaClusterSync,
            barrier_type_b=CudaCommitGroup,
        )
    msg = str(exc.value)
    assert "warpgroup" in msg
    assert "64" in msg


def test_mixed_syncs_Sm80_commit_group_unit(compiler):
    # Sm80_cp_async commit group requires execution by 1 thread, not 64
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mixed_syncs,
            clusterDim=4,
            unit_a=cuda_cluster,
            unit_b=cuda_cta_in_cluster,
            blockDim=64,
            first_sync_tl_a=cuda_in_order,
            first_sync_tl_b=Sm80_cp_async,
            barrier_type_a=CudaClusterSync,
            barrier_type_b=CudaCommitGroup,
        )
    msg = str(exc.value)
    assert "thread" in msg
    assert "64" in msg


def test_mixed_syncs_mismatch_first_sync_tl(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mixed_syncs,
            clusterDim=4,
            unit_a=cuda_cluster,
            unit_b=cuda_cta_in_cluster,
            blockDim=128,
            first_sync_tl_a=cuda_in_order,
            first_sync_tl_b=wgmma_async,
            barrier_type_a=CudaClusterSync,
            barrier_type_b=CudaCommitGroup,
            alt_first_sync_tl_a=cuda_temporal,
        )
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "Arrive" in msg
    assert "cuda_in_order" in msg
    assert "cuda_temporal" in msg


def test_mixed_syncs_mismatch_second_sync_tl(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mixed_syncs,
            clusterDim=4,
            unit_a=cuda_cluster,
            unit_b=cuda_cta_in_cluster,
            blockDim=128,
            first_sync_tl_a=cuda_in_order,
            first_sync_tl_b=wgmma_async,
            barrier_type_a=CudaClusterSync,
            barrier_type_b=CudaCommitGroup,
            alt_second_sync_tl_a=wgmma_async,
        )
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "Await" in msg
    assert "cuda_in_order" in msg
    assert "wgmma_async" in msg


def mkproc_cluster_sync_unit(
    unit, arrive_lo=0, arrive_hi=8, await_lo=0, await_hi=8, clusterDim=4
):
    @proc
    def test_proc():
        with CudaDeviceFunction(clusterDim=clusterDim, blockDim=256):
            for task in cuda_tasks(0, 1):
                for u in cuda_threads(0, 1, unit=unit):
                    sync: barrier @ CudaClusterSync
                    with CudaWarps(arrive_lo, arrive_hi):
                        Arrive(cuda_in_order, 1) >> sync
                    with CudaWarps(await_lo, await_hi):
                        Await(sync, cuda_in_order, 0)

    return test_proc


def test_cluster_sync_unit_baseline(compiler):
    # Correct usage of CudaClusterSync
    compiler.cuda_cpu_test(mkproc_cluster_sync_unit, unit=cuda_cluster, sm="90a")


def test_force_cluster_sync(compiler, golden):
    src = compiler.cuda_cpu_test(
        mkproc_cluster_sync_unit, unit=cuda_cluster, clusterDim=1, golden=golden
    )
    assert "cluster.arrive" in src.cuh_src
    assert "cluster.wait" in src.cuh_src


def test_cluster_sync_unit_cta(compiler):
    # Only 1 CTA involved in CudaClusterSync, expect full cluster
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cluster_sync_unit, unit=cuda_cta_in_cluster)
    msg = str(exc.value)
    assert "full cluster" in msg


def test_cluster_sync_unit_warp(compiler):
    # Only 1 warp per CTA involved in CudaClusterSync, expect full cluster
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_cluster_sync_unit, unit=cuda_warp, arrive_hi=1, await_hi=1
        )
    msg = str(exc.value)
    assert "full cluster" in msg


def test_cluster_sync_unit_await(compiler):
    # Partial warps missing in Await for CudaClusterSync.
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cluster_sync_unit, unit=cuda_cluster, await_lo=4)
    msg = str(exc.value)
    assert "Await" in msg


def mkproc_commit_group(
    first_sync_tl, second_sync_tl, unit, await_first=False, different_warps=False
):
    warps_lo = 8 if different_warps else 4
    warps_hi = 16 if different_warps else 12
    arrive_first = not await_first

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=512):
            for task in cuda_tasks(0, 1):
                cg: barrier[2] @ CudaCommitGroup
                if arrive_first:
                    with CudaWarps(4, 12):
                        for t in cuda_threads(0, 2, unit=unit):
                            Arrive(first_sync_tl, 1) >> cg[t]
                with CudaWarps(warps_lo, warps_hi):
                    for t in cuda_threads(0, 2, unit=unit):
                        Await(cg[t], second_sync_tl, 1)
                if await_first:
                    with CudaWarps(4, 12):
                        for t in cuda_threads(0, 2, unit=unit):
                            Arrive(first_sync_tl, 1) >> cg[t]

    return simplify(test_proc)


def test_wgmma_commit_group_async_proxy(compiler, golden):
    # wgmma -> async proxy is OK (wgmma is already in the async proxy)
    compiler.cuda_cpu_test(
        mkproc_commit_group,
        first_sync_tl=wgmma_async,
        second_sync_tl=cuda_generic_and_async_proxy,
        unit=cuda_warpgroup,
        golden=golden,
    )


def test_Sm80_commit_group_async_proxy(compiler):
    # Sm80_cp_async -> async proxy is not OK (Sm80_cp_async is in the generic proxy)
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_commit_group,
            first_sync_tl=Sm80_cp_async,
            second_sync_tl=cuda_generic_and_async_proxy,
            unit=cuda_thread,
        )
    msg = str(exc.value)
    assert "cg" in msg
    assert "Await" in msg
    assert "cuda_generic_and_async_proxy" in msg


def test_bad_first_sync_tl_commit_group(compiler):
    # tma_to_smem_async -> cuda_in_order is not supported by commit group
    # (this is handled by mbarrier completion mechanism)
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_commit_group,
            first_sync_tl=tma_to_smem_async,
            second_sync_tl=cuda_in_order,
            unit=cuda_thread,
        )
    msg = str(exc.value)
    assert "cg" in msg
    assert "Arrive" in msg
    assert "tma_to_smem_async" in msg


def test_commit_group_await_first_positive(compiler, golden):
    compiler.cuda_cpu_test(
        mkproc_commit_group,
        first_sync_tl=Sm80_cp_async,
        second_sync_tl=cuda_in_order,
        unit=cuda_thread,
        await_first=True,
        different_warps=False,
        golden=golden,
    )


def test_commit_group_await_first_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_commit_group,
            first_sync_tl=Sm80_cp_async,
            second_sync_tl=cuda_in_order,
            unit=cuda_thread,
            await_first=True,
            different_warps=True,
        )
    msg = str(exc.value)
    assert "inconsistent collective tiling with previous Await" in msg


def test_commit_group_arrive_first_positive(compiler, golden):
    compiler.cuda_cpu_test(
        mkproc_commit_group,
        first_sync_tl=Sm80_cp_async,
        second_sync_tl=cuda_in_order,
        unit=cuda_thread,
        await_first=False,
        different_warps=False,
        golden=golden,
    )


def test_commit_group_arrive_first_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_commit_group,
            first_sync_tl=Sm80_cp_async,
            second_sync_tl=cuda_in_order,
            unit=cuda_thread,
            await_first=False,
            different_warps=True,
        )
    msg = str(exc.value)
    assert "inconsistent collective tiling with previous Arrive" in msg


@dataclass(slots=True)
class MbarrierQualConfig:
    first_sync_tl: Sync_tl
    second_sync_tl: Sync_tl
    try_or_test: str
    arrive_cp_async: bool
    have_await_proxy_fence: bool
    have_init_proxy_fence: bool


# Sm80_cp_async -> cuda_temporal
# should use cp.async.mbarrier.arrive.noinc.shared::cta.b64
mbarrier_Sm80_cp_async_qc = MbarrierQualConfig(
    Sm80_cp_async, cuda_temporal, "test", True, False, False
)

# Same as before, but when compiled for sm_90a, switch from test_wait to try_wait
mbarrier_Sm90a_cp_async_qc = MbarrierQualConfig(
    Sm80_cp_async, cuda_temporal, "try", True, False, False
)

# cuda_in_order -> cuda_generic_and_async_proxy
# requires generic -> async proxy fence
mbarrier_in_order_to_wgmma_qc = MbarrierQualConfig(
    cuda_in_order, cuda_generic_and_async_proxy, "try", False, True, True
)

# cuda_temporal -> cuda_generic_and_async_proxy
# doesn't require the fence after the await
# (cuda_temporal resolves only WAR hazards),
# but we still need the proxy fence at startup.
mbarrier_temporal_to_wgmma_qc = MbarrierQualConfig(
    cuda_temporal, cuda_generic_and_async_proxy, "try", False, False, True
)

mbarrier_wrong_wgmma_qc = MbarrierQualConfig(
    cuda_in_order, wgmma_async, "try", False, True, True
)
mbarrier_wrong_cpu1_qc = MbarrierQualConfig(
    cuda_in_order, cpu_in_order, "try", False, True, True
)
mbarrier_wrong_cpu2_qc = MbarrierQualConfig(
    cpu_in_order, cuda_in_order, "try", False, True, True
)
mbarrier_wrong_tma_qc = MbarrierQualConfig(
    tma_to_smem_async, cuda_in_order, "try", False, True, True
)


# fmt: off
def mkproc_mbarriers(M_CTA: int, N_CTA: int, f_delay: int, b_delay: int, qc: MbarrierQualConfig):
    first_sync_tl = qc.first_sync_tl
    second_sync_tl = qc.second_sync_tl
    @proc
    def test_mbarriers():
        with CudaDeviceFunction(clusterDim=M_CTA * N_CTA, blockDim=64):
            for task in cuda_tasks(0, 2):
                for t2 in cuda_threads(0, 4, unit=16 * cuda_thread):
                    # Note: there are actually 4x as many queue barriers as there appear
                    # to be, because of the t2(0, 4) loop above. This is one of
                    # the tricky cases being tested by this test case. Essentially the
                    # compiler "lifts" the array to [M_CTA, N_CTA, 4, 2].
                    # 2025-09-18: handled with codegen_slices_to_root.
                    row_bars: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    col_bars: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    all_bars: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    f_rc_bars: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    b_rc_bars: barrier(f_rc_bars)[M_CTA, N_CTA, 2] @ CudaMbarrier
                    baseline: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    for i in seq(0, 5):
                        for t1 in cuda_threads(0, 2, unit=8 * cuda_thread):
                            for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                                for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                                    # Note baseline mbarrier doesn't use delay or parameterized sync-tl
                                    Arrive(cuda_in_order, 1) >> baseline[m_cta, n_cta, t1]

                                    # Only f_rc_bars and b_rc_bars are guarding each other.
                                    # Its ring buffer depth is f_delay + b_delay, instead of 1 + f_delay
                                    Await(b_rc_bars[m_cta, n_cta, t1], second_sync_tl, ~b_delay)
                                    Arrive(first_sync_tl, 1) >> f_rc_bars[m_cta, n_cta, t1]

                                    Arrive(first_sync_tl, 1) >> row_bars[m_cta, n_cta, t1] >> row_bars[m_cta, :, t1]
                                    Await(row_bars[m_cta, n_cta, t1], second_sync_tl, ~f_delay)
                                    Arrive(first_sync_tl, 1) >> col_bars[m_cta, n_cta, t1] >> col_bars[:, n_cta, t1]
                                    Await(col_bars[m_cta, n_cta, t1], second_sync_tl, ~f_delay)
                                    Arrive(first_sync_tl, 1) >> all_bars[m_cta, n_cta, t1] >> all_bars[:, :, t1]
                                    Await(all_bars[m_cta, n_cta, t1], second_sync_tl, ~f_delay)

                                    Await(f_rc_bars[m_cta, n_cta, t1], second_sync_tl, ~f_delay)
                                    Arrive(first_sync_tl, 1) >> b_rc_bars[m_cta, :, t1] >> b_rc_bars[:, n_cta, t1]

                                    Await(baseline[m_cta, n_cta, t1], cuda_in_order, ~0)
                # Need for the test not to crash due to "Cluster target block not present"
                Fence(cuda_temporal, cuda_temporal)
    return test_mbarriers
# fmt: on


def mkref_mbarriers(
    xrg: excut.ExcutReferenceGenerator,
    M_CTA: int,
    N_CTA: int,
    f_delay: int,
    b_delay: int,
    qc: MbarrierQualConfig,
):
    clusterDim = M_CTA * N_CTA
    blockDim = 64
    row_bars = xrg.new_varname("row_bars")
    col_bars = xrg.new_varname("col_bars")
    all_bars = xrg.new_varname("all_bars")
    f_rc_bars = xrg.new_varname("f_rc_bars")
    b_rc_bars = xrg.new_varname("b_rc_bars")
    baseline = xrg.new_varname("baseline")

    cta_arrive = f"mbarrier.arrive.shared::cta.b64"
    cluster_arrive = f"mbarrier.arrive.shared::cluster.b64"
    cta_async_arrive = f"cp.async.mbarrier.arrive.noinc.shared::cta.b64"
    cluster_async_arrive = f"cp.async.mbarrier.arrive.noinc.shared::cluster.b64"
    cta_await = f"mbarrier.{qc.try_or_test}_wait.parity.acquire.cta.shared::cta.b64"

    mbarrier_inits = (
        (row_bars, N_CTA, 1 + f_delay),
        (col_bars, M_CTA, 1 + f_delay),
        (all_bars, clusterDim, 1 + f_delay),
        (b_rc_bars, M_CTA + N_CTA - 1, f_delay + b_delay),
        (f_rc_bars, 1, f_delay + b_delay),
    )

    def device_setup(m_cta, n_cta):
        # 0th thread's initialization actions.
        with xrg.permuted():
            for t2 in range(4):
                for t1 in range(2):
                    # Initialize row_bars, col_bars, all_bars, rc_bars each
                    # with respective expected-arrive-count.
                    for var, cta_count, ring_size in mbarrier_inits:
                        for ring in range(ring_size):
                            expected_arrive = cta_count * 8
                            xrg(
                                "mbarrier.init.shared::cta.b64",
                                var[m_cta, n_cta, t2, t1, ring],
                                expected_arrive,
                            )
                    # Init baseline mbarriers (delay=0, no cross-cluster multicast)
                    for ring in range(1):
                        xrg(
                            "mbarrier.init.shared::cta.b64",
                            baseline[m_cta, n_cta, t2, t1, ring],
                            8,
                        )
        if qc.have_init_proxy_fence:
            xrg("fence.proxy.async")
        # End 0th thread init
        # Cross-thread sync
        for threadIdx in xrg.stride_threadIdx(blockDim):
            if clusterDim == 1:
                xrg("barrier.cta.sync", 0)
            else:
                xrg("barrier.cluster.arrive.aligned")
                xrg("barrier.cluster.wait.aligned")
        # End exo_deviceSetup

    def arrive_impl(m_cta, n_cta, t2, t1, i, match_cta, var, ring_size):
        other_ctas = []
        for m2 in range(0, M_CTA):
            for n2 in range(0, N_CTA):
                if match_cta(m2, n2) and (m_cta != m2 or n_cta != n2):
                    other_ctas.append((m2, n2))

        if qc.arrive_cp_async:
            ptx = cluster_async_arrive if other_ctas else cta_async_arrive
        else:
            ptx = cluster_arrive if other_ctas else cta_arrive

        # excut limitation: we assume that the mbarrier inside this CTA
        # is signalled first. This is not required for correct codegen.
        # The deduction algorithm might fail if we don't follow this assumption.
        xrg(ptx, var[m_cta, n_cta, t2, t1, i % ring_size])
        with xrg.permuted():
            for m2, n2 in other_ctas:
                xrg(ptx, var[m2, n2, t2, t1, i % ring_size])

    def await_impl(m_cta, n_cta, t2, t1, i, var, ring_size, delay):
        i -= delay
        if i >= 0:
            ring = i % ring_size
            parity = (i // ring_size) % 2
            xrg(cta_await, var[m_cta, n_cta, t2, t1, ring], parity)
            if qc.have_await_proxy_fence:
                xrg("fence.proxy.async")

    def thread_main(m_cta, n_cta, t2, t1):
        for i in range(5):
            assert (
                f_delay + b_delay <= 5
            ), "excut won't work if some of the mbarriers are never used (unfortunate design flaw with variable deduction)"

            # baseline mbarrier arrive (no ring buffering)
            xrg(cta_arrive, baseline[m_cta, n_cta, t2, t1, 0])

            match_one = lambda m, n: m == m_cta and n == n_cta
            match_row = lambda m, n: m == m_cta
            match_col = lambda m, n: n == n_cta
            match_any = lambda m, n: True
            match_rc = lambda m, n: m == m_cta or n == n_cta

            await_impl(m_cta, n_cta, t2, t1, i, b_rc_bars, b_delay + f_delay, b_delay)
            arrive_impl(
                m_cta, n_cta, t2, t1, i, match_one, f_rc_bars, b_delay + f_delay
            )

            arrive_impl(m_cta, n_cta, t2, t1, i, match_row, row_bars, 1 + f_delay)
            await_impl(m_cta, n_cta, t2, t1, i, row_bars, 1 + f_delay, f_delay)
            arrive_impl(m_cta, n_cta, t2, t1, i, match_col, col_bars, 1 + f_delay)
            await_impl(m_cta, n_cta, t2, t1, i, col_bars, 1 + f_delay, f_delay)
            arrive_impl(m_cta, n_cta, t2, t1, i, match_any, all_bars, 1 + f_delay)
            await_impl(m_cta, n_cta, t2, t1, i, all_bars, 1 + f_delay, f_delay)

            await_impl(m_cta, n_cta, t2, t1, i, f_rc_bars, b_delay + f_delay, f_delay)
            arrive_impl(m_cta, n_cta, t2, t1, i, match_rc, b_rc_bars, b_delay + f_delay)

            # baseline mbarrier await (no ring buffering)
            xrg(cta_await, baseline[m_cta, n_cta, t2, t1, 0], i % 2)

        if clusterDim == 1:
            xrg("barrier.cta.sync", 0)
        else:
            xrg("barrier.cluster.arrive.aligned")
            xrg("barrier.cluster.wait.aligned")

    xrg.begin_cuda()
    for task in xrg.stride_blockIdx(2, stride=clusterDim):
        for m_cta in xrg.stride_blockIdx(M_CTA, stride=N_CTA):
            for n_cta in xrg.stride_blockIdx(N_CTA):
                device_setup(m_cta, n_cta)
                for t2 in xrg.stride_threadIdx(4, stride=16):
                    for t1 in xrg.stride_threadIdx(2, stride=8):
                        for intra_t1 in xrg.stride_threadIdx(8):
                            thread_main(m_cta, n_cta, t2, t1)
    xrg.end_cuda()


mb_m1n1d1d2_Sm80_cp_async = dict(
    M_CTA=1, N_CTA=1, f_delay=1, b_delay=2, qc=mbarrier_Sm80_cp_async_qc
)
mb_m1n1d3d2_Sm80_cp_async = dict(
    M_CTA=1, N_CTA=1, f_delay=3, b_delay=2, qc=mbarrier_Sm80_cp_async_qc
)
mb_m1n1d0d0_Sm80_cp_async = dict(
    M_CTA=1, N_CTA=1, f_delay=0, b_delay=0, qc=mbarrier_Sm80_cp_async_qc
)
mb_m1n1d4d1_Sm90a_cp_async = dict(
    M_CTA=1, N_CTA=1, f_delay=4, b_delay=1, qc=mbarrier_Sm90a_cp_async_qc
)
mb_m4n2d1d2_in_order_to_wgmma = dict(
    M_CTA=4, N_CTA=2, f_delay=1, b_delay=2, qc=mbarrier_in_order_to_wgmma_qc
)
mb_m4n1d0d2_temporal_to_wgmma = dict(
    M_CTA=4, N_CTA=1, f_delay=0, b_delay=2, qc=mbarrier_temporal_to_wgmma_qc
)
mb_m1n4d2d2_temporal_to_wgmma = dict(
    M_CTA=1, N_CTA=4, f_delay=2, b_delay=2, qc=mbarrier_temporal_to_wgmma_qc
)

mb_m1n4d2d2_wrong_wgmma = dict(
    M_CTA=1, N_CTA=4, f_delay=2, b_delay=2, qc=mbarrier_wrong_wgmma_qc
)
mb_m1n4d2d2_wrong_cpu1 = dict(
    M_CTA=1, N_CTA=4, f_delay=2, b_delay=2, qc=mbarrier_wrong_cpu1_qc
)
mb_m1n4d2d2_wrong_cpu2 = dict(
    M_CTA=1, N_CTA=4, f_delay=2, b_delay=2, qc=mbarrier_wrong_cpu2_qc
)
mb_m1n4d2d2_wrong_tma = dict(
    M_CTA=1, N_CTA=4, f_delay=2, b_delay=2, qc=mbarrier_wrong_tma_qc
)

mb_m2n1d4_Sm90a_cp_async = dict(
    M_CTA=2, N_CTA=1, f_delay=4, b_delay=0, qc=mbarrier_Sm90a_cp_async_qc
)


def test_mbarriers_m1n1d1d2_Sm80_cp_async_excut(compiler_Sm80):
    compiler_Sm80.excut_test(
        mkproc_mbarriers, mkref_mbarriers, **mb_m1n1d1d2_Sm80_cp_async
    )


def test_mbarriers_m1n1d1d2_Sm80_cp_async_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_mbarriers, golden, **mb_m1n1d1d2_Sm80_cp_async)


def test_mbarriers_m1n1d3d2_Sm80_cp_async_excut(compiler_Sm80):
    compiler_Sm80.excut_test(
        mkproc_mbarriers, mkref_mbarriers, **mb_m1n1d3d2_Sm80_cp_async
    )


def test_mbarriers_m1n1d3d2_Sm80_cp_async_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_mbarriers, golden, **mb_m1n1d3d2_Sm80_cp_async)


def test_mbarriers_m1n1d4d1_Sm90a_cp_async_excut(compiler_Sm90a):
    compiler_Sm90a.excut_test(
        mkproc_mbarriers, mkref_mbarriers, **mb_m1n1d4d1_Sm90a_cp_async
    )


def test_mbarriers_m1n1d4d1_Sm90a_cp_async_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_mbarriers, golden, **mb_m1n1d4d1_Sm90a_cp_async)


def test_mbarriers_m4n2d1d2_in_order_to_wgmma_excut(compiler_Sm90a):
    compiler_Sm90a.excut_test(
        mkproc_mbarriers, mkref_mbarriers, **mb_m4n2d1d2_in_order_to_wgmma
    )


def test_mbarriers_m4n2d1d2_in_order_to_wgmma_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_mbarriers, golden, **mb_m4n2d1d2_in_order_to_wgmma)


def test_mbarriers_m4n1d0d2_temporal_to_wgmma_excut(compiler_Sm90a):
    compiler_Sm90a.excut_test(
        mkproc_mbarriers, mkref_mbarriers, **mb_m4n1d0d2_temporal_to_wgmma
    )


def test_mbarriers_m4n1d0d2_temporal_to_wgmma_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_mbarriers, golden, **mb_m4n1d0d2_temporal_to_wgmma)


def test_mbarriers_m1n4d2d2_temporal_to_wgmma_excut(compiler_Sm90a):
    compiler_Sm90a.excut_test(
        mkproc_mbarriers, mkref_mbarriers, **mb_m1n4d2d2_temporal_to_wgmma
    )


def test_mbarriers_m1n4d2d2_temporal_to_wgmma_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_mbarriers, golden, **mb_m1n4d2d2_temporal_to_wgmma)


def test_mbarriers_wrong_wgmma(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarriers, **mb_m1n4d2d2_wrong_wgmma)
    assert "consider cuda_generic_and_async_proxy" in str(exc.value)


def test_mbarriers_wrong_cpu1(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarriers, **mb_m1n4d2d2_wrong_cpu1)
    assert "cpu_in_order not supported" in str(exc.value)


def test_mbarriers_wrong_cpu2(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarriers, **mb_m1n4d2d2_wrong_cpu2)
    assert "cpu_in_order not supported" in str(exc.value)


def test_mbarriers_wrong_tma(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarriers, **mb_m1n4d2d2_wrong_tma)
    assert "tma_to_smem_async" in str(exc.value)
    assert "use cuda_temporal" in str(exc.value)


def test_mbarriers_Sm80_cp_async_1_CTA(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarriers, **mb_m2n1d4_Sm90a_cp_async)
    assert "Sm80_cp_async mbarrier must be within 1 CTA" in str(exc.value)


def test_mbarriers_invalid_0_delay(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarriers, **mb_m1n1d0d0_Sm80_cp_async)
    assert "must have some await with nonzero skips" in str(exc.value)
    assert "f_rc_bars" in str(exc.value)
    assert "b_rc_bars" in str(exc.value)


def mkproc_mbarrier_not_in_1_CTA():
    @proc
    def broken():
        with CudaDeviceFunction(clusterDim=4, blockDim=256):
            for task in cuda_tasks(0, 1):
                bad_bar: barrier[2] @ CudaMbarrier
                for cta_pair in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                    Arrive(cuda_in_order, 1) >> bad_bar[cta_pair]
                    Await(bad_bar[cta_pair], cuda_in_order, ~1)

    return broken


def test_mbarrier_not_in_1_CTA(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarrier_not_in_1_CTA)
    assert "bad_bar must be distributed so each mbarrier is resident in 1 CTA" in str(
        exc.value
    )


def mkproc_mbarrier_missing_idx(wrong):
    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                bar: barrier[2] @ CudaMbarrier
                for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                    if wrong:
                        for WaRp in cuda_threads(0, 4, unit=cuda_warp):
                            Arrive(cuda_in_order, 1) >> bar[wg]
                            Await(bar[wg], cuda_in_order, ~0)
                    else:
                        for WaRp in seq(0, 4):
                            Arrive(cuda_in_order, 1) >> bar[wg]
                            Await(bar[wg], cuda_in_order, ~0)

    return simplify(test_proc)


def test_mbarrier_missing_idx_positive(compiler):
    compiler.cuda_cpu_test(mkproc_mbarrier_missing_idx, wrong=False)


def test_mbarrier_missing_idx_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarrier_missing_idx, wrong=True)
    assert "Missing: WaRp" in str(exc.value)


def mkproc_mbarrier_warps_match(warps0, warps1):
    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=768):
            for task in cuda_tasks(0, 1):
                mbarrier: barrier @ CudaMbarrier
                with CudaWarps(*warps0):
                    Arrive(cuda_in_order, 1) >> mbarrier
                with CudaWarps(*warps1):
                    Await(mbarrier, cuda_in_order, ~0)

    return test_proc


def test_mbarrier_warps_match_positive(compiler):
    compiler.cuda_cpu_test(
        mkproc_mbarrier_warps_match, warps0=(12, 16), warps1=(12, 16)
    )


def test_offset_mbarrier_warps_match_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mbarrier_warps_match, warps0=(11, 15), warps1=(12, 16)
        )
    msg = str(exc.value)
    assert "Incompatible offsets" in msg
    assert "Incompatible box size" not in msg


def test_box_mbarrier_warps_match_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_mbarrier_warps_match, warps0=(12, 18), warps1=(12, 16)
        )
    msg = str(exc.value)
    assert "Incompatible offsets" not in msg
    assert "Incompatible box size" in msg


# "garden" means "garden variety fence", but the shorter name makes
# filenames in pytest /tmp dirs not be truncated as much.


def mkproc_garden_Sm80():
    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                # cp.async.wait_all + barrier.cta.sync
                Fence(Sm80_cp_async, cuda_in_order)

                for w in cuda_threads(0, 8, unit=cuda_warp):
                    # cp.async.wait_all + __syncwarp()
                    Fence(Sm80_generic, cuda_temporal)

                # barrier.cta.sync only
                Fence(cuda_in_order, cuda_in_order)

    return test_proc


def mkref_garden_Sm80(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for threadIdx in xrg.stride_threadIdx(256):
        xrg("cp.async.wait_all")
        xrg("barrier.cta.sync", 0)
        xrg("cp.async.wait_all")
        xrg("__syncwarp")
        xrg("barrier.cta.sync", 0)
    xrg.end_cuda()


def test_garden_Sm80_excut(compiler_Sm80):
    compiler_Sm80.excut_test(mkproc_garden_Sm80, mkref_garden_Sm80)


def test_garden_Sm80_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_garden_Sm80, golden)


def mkproc_garden_warps_threads():
    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                Fence(cuda_in_order, cuda_in_order)

                for tid in cuda_threads(0, 32, unit=cuda_thread):
                    Fence(Sm80_cp_async, cuda_in_order)

    return test_proc


def mkref_garden_warps_threads(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for threadIdx in xrg.stride_threadIdx(32):
        xrg("__syncwarp")
        xrg("cp.async.wait_all")
    xrg.end_cuda()


def test_garden_warps_threads_excut(compiler_Sm80):
    compiler_Sm80.excut_test(mkproc_garden_warps_threads, mkref_garden_warps_threads)


def test_garden_warps_threads_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_garden_warps_threads, golden)


def mkproc_garden_Sm90(
    special_lo,
    special_hi,
    test_first_sync_tl=cuda_in_order,
    test_second_sync_tl=cuda_in_order,
    special_first_sync_tl=cuda_in_order,
    special_second_sync_tl=wgmma_async_smem,
):
    @proc
    def test_proc():
        with CudaDeviceFunction(clusterDim=4, blockDim=256):
            for task in cuda_tasks(0, 1):
                # cluster: generic->async proxy
                Fence(cuda_in_order, cuda_generic_and_async_proxy)

                # We use the __syncwarp to separate blocks of code in mkref
                # and also for the (non-CUDA-device) invalid sync-tl tests.
                for cta in cuda_threads(0, 4, unit=cuda_cta_in_cluster):
                    # No proxy fence, as first-sync-tl is temporal-only
                    Fence(cuda_temporal, cuda_generic_and_async_proxy)
                    for w in cuda_threads(0, 8, unit=cuda_warp):
                        Fence(test_first_sync_tl, test_second_sync_tl)

                    # No proxy fence or cp.async.wait_all
                    Fence(cuda_in_order, cuda_in_order)
                    for w in cuda_threads(0, 8, unit=cuda_warp):
                        Fence(test_first_sync_tl, test_second_sync_tl)

                    # cp.async.wait_all
                    Fence(Sm80_cp_async, cuda_temporal)
                    for w in cuda_threads(0, 8, unit=cuda_warp):
                        Fence(test_first_sync_tl, test_second_sync_tl)

                    # Proxy fence for generic->async
                    Fence(cuda_in_order, cuda_generic_and_async_proxy)
                    for w in cuda_threads(0, 8, unit=cuda_warp):
                        Fence(test_first_sync_tl, test_second_sync_tl)

                    with CudaWarps(special_lo, special_hi):
                        # Testing special case; warpgroup
                        # cuda_in_order->wgmma_async_smem
                        Fence(special_first_sync_tl, special_second_sync_tl)

                # cluster: cp.async.wait_all
                cluster_sync: barrier @ CudaClusterSync
                Arrive(Sm80_generic, 1) >> cluster_sync
                Await(cluster_sync, cuda_in_order, 0)

                # Literally should do nothing
                for cta in cuda_threads(0, 4, unit=cuda_cta_in_cluster):
                    for tid in cuda_threads(0, 256, unit=cuda_thread):
                        Fence(cuda_in_order, cuda_in_order)

    return test_proc


def mkref_garden_Sm90(
    xrg: excut.ExcutReferenceGenerator,
    special_lo,
    special_hi,
    # Ignored args to match mkproc_garden_Sm90
    cluster_first_sync_tl=None,
    cluster_second_sync_tl=None,
    special_first_sync_tl=None,
    special_second_sync_tl=None,
):
    xrg.begin_cuda()
    for blockIdx in xrg.stride_blockIdx(4):
        for threadIdx in xrg.stride_threadIdx(256):
            # cluster: generic->async proxy
            xrg("barrier.cluster.arrive.aligned")
            xrg("barrier.cluster.wait.aligned")
            xrg("fence.proxy.async")

            # No proxy fence, as first-sync-tl is temporal-only
            xrg("barrier.cta.sync", 0)
            xrg("__syncwarp")

            # No proxy fence or cp.async.wait_all
            xrg("barrier.cta.sync", 0)
            xrg("__syncwarp")

            # cp.async.wait_all
            xrg("cp.async.wait_all")
            xrg("barrier.cta.sync", 0)
            xrg("__syncwarp")

            # Proxy fence for generic->wgmma_async
            xrg("barrier.cta.sync", 0)
            xrg("fence.proxy.async")
            xrg("__syncwarp")

            if 32 * special_lo <= threadIdx < 32 * special_hi:
                # NB currently this is disabled
                # Testing special case; warpgroup
                # cuda_in_order->wgmma_async_smem
                xrg("fence.proxy.async")

            # cluster: cp.async.wait_all
            xrg("cp.async.wait_all")
            xrg("barrier.cluster.arrive.aligned")
            xrg("barrier.cluster.wait.aligned")
    xrg.end_cuda()


# Adapt and re-enable these tests if we wish to support the special case
# for a warpgroup generating stuff in the generic proxy, then using
# that data in future wgmma instrs with ONLY a proxy fence, no cross-thread sync.
# I'm not sure if this is valid CUDA usage.
if False:

    def test_garden_Sm90_excut(compiler_Sm90a):
        compiler_Sm90a.excut_test(
            mkproc_garden_Sm90, mkref_garden_Sm90, special_lo=4, special_hi=8
        )

    def test_garden_Sm90_golden(compiler, golden):
        compiler.cuda_cpu_test(mkproc_garden_Sm90, golden, special_lo=4, special_hi=8)


def test_garden_wrong_L1(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_garden_Sm90,
            special_lo=4,
            special_hi=8,
            test_first_sync_tl=tma_to_smem_async,
        )
    msg = str(exc.value)
    assert "we allow Sm80_generic" in msg
    assert "tma_to_smem_async" in msg


def test_garden_wrong_L2(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_garden_Sm90,
            special_lo=4,
            special_hi=8,
            test_second_sync_tl=cpu_in_order,
        )
    msg = str(exc.value)
    assert "at most cuda_generic_and_async_proxy" in msg
    assert "cpu_in_order" in msg


def test_garden_wrong_coll_unit(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_garden_Sm90, special_lo=2, special_hi=6)
    assert "collective unit matched no known case" in str(exc.value)


def test_garden_wrong_special_case_L2(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_garden_Sm90,
            special_lo=4,
            special_hi=8,
            special_second_sync_tl=cuda_in_order,
        )
    assert "collective unit matched no known case" in str(exc.value)


def mkproc_cp_async_commit_group(ring, ntid, scale, K_tile, M=0, K=0, sync_check=False):
    # Overcomplicated kernel that sets h_out = scale * h_in.
    # M, K ignored (compatibility with mkref_cp_async_commit_group).
    del M
    del K

    @proc
    def p(M: size, K: size, h_out: f32[M, K], h_in: f32[M, K]):
        assert M % ntid == 0
        assert K % K_tile == 0
        d_in: f32[M, K] @ CudaGmemLinear
        d_out: f32[M, K] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2f32(M, K, d_in[:, :], h_in[:, :])

        # fmt: off
        with CudaDeviceFunction(blockDim=ntid):
            for m_task in cuda_tasks(0, M / ntid):
                cg: barrier[ntid] @ CudaCommitGroup
                smem: f32[ring, ntid, K_tile] @ CudaSmemLinear
                # Load K tiles 0, ..., ring - 2
                for warmup in seq(0, ring - 2):
                    for tid in cuda_threads(0, ntid):
                        if warmup < K / K_tile:
                            for k_cp in seq(0, K_tile / 4):
                                Sm80_cp_async_f32(
                                    smem[warmup, tid, 4 * k_cp : 4 * k_cp + 4],
                                    d_in[m_task * ntid + tid,
                                        K_tile * warmup + 4 * k_cp
                                      : K_tile * warmup + 4 * k_cp + 4],
                                                  size=4)
                        Arrive(Sm80_cp_async) >> cg[tid]
                for k_iter in seq(0, K / K_tile):
                    for tid in cuda_threads(0, ntid):
                        # Load K tile k_iter + ring - 2
                        if k_iter + (ring - 2) < K / K_tile:
                            for k_cp in seq(0, K_tile / 4):
                                Sm80_cp_async_f32(smem[(k_iter + ring - 2) % ring, tid, 4 * k_cp : 4 * k_cp + 4],
                                                  d_in[m_task * ntid + tid,
                                                      K_tile * (k_iter + ring - 2) + 4 * k_cp
                                                    : K_tile * (k_iter + ring - 2) + 4 * k_cp + 4],
                                                  size=4)
                        Arrive(Sm80_cp_async) >> cg[tid]
                        Await(cg[tid], cuda_in_order, ring - 2)
                    Fence(cuda_in_order, cuda_in_order)
                    for ms in seq(0, K_tile):
                        # Write out scaled version of K tile number k_iter
                        for mt in cuda_threads(0, ntid / K_tile, unit=K_tile*cuda_thread):
                            for k in cuda_threads(0, K_tile):
                                d_out[
                                    m_task * ntid + ms * (ntid / K_tile) + mt,
                                    k_iter * K_tile + k] = (scale *
                                        smem[k_iter % ring,
                                             ms * (ntid / K_tile) + mt,
                                             k])

                # Required to de-allocate SMEM safely.
                for tid in cuda_threads(0, ntid):
                    Await(cg[tid], cuda_in_order, 0)
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_2f32(M, K, h_out[:,:], d_out[:,:])

    p = simplify(p)
    p = rename(p, f"_{ring}_{ntid}_{K_tile}_cp_async_commit_group")
    if sync_check:
        p.sync_check(M=ntid * 2, K=K_tile * (ring + 2))
    return p


def cp_async_commit_group_test_value_impl(
    compiler_Sm80, ring, ntid, scale, K_tile, M, K
):
    cu = compiler_Sm80.cuda_test_context(
        mkproc_cp_async_commit_group(ring=ring, ntid=ntid, scale=scale, K_tile=K_tile)
    )
    h_in = np.ndarray(shape=(M, K), dtype=np.float32)
    h_out = np.ndarray(shape=(M, K), dtype=np.float32)

    rand = random.Random(5000)
    for m in range(0, M):
        for k in range(0, K):
            h_in[m, k] = rand.randrange(-10000, 10000)

    expected = scale * h_in
    cu(None, M, K, h_out, h_in)
    assert np.array_equal(h_out, expected)


def mkref_cp_async_commit_group(
    xrg: excut.ExcutReferenceGenerator,
    ring,
    ntid,
    scale,
    K_tile,
    M,
    K,
    sync_check=False,
):
    # scale, sync_check unused (compatibility with mkproc_cp_async_commit_group)
    cp_async_seq_len = K_tile // 4

    def log_cp_async_seq():
        for k_cp in range(0, cp_async_seq_len):
            xrg("cp.async.cg.shared.global", excut.sink, excut.sink, excut.sink)

    xrg.begin_cuda()
    for m_task in xrg.stride_blockIdx(M // ntid):
        # Load K tiles 0, ..., ring - 2
        for warmup in range(0, ring - 2):
            for tid in xrg.stride_threadIdx(ntid):
                if warmup < K // K_tile:
                    log_cp_async_seq()
                xrg("cp.async.commit_group")
        for k_iter in range(0, K // K_tile):
            for tid in xrg.stride_threadIdx(ntid):
                # Load K tile k_iter + ring - 2
                if k_iter + (ring - 2) < K / K_tile:
                    log_cp_async_seq()
                xrg("cp.async.commit_group")
                xrg("cp.async.wait_group", ring - 2)
                xrg("barrier.cta.sync", 0)
            # Write out scaled version of K tile number k_iter
            # No excut logging done here (that we care about).
        # Required to de-allocate SMEM safely.
        for tid in xrg.stride_threadIdx(ntid):
            xrg("cp.async.wait_group", 0)
            xrg("barrier.cta.sync", 0)
    xrg.end_cuda()


cp_async_commit_group_A_args = dict(
    ring=4,
    ntid=128,
    scale=3,
    K_tile=32,
)

cp_async_commit_group_B_args = dict(
    ring=6,
    ntid=256,
    scale=3,
    K_tile=8,
)


def test_golden_A_cp_async_commit_group(compiler, golden):
    compiler.cuda_cpu_test(
        mkproc_cp_async_commit_group,
        sync_check=True,
        golden=golden,
        M=1024,
        K=1536,
        **cp_async_commit_group_A_args,
    )


def test_golden_B_cp_async_commit_group(compiler, golden):
    compiler.cuda_cpu_test(
        mkproc_cp_async_commit_group,
        sync_check=True,
        golden=golden,
        M=1536,
        K=1024,
        **cp_async_commit_group_B_args,
    )


def test_value_A_cp_async_commit_group(compiler_Sm80):
    cp_async_commit_group_test_value_impl(
        compiler_Sm80, M=1024, K=1536, **cp_async_commit_group_A_args
    )


def test_value_B_cp_async_commit_group(compiler_Sm80):
    cp_async_commit_group_test_value_impl(
        compiler_Sm80, M=1536, K=1024, **cp_async_commit_group_B_args
    )


def test_excut_A_cp_async_commit_group(compiler_Sm80):
    M = 256
    K = 192
    h_in = np.ndarray(shape=(M, K), dtype=np.float32)
    h_out = np.ndarray(shape=(M, K), dtype=np.float32)
    compiler_Sm80.excut_test(
        mkproc_cp_async_commit_group,
        mkref_cp_async_commit_group,
        M,
        K,
        h_out,
        h_in,
        M=M,
        K=K,
        **cp_async_commit_group_A_args,
    )


def test_excut_B_cp_async_commit_group(compiler_Sm80):
    M = 256
    K = 192
    h_in = np.ndarray(shape=(M, K), dtype=np.float32)
    h_out = np.ndarray(shape=(M, K), dtype=np.float32)
    compiler_Sm80.excut_test(
        mkproc_cp_async_commit_group,
        mkref_cp_async_commit_group,
        M,
        K,
        h_out,
        h_in,
        M=M,
        K=K,
        **cp_async_commit_group_B_args,
    )


def mkproc_no_trailing_barrier():
    # fmt: off
    @proc
    def proc_no_trailing_barrier(gmem: f32[128] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                smem: f32[128] @ CudaSmemLinear
                bar: barrier @ CudaMbarrier
                Await(bar, cuda_in_order, 1)
                for tid in cuda_threads(0, 32):
                    Sm80_cp_async_f32(
                        smem[4 * tid : 4 * tid + 4],
                        gmem[4 * tid : 4 * tid + 4],
                        size=4,
                    ) >> bar
                Arrive(Sm80_cp_async, 1) >> bar
    return proc_no_trailing_barrier
    # fmt: on


def test_no_trailing_barrier(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_no_trailing_barrier)
    assert "does not take trailing barrier expression" in str(exc.value)


def test_pyparser_unexpected_shift():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: i32, y: i32):
            x >> y

    assert ">>" in str(exc.value)


def test_pyparser_unexpected_plus():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: i32, y: i32):
            x * y

    assert ">>" in str(exc.value)


def test_pyparser_fence_trailing_barrier_exprs():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: i32, y: i32):
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Fence(cuda_in_order, cuda_in_order) >> bar

    assert "Fence" in str(exc.value)
    assert ">>" in str(exc.value)


def test_pyparser_await_trailing_barrier_exprs():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: i32, y: i32):
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Await(bar, cuda_in_order, 0) >> bar

    assert "Await" in str(exc.value)
    assert ">>" in str(exc.value)


@instr
class bogus_test_instr:
    def behavior():
        pass

    def instance(self):
        self.coll_unit = cuda_thread
        self.instr_tl = cuda_in_order_instr
        self.instr_format = ["// bogus_test_instr"]


def test_pyparser_too_many_BarrierExpr():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    for tid in cuda_threads(0, 32):
                        bar: barrier @ CudaMbarrier
                        Arrive(cuda_in_order, 1) >> bar
                        bogus_test_instr() >> bar >> bar
                        Await(bar, cuda_in_order, 0)

    assert "bogus_test_instr cannot have more than 1 trailing barrier expr" in str(
        exc.value
    )


def test_pyparser_suggest_sync_tl():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    Fence(Sm80_cp_async_instr, cuda_in_order)

    assert "Sm80_cp_async?" in str(exc.value)


def test_pyparser_not_sync_tl():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    Fence("xyzzy", cuda_in_order)

    assert "xyzzy" in str(exc.value)


def test_pyparser_Arrive_kwarg():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Arrive(cuda_in_order, 1, bogus_kwarg=19) >> bar
                    Await(bar, cuda_in_order, 0)

    assert "bogus_kwarg" in str(exc.value)


def test_pyparser_Arrive_wrong_arg_count():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Arrive(cuda_in_order, 1, 19) >> bar
                    Await(bar, cuda_in_order, 0)

    assert "Arrive expects 2 arguments" in str(exc.value)


def test_pyparser_Await_wrong_arg_count():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Arrive(cuda_in_order, 1) >> bar
                    Await(bar, cuda_in_order)

    assert "Await expects 3 arguments" in str(exc.value)


def test_pyparser_Fence_wrong_arg_count():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    Fence()

    assert "Fence expects 2 arguments" in str(exc.value)


def test_pyparser_Await_N_not_int():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Arrive(cuda_in_order, 1) >> bar
                    Await(bar, cuda_in_order, 0.75)

    assert "Await" in str(exc.value)
    assert "0.75" in str(exc.value)


def test_pyparser_unexpected_BarrierType():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: f32 @ CudaMbarrier):
            pass

    msg = str(exc.value)
    assert "CudaMbarrier" in msg


def test_typecheck_barrier_type():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(xyzzy: f32):
            Arrive(cuda_in_order, 1) >> xyzzy

    msg = str(exc.value)
    assert "requires barrier type" in msg
    assert "xyzzy: f32" in msg


def test_typecheck_barrier_indices():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(xyzzy: f32):
            with CudaDeviceFunction(clusterDim=8, blockDim=256):
                for task in cuda_tasks(0, 4):
                    bars: barrier[2, 4] @ CudaMbarrier
                    for m in cuda_threads(0, 2, unit=4 * cuda_cta_in_cluster):
                        Arrive(cuda_in_order, 1) >> bars[m]

    msg = str(exc.value)
    assert "expected 2 indices" in msg


multicast_await_bug = 1
multicast_commit_group_bug = 2
home_barrier_different_barriers_bug = 3
home_barrier_binop_bug = 4
home_barrier_different_idxs_bug = 5
home_barrier_missing_vars_bug = 6
arrive_no_barriers_bug = 7


def mkproc_broken_barrier_exprs(bug):
    @proc
    def proc_broken_barrier_exprs():
        with CudaDeviceFunction(clusterDim=4, blockDim=128):
            for task in cuda_tasks(0, 4):
                mbar_a: barrier[2, 2] @ CudaMbarrier
                mbar_b: barrier[2, 2] @ CudaMbarrier
                for m_cta in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                    for n_cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                        if bug == home_barrier_missing_vars_bug:
                            Arrive(cuda_in_order) >> mbar_a[m_cta, :]
                        else:
                            (
                                Arrive(cuda_in_order)
                                >> mbar_a[m_cta, :]
                                >> mbar_a[m_cta, n_cta]
                            )
                        if bug == multicast_await_bug:
                            Await(mbar_a[m_cta, :], cuda_in_order, ~0)
                        else:
                            Await(mbar_a[m_cta, n_cta], cuda_in_order, ~0)

                        cg: barrier[128] @ CudaCommitGroup
                        for tid in cuda_threads(0, 128):
                            if bug == multicast_commit_group_bug:
                                Arrive(Sm80_cp_async) >> cg[tid] >> cg[:]
                            else:
                                Arrive(Sm80_cp_async) >> cg[tid]
                            Await(cg[tid], cuda_in_order, 0)

                        if bug == home_barrier_different_idxs_bug:
                            (
                                Arrive(cuda_in_order)
                                >> mbar_b[m_cta, n_cta]
                                >> mbar_b[:, m_cta]
                            )
                        else:
                            (
                                Arrive(cuda_in_order)
                                >> mbar_b[m_cta, n_cta]
                                >> mbar_b[:, n_cta]
                            )

                        if bug == home_barrier_binop_bug:
                            Await(mbar_b[m_cta, n_cta / 2], cuda_in_order, ~0)
                        else:
                            Await(mbar_b[m_cta, n_cta], cuda_in_order, ~0)

                        if bug == home_barrier_different_barriers_bug:
                            (
                                Arrive(cuda_in_order)
                                >> mbar_a[m_cta, n_cta]
                                >> mbar_b[m_cta, n_cta]
                            )
                        if bug == arrive_no_barriers_bug:
                            Arrive(cuda_in_order)

    return simplify(proc_broken_barrier_exprs)


def test_home_barrier_expr_positive(compiler):
    compiler.cuda_cpu_test(mkproc_broken_barrier_exprs, bug=0)


def test_forbid_multicast_await(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_broken_barrier_exprs, bug=multicast_await_bug)
    msg = str(exc.value)
    assert "Unsupported multicast" in msg
    assert "Await" in msg


def test_forbid_multicast_commit_group(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_broken_barrier_exprs, bug=multicast_commit_group_bug
        )
    msg = str(exc.value)
    assert "Unsupported multicast" in msg
    assert "CudaCommitGroup" in msg


def test_home_barrier_different_barriers(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_broken_barrier_exprs, bug=home_barrier_different_barriers_bug
        )
    msg = str(exc.value)
    assert "different barrier variables" in msg
    assert "mbar_a" in msg
    assert "mbar_b" in msg
    assert "cg" not in msg


def test_home_barrier_binop(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_broken_barrier_exprs, bug=home_barrier_binop_bug)
    msg = str(exc.value)
    assert "plain variable" in msg
    assert "n_cta / 2" in msg


def test_home_barrier_different_idxs(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_broken_barrier_exprs, bug=home_barrier_different_idxs_bug
        )
    msg = str(exc.value)
    assert "mismatches" in msg
    assert "m_cta" in msg
    assert "n_cta" in msg


def test_home_barrier_missing_vars(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_broken_barrier_exprs, bug=home_barrier_missing_vars_bug
        )
    msg = str(exc.value)
    assert "not an interval" in msg


def test_arrive_no_barriers(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_broken_barrier_exprs, bug=arrive_no_barriers_bug)
    msg = str(exc.value)
    assert "missing >>" in msg


def mkproc_await_wrong_N(barrier_mechanism, N, N2):
    first_sync_tl = (
        tma_to_gmem_async if barrier_mechanism == CudaCommitGroup else cuda_in_order
    )

    @proc
    def proc_await_wrong_N():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier @ barrier_mechanism
                Arrive(first_sync_tl) >> bar
                Await(bar, cuda_generic_and_async_proxy, N)
                Arrive(first_sync_tl) >> bar
                Await(bar, cuda_generic_and_async_proxy, N2)

    return proc_await_wrong_N


def test_await_wrong_N(compiler):
    mkproc = mkproc_await_wrong_N

    def helper(barrier_mechanism, N, N2, err):
        if not err:
            compiler.cuda_cpu_test(
                mkproc, barrier_mechanism=barrier_mechanism, N=N, N2=N2
            )
        else:
            with pytest.raises(Exception) as exc:
                compiler.cuda_cpu_test(
                    mkproc, barrier_mechanism=barrier_mechanism, N=N, N2=N2
                )
            msg = str(exc.value)
            assert err in msg

    helper(CudaClusterSync, -1, -1, "N = 0")
    helper(CudaClusterSync, 0, 0, None)
    helper(CudaClusterSync, +1, +1, "N = 0")
    helper(CudaMbarrier, -1, -1, None)
    helper(CudaMbarrier, 0, 0, "N < 0 (e.g. N = ~0)")
    helper(CudaMbarrier, +1, +1, "N < 0 (e.g. N = ~0)")
    helper(CudaCommitGroup, -1, -1, "N >= 0")
    helper(CudaCommitGroup, 0, 0, None)
    helper(CudaCommitGroup, +1, +1, None)

    helper(CudaCommitGroup, +1, +2, None)
    helper(CudaMbarrier, ~1, ~2, "uniform-N")


# This will be used to test the 3 guard cycle case and guarding
# failures due to thread mismatches.
# Note, I think Claude generated some other barrier guard tests,
# but not involving thread mismatch issues.
def mkproc_3cycle_mbarriers(
    A_delay,
    B_delay,
    C_delay,
    wrong_box=False,
    wrong_offset=False,
    wrong_thread_pitch=False,
    wrong_duplicate_await=False,
):
    warp_config = [
        CudaWarpConfig("A", 2),
        CudaWarpConfig("B", 2),
        CudaWarpConfig("C", 8),
    ]

    if wrong_offset:
        test_2, test_4 = 0, 2
    elif wrong_box:
        test_2, test_4 = 2, 3
    else:
        test_2, test_4 = 2, 4

    if wrong_thread_pitch:
        test_warpgroup = 2 * cuda_warp
        test_2, test_4 = 0, 1
    else:
        test_warpgroup = cuda_warpgroup

    if wrong_duplicate_await:
        test_0, test_1 = 1, 2
    else:
        test_0, test_1 = 0, 1

    @proc
    def proc_3cycle_mbarriers():
        with CudaDeviceFunction(warp_config=warp_config):
            for task in cuda_tasks(0, 1):
                AtoB: barrier[2] @ CudaMbarrier
                BtoC: barrier(AtoB)[2] @ CudaMbarrier
                CtoA: barrier(BtoC)[2] @ CudaMbarrier
                for n in seq(0, 25):
                    with CudaWarps(name="B"):
                        for w in cuda_threads(0, 2, unit=cuda_warp):
                            Await(AtoB[w], cuda_in_order, ~B_delay)
                            Arrive(cuda_in_order) >> BtoC[w]
                    with CudaWarps(name="C"):
                        for wg in cuda_threads(0, 2, unit=test_warpgroup):
                            with CudaWarps(test_2, test_4):
                                Await(BtoC[wg], cuda_in_order, ~C_delay)
                        for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                            with CudaWarps(2, 4):
                                Arrive(cuda_in_order) >> CtoA[wg]
                    with CudaWarps(name="A"):
                        # Single thread is valid!
                        if n == 0:
                            for tid in cuda_threads(0, 2, unit=cuda_thread):
                                Await(CtoA[tid], cuda_in_order, ~A_delay)
                                Arrive(cuda_in_order) >> AtoB[tid]
                        else:
                            with CudaWarps(test_0, test_1):
                                for tid in cuda_threads(0, 2, unit=cuda_thread):
                                    Await(CtoA[tid], cuda_in_order, ~A_delay)
                                    Arrive(cuda_in_order) >> AtoB[tid]

    return proc_3cycle_mbarriers


def mkref_3cycle_mbarriers(
    xrg: excut.ExcutReferenceGenerator,
    A_delay,
    B_delay,
    C_delay,
):
    xrg.begin_cuda()
    mbarriers = xrg.new_varname("mbarriers")
    depth = A_delay + B_delay + C_delay

    def Await(bar_name, n, i):
        assert i < 2
        if bar_name == "AtoB":
            delay = B_delay
            var_idx = 0
        elif bar_name == "BtoC":
            delay = C_delay
            var_idx = 1
        elif bar_name == "CtoA":
            delay = A_delay
            var_idx = 2
        else:
            assert 0

        if n >= delay:
            r = (n - delay) % depth
            parity = ((n - delay) // depth) & 1
            var = mbarriers[var_idx, i, r]
            xrg("mbarrier.test_wait.parity.acquire.cta.shared::cta.b64", var, parity)

    for task in xrg.stride_blockIdx(1):
        # Thread 0 sets up mbarriers
        with xrg.permuted():
            for r in range(0, depth):
                for i in range(0, 2):
                    # AtoB
                    xrg("mbarrier.init.shared::cta.b64", mbarriers[0, i, r], 1)
                    # BtoC
                    xrg("mbarrier.init.shared::cta.b64", mbarriers[1, i, r], 32)
                    # CtoA
                    xrg("mbarrier.init.shared::cta.b64", mbarriers[2, i, r], 64)
        for n in range(0, 25):
            # with CudaWarps("B")
            for w in xrg.stride_threadIdx(2, stride=32, offset=64):
                for lane in xrg.stride_threadIdx(32):
                    Await("AtoB", n, w)
                    xrg("mbarrier.arrive.shared::cta.b64", mbarriers[1, w, n % depth])
            # with CudaWarps("C")
            for wg in xrg.stride_threadIdx(2, stride=128, offset=128):
                # with CudaWarps(2, 4)
                for lane in xrg.stride_threadIdx(64, stride=1, offset=64):
                    Await("BtoC", n, wg)
                    xrg("mbarrier.arrive.shared::cta.b64", mbarriers[2, wg, n % depth])
            # with CudaWarps("A")
            for tid in xrg.stride_threadIdx(2, stride=1, offset=0):
                Await("CtoA", n, tid)
                xrg("mbarrier.arrive.shared::cta.b64", mbarriers[0, tid, n % depth])
    xrg.end_cuda()


def test_3cycle_mbarriers_excut(compiler_Sm80):
    compiler_Sm80.excut_test(
        mkproc_3cycle_mbarriers,
        mkref_3cycle_mbarriers,
        A_delay=1,
        B_delay=2,
        C_delay=3,
    )


def test_3cycle_mbarriers_positive(compiler, golden):
    compiler.cuda_cpu_test(
        mkproc_3cycle_mbarriers, A_delay=1, B_delay=2, C_delay=3, golden=golden
    )


def test_3cycle_mbarriers_wrong_box(compiler, golden):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_3cycle_mbarriers, A_delay=1, B_delay=2, C_delay=3, wrong_box=True
        )

    assert golden == get_remarks_golden(compiler, "proc_3cycle_mbarriers")

    msg = str(exc.value)
    assert "Incompatible box size" in msg


def test_3cycle_mbarriers_wrong_offset(compiler, golden):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_3cycle_mbarriers, A_delay=1, B_delay=2, C_delay=3, wrong_offset=True
        )

    assert golden == get_remarks_golden(compiler, "proc_3cycle_mbarriers")

    msg = str(exc.value)
    assert "Incompatible offset" in msg


def test_3cycle_mbarriers_wrong_thread_pitch(compiler, golden):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_3cycle_mbarriers,
            A_delay=1,
            B_delay=2,
            C_delay=3,
            wrong_thread_pitch=True,
        )

    assert golden == get_remarks_golden(compiler, "proc_3cycle_mbarriers")

    msg = str(exc.value)
    assert "inconsistent thread pitch with previous Await(BtoC" in msg


def test_3cycle_mbarriers_wrong_duplicate_await(compiler, golden):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_3cycle_mbarriers,
            A_delay=1,
            B_delay=2,
            C_delay=3,
            wrong_duplicate_await=True,
        )

    assert golden == get_remarks_golden(compiler, "proc_3cycle_mbarriers")

    msg = str(exc.value)
    assert ": Await(CtoA[tid]" in msg
    assert "has inconsistent collective tiling with previous Await(CtoA[tid]" in msg
