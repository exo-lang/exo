from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *

from exo.spork import excut


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
                        with CudaAsync(wgmma_async):
                            if have_fence:
                                Fence(first_sync_tl, second_sync_tl)

    return simplify(test_proc)


def test_wgmma_fence_positive(compiler, golden):
    compiler.cuda_cpu_test(mkproc_wgmma_fence, golden, sm="90a")


def test_wgmma_fence_missing_prologue(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_wgmma_fence, have_fence=False)
    assert "missing prologue sync in CudaAsync(wgmma_async_instr)" in str(exc.value)


def test_wgmma_fence_wrong_prologue(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_wgmma_fence,
            first_sync_tl=cuda_temporal,
            second_sync_tl=cuda_temporal,
            unit=cuda_cta_in_cluster,
            lo=0,
            hi=12,
            sm="90a",
        )
    assert "wrong prologue sync in CudaAsync(wgmma_async_instr)" in str(exc.value)


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


def mkproc_cluster_sync_unit(unit, await_lo=0, await_hi=8):
    @proc
    def test_proc():
        with CudaDeviceFunction(clusterDim=4, blockDim=256):
            for task in cuda_tasks(0, 1):
                for u in cuda_threads(0, 1, unit=unit):
                    sync: barrier @ CudaClusterSync
                    Arrive(cuda_in_order, 1) >> sync
                    with CudaWarps(await_lo, await_hi):
                        Await(sync, cuda_in_order, 0)

    return test_proc


def test_cluster_sync_unit_baseline(compiler):
    # Correct usage of CudaClusterSync
    compiler.cuda_cpu_test(mkproc_cluster_sync_unit, unit=cuda_cluster, sm="90a")


def test_cluster_sync_unit_cta(compiler):
    # Only 1 CTA involved in CudaClusterSync, expect full cluster
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cluster_sync_unit, unit=cuda_cta_in_cluster)
    msg = str(exc.value)
    assert "full cluster" in msg


def test_cluster_sync_unit_warp(compiler):
    # Only 1 warp per CTA involved in CudaClusterSync, expect full cluster
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cluster_sync_unit, unit=cuda_warp, await_hi=1)
    msg = str(exc.value)
    assert "full cluster" in msg


def test_cluster_sync_unit_await(compiler):
    # Partial warps missing in Await for CudaClusterSync.
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cluster_sync_unit, unit=cuda_cluster, await_lo=4)
    msg = str(exc.value)
    assert "Await" in msg


def mkproc_commit_group(first_sync_tl, second_sync_tl, unit):
    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                cg: barrier[2] @ CudaCommitGroup
                for t in cuda_threads(0, 2, unit=unit):
                    Arrive(first_sync_tl, 1) >> cg[t]
                    Await(cg[t], second_sync_tl, 0)

    return test_proc


def test_wgmma_commit_group_async_proxy(compiler, golden):
    # wgmma -> TMA is OK (wgmma is already in the async proxy)
    compiler.cuda_cpu_test(
        mkproc_commit_group,
        first_sync_tl=wgmma_async,
        second_sync_tl=tma_to_gmem_async,
        unit=cuda_warpgroup,
        golden=golden,
    )


def test_Sm80_commit_group_async_proxy(compiler):
    # Sm80_cp_async -> TMA is not OK (Sm80_cp_async is in the generic proxy)
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_commit_group,
            first_sync_tl=Sm80_cp_async,
            second_sync_tl=tma_to_gmem_async,
            unit=cuda_thread,
        )
    msg = str(exc.value)
    assert "cg" in msg
    assert "Await" in msg
    assert "tma_to_gmem_async" in msg


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

# cuda_in_order -> wgmma requires generic -> async proxy fence
mbarrier_in_order_to_wgmma_qc = MbarrierQualConfig(
    cuda_in_order, wgmma_async_smem, "try", False, True, True
)

# cuda_temporal -> wgmma doesn't require the fence after the await (cuda_temporal
# resolves only WAR hazards), but we still need the proxy fence at startup.
mbarrier_temporal_to_wgmma_qc = MbarrierQualConfig(
    cuda_temporal, wgmma_async_smem, "try", False, False, True
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
                    row_bars: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    col_bars: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    all_bars: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    rc_bars: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    baseline: barrier[M_CTA, N_CTA, 2] @ CudaMbarrier
                    for i in seq(0, 5):
                        for t1 in cuda_threads(0, 2, unit=8 * cuda_thread):
                            for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                                for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                                    # Note baseline mbarrier doesn't use delay or parameterized sync-tl
                                    Arrive(cuda_in_order, 1) >> baseline[m_cta, n_cta, t1]

                                    # Only rc_bars uses the back queue barrier array (-rc_bars)
                                    # Its ring buffer depth is f_delay + b_delay, instead of 1 + f_delay
                                    Await(-rc_bars[m_cta, n_cta, t1], second_sync_tl, ~b_delay)
                                    Arrive(first_sync_tl, 1) >> +rc_bars[m_cta, n_cta, t1]

                                    Arrive(first_sync_tl, 1) >> row_bars[m_cta, n_cta, t1] >> row_bars[m_cta, :, t1]
                                    Await(row_bars[m_cta, n_cta, t1], second_sync_tl, ~f_delay)
                                    Arrive(first_sync_tl, 1) >> col_bars[m_cta, n_cta, t1] >> col_bars[:, n_cta, t1]
                                    Await(col_bars[m_cta, n_cta, t1], second_sync_tl, ~f_delay)
                                    Arrive(first_sync_tl, 1) >> all_bars[m_cta, n_cta, t1] >> all_bars[:, :, t1]
                                    Await(all_bars[m_cta, n_cta, t1], second_sync_tl, ~f_delay)

                                    Await(+rc_bars[m_cta, n_cta, t1], second_sync_tl, ~f_delay)
                                    Arrive(first_sync_tl, 1) >> -rc_bars[m_cta, :, t1] >> -rc_bars[:, n_cta, t1]

                                    Await(baseline[m_cta, n_cta, t1], cuda_in_order, ~0)
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
    assert "consider wgmma_async_smem" in str(exc.value)


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
    assert "rc_bars must have some await with nonzero skips" in str(exc.value)


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
