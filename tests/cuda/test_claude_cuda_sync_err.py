"""Tests for error conditions in spork/cuda_sync_state.py

These tests cover error paths related to CUDA synchronization lowering.
"""

from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *


# =============================================================================
# barrier mechanism not supported in CUDA device function
# =============================================================================


def mkproc_unsupported_barrier_mechanism():
    """TODO: This test requires a BarrierMechanism that is not one of
    CudaMbarrier, CudaCommitGroup, or CudaClusterSync.
    Currently all available barrier mechanisms are supported.
    """
    pass


def test_unsupported_barrier_mechanism():
    # TODO: Implement when there's a barrier mechanism not supported in CUDA
    pytest.skip("All current barrier mechanisms are supported in CUDA device functions")


# =============================================================================
# mbarrier Arrive sync-tl tma_to_smem_async not supported
# =============================================================================


def mkproc_mbarrier_arrive_tma_to_smem():
    """Try to use tma_to_smem_async as Arrive sync-tl for mbarrier.
    The error message suggests using cuda_temporal and adding trailing barriers.
    """

    @proc
    def test_proc(foo: f32[128] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                # Barrier must be distributed per warpgroup to avoid distributed memory errors
                bar: barrier[1] @ CudaMbarrier
                for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
                    # tma_to_smem_async should not be directly used as Arrive sync-tl
                    Arrive(tma_to_smem_async, 1) >> bar[wg]
                    Await(bar[wg], cuda_in_order, ~1)

    return simplify(test_proc)


def test_mbarrier_arrive_tma_to_smem(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarrier_arrive_tma_to_smem)
    msg = str(exc.value)
    assert "tma_to_smem_async" in msg and "Arrive" in msg


# =============================================================================
# Sm80_cp_async_f32 does not take trailing barrier expression
# =============================================================================


def mkproc_sm80_cp_async_trailing_barrier():
    """Sm80_cp_async_f32 instruction does not support trailing barrier syntax"""

    @proc
    def test_proc(gmem: f32[128] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32, clusterDim=2):
            for task in cuda_tasks(0, 1):
                smem: f32[128] @ CudaSmemLinear
                bar: barrier[2] @ CudaMbarrier
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for tid in cuda_threads(0, 32):
                        # Sm80_cp_async_f32 does NOT support >> bar trailing syntax
                        (
                            Sm80_cp_async_f32(
                                smem[4 * tid : 4 * tid + 4],
                                gmem[4 * tid : 4 * tid + 4],
                                size=4,
                            )
                            >> bar[cta]
                        )
                    Arrive(Sm80_cp_async, 1) >> bar[cta]
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(bar[cta], cuda_in_order, ~1)

    return simplify(test_proc)


def test_sm80_cp_async_trailing_barrier(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_sm80_cp_async_trailing_barrier)
    msg = str(exc.value)
    assert "Sm80_cp_async_f32" in msg and "trailing barrier" in msg.lower()


# =============================================================================
# mbarrier Await sync-tl errors (wrong second sync-tl)
# =============================================================================


def mkproc_mbarrier_await_wrong_sync_tl():
    """Try to use an unsupported second sync-tl for mbarrier Await"""

    @proc
    def test_proc(foo: f32[128] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                # Barrier distributed per warpgroup
                bar: barrier[1] @ CudaMbarrier
                for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
                    Arrive(cuda_in_order, 1) >> bar[wg]
                    # wgmma_async is not valid as second sync-tl for mbarrier Await
                    Await(bar[wg], wgmma_async, ~1)

    return simplify(test_proc)


def test_mbarrier_await_wrong_sync_tl(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarrier_await_wrong_sync_tl)
    msg = str(exc.value)
    assert "mbarrier" in msg.lower() and "Await" in msg and "sync-tl" in msg.lower()


# =============================================================================
# Fence sync-tl errors
# =============================================================================


def mkproc_fence(first_sync_tl, second_sync_tl):
    """Create a Fence with specified sync-tl pair"""

    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, 32):
                    Fence(first_sync_tl, second_sync_tl)

    return simplify(test_proc)


def test_fence_unsupported_first_sync_tl_negative(compiler):
    with pytest.raises(Exception) as exc:
        # wgmma_async is not valid as first sync-tl for Fence
        compiler.cuda_cpu_test(
            mkproc_fence, first_sync_tl=wgmma_async, second_sync_tl=cuda_in_order
        )
    msg = str(exc.value)
    assert "Fence" in msg and "first" in msg.lower() and "sync-tl" in msg.lower()


def test_fence_sm80_cp_async_second_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Sm80_cp_async is not supported as second sync-tl in Fence
        compiler.cuda_cpu_test(
            mkproc_fence, first_sync_tl=cuda_in_order, second_sync_tl=Sm80_cp_async
        )
    msg = str(exc.value)
    assert (
        "Sm80_cp_async" in msg and "second" in msg.lower() and "sync-tl" in msg.lower()
    )


def test_fence_valid_positive(compiler):
    # cuda_in_order -> cuda_in_order is valid
    compiler.cuda_cpu_test(
        mkproc_fence, first_sync_tl=cuda_in_order, second_sync_tl=cuda_in_order
    )


# =============================================================================
# wgmma fence tests
# =============================================================================


def mkproc_wgmma_fence(second_sync_tl, use_warpgroup_unit=True, num_threads=128):
    """Create a wgmma fence with specified second sync-tl and thread configuration"""
    device_fn = CudaDeviceFunction(blockDim=num_threads)

    if use_warpgroup_unit:
        # Proper warpgroup usage: one warpgroup executes the fence
        @proc
        def test_proc(foo: f32 @ CudaGmemLinear):
            with device_fn:
                for task in cuda_tasks(0, 1):
                    for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
                        Fence(wgmma_fence_1, second_sync_tl)

    else:
        # Wrong: individual threads instead of warpgroup unit
        @proc
        def test_proc(foo: f32 @ CudaGmemLinear):
            with device_fn:
                for task in cuda_tasks(0, 1):
                    for tid in cuda_threads(0, num_threads):
                        Fence(wgmma_fence_1, second_sync_tl)

    return simplify(test_proc)


def test_wgmma_fence_wrong_second_negative(compiler):
    with pytest.raises(Exception) as exc:
        # wgmma_fence_1 requires wgmma_fence_2 as second, not cuda_in_order
        compiler.cuda_cpu_test(
            mkproc_wgmma_fence, second_sync_tl=cuda_in_order, use_warpgroup_unit=True
        )
    msg = str(exc.value)
    assert "wgmma" in msg.lower() and "fence" in msg.lower() and "second" in msg.lower()


def test_wgmma_fence_not_warpgroup_negative(compiler):
    with pytest.raises(Exception) as exc:
        # wgmma fence must be executed by exactly one warpgroup, not individual threads
        compiler.cuda_cpu_test(
            mkproc_wgmma_fence, second_sync_tl=wgmma_fence_2, use_warpgroup_unit=False
        )
    msg = str(exc.value)
    assert "wgmma" in msg.lower() and "warpgroup" in msg.lower()


def test_wgmma_fence_valid_positive(compiler):
    # Correct: wgmma_fence_1 -> wgmma_fence_2 executed by one warpgroup
    compiler.cuda_cpu_test(
        mkproc_wgmma_fence, second_sync_tl=wgmma_fence_2, use_warpgroup_unit=True
    )


# =============================================================================
# mbarrier must be distributed so each is resident in 1 CTA only
# =============================================================================


def mkproc_mbarrier_not_sub_cta(bar_count, cta_unit_scale):
    """Test that mbarrier must be distributed so each is resident in 1 CTA only.

    In a cluster with multiple CTAs, mbarrier elements must not span multiple CTAs.

    bar_count=4, cta_unit_scale=1: one barrier per CTA (valid)
    bar_count=2, cta_unit_scale=2: each barrier spans 2 CTAs (invalid)
    """
    device_fn = CudaDeviceFunction(clusterDim=4, blockDim=256)
    unit = cta_unit_scale * cuda_cta_in_cluster

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                bar: barrier[bar_count] @ CudaMbarrier
                for cta in cuda_threads(0, bar_count, unit=unit):
                    Arrive(cuda_in_order, 1) >> bar[cta]
                    Await(bar[cta], cuda_in_order, ~1)

    return simplify(test_proc)


def test_mbarrier_not_sub_cta_positive(compiler):
    # Valid: 4 barriers, one per CTA (unit=1*cuda_cta_in_cluster)
    compiler.cuda_cpu_test(mkproc_mbarrier_not_sub_cta, bar_count=4, cta_unit_scale=1)


def test_mbarrier_not_sub_cta_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: 2 barriers spanning 2 CTAs each (unit=2*cuda_cta_in_cluster)
        compiler.cuda_cpu_test(
            mkproc_mbarrier_not_sub_cta, bar_count=2, cta_unit_scale=2
        )
    msg = str(exc.value)
    assert "mbarrier" in msg.lower() and "resident in 1 CTA" in msg


# =============================================================================
# mbarrier ring buffer skip count errors
# =============================================================================


def mkproc_mbarrier_zero_skips(has_nonzero_skips):
    """Test that mbarrier ring buffer cycle must have some await with nonzero skips.

    For guarded barrier pairs (bar1 guards bar2), the total skips across
    the guard cycle must be > 0. If both Awaits use N=~0 (0 skips), error.

    has_nonzero_skips=True: At least one Await has N=~1 (valid)
    has_nonzero_skips=False: Both Awaits have N=~0, total skips=0 (invalid)
    """

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar1: barrier[1] @ CudaMbarrier
                bar2: barrier(bar1)[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    if has_nonzero_skips:
                        # Valid: at least one Await has nonzero skips
                        Await(bar2[w], cuda_in_order, ~1)  # 1 skip
                        Arrive(cuda_in_order, 1) >> bar1[w]
                        Await(bar1[w], cuda_in_order, ~0)  # 0 skips, but total=1
                        Arrive(cuda_in_order, 1) >> bar2[w]
                    else:
                        # Invalid: both Awaits have 0 skips, total=0
                        Await(bar2[w], cuda_in_order, ~0)  # 0 skips
                        Arrive(cuda_in_order, 1) >> bar1[w]
                        Await(bar1[w], cuda_in_order, ~0)  # 0 skips
                        Arrive(cuda_in_order, 1) >> bar2[w]

    return simplify(test_proc)


def test_mbarrier_zero_skips_positive(compiler):
    # Valid: at least one Await has nonzero skips
    compiler.cuda_cpu_test(mkproc_mbarrier_zero_skips, has_nonzero_skips=True)


def test_mbarrier_zero_skips_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: total skips across guard cycle is 0
        compiler.cuda_cpu_test(mkproc_mbarrier_zero_skips, has_nonzero_skips=False)
    msg = str(exc.value)
    assert "nonzero skips" in msg.lower() and ("bar1" in msg or "bar2" in msg)


# =============================================================================
# Sm80_cp_async mbarrier must be within 1 CTA (no cluster multicast)
# =============================================================================


def mkproc_sm80_cp_async_mbarrier_cluster(use_multicast):
    """Test that Sm80_cp_async mbarrier must be within 1 CTA.

    When using Sm80_cp_async as the Arrive sync-tl, the mbarrier cannot
    use cluster multicast (must stay within a single CTA).

    use_multicast=False: single CTA, no multicast (valid)
    use_multicast=True: multicast across CTAs (invalid)
    """
    device_fn = CudaDeviceFunction(clusterDim=2, blockDim=32)

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                bar: barrier[2] @ CudaMbarrier
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    if use_multicast:
                        # Invalid: Sm80_cp_async with multicast to other CTA
                        Arrive(Sm80_cp_async, 1) >> bar[cta] >> bar[:]
                    else:
                        # Valid: Sm80_cp_async within single CTA (no multicast)
                        Arrive(Sm80_cp_async, 1) >> bar[cta]
                    Await(bar[cta], cuda_in_order, ~1)

    return simplify(test_proc)


def test_sm80_cp_async_mbarrier_cluster_positive(compiler):
    # Valid: Sm80_cp_async within single CTA
    compiler.cuda_cpu_test(mkproc_sm80_cp_async_mbarrier_cluster, use_multicast=False)


def test_sm80_cp_async_mbarrier_cluster_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: Sm80_cp_async with cluster multicast
        compiler.cuda_cpu_test(
            mkproc_sm80_cp_async_mbarrier_cluster, use_multicast=True
        )
    msg = str(exc.value)
    assert "Sm80_cp_async" in msg and "1 CTA" in msg


# =============================================================================
# mbarrier multicast on intra-CTA dimension (not allowed)
# =============================================================================


def mkproc_mbarrier_intra_cta_multicast(multicast_on_warp):
    """Test that mbarrier multicast cannot be on intra-CTA dimensions.

    Multicast (using ':' interval syntax) is only allowed for CTA-level
    dimensions (thread_pitch >= blockDim). Trying to multicast on warp or
    thread dimensions within a CTA should fail.

    multicast_on_warp=False: multicast on CTA dimension (valid)
    multicast_on_warp=True: multicast on warp dimension (invalid)
    """
    device_fn = CudaDeviceFunction(clusterDim=2, blockDim=64)

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                # 2D barrier: [cta, warp]
                bar: barrier[2, 2] @ CudaMbarrier
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for w in cuda_threads(0, 2, unit=cuda_warp):
                        if multicast_on_warp:
                            # Invalid: multicast on warp dimension (intra-CTA)
                            Arrive(cuda_in_order, 1) >> bar[cta, w] >> bar[cta, :]
                        else:
                            # Valid: multicast on CTA dimension (inter-CTA)
                            Arrive(cuda_in_order, 1) >> bar[cta, w] >> bar[:, w]
                        Await(bar[cta, w], cuda_in_order, ~1)

    return simplify(test_proc)


def test_mbarrier_intra_cta_multicast_positive(compiler):
    # Valid: multicast on CTA dimension
    compiler.cuda_cpu_test(mkproc_mbarrier_intra_cta_multicast, multicast_on_warp=False)


def test_mbarrier_intra_cta_multicast_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: multicast on warp dimension (intra-CTA)
        compiler.cuda_cpu_test(
            mkproc_mbarrier_intra_cta_multicast, multicast_on_warp=True
        )
    msg = str(exc.value)
    # Error: thread_pitch not divisible by blockDim; cannot be multicast
    assert "cannot be multicast" in msg.lower() or "thread_pitch" in msg
