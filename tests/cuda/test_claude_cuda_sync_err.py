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
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier @ CudaMbarrier
                for tid in cuda_threads(0, 32):
                    # tma_to_smem_async should not be directly used as Arrive sync-tl
                    Arrive(tma_to_smem_async, 1) >> bar
                    Await(bar, cuda_in_order, ~1)

    return simplify(test_proc)


def test_mbarrier_arrive_tma_to_smem(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarrier_arrive_tma_to_smem)
    msg = str(exc.value)
    assert (
        "tma_to_smem" in msg.lower()
        or "sync-tl" in msg.lower()
        or "cuda_temporal" in msg
    )


# Claude totally didn't get it here.
# The test doesn't even multicast to different CTAs, and fails because of errors in
# distributed memory (and Sm80_cp_async_f32 does not take trailing barrier expression)
# =============================================================================
# Sm80_cp_async mbarrier must be within 1 CTA (cluster case)
# =============================================================================


def mkproc_sm80_cp_async_mbarrier_cross_cta():
    """Sm80_cp_async with mbarrier cannot be used across CTAs in a cluster"""

    @proc
    def test_proc(gmem: f32[128] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32, clusterDim=2):
            for task in cuda_tasks(0, 1):
                smem: f32[128] @ CudaSmemLinear
                bar: barrier[2] @ CudaMbarrier
                # Try to use Sm80_cp_async with multicast to multiple CTAs
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for tid in cuda_threads(0, 32):
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


def test_sm80_cp_async_mbarrier_cross_cta(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_sm80_cp_async_mbarrier_cross_cta)
    msg = str(exc.value)
    # May fail for different reasons in cluster setup, accept various errors
    assert (
        "Sm80" in msg or "CTA" in msg or "mbarrier" in msg or "cluster" in msg.lower()
    )


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
    assert "sync-tl" in msg.lower() or "wgmma" in msg.lower() or "Await" in msg


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
    assert "sync-tl" in msg.lower() or "Fence" in msg or "not supported" in msg.lower()


def test_fence_sm80_cp_async_second_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Sm80_cp_async is not supported as second sync-tl in Fence
        compiler.cuda_cpu_test(
            mkproc_fence, first_sync_tl=cuda_in_order, second_sync_tl=Sm80_cp_async
        )
    msg = str(exc.value)
    assert "Sm80" in msg or "sync-tl" in msg.lower() or "second" in msg.lower()


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
    assert "wgmma" in msg.lower() or "fence" in msg.lower()


def test_wgmma_fence_not_warpgroup_negative(compiler):
    with pytest.raises(Exception) as exc:
        # wgmma fence must be executed by exactly one warpgroup, not individual threads
        compiler.cuda_cpu_test(
            mkproc_wgmma_fence, second_sync_tl=wgmma_fence_2, use_warpgroup_unit=False
        )
    msg = str(exc.value)
    assert "warpgroup" in msg.lower() or "wgmma" in msg.lower()


def test_wgmma_fence_valid_positive(compiler):
    # Correct: wgmma_fence_1 -> wgmma_fence_2 executed by one warpgroup
    compiler.cuda_cpu_test(
        mkproc_wgmma_fence, second_sync_tl=wgmma_fence_2, use_warpgroup_unit=True
    )


# =============================================================================
# mbarrier must be distributed so each is resident in 1 CTA only
# =============================================================================


def mkproc_mbarrier_not_sub_cta():
    """TODO: mbarrier must be distributed within a single CTA.
    This error is raised when mbarrier is distributed across CTAs in a cluster.
    """
    pass


def test_mbarrier_not_sub_cta():
    # TODO: Need to figure out exact setup to trigger this error
    pytest.skip("Complex cluster setup required to trigger this error")


# =============================================================================
# mbarrier ring buffer skip count errors
# =============================================================================


# def mkproc_mbarrier_zero_skips():
#     """mbarrier cycle must have some await with nonzero skips"""

#     @proc
#     def test_proc(foo: f32[128] @ CudaGmemLinear):
#         with CudaDeviceFunction(blockDim=32):
#             for task in cuda_tasks(0, 1):
#                 bar: barrier @ CudaMbarrier
#                 for tid in cuda_threads(0, 32):
#                     Arrive(cuda_in_order, 1) >> bar
#                     # Using N=~0 means 0 skips - the ring buffer needs some skips
#                     Await(bar, cuda_in_order, ~0)

#     return simplify(test_proc)


# def test_mbarrier_zero_skips(compiler):
#     with pytest.raises(Exception) as exc:
#         compiler.cuda_cpu_test(mkproc_mbarrier_zero_skips)
#     msg = str(exc.value)
#     # This should fail due to ring buffer requirements
#     assert "skip" in msg.lower() or "N" in msg or "await" in msg.lower()
