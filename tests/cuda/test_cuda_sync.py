from __future__ import annotations

import numpy as np
import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *

from exo.spork import excut


def invoke_test(p, mkref, compiler, golden, sm=80):
    cu = compiler.cuda_test_context(p, sm=sm, excut=golden is None)
    if golden is None:
        cu(None)
        cu.excut_concordance(mkref)
    else:
        cu.compare_golden(golden)
    return cu


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
    invoke_test(mkproc_wgmma_fence(), None, compiler, golden, sm="90a")


def test_wgmma_fence_missing_prologue(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_wgmma_fence(have_fence=False), sm="90a")
    assert "missing prologue sync in CudaAsync(wgmma_async_instr)" in str(exc.value)


def test_wgmma_fence_wrong_prologue(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(
            mkproc_wgmma_fence(
                first_sync_tl=cuda_temporal,
                second_sync_tl=cuda_temporal,
                unit=cuda_cta_in_cluster,
                lo=0,
                hi=12,
            ),
            sm="90a",
        )
    assert "wrong prologue sync in CudaAsync(wgmma_async_instr)" in str(exc.value)


def test_wgmma_fence_wrong_coll_unit_size(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_wgmma_fence(unit=cuda_warp), sm="90a")
    assert "warpgroup" in str(exc.value)


def test_wgmma_fence_wrong_coll_unit_align(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_wgmma_fence(lo=5), sm="90a")
    assert "alignment" in str(exc.value)


def test_wgmma_fence_wrong_second_sync_tl(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(
            mkproc_wgmma_fence(second_sync_tl=cuda_temporal), sm="90a"
        )
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
    p = mkproc_mixed_syncs(
        clusterDim=4,
        unit_a=cuda_cluster,
        unit_b=cuda_cta_in_cluster,
        blockDim=128,
        first_sync_tl_a=cuda_in_order,
        first_sync_tl_b=wgmma_async,
        barrier_type_a=CudaClusterSync,
        barrier_type_b=CudaCommitGroup,
    )
    compiler.cuda_test_context(p, sm="90a")


def test_mixed_syncs_solitary_cluster_sync(compiler):
    # Two CudaClusterSync in scope not allowed
    p = mkproc_mixed_syncs(
        clusterDim=4,
        unit_a=cuda_cluster,
        unit_b=cuda_cluster,
        blockDim=128,
        first_sync_tl_a=cuda_in_order,
        first_sync_tl_b=cuda_in_order,
        barrier_type_a=CudaClusterSync,
        barrier_type_b=CudaClusterSync,
    )
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm="90a")
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "barrier_b" in msg
    assert "CudaClusterSync" in msg


def test_mixed_syncs_solitary_wgmma_commit_group(compiler):
    # Two wgmma CudaCommitGroup in scope not allowed
    p = mkproc_mixed_syncs(
        clusterDim=1,
        unit_a=cuda_cluster,
        unit_b=cuda_cluster,
        blockDim=128,
        first_sync_tl_a=wgmma_async,
        first_sync_tl_b=wgmma_async,
        barrier_type_a=CudaCommitGroup,
        barrier_type_b=CudaCommitGroup,
    )
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm="90a")
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "barrier_b" in msg
    assert "CudaCommitGroup" in msg


def test_mixed_syncs_mixed_commit_group(compiler):
    # Mixed commit group of Sm80 cp.async and wgmma, should be allowed, along with cluster fence
    p = mkproc_mixed_syncs(
        fence_first_sync_tl=cuda_in_order,
        clusterDim=1,
        unit_a=cuda_warpgroup,
        unit_b=cuda_thread,
        blockDim=128,
        first_sync_tl_a=wgmma_async,
        first_sync_tl_b=Sm80_cp_async,
        barrier_type_a=CudaCommitGroup,
        barrier_type_b=CudaCommitGroup,
    )
    compiler.cuda_test_context(p, sm="90a")


def test_mixed_syncs_solitary_Sm80_commit_group(compiler):
    # Two cp.async CudaCommitGroup in scope not allowed
    p = mkproc_mixed_syncs(
        clusterDim=1,
        unit_a=cuda_thread,
        unit_b=cuda_thread,
        blockDim=128,
        first_sync_tl_a=Sm80_cp_async,
        first_sync_tl_b=Sm80_cp_async,
        barrier_type_a=CudaCommitGroup,
        barrier_type_b=CudaCommitGroup,
    )
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm=80)
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "barrier_b" in msg
    assert "CudaCommitGroup" in msg


def test_mixed_cluster_sync_fence_positive(compiler):
    # Mixed CudaClusterSync and CTA fence, allowed
    p = mkproc_mixed_syncs(
        fence_first_sync_tl=cuda_in_order,
        clusterDim=1,
        unit_a=cuda_cluster,
        unit_b=cuda_cta_in_cluster,
        blockDim=128,
        first_sync_tl_a=cuda_in_order,
        first_sync_tl_b=wgmma_async,
        barrier_type_a=CudaClusterSync,
        barrier_type_b=CudaCommitGroup,
    )
    compiler.cuda_test_context(p, sm="90a")


def test_mixed_cluster_sync_fence_negative(compiler):
    # Mixed CudaClusterSync and cluster fence, not allowed
    # The only difference from above is we have clusterDim > 1 now.
    p = mkproc_mixed_syncs(
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
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm="90a")
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "Fence" in msg


def test_mixed_syncs_wgmma_commit_group_unit(compiler):
    # wgmma commit group requires execution by 128 threads, not 64
    p = mkproc_mixed_syncs(
        clusterDim=4,
        unit_a=cuda_cluster,
        unit_b=cuda_cta_in_cluster,
        blockDim=64,
        first_sync_tl_a=cuda_in_order,
        first_sync_tl_b=wgmma_async,
        barrier_type_a=CudaClusterSync,
        barrier_type_b=CudaCommitGroup,
    )
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm="90a")
    msg = str(exc.value)
    assert "warpgroup" in msg
    assert "64" in msg


def test_mixed_syncs_Sm80_commit_group_unit(compiler):
    # Sm80_cp_async commit group requires execution by 1 thread, not 64
    p = mkproc_mixed_syncs(
        clusterDim=4,
        unit_a=cuda_cluster,
        unit_b=cuda_cta_in_cluster,
        blockDim=64,
        first_sync_tl_a=cuda_in_order,
        first_sync_tl_b=Sm80_cp_async,
        barrier_type_a=CudaClusterSync,
        barrier_type_b=CudaCommitGroup,
    )
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm="90a")
    msg = str(exc.value)
    assert "thread" in msg
    assert "64" in msg


def test_mixed_syncs_mismatch_first_sync_tl(compiler):
    p = mkproc_mixed_syncs(
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
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm="90a")
    msg = str(exc.value)
    assert "barrier_a" in msg
    assert "Arrive" in msg
    assert "cuda_in_order" in msg
    assert "cuda_temporal" in msg


def test_mixed_syncs_mismatch_second_sync_tl(compiler):
    p = mkproc_mixed_syncs(
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
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm="90a")
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
    compiler.cuda_test_context(mkproc_cluster_sync_unit(cuda_cluster), sm="90a")


def test_cluster_sync_unit_cta(compiler):
    # Only 1 CTA involved in CudaClusterSync, expect full cluster
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(
            mkproc_cluster_sync_unit(cuda_cta_in_cluster), sm="90a"
        )
    msg = str(exc.value)
    assert "full cluster" in msg


def test_cluster_sync_unit_warp(compiler):
    # Only 1 warp per CTA involved in CudaClusterSync, expect full cluster
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(
            mkproc_cluster_sync_unit(cuda_warp, await_hi=1), sm="90a"
        )
    msg = str(exc.value)
    assert "full cluster" in msg


def test_cluster_sync_unit_await(compiler):
    # Partial warps missing in Await for CudaClusterSync.
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(
            mkproc_cluster_sync_unit(cuda_cluster, await_lo=4), sm="90a"
        )
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
    p = mkproc_commit_group(wgmma_async, tma_to_gmem_async, cuda_warpgroup)
    invoke_test(p, None, compiler, golden, sm="90a")


def test_Sm80_commit_group_async_proxy(compiler):
    # Sm80_cp_async -> TMA is not OK (Sm80_cp_async is in the generic proxy)
    p = mkproc_commit_group(Sm80_cp_async, tma_to_gmem_async, cuda_thread)
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm="90a")
    msg = str(exc.value)
    assert "cg" in msg
    assert "Await" in msg
    assert "tma_to_gmem_async" in msg


def test_bad_first_sync_tl_commit_group(compiler):
    # tma_to_smem_async -> cuda_in_order is not supported by commit group
    # (this is handled by mbarrier completion mechanism)
    p = mkproc_commit_group(tma_to_smem_async, cuda_in_order, cuda_thread)
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(p, sm="90a")
    msg = str(exc.value)
    assert "cg" in msg
    assert "Arrive" in msg
    assert "tma_to_smem_async" in msg
