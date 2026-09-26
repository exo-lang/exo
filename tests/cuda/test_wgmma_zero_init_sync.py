"""sync_check tests for zero-init wgmma instrs (Sm90_tk_mma_*_zi)

Zero-init used to be modeled by a separate instr (Sm90_tk_zero_scale_d)
whose fake write-only mutate hid a prior CUDA access to the D registers,
so the wgmma.fence required between that access and the wgmma
wasn't enforced. Now the zero-init is part of the wgmma instr itself.
"""

from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.cuda_tk import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *


def mkproc_cuda_write_then_wgmma(fence):
    """CUDA write of D -> fence (varies) -> zero-init wgmma(D)"""
    wgmma_fence = fence == "wgmma"
    cuda_fence = fence == "cuda"
    N = 64
    K = 64

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                D: f32[4, 16, N] @ Sm90_TkRmemTileD(N)
                A: bf16[64, K] @ Sm90_SmemSwizzled(128)
                B: bf16[N, K] @ Sm90_SmemSwizzled(128)
                cg: barrier @ Sm90_WgmmaCommitGroup
                for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
                    for w in cuda_threads(0, 4, unit=cuda_warp):
                        cuda_tk_tile_zero(D[w, :, :], dst=f32, rows=16, cols=N)
                    if wgmma_fence:
                        Fence(wgmma_fence_1, wgmma_fence_2)
                    if cuda_fence:
                        Fence(cuda_in_order, cuda_in_order)
                    Sm90_tk_mma_row_col_zi(
                        D[:, :, :], A[:, :], B[:, :], True, D=f32, A=bf16, B=bf16, N=N, K=K
                    )
                    Arrive(wgmma_async) >> cg
                    Await(cg, cuda_in_order, 0)

    return simplify(test_proc)


def test_cuda_write_wgmma_fence_positive(compiler):
    mkproc_cuda_write_then_wgmma("wgmma").sync_check()


@pytest.mark.parametrize("fence", ["cuda", "none"])
def test_cuda_write_no_wgmma_fence_negative(compiler, fence):
    p = mkproc_cuda_write_then_wgmma(fence)
    with pytest.raises(Exception) as exc:
        p.sync_check()
    assert "WAW" in str(exc.value)


def mkproc_handwritten_zero_init_loop():
    """Handwritten style: zero-init on the first iteration of a K loop,
    with CUDA writes of D (after the wgmma retires) between batches."""
    N = 64
    K = 64

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                D: f32[4, 16, N] @ Sm90_TkRmemTileD(N)
                A: bf16[4, 64, K] @ Sm90_SmemSwizzled(128)
                B: bf16[4, N, K] @ Sm90_SmemSwizzled(128)
                cg: barrier @ Sm90_WgmmaCommitGroup
                for batch in seq(0, 3):
                    for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
                        Fence(wgmma_fence_1, wgmma_fence_2)
                        for hdim64 in seq(0, 4):
                            Sm90_tk_mma_row_col_zi(
                                D[:, :, :],
                                A[hdim64, :, :],
                                B[hdim64, :, :],
                                hdim64 == 0,
                                D=f32, A=bf16, B=bf16, N=N, K=K,
                            )
                        Arrive(wgmma_async) >> cg
                        Await(cg, cuda_in_order, 0)
                        for w in cuda_threads(0, 4, unit=cuda_warp):
                            cuda_tk_tile_zero(D[w, :, :], dst=f32, rows=16, cols=N)

    return simplify(test_proc)


def test_handwritten_zero_init_loop_positive(compiler):
    p = mkproc_handwritten_zero_init_loop()
    p.sync_check()
    compiler.cuda_cpu_test(lambda: p)

