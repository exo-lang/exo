from __future__ import annotations

import pytest
import numpy as np

from exo.platforms.cuda import *
from exo.platforms.cuda_tk import *
from exo.platforms.Sm90 import *


def test_tk_load_sg(compiler_Sm80):
    @proc
    def p(h_dst: f32[64, 128] @ DRAM, h_src: f32[64, 128] @ DRAM):
        d_src: f32[64, 128] @ CudaGmemLinear
        d_dst: f32[64, 128] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2f32(64, 128, d_src[:, :], h_src[:, :])

        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                smem: f32[4, 64, 32] @ Sm90_SmemSwizzled(128)
                for col in cuda_threads(0, 4, unit=cuda_warp):
                    for s in seq(0, 2):  # Test offsetting
                        tk_load_sg(
                            smem[col, 32 * s : 32 * s + 32, :],
                            d_src[s * 32 : s * 32 + 32 :, 32 * col : 32 * col + 32],
                            dst=f32,
                            src=f32,
                            size0=32,
                            size1=32,
                        )
                Fence(cuda_in_order, cuda_in_order)
                for r in seq(0, 64):
                    for c in cuda_threads(0, 128):
                        d_dst[r, c] = smem[c / 32, r, c % 32]
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_2f32(64, 128, h_dst[:, :], d_dst[:, :])

    lib = compiler_Sm80.nvcc_compile(p)
    h_dst = np.ndarray(shape=(64, 128), dtype=np.float32)
    h_src = np.ndarray(shape=(64, 128), dtype=np.float32)

    for r in range(64):
        for c in range(128):
            h_src[r, c] = 100 * c + r
    lib(None, h_dst, h_src)

    print([int(h_src[r, 0]) for r in range(64)])
    print([int(h_dst[r, 0]) for r in range(64)])

    assert np.array_equal(h_src, h_dst)
