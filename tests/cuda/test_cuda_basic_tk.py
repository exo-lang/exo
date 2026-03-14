from __future__ import annotations

import pytest
import numpy as np

from exo import *
from exo.scalars import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.cuda_tk import *
from exo.platforms.Sm90 import *


def test_tk_load_rs_advice_simple():
    size0, size1 = 160, 64
    advice = get_tk_load_rs_advice(size0, size1, dst=f16, src=f32, swizzle=128)
    assert f16.bits == 16
    assert f32.bits == 32
    assert advice.rmem == CudaTkWarpTile(size0, size1, "row")
    assert advice.smem == Sm90_SmemSwizzled(128)
    assert advice.swizzle_elements == 32
    # fmt: off
    assert advice.instr == tk_load_rs_inner_cols_32(outer_cols=size1 // 32, rows=size0, dst=f16, src=f32)


def test_tk_store_rs_advice_simple():
    size0, size1 = 160, 64
    advice = get_tk_store_rs_advice(size0, size1, dst=f16, src=f32, swizzle=32)
    assert f16.bits == 16
    assert f32.bits == 32
    assert advice.rmem == CudaTkWarpTile(size0, size1, "row")
    assert advice.smem == Sm90_SmemSwizzled(32)
    assert advice.swizzle_elements == 16
    # fmt: off
    assert advice.instr == tk_store_rs_inner_cols_16(outer_cols=size1 // 16, rows=size0, dst=f16, src=f32)


def test_tk_load_store_rs_simple(compiler_Sm80):
    R, C = 192, 128

    @proc
    def p(h_dst: f32[R, C] @ DRAM, h_src: f32[R, C] @ DRAM):
        d_dst: f32[R, C] @ CudaGmemLinear
        d_src: f32[R, C] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2f32(R, C, d_src[:, :], h_src[:, :])

        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                # dummy_semm forces allocation not aligned to 1024B boundary.
                dummy_smem: f32[1] @ CudaSmemLinear

                smem: f32[C / 32, R, 32] @ Sm90_SmemSwizzled(128)
                # Manually (inefficently) load GMEM tile to SMEM
                for co in seq(0, C / 32):
                    for r in seq(0, R):
                        for ci in cuda_threads(0, 32):
                            smem[co, r, ci] = d_src[r, co * 32 + ci]
                # Overwrite one entry to 137
                # Along with manual loads, this tests that ThunderKittens
                # and Exo-GPU are indexing SMEM the same way.
                Fence(cuda_in_order, cuda_in_order)
                for tid in cuda_threads(0, 1, unit=cuda_thread):
                    smem[1, 3, 7] = 137
                    dummy_smem[0] = 0  # Prevent compiler warning
                # Copy SMEM tile to RMEM, then back to SMEM.
                # Conversion to bf16 will truncate some bits!
                Fence(cuda_in_order, cuda_in_order)
                for w in cuda_threads(0, 4, unit=cuda_warp):
                    tile: bf16[R / 4, C] @ CudaTkWarpTile(R // 4, C, "row")
                    tk_load_rs_inner_cols_32(
                        tile[:, :],
                        smem[:, R / 4 * w : R / 4 * w + R / 4, :],
                        dst=bf16,
                        src=f32,
                        rows=R // 4,
                        outer_cols=C // 32,
                    )
                    Fence(cuda_in_order, cuda_in_order)
                    tk_store_rs_inner_cols_32(
                        smem[:, R / 4 * w : R / 4 * w + R / 4, :],
                        tile[:, :],
                        dst=f32,
                        src=bf16,
                        rows=R // 4,
                        outer_cols=C // 32,
                    )
                Fence(cuda_in_order, cuda_in_order)
                # Manually (inefficently) store SMEM tile to GMEM
                for co in seq(0, C / 32):
                    for r in seq(0, R):
                        for ci in cuda_threads(0, 32):
                            d_dst[r, co * 32 + ci] = smem[co, r, ci]
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_2f32(R, C, h_dst[:, :], d_dst[:, :])

    p = simplify(p)
    lib = compiler_Sm80.nvcc_compile(p)
    # p.sync_check()

    h_dst = np.ndarray(shape=(R, C), dtype=np.float32)
    h_ref = np.ndarray(shape=(R, C), dtype=np.float32)
    h_src = np.ndarray(shape=(R, C), dtype=np.float32)

    for r in range(0, R):
        for c in range(0, C):
            test_bits = ((r * 3) + (c * 5)) & 127
            h_src[r, c] = 513 + test_bits * 4
            h_ref[r, c] = 512 + test_bits * 4  # Bit 0 should be truncated f32->bf16
            if r == 3 and c == 32 + 7:
                h_ref[r, c] = 137

    lib(None, h_dst, h_src)

    assert np.array_equal(h_dst, h_ref)


def test_tk_load_sg_simple(compiler_Sm80):
    @proc
    def p(h_dst: f32[192, 128] @ DRAM, h_src: f32[192, 128] @ DRAM):
        d_dst: f32[192, 128] @ CudaGmemLinear
        d_src: f32[192, 128] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2f32(192, 128, d_src[:, :], h_src[:, :])

        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                smem: f32[128 / 32, 192, 32] @ Sm90_SmemSwizzled(128)
                for col in cuda_threads(0, 4, unit=cuda_warp):
                    for s in seq(0, 2):  # Test offsetting
                        tk_load_sg(
                            smem[col, 96 * s : 96 * s + 96, :],
                            d_src[s * 96 : s * 96 + 96 :, 32 * col : 32 * col + 32],
                            dst=f32,
                            src=f32,
                            size0=96,
                            size1=32,
                        )
                Fence(cuda_in_order, cuda_in_order)
                for r in seq(0, 192):
                    for c in cuda_threads(0, 128):
                        d_dst[r, c] = smem[c / 32, r, c % 32]
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_2f32(192, 128, h_dst[:, :], d_dst[:, :])

    p = simplify(p)
    lib = compiler_Sm80.nvcc_compile(p)
    # p.sync_check()

    h_dst = np.ndarray(shape=(192, 128), dtype=np.float32)
    h_src = np.ndarray(shape=(192, 128), dtype=np.float32)

    for r in range(192):
        for c in range(128):
            h_src[r, c] = 100 * c + r
    lib(None, h_dst, h_src)

    assert np.array_equal(h_src, h_dst)


def test_tk_store_sg_simple(compiler_Sm80):
    R = 160
    C = 32

    @proc
    def p(h_dst: f16[R, C] @ DRAM, h_src: f16[R, C] @ DRAM):
        d_dst: f16[R, C] @ CudaGmemLinear
        d_src: f16[R, C] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2d(R, C, d_src[:, :], h_src[:, :], dst=f16, src=f16)

        with CudaDeviceFunction(blockDim=64):
            for task in cuda_tasks(0, 1):
                # Here we DON'T have a 3D tile.
                # Test that this code path also works (requires C = 64/sizeof(*smem))
                smem: f16[R, C] @ Sm90_SmemSwizzled(64)

                for r in seq(0, R):
                    for c in cuda_threads(0, C, unit=cuda_thread):
                        smem[r, c] = d_src[r, c]

                Fence(cuda_in_order, cuda_in_order)

                for row in cuda_threads(0, 2, unit=cuda_warp):
                    tk_store_sg(
                        d_dst[R / 2 * row : R / 2 * row + R / 2, :],
                        smem[R / 2 * row : R / 2 * row + R / 2, :],
                        dst=f16,
                        src=f16,
                        size0=R // 2,
                        size1=C,
                    )
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_2d(R, C, h_dst[:, :], d_dst[:, :], dst=f16, src=f16)

    p = simplify(p)
    lib = compiler_Sm80.nvcc_compile(p)
    p.sync_check()

    h_dst = np.ndarray(shape=(R, C), dtype=np.int16)  # i16 is good enough
    h_src = np.ndarray(shape=(R, C), dtype=np.int16)

    for r in range(R):
        for c in range(C):
            h_src[r, c] = 100 * c + r
    lib(None, h_dst, h_src)

    assert np.array_equal(h_src, h_dst)


def test_load_store_rg_simple(compiler_Sm80):
    R = 160
    C = 192 + 100

    @proc
    def p(h_inout: f32[R, C] @ DRAM):
        d_inout: f32[R, C] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2d(R, C, d_inout[:, :], h_inout[:, :], dst=f32, src=f32)

        with CudaDeviceFunction(blockDim=320):
            for task in cuda_tasks(0, 1):
                tiles: bf16[10, 4, 16, 48] @ CudaTkWarpTile(16, 48, "row")

                for w in cuda_threads(0, 10, unit=cuda_warp):
                    for s in seq(0, 4):
                        cuda_tk_load_rg(
                            tiles[w, s, :, :],
                            d_inout[16 * w : 16 * w + 16, 48 * s : 48 * s + 48],
                            dst=bf16,
                            src=f32,
                            size0=16,
                            size1=48,
                        )
                Fence(cuda_in_order, cuda_in_order)
                # Copy it back shifted by +100 columns
                for w in cuda_threads(0, 10, unit=cuda_warp):
                    for s in seq(0, 4):
                        cuda_tk_store_rg(
                            d_inout[16 * w : 16 * w + 16, 48 * s + 100 : 48 * s + 148],
                            tiles[w, s, :, :],
                            dst=f32,
                            src=bf16,
                            size0=16,
                            size1=48,
                        )

        cudaMemcpyAsync_dtoh_2d(R, C, h_inout[:, :], d_inout[:, :], dst=f32, src=f32)

    p = simplify(p)
    lib = compiler_Sm80.nvcc_compile(p)
    # p.sync_check()

    h_inout = np.ndarray(shape=(R, C), dtype=np.float32)
    h_ref = np.ndarray(shape=(R, C), dtype=np.float32)

    for r in range(R):
        for c in range(C):
            test_bits = ((r * 3) + (c * 5)) & 127
            h_inout[r, c] = 513 + test_bits * 4
            h_ref[r, c] = 513 + test_bits * 4

    # The copy should do two things
    # * Copy [0:64, 0:192] tile shifted +100 columns.
    # * Truncate bottom bit (due to bf16 conversion)
    assert C == 192 + 100
    for r in range(R):
        for c in range(192):
            h_ref[r, c + 100] = int(h_inout[r, c]) & ~1

    lib(None, h_inout)
    assert np.array_equal(h_inout, h_ref)
