from __future__ import annotations

import math
import numpy as np
import pytest
import random

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *


import random


f16_table = (
    (-4, 0xC400),
    (-3, 0xC200),
    (-2, 0xC000),
    (-1, 0xBC00),
    (0, 0x0000),
    (1, 0x3C00),
    (2, 0x4000),
    (3, 0x4200),
    (4, 0x4400),
    (5, 0x4500),
    (6, 0x4600),
    (7, 0x4700),
    (8, 0x4800),
    (9, 0x4880),
    (10, 0x4900),
    (11, 0x4980),
    (12, 0x4A00),
    (13, 0x4A80),
    (14, 0x4B00),
    (15, 0x4B80),
)

bf16_table = (
    (-4, 0xC080),
    (-3, 0xC040),
    (-2, 0xC000),
    (-1, 0xBF80),
    (0, 0x0000),
    (1, 0x3F80),
    (2, 0x4000),
    (3, 0x4040),
    (4, 0x4080),
    (5, 0x40A0),
    (6, 0x40C0),
    (7, 0x40E0),
    (8, 0x4100),
    (9, 0x4110),
    (10, 0x4120),
    (11, 0x4130),
    (12, 0x4140),
    (13, 0x4150),
    (14, 0x4160),
    (15, 0x4170),
)


def mkproc_naive_gemm(typA, typB, typC):
    @proc
    def p(
        M: size,
        N: size,
        K: size,
        h_A: cu_f16[M, K] @ DRAM,
        h_B: cu_f16[N, K] @ DRAM,
        h_C: f32[N, M] @ DRAM,
    ):
        assert M % 128 == 0
        assert N % 128 == 0
        assert K % 32 == 0

        d_A: cu_f16[M, K] @ CudaGmemLinear
        d_B: cu_f16[N, K] @ CudaGmemLinear
        d_C: f32[N, M] @ CudaGmemLinear

        # Replaced with cudaMemcpyAsync_htod_2{b}f16
        for memcpy_m in seq(0, M):
            for k in seq(0, K):
                d_A[memcpy_m, k] = h_A[memcpy_m, k]
        for memcpy_n in seq(0, N):
            for k in seq(0, K):
                d_B[memcpy_n, k] = h_B[memcpy_n, k]

        with CudaDeviceFunction(blockDim=128, blocks_per_sm=3):
            for m_cta in cuda_tasks(0, M / 128):
                for n_cta in cuda_tasks(0, N / 128):
                    s_A: cu_f16[128, 32] @ Sm90_SmemSwizzled(128)
                    s_B: cu_f16[128, 32] @ CudaSmemLinear
                    r_C: f32[128, 128] @ CudaRmem

                    for m in cuda_threads(0, 128):
                        for n in seq(0, 128):
                            r_C[m, n] = 0

                    for ks in seq(0, K / 32):
                        for ms in seq(0, 32):
                            for mt in cuda_threads(0, 4, unit=32 * cuda_thread):
                                for k in cuda_threads(0, 32):
                                    s_A[ms * 4 + mt, k] = d_A[
                                        m_cta * 128 + ms * 4 + mt, ks * 32 + k
                                    ]
                                    s_B[ms * 4 + mt, k] = d_B[
                                        n_cta * 128 + ms * 4 + mt, ks * 32 + k
                                    ]
                        Fence(cuda_in_order, cuda_in_order)

                        for m in cuda_threads(0, 128):
                            for n in seq(0, 128):
                                for k in seq(0, 32):
                                    a_f32: f32 @ CudaRmem
                                    b_f32: f32 @ CudaRmem
                                    a_f32 = s_A[m, k]
                                    b_f32 = s_B[n, k]
                                    r_C[m, n] += a_f32 * b_f32
                        Fence(cuda_in_order, cuda_in_order)

                    for m in cuda_threads(0, 128):
                        for n in seq(0, 128):
                            d_C[n_cta * 128 + n, m_cta * 128 + m] = r_C[m, n]

        cudaMemcpyAsync_dtoh_2f32(N, M, h_C[:, :], d_C[:, :])

    p = set_precision(p, "h_A", typA)
    p = set_precision(p, "d_A", typA)
    p = set_precision(p, "s_A", typA)
    p = set_precision(p, "h_B", typB)
    p = set_precision(p, "d_B", typB)
    p = set_precision(p, "s_B", typB)
    p = set_precision(p, "d_C", typC)
    p = set_precision(p, "h_C", typC)  # Leave r_C as f32

    if typA == "cu_f16":
        memcpyA = cudaMemcpyAsync_htod_2f16()
    else:
        memcpyA = cudaMemcpyAsync_htod_2bf16()
    if typB == "cu_f16":
        memcpyB = cudaMemcpyAsync_htod_2f16()
    else:
        memcpyB = cudaMemcpyAsync_htod_2bf16()
    p = replace(p, p.find_loop("memcpy_m"), memcpyA)
    p = replace(p, p.find_loop("memcpy_n"), memcpyB)

    p = rename(p, f"cu_f16_naive_gemm_{typA}_{typB}_{typC}")
    return p


def naive_gemm_test_impl(compiler_Sm80, typA, typB, typC):
    assert typA in ("cu_f16", "cu_bf16")
    assert typB in ("cu_f16", "cu_bf16")
    assert typC == "f32"

    cu = compiler_Sm80.cuda_test_context(
        mkproc_naive_gemm(typA=typA, typB=typB, typC=typC)
    )

    for M, N, K in ((512, 256, 384), (384, 768, 160)):
        A_f32 = np.ndarray(shape=(M, K), dtype=np.float32, order="C")
        B_f32 = np.ndarray(shape=(K, N), dtype=np.float32, order="F")
        A_bits16 = np.ndarray(shape=(M, K), dtype=np.uint16, order="C")
        B_bits16 = np.ndarray(shape=(K, N), dtype=np.uint16, order="F")
        C_f32_result = np.ndarray(shape=(M, N), dtype=np.float32, order="F")

        rand = random.Random(3090)

        table = f16_table if typA == "cu_f16" else bf16_table
        for m in range(0, M):
            for k in range(0, K):
                value, bits = rand.choice(table)
                A_f32[m, k] = value
                A_bits16[m, k] = bits
        table = f16_table if typB == "cu_f16" else bf16_table
        for n in range(0, N):
            for k in range(0, K):
                value, bits = rand.choice(table)
                B_f32[k, n] = value
                B_bits16[k, n] = bits

        cu(None, M, N, K, A_bits16, B_bits16, C_f32_result)
        C_f32_expected = A_f32 @ B_f32

        assert np.array_equal(C_f32_result, C_f32_expected)


def test_golden_naive_cu_f16_gemm(compiler, golden):
    compiler.cuda_cpu_test(
        mkproc_naive_gemm, golden=golden, typA="cu_f16", typB="cu_f16", typC="f32"
    )


def test_golden_naive_cu_bf16_gemm(compiler, golden):
    compiler.cuda_cpu_test(
        mkproc_naive_gemm, golden=golden, typA="cu_bf16", typB="cu_bf16", typC="f32"
    )


def test_run_naive_cu_f16_gemm(compiler_Sm80):
    naive_gemm_test_impl(compiler_Sm80, "cu_f16", "cu_f16", "f32")


def test_run_naive_cu_bf16_gemm(compiler_Sm80):
    naive_gemm_test_impl(compiler_Sm80, "cu_bf16", "cu_bf16", "f32")


def mkproc_vec_add(in_typ):
    if in_typ == "cu_f16":
        memcpy = cudaMemcpyAsync_htod_1f16()
        ld = cuda_packed_load_f16()
        num_packed = 2
    elif in_typ == "cu_bf16":
        memcpy = cudaMemcpyAsync_htod_1bf16()
        ld = cuda_packed_load_bf16()
        num_packed = 2
    elif in_typ == "f32":
        memcpy = cudaMemcpyAsync_htod_1f32()
        ld = cuda_packed_load_f32()
        num_packed = 1
    elif in_typ == "i32":
        memcpy = cudaMemcpyAsync_htod_1i32()
        ld = cuda_packed_load_i32()
        num_packed = 1
    else:
        assert 0, in_typ

    task_size = 256 * num_packed

    @proc
    def p(N: size, h_x: f32[N], h_y: f32[N], h_out: f32[N]):
        assert N % task_size == 0

        d_x: f32[N] @ CudaGmemLinear
        d_y: f32[N] @ CudaGmemLinear
        d_out: f32[N] @ CudaGmemLinear

        for d_x_init_n in seq(0, N):
            d_x[d_x_init_n] = h_x[d_x_init_n]
        for d_y_init_n in seq(0, N):
            d_y[d_y_init_n] = h_y[d_y_init_n]

        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, N / task_size):
                rmem_x: f32[256, num_packed] @ CudaRmemPacked32
                for tid in cuda_threads(0, 256):
                    for pack_ld_x in seq(0, num_packed):
                        rmem_x[tid, pack_ld_x] = d_x[
                            task * task_size + tid * num_packed + pack_ld_x
                        ]
                    rmem_y: f32[num_packed] @ CudaRmemPacked32
                    for pack_ld_y in seq(0, num_packed):
                        rmem_y[pack_ld_y] = d_y[
                            task * task_size + tid * num_packed + pack_ld_y
                        ]
                    for pack_st_out in seq(0, num_packed):
                        d_out[task * task_size + tid * num_packed + pack_st_out] = (
                            rmem_x[tid, pack_st_out] + rmem_y[pack_st_out]
                        )

        cudaMemcpyAsync_dtoh_1f32(N, h_out[:], d_out[:])

    p = set_precision(p, "h_x", in_typ)
    p = set_precision(p, "d_x", in_typ)
    p = set_precision(p, "rmem_x", in_typ)
    p = set_precision(p, "h_y", in_typ)
    p = set_precision(p, "d_y", in_typ)
    p = set_precision(p, "rmem_y", in_typ)

    p = replace(p, p.find_loop("d_x_init_n"), memcpy)
    p = replace(p, p.find_loop("d_y_init_n"), memcpy)
    p = replace(p, p.find_loop("pack_ld_x"), ld)
    p = replace(p, p.find_loop("pack_ld_y"), ld)
    p = rename(p, f"cu_f16_test_vec_add_{in_typ}")
    p = simplify(p)
    return p


def vec_add_test_impl(compiler_Sm80, in_typ):
    cu = compiler_Sm80.cuda_test_context(mkproc_vec_add(in_typ=in_typ))
    N = 2048
    x_f32 = np.ndarray(shape=(N,), dtype=np.float32)
    y_f32 = np.ndarray(shape=(N,), dtype=np.float32)
    z_result = np.ndarray(shape=(N,), dtype=np.float32)

    rand = random.Random(3090)

    if in_typ == "cu_f16" or in_typ == "cu_bf16":
        table = f16_table if in_typ == "cu_f16" else bf16_table
        x_bits = np.ndarray(shape=(N,), dtype=np.uint16)
        y_bits = np.ndarray(shape=(N,), dtype=np.uint16)
        for n in range(N):
            value, bits = rand.choice(table)
            x_f32[n] = value
            x_bits[n] = bits
            value, bits = rand.choice(table)
            y_f32[n] = value
            y_bits[n] = bits
    elif in_typ == "f32" or in_typ == "i32":
        for n in range(N):
            x_f32[n] = rand.randrange(-4095, 4096)
            y_f32[n] = rand.randrange(-4095, 4096)
        if in_typ == "f32":
            x_bits = x_f32
            y_bits = y_f32
        else:
            x_bits = np.ndarray(shape=(N,), dtype=np.int32)
            y_bits = np.ndarray(shape=(N,), dtype=np.int32)
            for n in range(N):
                x_bits[n] = int(x_f32[n])
                y_bits[n] = int(y_f32[n])
    else:
        assert 0, in_typ

    cu(None, N, x_bits, y_bits, z_result)
    z_expected = x_f32 + y_f32

    assert np.array_equal(z_result, z_expected)


def test_golden_vec_add_f16(compiler, golden):
    compiler.cuda_cpu_test(mkproc_vec_add, golden=golden, in_typ="cu_f16")


def test_run_vec_add_f16(compiler_Sm80):
    vec_add_test_impl(compiler_Sm80, in_typ="cu_f16")


def test_golden_vec_add_bf16(compiler, golden):
    compiler.cuda_cpu_test(mkproc_vec_add, golden=golden, in_typ="cu_bf16")


def test_run_vec_add_bf16(compiler_Sm80):
    vec_add_test_impl(compiler_Sm80, in_typ="cu_bf16")


def test_golden_vec_add_f32(compiler, golden):
    compiler.cuda_cpu_test(mkproc_vec_add, golden=golden, in_typ="f32")


def test_run_vec_add_f32(compiler_Sm80):
    vec_add_test_impl(compiler_Sm80, in_typ="f32")


def test_golden_vec_add_i32(compiler, golden):
    compiler.cuda_cpu_test(mkproc_vec_add, golden=golden, in_typ="i32")


def test_run_vec_add_i32(compiler_Sm80):
    vec_add_test_impl(compiler_Sm80, in_typ="i32")
