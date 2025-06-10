from __future__ import annotations

import numpy as np
import pytest
import random

from exo import proc
from exo.core.LoopIR import T
from exo.core.extern import Extern, _EErr
from exo.platforms.cuda import *
from exo.stdlib.scheduling import *


def test_cuda_add_vec(compiler):
    """
    Hello world: Compute c = a + b
    """

    @proc
    def cuda_add_vec(n: size, a: i32[n], b: i32[n], c: i32[n]):
        assert n % 32 == 0
        device_a: i32[n] @ CudaGmemLinear
        device_d: i32[n] @ CudaGmemLinear
        cudaMemcpyAsync_htod_1i32(n, device_a, a)
        cudaMemcpyAsync_htod_1i32(n, device_d, b)
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, n / 32):
                for tid in cuda_threads(0, 32):
                    device_d[task * 32 + tid] += device_a[task * 32 + tid]
        cudaMemcpyAsync_dtoh_1i32(n, c, device_d)

    fn = compiler.nvcc_compile(cuda_add_vec)

    for n in (32, 32000):
        a = np.ndarray(shape=(n,), dtype=np.int32)
        b = np.ndarray(shape=(n,), dtype=np.int32)
        c_test = np.ndarray(shape=(n,), dtype=np.int32)
        for i in range(0, n):
            a[i] = i % 42
            b[i] = i + 101
        c_expected = a + b
        fn(None, n, a, b, c_test)
        assert np.array_equal(c_test, c_expected)


def test_tmp_excut(compiler):
    """
    Hello world: Compute c = a + b
    """

    @proc
    def cuda_add_vec(n: size, a: i32[n], b: i32[n], c: i32[n]):
        assert n % 32 == 0
        device_a: i32[n] @ CudaGmemLinear
        device_d: i32[n] @ CudaGmemLinear
        cudaMemcpyAsync_htod_1i32(n, device_a, a)
        cudaMemcpyAsync_htod_1i32(n, device_d, b)
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, n / 32):
                for tid in cuda_threads(0, 32):
                    device_d[task * 32 + tid] += device_a[task * 32 + tid]
        cudaMemcpyAsync_dtoh_1i32(n, c, device_d)

    fn = compiler.nvcc_compile(
        cuda_add_vec,
        excut=True,
    )
    fn.exo_excut_begin_log_file(compiler.workdir / "excut_trace.json", 1 << 24)

    for n in (32, 32000):
        a = np.ndarray(shape=(n,), dtype=np.int32)
        b = np.ndarray(shape=(n,), dtype=np.int32)
        c_test = np.ndarray(shape=(n,), dtype=np.int32)
        for i in range(0, n):
            a[i] = i % 42
            b[i] = i + 101
        c_expected = a + b
        fn(None, n, a, b, c_test)
        assert np.array_equal(c_test, c_expected)

    fn.exo_excut_end_log_file()


def test_cuda_simple_saxpy(compiler):
    """
    Compute y = ay + x
    """

    @proc
    def saxpy(n: size, a: f32[1], y: f32[n], x: f32[n]):
        assert n % 128 == 0
        device_x: f32[n] @ CudaGmemLinear
        device_y: f32[n] @ CudaGmemLinear
        device_a: f32 @ CudaGridConstant

        device_a = a[0]
        cudaMemcpyAsync_htod_1f32(n, device_x, x)
        cudaMemcpyAsync_htod_1f32(n, device_y, y)

        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, n / 128):
                for tid in cuda_threads(0, 128):
                    device_y[task * 128 + tid] = (
                        device_y[task * 128 + tid] * device_a
                        + device_x[task * 128 + tid]
                    )
        cudaMemcpyAsync_dtoh_1f32(n, y, device_y)

    fn = compiler.nvcc_compile(saxpy)

    for a in (1.0, 3.75):
        for n in (128, 128 * 300):
            x = np.ndarray(shape=(n,), dtype=np.float32)
            y = np.ndarray(shape=(n,), dtype=np.float32)
            for i in range(0, n):
                x[i] = i % 42
                y[i] = i + 101.5
            y_expected = a * y + x
            fn(None, n, np.array([a], dtype=np.float32), y, x)
            assert np.array_equal(y, y_expected)


def test_cuda_two_device_functions(compiler):
    """
    Compute y = ay + x but really inefficiently, using 2 cuda kernels
    """

    @proc
    def saxpy(n: size, a: f32[1], y: f32[n], x: f32[n]):
        assert n % 128 == 0
        device_x: f32[n] @ CudaGmemLinear
        device_y: f32[n] @ CudaGmemLinear
        device_a: f32 @ CudaGridConstant

        device_a = a[0]
        cudaMemcpyAsync_htod_1f32(n, device_x, x)
        cudaMemcpyAsync_htod_1f32(n, device_y, y)

        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, n / 128):
                for tid in cuda_threads(0, 128):
                    device_y[task * 128 + tid] = device_y[task * 128 + tid] * device_a

        with CudaDeviceFunction(blockDim=64):
            for task in cuda_tasks(0, n / 64):
                for tid in cuda_threads(0, 64):
                    device_y[task * 64 + tid] += device_x[task * 64 + tid]

        cudaMemcpyAsync_dtoh_1f32(n, y, device_y)

    fn = compiler.nvcc_compile(saxpy)

    for a in (1.0, 3.75):
        for n in (128, 128 * 300):
            x = np.ndarray(shape=(n,), dtype=np.float32)
            y = np.ndarray(shape=(n,), dtype=np.float32)
            for i in range(0, n):
                x[i] = i % 42
                y[i] = i + 101.5
            y_expected = a * y + x
            fn(None, n, np.array([a], dtype=np.float32), y, x)
            assert np.array_equal(y, y_expected)


@instr
class gemm_init_pcg3d_mod:
    def behavior(
        M: size,
        K: size,
        A: [f32][M, K] @ CudaGmemLinear,
        m: index,
        k: index,
        seed: size,
        modulus: size,
    ):
        assert m >= 0
        assert k >= 0
        assert m < M
        assert k < K
        A[m, k] = 12345  # Lie to Exo since Exo can't express this prng

    def instance(self, seed, modulus):
        self.actor_kind = cuda_classic
        self.access_info["A"].actor_signature = sig_cuda_classic
        self.coll_unit = cuda_thread
        self.cu_utils = [self._cu_util]
        self.instr_format = [self._line_format % (seed, modulus)]

    _line_format = "exo_CudaUtil::gemm_init_pcg3d_mod({K}, {A}, {m}, {k}, %s, %s);"

    _cu_util = """__device__ void gemm_init_pcg3d_mod(
        uint32_t K, struct exo_win_2f32 A,
        uint32_t m, uint32_t k, uint32_t seed, uint32_t modulus)
{
    uint32_t x = m;
    uint32_t y = k;
    uint32_t z = seed;

    // Copied pseudo random number generation code.
    // http://www.jcgt.org/published/0009/03/02/
    // Hash Functions for GPU Rendering, Mark Jarzynski, Marc Olano, NVIDIA
    x = x*1664525u + 1013904223u;
    y = y*1664525u + 1013904223u;
    z = z*1664525u + 1013904223u;

    x += y*z;
    y += z*x;
    z += x*y;

    x ^= x >> 16u;
    y ^= y >> 16u;
    z ^= z >> 16u;

    x += y*z;
    y += z*x;
    z += x*y;

    float value = (float)((x + y + z) % modulus);

    A.data[A.strides[0] * m + k] = value;
}
"""


def test_cuda_simple_matmul(compiler):
    """
    Initialize A, B with data, and compute C = A * B
    """
    # fmt: off
    @proc
    def gemm_test(M: size, N: size, K: size, A_cpu: f32[M,K] @ DRAM, B_cpu: f32[N,K] @ DRAM, C_cpu: f32[N,M] @ DRAM):
        A: f32[M, K] @ CudaGmemLinear
        B: f32[N, K] @ CudaGmemLinear
        C: f32[N, M] @ CudaGmemLinear

        # Cuda0: initialize A with "random" data
        with CudaDeviceFunction(blockDim=32):
            for m in cuda_tasks(0, M):
                for k_task in cuda_tasks(0, K / 32):
                    for k_seq in cuda_threads(0, 32):
                        gemm_init_pcg3d_mod(M, K, A, m, k_task * 32 + k_seq, seed=1337, modulus=7)

        # Cuda1: initialize B with "random" data
        with CudaDeviceFunction(blockDim=32):
            for n in cuda_tasks(0, N):
                for k_task in cuda_tasks(0, K / 32):
                    for k_seq in cuda_threads(0, 32):
                        gemm_init_pcg3d_mod(N, K, B, n, k_task * 32 + k_seq, seed=42, modulus=14)

        # Cuda2: C = A * B
        with CudaDeviceFunction(blockDim=256):
            for m2 in cuda_tasks(0, M / 128):
                for n2 in cuda_tasks(0, N / 256):
                    # TeX: begin working_smem
                    # TeX: remark! working_smem[0]
                    # Per-CTA allocations
                    # TeX: color remark working_smem[0]
                    # yyyy
                    # SMEM allocations: not distributed, as SMEM expects to be allocated by a CTA
                    # TeX: color line *
                    #                      yyyyyyyyyyyyyy
                    A_smem: f32[128, 32] @ CudaSmemLinear
                    # TeX: color line *
                    #                      yyyyyyyyyyyyyy
                    B_smem: f32[256, 32] @ CudaSmemLinear
                    # TeX: color remark working_smem[0]
                    # rrrr
                    # RMEM allocation: distributed, as registers are allocated per-thread
                    # TeX: color line *
                    #          gg  vv           rrrrrrrr
                    accum: f32[16, 16, 8, 16] @ CudaRmem
                    # TeX: end working_smem

                    for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                        for n1 in cuda_threads(0, 16, unit=cuda_thread):
                            for m0 in seq(0, 8):
                                for n0 in seq(0, 16):
                                    # TeX: summary
                                    # Zero per-thread accumulators
                                    accum[m1, n1, m0, n0] = 0

                    # TeX: begin working_smem
                    # TeX: color line *
                    #   bb
                    for k1 in seq(0, K / 32):
                        # TeX: end working_smem
                        # TeX: begin working_smem[1]
                        for m1 in seq(0, 16):
                            for m0 in cuda_threads(0, 8, unit=32*cuda_thread):
                                for k0 in cuda_threads(0, 32):
                                    # TeX: summary!
                                    # Load A_smem
                                    # TeX: color line *
                                   #yyyyyyy             y
                                    A_smem[m1*8 + m0, k0] = A[m2*128 + m1*8 + m0, k1*32 + k0]
                        for n1 in seq(0, 32):
                            for n0 in cuda_threads(0, 8, unit=32*cuda_thread):
                                for k0 in cuda_threads(0, 32):
                                    # TeX: summary!
                                    # Load B_smem
                                    # TeX: color line *
                                   #yyyyyyy             y
                                    B_smem[n1*8 + n0, k0] = B[n2*256 + n1*8 + n0, k1*32 + k0]
                        # TeX: end working_smem[1]

                        # TeX: begin working_smem
                        Fence(cuda_classic, cuda_classic)  # __syncthreads()
                        # TeX: end working_smem

                        # TeX: begin working_smem
                        # TeX: color line *
                        #   gg
                        for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                            # TeX: color line *
                            #   vv
                            for n1 in cuda_threads(0, 16, unit=cuda_thread):
                                # TeX: end working_smem
                                # TeX: begin working_smem[2]
                                for k0 in seq(0, 32):
                                    # TeX: summary!
                                    # accum += A_smem @ B_smem
                                    for m0 in seq(0, 8):
                                        for n0 in seq(0, 16):
                                            # TeX: color line *
                                           #rrrrrrgg  vv        r     yyyyyyy             y
                                            accum[m1, n1, m0, n0] += (A_smem[m1*8 + m0, k0]
                                            # TeX: color line *
                                            #                              yyyyyyy              y
                                                                         * B_smem[n1*16 + n0, k0])
                                # TeX: end working_smem[2]

                        # TeX: begin working_smem
                        Fence(cuda_classic, cuda_classic)  # __syncthreads()
                    # TeX: color line *
                    #     bb
                    # End k1 loop
                    # TeX: end working_smem
                    for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                        for n1 in cuda_threads(0, 16, unit=cuda_thread):
                            for m0 in seq(0, 8):
                                for n0 in seq(0, 16):
                                    # TeX: summary
                                    # Each thread writes out accumulators
    # TeX: begin working_smem
    # TeX: color line *
    #                                                                            rrrrrrgg  vv        r
                                    C[n2*256 + n1*16 + n0, m2*128 + m1*8 + m0] = accum[m1, n1, m0, n0]
    # TeX: end working_smem
    # fmt: on

        # Epilogue: copy A, B, C to CPU, for use by np.matmul
        cudaMemcpyAsync_dtoh_2f32(M, K, A_cpu, A)
        cudaMemcpyAsync_dtoh_2f32(N, K, B_cpu, B)
        cudaMemcpyAsync_dtoh_2f32(N, M, C_cpu, C)

    fn = compiler.nvcc_compile(gemm_test)

    for M in (512, 4096):
        for N in (512, 65536):
            for K in (512, 65536):
                if M * N * K > 4096 * 512 * 65536:
                    continue
                A = np.ndarray(shape=(M, K), dtype=np.float32, order="C")
                B = np.ndarray(shape=(K, N), dtype=np.float32, order="F")
                C_test = np.ndarray(shape=(M, N), dtype=np.float32, order="F")
                fn(None, M, N, K, A, B, C_test)
                # A and B initialized by gemm_test, then used by numpy
                C_expected = np.ndarray(shape=(M, N), dtype=np.float32, order="F")
                np.matmul(A, B, C_expected)
                assert np.array_equal(C_test, C_expected)


def test_cuda_arrays(compiler):
    """Test correct behavior of passing arrays from host to device:

    * grid constant support
    * runtime-size arrays
    * test exo_* prefix args; which are mangled
    * conversion
    """

    @proc
    def array_test(
        exo_ZZ0a: size,
        exo_ZZ0b: size,
        exo_ZZ1a: size,
        exo_ZZ1b: size,
        exo_ZZ2: i32[exo_ZZ0a + exo_ZZ0b, exo_ZZ1a + exo_ZZ1b],
        exo_ZZ3: f32[4],
        exo_ZZ4: f32[4],
    ):
        assert exo_ZZ0b > 4
        assert exo_ZZ1b > 4
        exo_ZZ2_device: i32[exo_ZZ0a + exo_ZZ0b, exo_ZZ1a + exo_ZZ1b] @ CudaGmemLinear
        exo_ZZ3_device: f32[4] @ CudaGridConstant
        exo_ZZ4_device: f32[4] @ CudaGridConstant
        for i in seq(0, 4):
            exo_ZZ3_device[i] = exo_ZZ3[i]
            exo_ZZ4_device[i] = exo_ZZ4[i]
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, 4):
                    # Note: exo_ZZ1b is not used explicitly here, but is used
                    # in the generated C++ for stride calculations.
                    exo_ZZ2_device[exo_ZZ0a + tid, exo_ZZ1a + 4 - tid] = (
                        exo_ZZ3_device[tid] + exo_ZZ4_device[tid]
                    )
        cudaMemcpyAsync_dtoh_2i32(
            exo_ZZ0a + exo_ZZ0b, exo_ZZ1a + exo_ZZ1b, exo_ZZ2, exo_ZZ2_device
        )

    fn = compiler.nvcc_compile(array_test)

    ZZ0a = 129
    ZZ0b = 100
    ZZ1a = 137
    ZZ1b = 10
    ZZ2 = np.ndarray(shape=(ZZ0a + ZZ0b, ZZ1a + ZZ1b), dtype=np.int32, order="C")
    ZZ3 = np.array([2.5, 7.5, 1.5, 8.5], dtype=np.float32, order="C")
    ZZ4 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32, order="C")
    fn(None, ZZ0a, ZZ0b, ZZ1a, ZZ1b, ZZ2, ZZ3, ZZ4)

    for i in range(4):
        assert ZZ2[ZZ0a + i, ZZ1a + 4 - i] == ZZ3[i] + ZZ4[i]


# TODO seperate Sm80 test
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *

Mw = 96
Nw = 64

M1 = 192
N1 = 256  # Does not change gracefully

K0 = 16
MMA_K = 8


# fmt: off
@proc
def xgemm_Sm80_fence(M: size, N: size, K: size, A_host: f32[M,K], B_host: f32[K,N], C_host: f32[M,N]):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    A: f32[M, K] @ CudaGmemLinear
    B: f32[K, N] @ CudaGmemLinear
    C: f32[M, N] @ CudaGmemLinear

    cudaMemcpyAsync_htod_2f32(M, K, A, A_host)
    cudaMemcpyAsync_htod_2f32(K, N, B, B_host)

    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code

                # Tiles (double buffered)
                A_smem : f32[2, M1, K0] @ CudaSmemLinear
                B_smem : f32[2, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                Sm80_mma_zero_d_tf32(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, double buffered
                # Don't accum tile in first iteration.
                # Don't load tile in last iteration.
                # 1 iteration delay between load and use.
                for k1 in seq(0, K / K0 + 1):
                    if k1 < K / K0:
                        with CudaAsync(Sm80_cp_async):
                            # Load A tile
                            for m1 in seq(0, M1 / 64):
                                for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                    for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                        Sm80_cp_async_f32(A_smem[k1 % 2, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                          A[m2 * M1 + m1 * 64 + m0,
                                                          k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4], size=4)

                            # Load B tile
                            for k0_seq in seq(0, 4):
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        Sm80_cp_async_f32(B_smem[k1 % 2, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                          B[k1 * K0 + k0_seq * 4 + k0_par,
                                                          n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4], size=4)
                        # end CudaAsync(Sm80_cp_async)
                # for-k1 (K tiles) loop continues
                    if k1 > 0:
                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                                             B_smem[1 - k1 % 2,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_a_tf32(A_rmem[k_seq,:,:],
                                                             A_smem[1 - k1 % 2,
                                                             mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                          A_rmem[k_seq,:,:],
                                                          B_rmem[k_seq,n_seq,:,:], K=MMA_K)

                    # Sm80_generic actor kind = (cuda_classic | Sm80_cp_async)
                    Fence(Sm80_generic, Sm80_generic)

                # for-k1 (K tiles) loop ends

                # Write out accumulator
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                Sm80_mma_store_d_tf32(
                                    C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                    n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                    D_rmem[mw,nw,m_seq,n_seq,:,:])

    cudaMemcpyAsync_dtoh_2f32(M, N, C_host, C)
# fmt: on


xgemm_Sm80_fence = simplify(xgemm_Sm80_fence)


def test_tmp_Sm80(compiler):
    fn = compiler.nvcc_compile(
        xgemm_Sm80_fence,
        excut=True,
    )
    fn.exo_excut_begin_log_file(compiler.workdir / "excut_Sm80_fence.json", 1 << 30)

    M, N, K = 192, 256, 128
    A = np.zeros(shape=(M, K), dtype=np.float32, order="C")
    B = np.zeros(shape=(K, N), dtype=np.float32, order="C")
    C_test = np.ndarray(shape=(M, N), dtype=np.float32, order="C")
    fn(None, M, N, K, A, B, C_test)

    fn.exo_excut_end_log_file()
