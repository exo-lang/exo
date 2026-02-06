from __future__ import annotations

import math
import numpy as np
import pytest
import random

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

from exo.core.LoopIR import get_global_debug_log

# TODO find a better home for these than exo.platforms
# For now they're there in order to piggyback off exo's package structure.
# I don't have patience to deal with "attempted relative input without XXX" BS.
from exo.platforms.gemm_Sm90a_test.make_Sm90a_gemm import make_Sm90a_gemm
from exo.platforms.gemm_Sm90a_test.schedule_Sm90a_gemm import schedule_Sm90a_gemm
from exo.platforms.gemm_Sm90a_test.Sm90a_gemm_pre_config import Sm90aGemmConfig


config_A = Sm90aGemmConfig(
    smem_M=128, smem_N=256, tma_to_gmem=False, enable_split_k=False
)
config_B = Sm90aGemmConfig(
    smem_M=256, smem_N=192, tma_to_gmem=False, enable_split_k=False
)
config_K = Sm90aGemmConfig(
    smem_M=256, smem_N=192, tma_to_gmem=True, enable_split_k=True
)


def mkproc_gemm(config: Sm90aGemmConfig, ncta_M, ncta_N, scheduled):
    if scheduled:
        gpu_gemm = schedule_Sm90a_gemm(config, ncta_M, ncta_N)
    else:
        gpu_gemm = make_Sm90a_gemm(config, ncta_M, ncta_N)

    @proc
    def cpu_gemm_wrapper(
        L: size,
        M: size,
        N: size,
        K_split: size,
        cluster_K: size,
        A: f32[L, M, K_split, cluster_K] @ DRAM,
        B: f32[L, N, K_split, cluster_K] @ DRAM,
        C: f32[L, N, M] @ DRAM,
    ):
        assert cluster_K % 4 == 0
        d_A: f32[L, M, K_split, cluster_K] @ CudaGmemLinear
        d_B: f32[L, N, K_split, cluster_K] @ CudaGmemLinear
        d_C: f32[L, N, M] @ CudaGmemLinear

        cudaMemcpyAsync_htod_4f32(
            L, M, K_split, cluster_K, d_A[:, :, :, :], A[:, :, :, :]
        )
        cudaMemcpyAsync_htod_4f32(
            L, N, K_split, cluster_K, d_B[:, :, :, :], B[:, :, :, :]
        )
        gpu_gemm(L, M, N, K_split, cluster_K, d_A, d_B, d_C)
        cudaMemcpyAsync_dtoh_3f32(L, N, M, C[:, :, :], d_C[:, :, :])

    return cpu_gemm_wrapper


def mktest_golden(config: Sm90aGemmConfig, ncta_M, ncta_N, scheduled):
    def test(compiler, golden):
        compiler.cuda_cpu_test(
            mkproc_gemm,
            golden=golden,
            config=config,
            ncta_M=ncta_M,
            ncta_N=ncta_N,
            scheduled=scheduled,
        )

    return test


def mktest_run(config: Sm90aGemmConfig, ncta_M, ncta_N, scheduled, K_split):
    # More thorough runtime testing will remain in sporkbench.
    L = 1
    M = 2000
    N = 2004
    K = 1600
    cluster_K = K // K_split

    def test(compiler_Sm90a):
        rand = random.Random(100)
        A = np.ndarray(shape=(M, K), dtype=np.float32, order="C")
        B = np.ndarray(shape=(K, N), dtype=np.float32, order="F")
        C = np.ndarray(shape=(M, N), dtype=np.float32, order="F")

        for m in range(0, M):
            for k in range(0, K):
                A[m, k] = rand.randrange(-10, +10)
        for k in range(0, K):
            for n in range(0, N):
                B[k, n] = rand.randrange(-5, 15)

        cu = compiler_Sm90a.cuda_test_context(
            mkproc_gemm(config, ncta_M, ncta_N, scheduled)
        )

        C_expected = A @ B
        cu(None, L, M, N, K_split, cluster_K, A, B, C)
        assert np.array_equal(C, C_expected)

    return test


test_golden_Sm90a_gemm_m1n1_A = mktest_golden(config_A, 1, 1, False)
test_golden_Sm90a_gemm_m1n1_B = mktest_golden(config_B, 1, 1, False)
test_golden_Sm90a_gemm_m1n1_K = mktest_golden(config_K, 1, 1, False)

test_golden_Sm90a_gemm_m2n1_A = mktest_golden(config_A, 2, 1, False)
test_golden_Sm90a_gemm_m1n2_B = mktest_golden(config_B, 1, 2, False)
test_golden_Sm90a_gemm_m2n2_K = mktest_golden(config_K, 2, 2, False)
test_run_Sm90a_gemm_m2n2_K = mktest_run(config_K, 2, 2, False, K_split=2)

test_golden_Sm90a_sch_gemm_m1n2_B = mktest_golden(config_B, 1, 2, True)
test_run_Sm90a_sch_gemm_m1n2_B = mktest_run(config_B, 1, 2, True, K_split=1)


def test_Sm90a_gemm_remarks(compiler, golden):
    # This is a very fragile test.
    # Anytime we add more logging, the golden will change.
    # We just want to force some testing code coverage.

    def mkproc():
        gpu_gemm = make_Sm90a_gemm(config_K, 2, 1)
        gpu_gemm = rename(gpu_gemm, "gpu_gemm")
        return gpu_gemm

    compiler.cuda_cpu_test(mkproc)
    debug_log = get_global_debug_log()
    debug_log.write_all_impl()
    with open(str(compiler.workdir / "debug" / "gpu_gemm-analysis.py")) as f:
        assert f.read() == golden
