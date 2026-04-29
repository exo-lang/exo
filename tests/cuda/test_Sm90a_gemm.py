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

import exo.platforms.Sm90.tk_gemm_util as gemm_util
from exo.platforms.Sm90.tk_gemm_util import (
    L_divisor,
    M_divisor,
    N_divisor,
    K_cluster_divisor,
)

from dataclasses import replace  # Name conflict with exo.stdlib.scheduling...


f32_config = gemm_util.GemmConfig(
    A_type="f32",
    A_major="row",
    B_type="f32",
    B_major="col",
    C_type="f32",
    C_major="row",
)

config_A = replace(f32_config, cta_M=128, cta_N=256, enable_split_k=False)
config_B = replace(f32_config, cta_M=256, cta_N=192, enable_split_k=False)
config_K = replace(f32_config, cta_M=256, cta_N=192, enable_split_k=True)


def mkproc_gemm(config: gemm_util.GemmConfig, ncta_M, ncta_N, scheduled):
    config = replace(config, ncta_M=ncta_M, ncta_N=ncta_N)
    if scheduled:
        assert 0, "TODO schedule"
    else:
        gpu_gemm = gemm_util.handwrite_gemm(config)

    @proc
    def cpu_gemm_wrapper(
        L: size,
        M: size,
        N: size,
        K_split: size,
        K_cluster: size,
        A: f32[L, M, K_split, K_cluster] @ DRAM,
        B: f32[L, N, K_split, K_cluster] @ DRAM,
        C: f32[L, M, N] @ DRAM,
    ):
        assert L % L_divisor == 0
        assert M % M_divisor == 0
        assert N % N_divisor == 0
        assert K_cluster % K_cluster_divisor == 0
        d_A: f32[L, M, K_split, K_cluster] @ CudaGmemLinear
        d_B: f32[L, N, K_split, K_cluster] @ CudaGmemLinear
        d_C: f32[L, M, N] @ CudaGmemLinear

        cudaMemcpyAsync_htod_4f32(
            L, M, K_split, K_cluster, d_A[:, :, :, :], A[:, :, :, :]
        )
        cudaMemcpyAsync_htod_4f32(
            L, N, K_split, K_cluster, d_B[:, :, :, :], B[:, :, :, :]
        )
        gpu_gemm(L, M, N, K_split, K_cluster, d_A, d_B, d_C)
        cudaMemcpyAsync_dtoh_3f32(L, M, N, C[:, :, :], d_C[:, :, :])

    return cpu_gemm_wrapper


def mktest_golden(config: gemm_util.GemmConfig, ncta_M, ncta_N, scheduled):
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


def mktest_run(config: gemm_util.GemmConfig, ncta_M, ncta_N, scheduled, K_split):
    # More thorough runtime testing will remain in sporkbench.
    L = 1
    M = 2000
    N = 2004
    K = 1600
    K_cluster = K // K_split

    def test(compiler_Sm90a):
        rand = random.Random(100)
        A = np.ndarray(shape=(M, K), dtype=np.float32, order="C")
        B = np.ndarray(shape=(K, N), dtype=np.float32, order="F")
        C = np.ndarray(shape=(M, N), dtype=np.float32, order="C")

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
        cu(None, L, M, N, K_split, K_cluster, A, B, C)
        assert np.array_equal(C, C_expected)

    return test


test_golden_Sm90a_gemm_m1n1_A = mktest_golden(config_A, 1, 1, False)
test_golden_Sm90a_gemm_m1n1_B = mktest_golden(config_B, 1, 1, False)
test_golden_Sm90a_gemm_m1n1_K = mktest_golden(config_K, 1, 1, False)

test_golden_Sm90a_gemm_m2n1_A = mktest_golden(config_A, 2, 1, False)
test_golden_Sm90a_gemm_m1n2_B = mktest_golden(config_B, 1, 2, False)
test_golden_Sm90a_gemm_m2n2_K = mktest_golden(config_K, 2, 2, False)
test_run_Sm90a_gemm_m2n2_K = mktest_run(config_K, 2, 2, False, K_split=2)

# test_golden_Sm90a_sch_gemm_m1n2_B = mktest_golden(config_B, 1, 2, True)
# test_run_Sm90a_sch_gemm_m1n2_B = mktest_run(config_B, 1, 2, True, K_split=1)


def test_Sm90a_gemm_remarks(compiler, golden):
    # This is a very fragile test.
    # Anytime we add more logging, the golden will change.
    # We just want to force some logging code coverage.

    def mkproc():
        config = replace(config_K, ncta_M=2, ncta_N=1)
        gpu_gemm = gemm_util.handwrite_gemm(config)
        gpu_gemm = rename(gpu_gemm, "gpu_gemm")
        return gpu_gemm

    compiler.cuda_cpu_test(mkproc)
    debug_log = get_global_debug_log()
    debug_log.write_all_impl()
    with open(str(compiler.workdir / "debug" / "gpu_gemm-analysis.py")) as f:
        assert f.read() == golden
