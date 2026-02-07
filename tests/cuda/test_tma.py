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


tma_tester_smem_M = 248
tma_tester_smem_K_dict = {0: 32, 32: 8, 64: 16, 128: 32}
tma_tester_tasks_M = 3
tma_tester_tasks_K = 4


def mkproc_tma_tester(swizzle: int):
    smem_M = tma_tester_smem_M
    smem_K = tma_tester_smem_K_dict[swizzle]
    M = smem_M * tma_tester_tasks_M
    K = smem_K * tma_tester_tasks_K
    smem_type = CudaSmemLinear if swizzle == 0 else Sm90_SmemSwizzled(swizzle)

    # h_sum = h_x + h_y
    # x is loaded to SMEM with TMA
    # y is loaded to SMEM with Sm80 cp.async
    # sum = x + y computed with cuda_in_order code (not using instrs)
    # sum is stored using TMA
    # This is testing that cp.async and cuda_in_order code is interacting with
    # swizzled memory in the format that TMA defines.
    # Futhermore the fact that we have smem_M = 248 (weird number) is testing
    # that the non-aligned-to-1024B memory (smem_y, smem_sum) isn't causing problems.
    @proc
    def tma_tester_proc(h_sum: f32[M, K], h_x: f32[M, K], h_y: f32[M, K]):
        # fmt: off
        d_x: f32[M, K + 64] @ CudaGmemLinear
        d_y: f32[M, K] @ CudaGmemLinear
        d_sum: f32[1 + M, K] @ CudaGmemLinear

        for m in seq(0, K):
            cudaMemcpyAsync_htod_1f32(K, d_x[m, 0:K], h_x[m, 0:K])
        cudaMemcpyAsync_htod_2f32(M, K, d_y[:, :], h_y[:, :])

        # NB with the K + 64 in d_x, we are testing what happens if
        # you create a tensorMap that is not densely packed.

        # Backdoor for testing, _debug is supposed to trigger writing out the
        # WindowFeatures as comments in the generated C++
        x_tensorMap_debug = d_x[:, 0:K] @ Sm90_tensorMap(swizzle, smem_M, smem_K)
        sum_tensorMap_debug = d_sum[:, :] @ Sm90_tensorMap(swizzle, smem_M, smem_K)

        # We skip d_sum[0, :]
        # This is testing TMA WindowStmt on the CPU.
        sum_tensorMap_window = sum_tensorMap_debug[1:, :]

        with CudaDeviceFunction(blockDim=256):
            for task_m in cuda_tasks(0, tma_tester_tasks_M):
                for task_k in cuda_tasks(0, tma_tester_tasks_K):
                    smem_x: f32[smem_M, smem_K] @ smem_type
                    smem_y: f32[smem_M, smem_K] @ smem_type
                    smem_sum: f32[smem_M, smem_K] @ smem_type
                    raw: barrier @ CudaMbarrier
                    war: barrier(raw) @ CudaMbarrier

                    # Warp 0 copies x to SMEM using TMA.
                    with CudaWarps(0, 1):
                        Await(war, cuda_temporal, ~1)
                        # Test for TMA WindowStmt on the GPU
                        x_input = x_tensorMap_debug[
                            task_m * smem_M : task_m * smem_M + smem_M,
                            task_k * smem_K : task_k * smem_K + smem_K,
                        ]
                        if swizzle == 0:
                            Sm90_copy_tensor_to_smem_linear_2f32(smem_x[:, :], x_input,
                                size0=smem_M, size1=smem_K) >> raw
                        else:
                            Sm90_copy_tensor_to_smem_swizzled_2f32(smem_x[:, :], x_input,
                                size0=smem_M, size1=smem_K) >> raw
                        Arrive(cuda_temporal, 1) >> raw

                    # All warps copy y to SMEM using cp.async (lazy threading)
                    for m in cuda_threads(0, smem_M):
                        for ko in seq(0, smem_K / 4):
                            Sm80_cp_async_f32(
                                smem_y[m, ko * 4: ko * 4 + 4],
                                d_y[task_m * smem_M + m, task_k * smem_K + ko * 4: task_k * smem_K + ko * 4 + 4],
                                size=4,
                            )
                    Fence(Sm80_cp_async, cuda_in_order)

                    # Compute the sum (also lazy threading)
                    Await(raw, cuda_in_order, ~0)
                    for m in cuda_threads(0, smem_M):
                        for k in seq(0, smem_K):
                            smem_sum[m, k] = smem_x[m, k] + smem_y[m, k]
                    Arrive(cuda_in_order, 1) >> war

                    # Warp 0 copies sum to GMEM using TMA.
                    Fence(cuda_in_order, cuda_generic_and_async_proxy)
                    tma_window = sum_tensorMap_window[
                        task_m * smem_M : task_m * smem_M + smem_M,
                        task_k * smem_K : task_k * smem_K + smem_K,
                    ]
                    with CudaWarps(0, 1):
                        if swizzle == 0:
                            Sm90_copy_tensor_to_gmem_linear_2f32(tma_window, smem_sum[:, :], size0=smem_M, size1=smem_K)
                        else:
                            Sm90_copy_tensor_to_gmem_swizzled_2f32(tma_window, smem_sum[:, :], size0=smem_M, size1=smem_K)
                        cg: barrier @ CudaCommitGroup
                        Arrive(tma_to_gmem_async, 1) >> cg
                        Await(cg, cuda_in_order, 0)
                    Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_2f32(M, K, h_sum[:, :], d_sum[1:, :])

    tma_tester_proc = simplify(tma_tester_proc)
    tma_tester_proc.sync_check()
    tma_tester_proc = rename(tma_tester_proc, "tma_tester_SW" + str(swizzle))
    return tma_tester_proc


def test_tma_tester_proc_SW0_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_tma_tester, swizzle=0, golden=golden)


def test_tma_tester_proc_SW32_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_tma_tester, swizzle=32, golden=golden)


def test_tma_tester_proc_SW64_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_tma_tester, swizzle=64, golden=golden)


def test_tma_tester_proc_SW128_golden(compiler, golden):
    compiler.cuda_cpu_test(mkproc_tma_tester, swizzle=128, golden=golden, sm="90a")


def test_tma_Sm90a(compiler_Sm90a):
    # Compile them all together to make sure there's no symbol name conflicts
    # when using different swizzle modes.
    procs = [mkproc_tma_tester(sw) for sw in (0, 32, 64, 128)]
    cu = compiler_Sm90a.cuda_test_context(procs)

    for swizzle in (0, 32, 64, 128):
        M = tma_tester_tasks_M * tma_tester_smem_M
        K = tma_tester_tasks_K * tma_tester_smem_K_dict[swizzle]
        h_x = np.ndarray(shape=(M, K), dtype=np.float32, order="C")
        h_y = np.ndarray(shape=(M, K), dtype=np.float32, order="C")
        h_sum = np.ndarray(shape=(M, K), dtype=np.float32, order="C")
        h_sum_expected = np.ndarray(shape=(M, K), dtype=np.float32, order="C")

        for m in range(0, M):
            for k in range(0, K):
                h_x[m, k] = 3 * m + 5 * k
                h_y[m, k] = 10 * m + swizzle
                h_sum_expected[m, k] = h_x[m, k] + h_y[m, k]
                h_sum[m, k] = 0

        cu.run(f"tma_tester_SW{swizzle}", None, h_sum, h_x, h_y)

        for m in range(0, M):
            for k in range(0, K):
                assert h_sum[m, k] == h_sum_expected[m, k]
