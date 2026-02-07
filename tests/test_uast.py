from __future__ import annotations

import pytest

from exo import DRAM
from exo.frontend.pyparser import (
    Parser,
    get_parent_scope,
    get_ast_from_python,
    ParseError,
)
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *


def to_uast(f):
    body, getsrcinfo = get_ast_from_python(f)
    parser = Parser(
        body,
        getsrcinfo,
        parent_scope=get_parent_scope(depth=2),
        as_func=True,
    )
    return parser.result()


def test_conv1d(golden):
    def conv1d(
        n: size, m: size, r: size, x: R[n], w: R[m], res: R[r]
    ):  # pragma: no cover
        for i in seq(0, r):
            res[i] = 0.0
        for i in seq(0, r):
            for j in seq(0, n):
                if i <= j < i + m:
                    res[i] += x[j] * w[i - j + m - 1]

    assert str(to_uast(conv1d)) == golden


def test_unary_neg(golden):
    def negate_array(n: size, x: R[n], res: R[n] @ DRAM):  # pragma: no cover
        for i in seq(0, n):
            res[i] = -x[i] + -(x[i]) - -(x[i] + 0.0)

    assert str(to_uast(negate_array)) == golden


def test_alloc_nest(golden):
    def alloc_nest(
        n: size, m: size, x: R[n, m], y: R[n, m] @ DRAM, res: R[n, m] @ DRAM
    ):  # pragma: no cover
        for i in seq(0, n):
            rloc: R[m] @ DRAM
            xloc: R[m] @ DRAM
            yloc: R[m] @ DRAM
            for j in seq(0, m):
                xloc[j] = x[i, j]
            for j in seq(0, m):
                yloc[j] = y[i, j]
            for j in seq(0, m):
                rloc[j] = xloc[j] + yloc[j]
            for j in seq(0, m):
                res[i, j] = rloc[j]

    assert str(to_uast(alloc_nest)) == golden


global_str = "What is 6 times 9?"
global_num = 42


def test_variable_lookup_positive():
    def func(f: f32):
        for i in seq(0, 42):
            f += 1

    reference = to_uast(func)

    def func(f: f32):
        for i in seq(0, global_num):
            f += 1

    test_global = to_uast(func)
    assert str(test_global) == str(reference)

    local_num = 42

    def func(f: f32):
        for i in seq(0, local_num):
            f += 1

    test_local = to_uast(func)
    assert str(test_local) == str(reference)


def test_variable_lookup_type_error():
    def func(f: f32):
        for i in seq(0, global_str):
            f += 1

    with pytest.raises(
        ParseError, match="Unquote received input that couldn't be unquoted"
    ):
        to_uast(func)

    local_str = "xyzzy"

    def func(f: f32):
        for i in seq(0, local_str):
            f += 1

    with pytest.raises(
        ParseError, match="Unquote received input that couldn't be unquoted"
    ):
        to_uast(func)


def test_variable_lookup_name_error():
    def func(f: f32):
        for i in seq(0, xyzzy):
            f += 1

    with pytest.raises(ParseError, match="'xyzzy' undefined"):
        to_uast(func)


def test_call_not_a_proc():
    fake_proc = 137.0

    def func(f: f32):
        fake_proc(f)

    with pytest.raises(ParseError, match="procedure or InstrTemplate"):
        to_uast(func)


def test_cuda_uast(golden):
    swizzle = 128
    tma_tester_smem_M = 248
    tma_tester_smem_K_dict = {0: 32, 32: 8, 64: 16, 128: 32}
    tma_tester_tasks_M = 3
    tma_tester_tasks_K = 4

    smem_M = tma_tester_smem_M
    smem_K = tma_tester_smem_K_dict[swizzle]
    M = smem_M * tma_tester_tasks_M
    K = smem_K * tma_tester_tasks_K
    smem_type = CudaSmemLinear if swizzle == 0 else Sm90_SmemSwizzled(swizzle)

    # Copied from cuda/test_tma.py
    def tma_tester_gpu_proc(
        sum_tensorMap_window: [f32][M, K] @ Sm90_tensorMap(swizzle, smem_M, smem_K),
        d_x: [f32][M, K] @ CudaGmemLinear,
        d_y: f32[M, K] @ CudaGmemLinear,
    ):
        assert stride(d_x, 1) == 1
        assert stride(d_y, 1) == 1
        assert stride(sum_tensorMap_window, 1) == 1
        # fmt: off
        # Backdoor for testing, _debug is supposed to trigger writing out the
        # WindowFeatures as comments in the generated C++
        x_tensorMap_debug = d_x[:, :] @ Sm90_tensorMap(swizzle, smem_M, smem_K)

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
                    with CudaWarps(0, 1):
                        tma_window = sum_tensorMap_window[
                            task_m * smem_M : task_m * smem_M + smem_M,
                            task_k * smem_K : task_k * smem_K + smem_K,
                        ]
                        if swizzle == 0:
                            Sm90_copy_tensor_to_gmem_linear_2f32(
                                tma_window[:, :], smem_sum[:, :],
                                size0=smem_M, size1=smem_K
                            )
                        else:
                            Sm90_copy_tensor_to_gmem_swizzled_2f32(
                                tma_window[:, :], smem_sum[:, :],
                                size0=smem_M, size1=smem_K
                            )
                        cg: barrier @ CudaCommitGroup
                        Arrive(tma_to_gmem_async, 1) >> cg
                        Await(cg, cuda_in_order, 0)
                    Fence(cuda_in_order, cuda_in_order)

    result = str(to_uast(tma_tester_gpu_proc))
    lines = [line.strip() for line in result.split("\n")]

    assert result == golden

    # fmt: off
    assert "with CudaWarps(0, 1):" in lines
    assert "with CudaDeviceFunction(blockDim=256):" in lines
    assert "Await(war, cuda_temporal, ~1)" in lines
    assert "raw: barrier @ CudaMbarrier" in lines
    assert "war: barrier(raw) @ CudaMbarrier" in lines
    assert "Fence(cuda_in_order, cuda_generic_and_async_proxy)" in lines
    assert "Arrive(tma_to_gmem_async, 1) >> cg" in lines
    assert "smem_box=None) >> raw" in lines  # Fragile check for trailing barrier expr of TMA instr

    # Trickiest part, WindowStmt handled weirdly in UAST pretty-printer
    assert f"x_tensorMap_debug = d_x[:, :] @ Sm90_tensorMap(128, 248, 32)" in lines


def test_cuda_uast_multicast(golden):
    # Test UAST multicast and also warp_config

    my_warp_config = [
        CudaWarpConfig("producer", 1, setmaxnreg_dec=40),
        CudaWarpConfig("unused", 3, setmaxnreg_dec=40),
        CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),
    ]

    ncta_M = 4
    ncta_N = 2
    RING = 3

    def that_cuda_proc():
        # fmt: off
        with CudaDeviceFunction(clusterDim=ncta_M * ncta_N, warp_config=my_warp_config):
            for task in cuda_tasks(0, 1):
                raw : barrier[ncta_M, ncta_N] @ CudaMbarrier
                war : barrier(raw)[ncta_M, ncta_N] @ CudaMbarrier
                for iter_k in seq(0, 100):
                    with CudaWarps(name="producer"):
                        for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                            for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                Await(war[cta_m,cta_n], cuda_temporal, ~(RING-1))
                        for cta_n in cuda_threads(0, ncta_N, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N)):
                            for cta_m in cuda_threads(0, ncta_M, unit=cuda_cta_in_cluster):
                                Arrive(cuda_temporal) >> raw[cta_m,:] >> raw[:,cta_n]
                    with CudaWarps(name="consumer"):
                        for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                            for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                Await(raw[cta_m,cta_n], cuda_generic_and_async_proxy, ~0)
                                Arrive(cuda_in_order) >> war[cta_m,:] >> war[:,cta_n]

    result = str(to_uast(that_cuda_proc))
    assert result == golden

    lines = [line.strip() for line in result.split("\n")]
    glued = " ".join(lines)

    def substr(find_me):
        return glued.find(find_me) >= 0

    # fmt: off
    assert substr("clusterDim=8")
    assert substr("setmaxnreg_dec=40")
    assert substr("for cta_n in cuda_threads(0, 2, unit=4 * cuda_cta_in_cluster_strided(2)):")
    assert substr("unit=2 * cuda_cta_in_cluster")

    assert substr("raw: barrier[4, 2] @ CudaMbarrier")
    assert substr("war: barrier(raw)[4, 2] @ CudaMbarrier")
    assert substr("Await(war[cta_m, cta_n], cuda_temporal, ~2)")
    assert substr("Arrive(cuda_temporal, 1) >> raw[cta_m, :] >> raw[:, cta_n]")
    assert substr("Await(raw[cta_m, cta_n], cuda_generic_and_async_proxy, ~0)")
    assert substr("Arrive(cuda_in_order, 1) >> war[cta_m, :] >> war[:, cta_n]")
