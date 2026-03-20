from __future__ import annotations

import math
import exo
import numpy as np
from dataclasses import dataclass
from random import Random
from typing import List, Tuple, Set, Dict, Optional, Callable
import operator

from exo import *
from exo.API import InstrTemplate
from exo.platforms.cuda import *
from exo.platforms.cuda_tk import *
from exo.scalars import inf, f16, bf16, f32, e4m3, e5m2
from exo.stdlib.scheduling import *

# Not intended as a public module, but we do this to autogenerate
# tests for all the instructions.
import exo.platforms.kittens_impl.tk_vec_ops as ops_module


vec_instr_names = []


for attr in sorted(dir(ops_module)):
    obj = getattr(ops_module, attr)
    if isinstance(obj, InstrTemplate):
        if attr.endswith("_inf"):
            # Ignore convenience pos_infty -> pos_inf renames
            assert hasattr(ops_module, attr + "ty")
        else:
            assert attr.startswith("cuda_tk_vec_"), attr
            assert hasattr(exo.platforms.cuda_tk, attr), attr
            vec_instr_names.append(attr)


assert len(vec_instr_names) >= 44, "Add or remove test coverage"


@dataclass(slots=True)
class VecTester:
    p: proc
    run: Callable[None, [CudaTestContext]]
    instr_name: str
    only: bool


def make_copy_tester(
    dst_layout, src_layout, T_dst, T_src, only=False, *, expected_tuple
):
    rng = Random(137)
    length = 32 * rng.randrange(1, 4)
    if dst_layout == src_layout:
        instr_name = "cuda_tk_vec_copy"
        copy_instr = getattr(ops_module, instr_name)(
            length=length, layout=dst_layout, dst=T_dst, src=T_src
        )
    else:
        instr_name = "cuda_tk_vec_copy_layout"
        copy_instr = getattr(ops_module, instr_name)(
            length=length,
            dst_layout=dst_layout,
            src_layout=src_layout,
            dst=T_dst,
            src=T_src,
        )

    @proc
    def p(
        h_dst: f32[length],
        h_src: f32[length],
    ):
        # fmt: off
        d_dst: f32[length] @ CudaGmemLinear
        d_src: f32[length] @ CudaGmemLinear
        cudaMemcpyAsync_htod_1f32(length, d_src[:], h_src[:])

        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                s_src: f32[length] @ CudaSmemLinear
                s_dst: f32[length] @ CudaSmemLinear
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        s_src[s * 32 + tid] = d_src[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)
                r_dst: T_dst[length] @ CudaTkWarpVec(length, dst_layout)
                r_src: T_src[length] @ CudaTkWarpVec(length, src_layout)
                cuda_tk_load_vec_rs(r_src[:], s_src[:], length=length, layout=src_layout, dst=T_src, src=f32)
                copy_instr(r_dst[:], r_src[:])
                cuda_tk_store_vec_rs(s_dst[:], r_dst[:], length=length, layout=dst_layout, dst=f32, src=T_dst)
                Fence(cuda_in_order, cuda_in_order)
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        d_dst[s * 32 + tid] = s_dst[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_1f32(length, h_dst[:], d_dst[:])

    proc_name = f"copy_tester_{T_dst}{dst_layout}_{T_src}{src_layout}"
    p = simplify(p)
    p = rename(p, proc_name)

    def run(cu: CudaTestContext):
        rng_start = -100
        rng_end = 100
        h_dst = np.zeros((length,), dtype=np.float32)
        h_src = np.zeros((length,), dtype=np.float32)
        for i in range(0, length):
            h_src[i] = rng.randrange(rng_start, rng_end)

        assert len(expected_tuple) == 2
        h_src[29] = expected_tuple[1]

        cu.run(proc_name, None, h_dst, h_src)

        # Exact comparisons here.
        for i in range(0, length):
            if i == 29:
                assert h_dst[i] == expected_tuple[0], proc_name
            else:
                assert h_dst[i] == h_src[i], proc_name

    return VecTester(p, run, instr_name, only)


def make_0ary_tester(instr_name, T, only=False, *, expected_value):
    rng = Random(instr_name + str(T))
    length = 32 * rng.randrange(1, 4)
    layout = rng.choice(("align", "ortho", "naive"))
    vec_instr = getattr(ops_module, instr_name)

    @proc
    def p(h_dst: f32[length]):
        # fmt: off
        d_dst: f32[length] @ CudaGmemLinear
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                r_dst: T[length] @ CudaTkWarpVec(length, layout)
                vec_instr(r_dst[:], dst=T, length=length, layout=layout)
                s_dst: f32[length] @ CudaSmemLinear
                cuda_tk_store_vec_rs(s_dst[:], r_dst[:], length=length, layout=layout, dst=f32, src=T)
                Fence(cuda_in_order, cuda_in_order)
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        d_dst[s * 32 + tid] = s_dst[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)
        cudaMemcpyAsync_dtoh_1f32(length, h_dst[:], d_dst[:])

    proc_name = f"tester_{instr_name}_{T}{layout}"
    p = simplify(p)
    p = rename(p, proc_name)

    def run(cu: CudaTestContext):
        h_dst = np.zeros((length,), dtype=np.float32)
        h_dst[0] = 1234

        cu.run(proc_name, None, h_dst)

        for i in range(0, length):
            assert h_dst[i] == expected_value, proc_name

    return VecTester(p, run, instr_name, only)


def make_unary_tester(instr_name, T, only=False, *, expected_tuple=None):
    rng = Random(instr_name + str(T))
    length = 32 * rng.randrange(1, 4)
    layout = rng.choice(("align", "ortho", "naive"))
    vec_instr = getattr(ops_module, instr_name)

    @proc
    def p(
        h_cpu_dst: f32[length],
        h_cuda_dst: f32[length],
        h_src: f32[length],
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test works by comparing the CPU output generated here to CUDA.
        vec_instr(h_cpu_dst[:], h_src[:], length=length, dst=f32, src=f32, layout=layout)

        d_dst: f32[length] @ CudaGmemLinear
        d_src: f32[length] @ CudaGmemLinear
        cudaMemcpyAsync_htod_1f32(length, d_src[:], h_src[:])

        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                s_src: f32[length] @ CudaSmemLinear
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        s_src[s * 32 + tid] = d_src[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)
                r_src: T[length] @ CudaTkWarpVec(length, layout)
                r_dst: T[length] @ CudaTkWarpVec(length, layout)
                s_dst: f32[length] @ CudaSmemLinear
                cuda_tk_load_vec_rs(r_src[:], s_src[:], length=length, layout=layout, dst=T, src=f32)
                vec_instr(r_dst[:], r_src[:], length=length, layout=layout, dst=T, src=T)
                cuda_tk_store_vec_rs(s_dst[:], r_dst[:], length=length, layout=layout, dst=f32, src=T)
                Fence(cuda_in_order, cuda_in_order)
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        d_dst[s * 32 + tid] = s_dst[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_1f32(length, h_cuda_dst[:], d_dst[:])

    proc_name = f"tester_{instr_name}_{T}{layout}"
    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, proc_name)

    if T.bits <= 8:
        rng_start = -1
        rng_end = 1
    elif "_log" in instr_name:
        rng_start = 0.25
        rng_end = 125.0
    elif "_exp" in instr_name:
        rng_start = -10
        rng_end = 3
    else:
        rng_start = -15
        rng_end = 15

    def run(cu: CudaTestContext):
        h_cpu_dst = np.zeros((length,), dtype=np.float32)
        h_cuda_dst = np.zeros((length,), dtype=np.float32)
        h_src = np.zeros((length,), dtype=np.float32)
        for i in range(0, length):
            h_src[i] = 0.25 * rng.randrange(int(4 * rng_start), int(4 * rng_end))
        if expected_tuple:
            assert len(expected_tuple) == 2
            h_src[29] = expected_tuple[1]

        cu.run(proc_name, None, h_cpu_dst, h_cuda_dst, h_src)

        for i in range(0, length):
            if i == 29 and expected_tuple:
                # Exact comparison
                assert h_cuda_dst[i] == expected_tuple[0], proc_name
            else:
                assert math.fabs(h_cpu_dst[i] - h_cuda_dst[i]) <= 0.0625, proc_name

    return VecTester(p, run, instr_name, only)


def make_binary_tester(instr_name, T, only=False, *, expected_tuple):
    rng = Random(instr_name + str(T))
    length = 32 * rng.randrange(1, 4)
    layout = rng.choice(("align", "ortho", "naive"))

    fragments = instr_name.split("_")
    if fragments[-1] == "3op":
        instr_name_2op = "_".join(fragments[:-1]) + "_lhs"
        instr_name_3op = instr_name
        writes_lhs = False
        writes_rhs = False
    else:
        instr_name_2op = instr_name
        instr_name_3op = "_".join(fragments[:-1]) + "_3op"
        assert fragments[-1] in ("lhs", "rhs", "reduce")
        writes_lhs = fragments[-1] != "rhs"
        writes_rhs = fragments[-1] == "rhs"
    vec_instr_2op = getattr(ops_module, instr_name_2op)
    vec_instr_3op = getattr(ops_module, instr_name_3op)

    @proc
    def p(
        h_cpu_dst: f32[length],
        h_cuda_dst: f32[length],
        h_lhs: f32[length],
        h_rhs: f32[length],
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test works by comparing the CPU output generated here to CUDA.
        vec_instr_3op(
            h_cpu_dst[:], h_lhs[:], h_rhs[:],
            length=length, dst=f32, lhs=f32, rhs=f32, layout=layout,
        )

        d_dst: f32[length] @ CudaGmemLinear
        d_lhs: f32[length] @ CudaGmemLinear
        d_rhs: f32[length] @ CudaGmemLinear
        cudaMemcpyAsync_htod_1f32(length, d_lhs[:], h_lhs[:])
        cudaMemcpyAsync_htod_1f32(length, d_rhs[:], h_rhs[:])

        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                s_lhs: f32[length] @ CudaSmemLinear
                s_rhs: f32[length] @ CudaSmemLinear
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        s_lhs[s * 32 + tid] = d_lhs[s * 32 + tid]
                        s_rhs[s * 32 + tid] = d_rhs[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)
                r_lhs: T[length] @ CudaTkWarpVec(length, layout)
                r_rhs: T[length] @ CudaTkWarpVec(length, layout)
                r_dst: T[length] @ CudaTkWarpVec(length, layout)
                s_dst: f32[length] @ CudaSmemLinear
                cuda_tk_load_vec_rs(r_lhs[:], s_lhs[:], length=length, layout=layout, dst=T, src=f32)
                cuda_tk_load_vec_rs(r_rhs[:], s_rhs[:], length=length, layout=layout, dst=T, src=f32)

                if writes_lhs:
                    cuda_tk_vec_copy(r_dst[:], r_lhs[:], length=length, layout=layout, dst=T, src=T)
                    vec_instr_2op(r_dst[:], r_rhs[:], length=length, layout=layout, dst=T, src=T)
                elif writes_rhs:
                    cuda_tk_vec_copy(r_dst[:], r_rhs[:], length=length, layout=layout, dst=T, src=T)
                    vec_instr_2op(r_lhs[:], r_dst[:], length=length, layout=layout, dst=T, src=T)
                else:
                    vec_instr_3op(r_dst[:], r_lhs[:], r_rhs[:], length=length, layout=layout, dst=T, lhs=T, rhs=T)

                cuda_tk_store_vec_rs(s_dst[:], r_dst[:], length=length, layout=layout, dst=f32, src=T)
                Fence(cuda_in_order, cuda_in_order)
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        d_dst[s * 32 + tid] = s_dst[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_1f32(length, h_cuda_dst[:], d_dst[:])

    proc_name = f"tester_{instr_name}_{T}{layout}"
    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, proc_name)

    lhs_magn = 128 if T.bits >= 32 else 8
    rhs_magn = 129 if T.bits >= 32 else 5

    def run(cu: CudaTestContext):
        h_cpu_dst = np.zeros((length,), dtype=np.float32)
        h_cuda_dst = np.zeros((length,), dtype=np.float32)
        h_lhs = np.zeros((length,), dtype=np.float32)
        h_rhs = np.zeros((length,), dtype=np.float32)
        for i in range(length):
            h_lhs[i] = rng.randrange(-lhs_magn, lhs_magn)
            h_rhs[i] = rng.randrange(1, rhs_magn)

        assert len(expected_tuple) == 3
        h_lhs[29] = expected_tuple[1]
        h_rhs[29] = expected_tuple[2]

        cu.run(proc_name, None, h_cpu_dst, h_cuda_dst, h_lhs, h_rhs)

        for i in range(0, length):
            if i == 29 and expected_tuple:
                # Exact comparison
                assert h_cuda_dst[i] == expected_tuple[0], proc_name
            else:
                assert math.fabs(h_cpu_dst[i] - h_cuda_dst[i]) <= 0.0625, proc_name

    return VecTester(p, run, instr_name, only)


def make_binary_vec_scalar_tester(instr_name, T, only=False, *, expected_tuple):
    rng = Random(instr_name + str(T))
    length = 32 * rng.randrange(1, 4)
    layout = rng.choice(("align", "ortho", "naive"))

    fragments = instr_name.split("_")
    assert len(fragments) >= 2
    assert fragments[-1] == "scalar"
    if fragments[-2] == "3op":
        instr_name_2op = "_".join(fragments[:-2]) + "_lhs_scalar"
        instr_name_3op = instr_name
        writes_lhs = False
    else:
        instr_name_2op = instr_name
        instr_name_3op = "_".join(fragments[:-2]) + "_3op_scalar"
        assert fragments[-2] in ("lhs", "reduce")
        writes_lhs = True
    vec_instr_2op = getattr(ops_module, instr_name_2op)
    vec_instr_3op = getattr(ops_module, instr_name_3op)

    @proc
    def p(
        h_cpu_dst: f32[length],
        h_cuda_dst: f32[length],
        h_lhs: f32[length],
        h_cpu_rhs: f32,
        h_cuda_rhs: T,
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test works by comparing the CPU output generated here to CUDA.
        vec_instr_3op(
            h_cpu_dst[:], h_lhs[:], h_cpu_rhs,
            length=length, dst=f32, lhs=f32, rhs=f32, layout=layout,
        )


        d_dst: f32[length] @ CudaGmemLinear
        d_lhs: f32[length] @ CudaGmemLinear
        d_rhs: T @ CudaGridConstant
        cudaMemcpyAsync_htod_1f32(length, d_lhs[:], h_lhs[:])
        d_rhs = h_cuda_rhs

        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                s_lhs: f32[length] @ CudaSmemLinear
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        s_lhs[s * 32 + tid] = d_lhs[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)
                r_lhs: T[length] @ CudaTkWarpVec(length, layout)
                r_dst: T[length] @ CudaTkWarpVec(length, layout)
                s_dst: f32[length] @ CudaSmemLinear
                cuda_tk_load_vec_rs(r_lhs[:], s_lhs[:], length=length, layout=layout, dst=T, src=f32)

                if writes_lhs:
                    cuda_tk_vec_copy(r_dst[:], r_lhs[:], length=length, layout=layout, dst=T, src=T)
                    vec_instr_2op(r_dst[:], d_rhs, length=length, layout=layout, dst=T, src=T)
                else:
                    vec_instr_3op(r_dst[:], r_lhs[:], d_rhs, length=length, layout=layout, dst=T, lhs=T, rhs=T)

                cuda_tk_store_vec_rs(s_dst[:], r_dst[:], length=length, layout=layout, dst=f32, src=T)
                Fence(cuda_in_order, cuda_in_order)
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        d_dst[s * 32 + tid] = s_dst[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_1f32(length, h_cuda_dst[:], d_dst[:])

    proc_name = f"tester_{instr_name}_{T}{layout}"
    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, proc_name)

    lhs_magn = 128 if T.bits >= 32 else 8
    rhs_magn = 129 if T.bits >= 32 else 5

    def run(cu: CudaTestContext):
        h_cpu_dst = np.zeros((length,), dtype=np.float32)
        h_cuda_dst = np.zeros((length,), dtype=np.float32)
        h_lhs = np.zeros((length,), dtype=np.float32)
        for i in range(length):
            h_lhs[i] = rng.randrange(-lhs_magn, lhs_magn)

        assert len(expected_tuple) == 3
        h_lhs[29] = expected_tuple[1]
        h_cpu_rhs = np.array((expected_tuple[2],), dtype=np.float32)
        if T == f32:
            h_cuda_rhs = np.array((expected_tuple[2],), dtype=np.float32)
        else:
            assert T in (f16, f32)
            h_cuda_rhs = np.array((expected_tuple[2],), dtype=np.float16)

        cu.run(proc_name, None, h_cpu_dst, h_cuda_dst, h_lhs, h_cpu_rhs, h_cuda_rhs)

        for i in range(0, length):
            if i == 29 and expected_tuple:
                # Exact comparison
                assert h_cuda_dst[i] == expected_tuple[0], proc_name
            else:
                assert math.fabs(h_cpu_dst[i] - h_cuda_dst[i]) <= 0.0625, proc_name

    return VecTester(p, run, instr_name, only)


def test_tk_vec_ops(compiler_Sm80):
    # Note, because ThunderKittens takes so long to compile, we amortize
    # the time by compiling all the tests together.
    # Pass only=True to one of the testers to select just that sub-test to run.
    # fmt: off
    testers = [
        #
        make_copy_tester("align", "align", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_copy_tester("align", "ortho", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_copy_tester("align", "naive", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_copy_tester("ortho", "align", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_copy_tester("ortho", "ortho", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_copy_tester("ortho", "naive", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_copy_tester("naive", "align", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_copy_tester("naive", "ortho", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_copy_tester("naive", "naive", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_copy_tester("align", "align", f32, f16, expected_tuple=(19.25, 19.25)),
        make_copy_tester("align", "ortho", f32, f16, expected_tuple=(19.25, 19.25)),
        make_copy_tester("align", "naive", f32, f16, expected_tuple=(19.25, 19.25)),
        make_copy_tester("ortho", "align", f32, bf16, expected_tuple=(19.25, 19.25)),
        make_copy_tester("ortho", "ortho", f32, bf16, expected_tuple=(19.25, 19.25)),
        make_copy_tester("ortho", "naive", f32, bf16, expected_tuple=(19.25, 19.25)),
        make_copy_tester("align", "ortho", f16, f32, expected_tuple=(inf, 102400.125)),
        make_copy_tester("align", "ortho", bf16, f32, expected_tuple=(102400, 102400.125)),
        #
        make_0ary_tester("cuda_tk_vec_zero", f32, expected_value=0),
        make_0ary_tester("cuda_tk_vec_one", f32, expected_value=1),
        make_0ary_tester("cuda_tk_vec_pos_infty", f32, expected_value=inf),
        make_0ary_tester("cuda_tk_vec_neg_infty", f32, expected_value=-inf),
        make_0ary_tester("cuda_tk_vec_zero", f16, expected_value=0),
        make_0ary_tester("cuda_tk_vec_one", bf16, expected_value=1),
        make_0ary_tester("cuda_tk_vec_pos_infty", f16, expected_value=inf),
        make_0ary_tester("cuda_tk_vec_neg_infty", bf16, expected_value=-inf),
        #
        make_unary_tester("cuda_tk_vec_exp", f16),
        make_unary_tester("cuda_tk_vec_exp2", f16, expected_tuple=(8, 3)),
        make_unary_tester("cuda_tk_vec_log", f16),
        make_unary_tester("cuda_tk_vec_log2", f16, expected_tuple=(3, 8)),
        make_unary_tester("cuda_tk_vec_abs", f16, expected_tuple=(inf, -100000)),
        make_unary_tester("cuda_tk_vec_relu", f16, expected_tuple=(0, -19.125)),
        #
        make_unary_tester("cuda_tk_vec_exp", f32),
        make_unary_tester("cuda_tk_vec_exp2", f32, expected_tuple=(8, 3)),
        make_unary_tester("cuda_tk_vec_log", f32),
        make_unary_tester("cuda_tk_vec_log2", f32, expected_tuple=(3, 8)),
        make_unary_tester("cuda_tk_vec_abs", f32, expected_tuple=(19.125, -19.125)),
        make_unary_tester("cuda_tk_vec_relu", f32, expected_tuple=(19.125, 19.125)),
        #
        make_binary_tester("cuda_tk_vec_add_3op", f32, expected_tuple=(9, 3, 6)),
        make_binary_tester("cuda_tk_vec_sub_3op", f32, expected_tuple=(-3, 3, 6)),
        make_binary_tester("cuda_tk_vec_mul_3op", f32, expected_tuple=(18, 3, 6)),
        make_binary_tester("cuda_tk_vec_div_3op", f32, expected_tuple=(0.375, 3, 8)),
        make_binary_tester("cuda_tk_vec_max_3op", f32, expected_tuple=(8, 3, 8)),
        make_binary_tester("cuda_tk_vec_min_3op", f32, expected_tuple=(3, 3, 8)),
        #
        make_binary_vec_scalar_tester("cuda_tk_vec_add_3op_scalar", f32, expected_tuple=(9, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_sub_3op_scalar", f32, expected_tuple=(-3, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_mul_3op_scalar", f32, expected_tuple=(18, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_div_3op_scalar", f32, expected_tuple=(0.375, 3, 8)),
        make_binary_vec_scalar_tester("cuda_tk_vec_max_3op_scalar", f32, expected_tuple=(8, 3, 8)),
        make_binary_vec_scalar_tester("cuda_tk_vec_min_3op_scalar", f32, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester("cuda_tk_vec_add_reduce", f32, expected_tuple=(9, 3, 6)),
        make_binary_tester("cuda_tk_vec_add_lhs", f32, expected_tuple=(9, 3, 6)),
        make_binary_tester("cuda_tk_vec_sub_lhs", f32, expected_tuple=(-3, 3, 6)),
        make_binary_tester("cuda_tk_vec_mul_lhs", f32, expected_tuple=(18, 3, 6)),
        make_binary_tester("cuda_tk_vec_div_lhs", f32, expected_tuple=(0.375, 3, 8)),
        make_binary_tester("cuda_tk_vec_max_lhs", f32, expected_tuple=(8, 3, 8)),
        make_binary_tester("cuda_tk_vec_min_lhs", f32, expected_tuple=(3, 3, 8)),
        #
        make_binary_vec_scalar_tester("cuda_tk_vec_add_reduce_scalar", f32, expected_tuple=(9, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_add_lhs_scalar", f32, expected_tuple=(9, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_sub_lhs_scalar", f32, expected_tuple=(-3, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_mul_lhs_scalar", f32, expected_tuple=(18, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_div_lhs_scalar", f32, expected_tuple=(0.375, 3, 8)),
        make_binary_vec_scalar_tester("cuda_tk_vec_max_lhs_scalar", f32, expected_tuple=(8, 3, 8)),
        make_binary_vec_scalar_tester("cuda_tk_vec_min_lhs_scalar", f32, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester("cuda_tk_vec_add_rhs", f32, expected_tuple=(9, 3, 6)),
        make_binary_tester("cuda_tk_vec_sub_rhs", f32, expected_tuple=(-3, 3, 6)),
        make_binary_tester("cuda_tk_vec_mul_rhs", f32, expected_tuple=(18, 3, 6)),
        make_binary_tester("cuda_tk_vec_div_rhs", f32, expected_tuple=(0.375, 3, 8)),
        make_binary_tester("cuda_tk_vec_max_rhs", f32, expected_tuple=(8, 3, 8)),
        make_binary_tester("cuda_tk_vec_min_rhs", f32, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester("cuda_tk_vec_add_3op", f16, expected_tuple=(9, 3, 6)),
        make_binary_tester("cuda_tk_vec_sub_3op", f16, expected_tuple=(-3, 3, 6)),
        make_binary_tester("cuda_tk_vec_mul_3op", f16, expected_tuple=(18, 3, 6)),
        make_binary_tester("cuda_tk_vec_div_3op", f16, expected_tuple=(0.375, 3, 8)),
        make_binary_tester("cuda_tk_vec_max_3op", f16, expected_tuple=(8, 3, 8)),
        make_binary_tester("cuda_tk_vec_min_3op", f16, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester("cuda_tk_vec_add_lhs", bf16, expected_tuple=(9, 3, 6)),
        make_binary_tester("cuda_tk_vec_sub_lhs", bf16, expected_tuple=(-3, 3, 6)),
        make_binary_tester("cuda_tk_vec_mul_lhs", bf16, expected_tuple=(18, 3, 6)),
        make_binary_tester("cuda_tk_vec_div_lhs", bf16, expected_tuple=(0.375, 3, 8)),
        make_binary_tester("cuda_tk_vec_max_lhs", bf16, expected_tuple=(8, 3, 8)),
        make_binary_tester("cuda_tk_vec_min_lhs", bf16, expected_tuple=(3, 3, 8)),
        #
        make_binary_vec_scalar_tester("cuda_tk_vec_add_3op_scalar", f16, expected_tuple=(9, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_mul_3op_scalar", f16, expected_tuple=(18, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_add_reduce_scalar", f16, expected_tuple=(9, 3, 6)),
        make_binary_vec_scalar_tester("cuda_tk_vec_mul_lhs_scalar", f16, expected_tuple=(18, 3, 6)),
    ]
    # fmt: on

    # Implements only=True from comment.
    have_only = any(tester.only for tester in testers)
    if have_only:
        testers = [tester for tester in testers if tester.only]

    procs = [tester.p for tester in testers]

    cu = compiler_Sm80.cuda_test_context(procs)

    for tester in testers:
        tester.run(cu)

    assert not have_only, "Test passed. Please remove only=True."

    tested_instr_names = set(tester.instr_name for tester in testers)
    missing_instr_names = set(vec_instr_names) - tested_instr_names
    assert not missing_instr_names, "Missing coverage"
