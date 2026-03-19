from __future__ import annotations

import math
import exo
import numpy as np
from dataclasses import dataclass
from random import Random
from typing import List, Tuple, Set, Dict, Optional, Callable

from exo import *
from exo.API import InstrTemplate
from exo.platforms.cuda import *
from exo.platforms.cuda_tk import *
from exo.scalars import inf, f16, bf16, f32, e4m3, e5m2
from exo.stdlib.scheduling import *

# Not intended as a public module, but we do this to autogenerate
# tests for all the instructions.
import exo.platforms.kittens_impl.tk_tile_ops as ops_module


tile_instr_names = []


for name in sorted(dir(ops_module)):
    obj = getattr(ops_module, name)
    if isinstance(obj, InstrTemplate):
        if name.endswith("_inf"):
            # Ignore convenience pos_infty -> pos_inf renames
            assert hasattr(ops_module, name + "ty")
        else:
            assert name.startswith("cuda_tk_"), name
            assert hasattr(exo.platforms.cuda_tk, name), name
            tile_instr_names.append(name)


assert len(tile_instr_names) >= 56, "Add or remove test coverage"


@dataclass(slots=True)
class TileTester:
    p: proc
    run: Callable[None, [CudaTestContext]]
    instr_name: str
    only: bool
    T_dst: ScalarInfo
    T_src: ScalarInfo


def make_reduce_tester(
    instr_name,
    tile_base_type,
    np_reduce,
    only=False,
):
    # Tester for tile -> vector row/col reductions.
    T = tile_base_type
    tile_instr = getattr(ops_module, instr_name)
    rng = Random(instr_name)

    if "_prod" in instr_name:
        rng_start = 1
        rng_end = 4
        rows = 32
        cols = 32
    elif T.bits <= 8:
        rng_start = -3
        rng_end = +3
        rows = 32 * rng.randrange(1, 4)
        cols = 32 * rng.randrange(1, 4)
    else:
        rng_start = -10
        rng_end = 10
        rows = 32 * rng.randrange(1, 4)
        cols = 32 * rng.randrange(1, 4)

    tile_rmem = CudaTkWarpTile(rows, cols, "row")

    if instr_name.startswith("cuda_tk_row_"):
        axis = 1
        length = rows
        vec_rmem = tile_rmem.col_vec
    else:
        assert instr_name.startswith("cuda_tk_col_")
        axis = 0
        length = cols
        vec_rmem = tile_rmem.row_vec

    @proc
    def p(
        h_cpu_dst: f32[length],
        h_cuda_dst: f32[length],
        h_src: f32[rows, cols],
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test works by comparing the CPU output generated here to CUDA.
        tile_instr(h_cpu_dst[:], h_src[:, :], rows=rows, cols=cols, dst=f32, src=f32)

        d_src: f32[rows, cols] @ CudaGmemLinear
        d_dst: f32[length] @ CudaGmemLinear

        cudaMemcpyAsync_htod_2f32(rows, cols, d_src[:, :], h_src[:, :])

        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                r_dst: T[length] @ vec_rmem
                r_src: T[rows, cols] @ tile_rmem
                cuda_tk_load_rg(r_src[:, :], d_src[:, :], size0=rows, size1=cols, dst=T, src=f32)
                tile_instr(r_dst[:], r_src[:, :], rows=rows, cols=cols, dst=T, src=T)
                s_dst: f32[length] @ CudaSmemLinear
                cuda_tk_store_vec_rs(s_dst[:], r_dst[:], length=length, layout=vec_rmem.layout, dst=f32, src=T)
                Fence(cuda_in_order, cuda_in_order)
                for s in seq(0, length / 32):
                    for tid in cuda_threads(0, 32):
                        d_dst[s * 32 + tid] = s_dst[s * 32 + tid]
                Fence(cuda_in_order, cuda_in_order)

        cudaMemcpyAsync_dtoh_1f32(length, h_cuda_dst[:], d_dst[:])

    name = f"tester_{T}_{instr_name}"
    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, name)

    def run(cu: CudaTestContext):
        h_cpu_dst = np.zeros((length,), dtype=np.float32)
        h_cuda_dst = np.zeros((length,), dtype=np.float32)
        h_ref = np.zeros((length,), dtype=np.float32)
        h_src = np.zeros((rows, cols), dtype=np.float32)
        for r in range(0, rows):
            for c in range(0, cols):
                h_src[r, c] = rng.randrange(rng_start, rng_end)

        h_ref = np_reduce.reduce(h_src, axis=axis, dtype=np.float32)

        cu.run(name, None, h_cpu_dst, h_cuda_dst, h_src)

        # Exact comparisons here.
        # This should work for everything except low-precision product
        # (so we don't test that).
        for i in range(0, length):
            assert h_cuda_dst[i] == h_ref[i], (name, i)
            assert h_cuda_dst[i] == h_cpu_dst[i], (name, i)

    return TileTester(p, run, instr_name, only, T, T)


def make_0ary_tester(
    instr_name,
    tile_base_type,
    coordinate_values: Dict[Tuple[int, int], Tuple[int, int]],
    only=False,
):
    # Tester for 0-ary operators that overwrite tiles in-place.
    # Coordinate values: given pair=coordinate_values[r, c] isn't giving KeyError,
    # Initialize tensor[r, c] to pair[1] and expect it to mutate to pair[0].
    T = tile_base_type
    tile_instr = getattr(ops_module, instr_name)

    rng = Random(instr_name)
    rows = 16 * rng.randrange(3, 6)
    cols = 16 * rng.randrange(4, 6)

    @proc
    def p(
        h_cpu_inout: f32[rows, cols],
        h_cuda_inout: f32[rows, cols],
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test works by comparing the CPU output generated here to CUDA.
        tile_instr(h_cpu_inout[:, :], rows=rows, cols=cols, dst=f32)

        d_inout: f32[rows, cols] @ CudaGmemLinear

        cudaMemcpyAsync_htod_2f32(rows, cols, d_inout[:, :], h_cuda_inout[:, :])

        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                r_inout: T[rows, cols] @ CudaTkWarpTile(rows, cols)
                cuda_tk_load_rg(r_inout[:, :], d_inout[:, :], size0=rows, size1=cols, dst=T, src=f32)
                tile_instr(r_inout[:, :], rows=rows, cols=cols, dst=T)
                Fence(cuda_in_order, cuda_in_order)
                cuda_tk_store_rg(d_inout[:, :], r_inout[:, :], size0=rows, size1=cols, dst=f32, src=T)

        cudaMemcpyAsync_dtoh_2f32(rows, cols, h_cuda_inout[:, :], d_inout[:, :])

    name = f"tester_{T}_{instr_name}"
    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, name)

    if T.bits <= 8:
        rng_start = -1
        rng_end = +1
    else:
        rng_start = -15
        rng_end = 15

    def run(cu: CudaTestContext):
        h_cpu_inout = np.zeros((rows, cols), dtype=np.float32)
        h_cuda_inout = np.zeros((rows, cols), dtype=np.float32)
        for r in range(0, rows):
            for c in range(0, cols):
                try:
                    pair = coordinate_values[r, c]
                    assert len(pair) == 2
                    init = pair[1]
                except KeyError:
                    init = 0.25 * rng.randrange(int(4 * rng_start), int(4 * rng_end))
                h_cpu_inout[r, c] = init
                h_cuda_inout[r, c] = init

        cu.run(name, None, h_cpu_inout, h_cuda_inout)

        for r in range(0, rows):
            for c in range(0, cols):
                try:
                    pair = coordinate_values[r, c]
                    # Exact comparison
                    assert h_cpu_inout[r, c] == pair[0], name
                except KeyError:
                    # Weird inf-tolerant comparison that still rejects NaN.
                    # Note != is not the same thing as not == for NaN.
                    if not (h_cpu_inout[r, c] == h_cuda_inout[r, c]):
                        # fmt: off
                        assert math.fabs(h_cpu_inout[r, c] - h_cuda_inout[r, c]) <= 0.0625, name

    return TileTester(p, run, instr_name, only, T, T)


def make_unary_tester(
    instr_name, T_dst, T_src=None, only=False, *, expected_tuple=None
):
    # Tester for unary operators `dst = op(src)`
    tile_instr = getattr(ops_module, instr_name)

    if T_src is None:
        T_src = T_dst

    rng = Random(instr_name)
    rows = 16 * rng.randrange(3, 6)
    cols = 16 * rng.randrange(4, 6)

    @proc
    def p(
        h_cpu_dst: f32[rows, cols],
        h_cuda_dst: f32[rows, cols],
        h_src: f32[rows, cols],
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test works by comparing the CPU output generated here to CUDA.
        tile_instr(
            h_cpu_dst[:, :], h_src[:, :],
            rows=rows, cols=cols, dst=f32, src=f32,
        )

        d_dst: f32[rows, cols] @ CudaGmemLinear
        d_src: f32[rows, cols] @ CudaGmemLinear

        cudaMemcpyAsync_htod_2f32(rows, cols, d_src[:, :], h_src[:, :])

        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                r_dst: T_dst[rows, cols] @ CudaTkWarpTile(rows, cols)
                r_src: T_src[rows, cols] @ CudaTkWarpTile(rows, cols)
                cuda_tk_load_rg(r_src[:, :], d_src[:, :], size0=rows, size1=cols, dst=T_src, src=f32)
                tile_instr(r_dst[:, :], r_src[:, :], rows=rows, cols=cols, dst=T_dst, src=T_src)
                cuda_tk_store_rg(d_dst[:, :], r_dst[:, :], size0=rows, size1=cols, dst=f32, src=T_dst)

        cudaMemcpyAsync_dtoh_2f32(rows, cols, h_cuda_dst[:, :], d_dst[:, :])

    name = f"tester_{T_dst}_{T_src}_{instr_name}"
    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, name)

    if T_src.bits <= 8 or T_dst.bits <= 8:
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
        h_cpu_dst = np.zeros((rows, cols), dtype=np.float32)
        h_cuda_dst = np.zeros((rows, cols), dtype=np.float32)
        h_src = np.zeros((rows, cols), dtype=np.float32)
        for r in range(0, rows):
            for c in range(0, cols):
                h_src[r, c] = 0.25 * rng.randrange(int(4 * rng_start), int(4 * rng_end))

        if expected_tuple:
            assert len(expected_tuple) == 2
            h_src[42, 49] = expected_tuple[1]

        cu.run(name, None, h_cpu_dst, h_cuda_dst, h_src)

        for r in range(0, rows):
            for c in range(0, cols):
                if r == 42 and c == 49 and expected_tuple:
                    # Exact comparison
                    assert h_cuda_dst[42, 49] == expected_tuple[0], name
                else:
                    assert math.fabs(h_cpu_dst[r, c] - h_cuda_dst[r, c]) <= 0.0625, name

    return TileTester(p, run, instr_name, only, T_dst, T_src)


def make_binary_run(name, rows, cols, lhs_magn, rhs_magn, expected_tuple):
    def run(cu: CudaTestContext):
        rng = Random(20010106)

        h_cpu_dst = np.zeros((rows, cols), dtype=np.float32)
        h_cuda_dst = np.zeros((rows, cols), dtype=np.float32)
        h_lhs = np.zeros((rows, cols), dtype=np.float32)
        h_rhs = np.zeros((rows, cols), dtype=np.float32)
        for r in range(0, rows):
            for c in range(0, cols):
                h_lhs[r, c] = rng.randrange(-lhs_magn, lhs_magn)
                h_rhs[r, c] = rng.randrange(1, rhs_magn)

        if expected_tuple:
            assert len(expected_tuple) == 3
            h_lhs[42, 49] = expected_tuple[1]
            h_rhs[42, 49] = expected_tuple[2]

        cu.run(name, None, h_cpu_dst, h_cuda_dst, h_lhs, h_rhs)

        for r in range(0, rows):
            for c in range(0, cols):
                if r == 42 and c == 49 and expected_tuple:
                    # Exact comparison
                    assert h_cuda_dst[42, 49] == expected_tuple[0], name
                else:
                    assert math.fabs(h_cpu_dst[r, c] - h_cuda_dst[r, c]) <= 0.0625, name

    return run


def make_binary_tester_3(
    instr_name, tile_base_type, only=False, *, expected_tuple=None
):
    # Tester for 3 operand tile instructions `dst = lhs op rhs`
    tile_instr_3operand = getattr(ops_module, instr_name)
    tile_instr = getattr(ops_module, instr_name)
    T = tile_base_type

    rng = Random(instr_name)
    rows = 16 * rng.randrange(3, 6)
    cols = 16 * rng.randrange(4, 6)

    @proc
    def p(
        h_cpu_dst: f32[rows, cols],
        h_cuda_dst: f32[rows, cols],
        h_lhs: f32[rows, cols],
        h_rhs: f32[rows, cols],
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test works by comparing the CPU output generated here to CUDA.
        tile_instr_3operand(
            h_cpu_dst[:, :], h_lhs[:, :], h_rhs[:, :],
            rows=rows, cols=cols, dst=f32, lhs=f32, rhs=f32,
        )

        d_lhs: f32[rows, cols] @ CudaGmemLinear
        d_rhs: f32[rows, cols] @ CudaGmemLinear
        d_dst: f32[rows, cols] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2f32(rows, cols, d_lhs[:, :], h_lhs[:, :])
        cudaMemcpyAsync_htod_2f32(rows, cols, d_rhs[:, :], h_rhs[:, :])
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                r_lhs: T[rows, cols] @ CudaTkWarpTile(rows, cols)
                r_rhs: T[rows, cols] @ CudaTkWarpTile(rows, cols)
                r_dst: T[rows, cols] @ CudaTkWarpTile(rows, cols)
                cuda_tk_load_rg(r_lhs[:, :], d_lhs[:, :], size0=rows, size1=cols, dst=T, src=f32)
                cuda_tk_load_rg(r_rhs[:, :], d_rhs[:, :], size0=rows, size1=cols, dst=T, src=f32)
                tile_instr(r_dst[:, :], r_lhs[:, :], r_rhs[:, :], rows=rows, cols=cols, dst=T, lhs=T, rhs=T)
                cuda_tk_store_rg(d_dst[:, :], r_dst[:, :], size0=rows, size1=cols, dst=f32, src=T)
        cudaMemcpyAsync_dtoh_2f32(rows, cols, h_cuda_dst[:, :], d_dst[:, :])

    proc_name = f"tester_{T}_{instr_name}"

    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, proc_name)

    lhs_magn = 128 if T.bits >= 32 else 8
    rhs_magn = 129 if T.bits >= 32 else 5

    run = make_binary_run(p.name(), rows, cols, lhs_magn, rhs_magn, expected_tuple)
    return TileTester(p, run, instr_name, only, T, T)


def make_binary_tester_2(
    instr_name, tile_base_type, only=False, *, expected_tuple=None
):
    # Tester for 2 operand tile instructions `dst = dst op src` or `dst = src op dst`
    fragments = instr_name.split("_")
    if fragments[-1] == "rhs":
        modifies_rhs = True
    else:
        modifies_rhs = False
        assert fragments[-1] == "lhs" or fragments[-1] == "reduce"
    instr_name_3operand = "_".join(fragments[:-1])
    tile_instr_3operand = getattr(ops_module, instr_name_3operand)
    tile_instr = getattr(ops_module, instr_name)
    T = tile_base_type

    rng = Random(instr_name)
    rows = 16 * rng.randrange(3, 6)
    cols = 16 * rng.randrange(4, 6)

    @proc
    def p(
        h_cpu_dst: f32[rows, cols],
        h_cuda_dst: f32[rows, cols],
        h_lhs: f32[rows, cols],
        h_rhs: f32[rows, cols],
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test works by comparing the CPU output generated here to CUDA.
        tile_instr_3operand(
            h_cpu_dst[:, :], h_lhs[:, :], h_rhs[:, :],
            rows=rows, cols=cols, dst=f32, lhs=f32, rhs=f32,
        )

        d_lhs: f32[rows, cols] @ CudaGmemLinear
        d_rhs: f32[rows, cols] @ CudaGmemLinear
        d_dst: f32[rows, cols] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2f32(rows, cols, d_lhs[:, :], h_lhs[:, :])
        cudaMemcpyAsync_htod_2f32(rows, cols, d_rhs[:, :], h_rhs[:, :])
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                r_lhs: T[rows, cols] @ CudaTkWarpTile(rows, cols)
                r_rhs: T[rows, cols] @ CudaTkWarpTile(rows, cols)
                cuda_tk_load_rg(r_lhs[:, :], d_lhs[:, :], size0=rows, size1=cols, dst=T, src=f32)
                cuda_tk_load_rg(r_rhs[:, :], d_rhs[:, :], size0=rows, size1=cols, dst=T, src=f32)
                tile_instr(r_lhs[:, :], r_rhs[:, :], rows=rows, cols=cols, dst=T, src=T)
                if modifies_rhs:
                    cuda_tk_store_rg(d_dst[:, :], r_rhs[:, :], size0=rows, size1=cols, dst=f32, src=T)
                else:
                    cuda_tk_store_rg(d_dst[:, :], r_lhs[:, :], size0=rows, size1=cols, dst=f32, src=T)
        cudaMemcpyAsync_dtoh_2f32(rows, cols, h_cuda_dst[:, :], d_dst[:, :])

    proc_name = f"tester_{T}_{instr_name}"

    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, proc_name)

    lhs_magn = 128 if T.bits >= 32 else 8
    rhs_magn = 129 if T.bits >= 32 else 5

    run = make_binary_run(p.name(), rows, cols, lhs_magn, rhs_magn, expected_tuple)
    return TileTester(p, run, instr_name, only, T, T)


def test_tk_tile_ops(compiler_Sm80):
    # Note, because ThunderKittens takes so long to compile, we amortize
    # the time by compiling all the tests together.
    # Pass only=True to one of the testers to select just that sub-test to run.
    # fmt: off
    testers = [
        #
        make_reduce_tester("cuda_tk_row_sum", f32, np.add),
        make_reduce_tester("cuda_tk_col_sum", f32, np.add),
        make_reduce_tester("cuda_tk_row_prod", f32, np.multiply),
        make_reduce_tester("cuda_tk_col_prod", f32, np.multiply),
        make_reduce_tester("cuda_tk_row_max", f32, np.maximum),
        make_reduce_tester("cuda_tk_col_max", f32, np.maximum),
        make_reduce_tester("cuda_tk_row_min", f32, np.minimum),
        make_reduce_tester("cuda_tk_col_min", f32, np.minimum),
        #
        make_reduce_tester("cuda_tk_row_sum", f16, np.add),
        make_reduce_tester("cuda_tk_row_max", f16, np.maximum),
        make_reduce_tester("cuda_tk_row_max", bf16, np.maximum),
        make_reduce_tester("cuda_tk_col_max", f16, np.maximum),
        make_reduce_tester("cuda_tk_col_max", bf16, np.maximum),
        #
        make_0ary_tester("cuda_tk_tile_zero", f32, {(1, 3): (0, 1337)}),
        make_0ary_tester("cuda_tk_tile_one", f32, {(1, 3): (1, 1337)}),
        make_0ary_tester("cuda_tk_tile_pos_infty", f32, {(1, 3): (inf, 1337)}),
        make_0ary_tester("cuda_tk_tile_neg_infty", f32, {(1, 3): (-inf, 1337)}),
        #
        make_0ary_tester("cuda_tk_make_causal_zero", f32, {(20, 40): (0, 2), (40, 20): (2, 2)}),
        make_0ary_tester("cuda_tk_make_causal_one", f32, {(20, 40): (1, 2), (40, 20): (2, 2)}),
        make_0ary_tester("cuda_tk_make_causal_pos_infty", f32, {(20, 40): (inf, 2), (40, 20): (2, 2)}),
        make_0ary_tester("cuda_tk_make_causal_neg_infty", f32, {(20, 40): (-inf, 2), (40, 20): (2, 2)}),
        make_0ary_tester("cuda_tk_make_causal_t_zero", f32, {(40, 30): (0, 2), (10, 30): (2, 2)}),
        make_0ary_tester("cuda_tk_make_causal_t_one", f32, {(40, 30): (1, 2), (10, 30): (2, 2)}),
        make_0ary_tester("cuda_tk_make_causal_t_pos_infty", f32, {(40, 30): (inf, 2), (10, 30): (2, 2)}),
        make_0ary_tester("cuda_tk_make_causal_t_neg_infty", f32, {(40, 30): (-inf, 2), (10, 30): (2, 2)}),
        #
        make_0ary_tester("cuda_tk_tile_one", f16, {(1, 3): (1, 1337)}),
        make_0ary_tester("cuda_tk_tile_neg_infty", bf16, {(1, 3): (-inf, 1337)}),
        make_0ary_tester("cuda_tk_make_causal_pos_infty", f16, {(20, 40): (inf, 2), (40, 20): (2, 2)}),
        make_0ary_tester("cuda_tk_make_causal_t_zero", bf16, {(40, 30): (0, 2), (10, 30): (2, 2)}),
        #
        make_unary_tester("cuda_tk_tile_copy", f32, f32, expected_tuple=(102400.125, 102400.125)),
        make_unary_tester("cuda_tk_tile_copy", bf16, f32, expected_tuple=(102400, 102400.125)),
        make_unary_tester("cuda_tk_tile_copy", f16, f32, expected_tuple=(inf, 102400.125)),
        make_unary_tester("cuda_tk_tile_copy", f32, f16, expected_tuple=(-1280, -1280)),
        make_unary_tester("cuda_tk_tile_copy", f32, bf16, expected_tuple=(-1280, -1280)),
        # TODO test these on Hopper
        # make_unary_tester("cuda_tk_tile_copy", e4m3, f32, only=True, expected_tuple=(15, 15.1)),
        # make_unary_tester("cuda_tk_tile_copy", e5m2, f32, only=True, expected_tuple=(16, 15.1)),
        # make_unary_tester("cuda_tk_tile_copy", f32, e4m3, only=True),
        # make_unary_tester("cuda_tk_tile_copy", f32, e5m2, only=True),
        # make_unary_tester("cuda_tk_tile_copy", e4m3, f16, only=True),
        #
        make_unary_tester("cuda_tk_tile_exp", f16),
        make_unary_tester("cuda_tk_tile_exp2", f16, expected_tuple=(8, 3)),
        make_unary_tester("cuda_tk_tile_log", f16),
        make_unary_tester("cuda_tk_tile_log2", f16, expected_tuple=(3, 8)),
        make_unary_tester("cuda_tk_tile_abs", f16, expected_tuple=(inf, -100000)),
        make_unary_tester("cuda_tk_tile_relu", f16, expected_tuple=(0, -19.125)),
        #
        make_unary_tester("cuda_tk_tile_exp", f32),
        make_unary_tester("cuda_tk_tile_exp2", f32, expected_tuple=(8, 3)),
        make_unary_tester("cuda_tk_tile_log", f32),
        make_unary_tester("cuda_tk_tile_log2", f32, expected_tuple=(3, 8)),
        make_unary_tester("cuda_tk_tile_abs", f32, expected_tuple=(19.125, -19.125)),
        make_unary_tester("cuda_tk_tile_relu", f32, expected_tuple=(19.125, 19.125)),
        #
        make_binary_tester_3("cuda_tk_tile_add", f32, expected_tuple=(9, 3, 6)),
        make_binary_tester_3("cuda_tk_tile_sub", f32, expected_tuple=(-3, 3, 6)),
        make_binary_tester_3("cuda_tk_tile_mul", f32, expected_tuple=(18, 3, 6)),
        make_binary_tester_3("cuda_tk_tile_div", f32, expected_tuple=(0.375, 3, 8)),
        make_binary_tester_3("cuda_tk_tile_max", f32, expected_tuple=(8, 3, 8)),
        make_binary_tester_3("cuda_tk_tile_min", f32, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester_2("cuda_tk_tile_add_reduce", f32, expected_tuple=(9, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_add_lhs", f32, expected_tuple=(9, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_sub_lhs", f32, expected_tuple=(-3, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_mul_lhs", f32, expected_tuple=(18, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_div_lhs", f32, expected_tuple=(0.375, 3, 8)),
        make_binary_tester_2("cuda_tk_tile_max_lhs", f32, expected_tuple=(8, 3, 8)),
        make_binary_tester_2("cuda_tk_tile_min_lhs", f32, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester_2("cuda_tk_tile_add_rhs", f32, expected_tuple=(9, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_sub_rhs", f32, expected_tuple=(-3, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_mul_rhs", f32, expected_tuple=(18, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_div_rhs", f32, expected_tuple=(0.375, 3, 8)),
        make_binary_tester_2("cuda_tk_tile_max_rhs", f32, expected_tuple=(8, 3, 8)),
        make_binary_tester_2("cuda_tk_tile_min_rhs", f32, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester_3("cuda_tk_tile_add", f16, expected_tuple=(9, 3, 6)),
        make_binary_tester_3("cuda_tk_tile_sub", f16, expected_tuple=(-3, 3, 6)),
        make_binary_tester_3("cuda_tk_tile_mul", f16, expected_tuple=(18, 3, 6)),
        make_binary_tester_3("cuda_tk_tile_div", f16, expected_tuple=(0.375, 3, 8)),
        make_binary_tester_3("cuda_tk_tile_max", f16, expected_tuple=(8, 3, 8)),
        make_binary_tester_3("cuda_tk_tile_min", f16, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester_2("cuda_tk_tile_add_lhs", bf16, expected_tuple=(9, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_sub_lhs", bf16, expected_tuple=(-3, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_mul_lhs", bf16, expected_tuple=(18, 3, 6)),
        make_binary_tester_2("cuda_tk_tile_div_lhs", bf16, expected_tuple=(0.375, 3, 8)),
        make_binary_tester_2("cuda_tk_tile_max_lhs", bf16, expected_tuple=(8, 3, 8)),
        make_binary_tester_2("cuda_tk_tile_min_lhs", bf16, expected_tuple=(3, 3, 8)),
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
    missing_instr_names = set(tile_instr_names) - tested_instr_names
    assert not missing_instr_names, "Missing coverage"
