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
from exo.scalars import inf, f16, bf16, f32
from exo.stdlib.scheduling import *

# Not intended as a public module, but we do this to autogenerate
# tests for all the instructions.
import exo.platforms.kittens_impl.tk_tile_ops as ops_module


tile_instr_names = []


for name in sorted(dir(ops_module)):
    obj = getattr(ops_module, name)
    if isinstance(obj, InstrTemplate):
        assert name.startswith("cuda_tk_"), name
        assert hasattr(exo.platforms.cuda_tk, name), name
        tile_instr_names.append(name)


assert len(tile_instr_names) >= 62, "Add or remove test coverage"


@dataclass(slots=True)
class TileTester:
    p: proc
    run: Callable[None, [CudaTestContext]]
    instr_name: str
    only: bool
    T: ScalarInfo


def make_binary_run(name, rows, cols, lhs_magn, rhs_magn, expected_tuple):
    def run(cu: CudaTestContext):
        rng = Random(20010106)

        h_cpu_out = np.zeros((rows, cols), dtype=np.float32)
        h_cuda_out = np.zeros((rows, cols), dtype=np.float32)
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

        cu.run(name, None, h_cpu_out, h_cuda_out, h_lhs, h_rhs)

        if expected_tuple:
            assert h_cuda_out[42, 49] == expected_tuple[0], name
        for r in range(0, rows):
            for c in range(0, cols):
                assert math.fabs(h_cpu_out[r, c] - h_cuda_out[r, c]) <= 0.0625, name

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
        h_cpu_out: f32[rows, cols],
        h_cuda_out: f32[rows, cols],
        h_lhs: f32[rows, cols],
        h_rhs: f32[rows, cols],
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test functions by comparing the CPU output generated here to CUDA.
        tile_instr_3operand(
            h_cpu_out[:, :], h_lhs[:, :], h_rhs[:, :],
            rows=rows, cols=cols, dst=f32, lhs=f32, rhs=f32,
        )

        d_lhs: f32[rows, cols] @ CudaGmemLinear
        d_rhs: f32[rows, cols] @ CudaGmemLinear
        d_out: f32[rows, cols] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2f32(rows, cols, d_lhs[:, :], h_lhs[:, :])
        cudaMemcpyAsync_htod_2f32(rows, cols, d_rhs[:, :], h_rhs[:, :])
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                r_lhs: T[rows, cols] @ CudaTkWarpTile(rows, cols)
                r_rhs: T[rows, cols] @ CudaTkWarpTile(rows, cols)
                r_out: T[rows, cols] @ CudaTkWarpTile(rows, cols)
                cuda_tk_load_rg(r_lhs[:, :], d_lhs[:, :], size0=rows, size1=cols, dst=T, src=f32)
                cuda_tk_load_rg(r_rhs[:, :], d_rhs[:, :], size0=rows, size1=cols, dst=T, src=f32)
                tile_instr(r_out[:, :], r_lhs[:, :], r_rhs[:, :], rows=rows, cols=cols, dst=T, lhs=T, rhs=T)
                cuda_tk_store_rg(d_out[:, :], r_out[:, :], size0=rows, size1=cols, dst=f32, src=T)
        cudaMemcpyAsync_dtoh_2f32(rows, cols, h_cuda_out[:, :], d_out[:, :])

    proc_name = f"tester_{T}_" + instr_name

    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, proc_name)

    lhs_magn = 128 if T.bits >= 32 else 8
    rhs_magn = 129 if T.bits >= 32 else 5

    run = make_binary_run(p.name(), rows, cols, lhs_magn, rhs_magn, expected_tuple)
    return TileTester(p, run, instr_name, only, T)


def make_binary_tester_2(
    instr_name, tile_base_type, only=False, *, expected_tuple=None
):
    # Tester for 2 operand tile instructions `lhs = lhs op rhs` or `rhs = lhs op rhs`
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
        h_cpu_out: f32[rows, cols],
        h_cuda_out: f32[rows, cols],
        h_lhs: f32[rows, cols],
        h_rhs: f32[rows, cols],
    ):
        # fmt: off
        # This will be inlined, unpacking the CUDA instr's behavior as CPU code.
        # The test functions by comparing the CPU output generated here to CUDA.
        tile_instr_3operand(
            h_cpu_out[:, :], h_lhs[:, :], h_rhs[:, :],
            rows=rows, cols=cols, dst=f32, lhs=f32, rhs=f32,
        )

        d_lhs: f32[rows, cols] @ CudaGmemLinear
        d_rhs: f32[rows, cols] @ CudaGmemLinear
        d_out: f32[rows, cols] @ CudaGmemLinear
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
                    cuda_tk_store_rg(d_out[:, :], r_rhs[:, :], size0=rows, size1=cols, dst=f32, src=T)
                else:
                    cuda_tk_store_rg(d_out[:, :], r_lhs[:, :], size0=rows, size1=cols, dst=f32, src=T)
        cudaMemcpyAsync_dtoh_2f32(rows, cols, h_cuda_out[:, :], d_out[:, :])

    proc_name = f"tester_{T}_" + instr_name

    p = inline(p, p.body()[0])
    p = simplify(p)
    p = rename(p, proc_name)

    lhs_magn = 128 if T.bits >= 32 else 8
    rhs_magn = 129 if T.bits >= 32 else 5

    run = make_binary_run(p.name(), rows, cols, lhs_magn, rhs_magn, expected_tuple)
    return TileTester(p, run, instr_name, only, T)


def test_tk_tile_ops(compiler_Sm80):
    # Note, because ThunderKittens takes so long to compile, we amortize
    # the time by compiling all the tests together.
    # Pass only=True to one of the testers to select just that sub-test to run.
    # fmt: off
    testers = [
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

    # fmt: true
    # Implements only=True from comment.
    have_only = any(tester.only for tester in testers)
    if have_only:
        testers = filter(testers, lambda tester: tester.only)
    else:
        pass  # TODO assert test count

    procs = [tester.p for tester in testers]

    cu = compiler_Sm80.cuda_test_context(procs)

    for tester in testers:
        tester.run(cu)

    assert not have_only, "Test passed. Please remove only=True before pushing"
