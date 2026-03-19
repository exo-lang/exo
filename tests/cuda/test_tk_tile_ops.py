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


tile_instrs = []


for name in sorted(dir(ops_module)):
    obj = getattr(ops_module, name)
    if isinstance(obj, InstrTemplate):
        assert name.startswith("cuda_tk_"), name
        assert hasattr(exo.platforms.cuda_tk, name), name
        tile_instrs.append(obj)


assert len(tile_instrs) == 62, "Add or remove test coverage"


@dataclass(slots=True)
class TileTester:
    p: proc
    run: Callable[None, [CudaTestContext]]


def make_binary_tester(instr_name, tile_base_type, expected_tuple=None):
    tile_instr = getattr(ops_module, instr_name)
    T = tile_base_type

    rows = 48
    cols = 80

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
        tile_instr(h_cpu_out[:, :], h_lhs[:, :], h_rhs[:, :], rows=rows, cols=cols, dst=f32, lhs=f32, rhs=f32)

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
    rhs_end = 129 if T.bits >= 32 else 5

    def run(cu: CudaTestContext):
        rng = Random(20010106)

        h_cpu_out = np.zeros((rows, cols), dtype=np.float32)
        h_cuda_out = np.zeros((rows, cols), dtype=np.float32)
        h_lhs = np.zeros((rows, cols), dtype=np.float32)
        h_rhs = np.zeros((rows, cols), dtype=np.float32)
        for r in range(0, rows):
            for c in range(0, cols):
                h_lhs[r, c] = rng.randrange(-lhs_magn, lhs_magn)
                h_rhs[r, c] = rng.randrange(1, rhs_end)

        if expected_tuple:
            assert len(expected_tuple) == 3
            h_lhs[0, 0] = expected_tuple[1]
            h_rhs[0, 0] = expected_tuple[2]

        cu.run(p.name(), None, h_cpu_out, h_cuda_out, h_lhs, h_rhs)

        if expected_tuple:
            assert h_cuda_out[0, 0] == expected_tuple[0], p.name()
        for r in range(0, rows):
            for c in range(0, cols):
                assert math.fabs(h_cpu_out[r, c] - h_cuda_out[r, c]) <= 0.0625, p.name()

    return TileTester(p, run)


def test_tk_tile_ops(compiler_Sm80):
    testers = [
        make_binary_tester("cuda_tk_tile_add", f32, expected_tuple=(9, 3, 6)),
        make_binary_tester("cuda_tk_tile_sub", f32, expected_tuple=(-3, 3, 6)),
        make_binary_tester("cuda_tk_tile_mul", f32, expected_tuple=(18, 3, 6)),
        make_binary_tester("cuda_tk_tile_div", f32, expected_tuple=(0.375, 3, 8)),
        make_binary_tester("cuda_tk_tile_max", f32, expected_tuple=(8, 3, 8)),
        make_binary_tester("cuda_tk_tile_min", f32, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester("cuda_tk_tile_add", f16, expected_tuple=(9, 3, 6)),
        make_binary_tester("cuda_tk_tile_sub", f16, expected_tuple=(-3, 3, 6)),
        make_binary_tester("cuda_tk_tile_mul", f16, expected_tuple=(18, 3, 6)),
        make_binary_tester("cuda_tk_tile_div", f16, expected_tuple=(0.375, 3, 8)),
        make_binary_tester("cuda_tk_tile_max", f16, expected_tuple=(8, 3, 8)),
        make_binary_tester("cuda_tk_tile_min", f16, expected_tuple=(3, 3, 8)),
        #
        make_binary_tester("cuda_tk_tile_add", bf16, expected_tuple=(9, 3, 6)),
        make_binary_tester("cuda_tk_tile_sub", bf16, expected_tuple=(-3, 3, 6)),
        make_binary_tester("cuda_tk_tile_mul", bf16, expected_tuple=(18, 3, 6)),
        make_binary_tester("cuda_tk_tile_div", bf16, expected_tuple=(0.375, 3, 8)),
        make_binary_tester("cuda_tk_tile_max", bf16, expected_tuple=(8, 3, 8)),
        make_binary_tester("cuda_tk_tile_min", bf16, expected_tuple=(3, 3, 8)),
    ]

    procs = [tester.p for tester in testers]

    cu = compiler_Sm80.cuda_test_context(procs)

    for tester in testers:
        tester.run(cu)
