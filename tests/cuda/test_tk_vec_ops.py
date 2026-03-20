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


# assert len(tile_instr_names) >= 56, "Add or remove test coverage"


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
