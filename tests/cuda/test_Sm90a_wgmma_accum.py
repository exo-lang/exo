"""Accumulate-only (non-_zi) wgmma: load data into D, then D += A @ B

This is a strawman use case (TMA reduce is the sensible way to add to
existing data), but it tests the behavior() and codegen() of the
accumulate-only Sm90_tk_mma_{a}_{b} instrs, for all A/B modes:
scale-d must be 1 for every native k-step, so the data loaded into D
is kept and accumulated into.

The loads/stores of registers and the wgmma are substituted with
`replace` from plain loop nests, so instr unification checks that
each instr's behavior matches the plain loop nest.

Inputs are small integers so all results are exact (in bf16/f16/tf32
inputs and f32/f16 accumulators), allowing exact comparison to numpy.
"""

from __future__ import annotations

import numpy as np
import random

from exo import *
from exo.scalars import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.cuda_tk import *
from exo.platforms.Sm90 import *


# Number of wgmma calls; each call has K = KS = 128 / sizeof(A/B type).
KT = 2


def mkproc_wgmma_accum(a_mode, b_mode, ab_t, d_t):
    """C_out = C_in + A @ B^T; A: [64, K], B: [N, K], C: [64, N], all f32 in GMEM.

    D (d_t) is loaded from C_in; A and B are converted to ab_t
    and staged in SMEM (or A in registers) in the layout required
    by the A/B mode; then KT accumulate-only wgmma calls.
    """
    assert a_mode in ("row", "col", "rmem")
    assert b_mode in ("row", "col")
    a_row = a_mode == "row"
    a_col = a_mode == "col"
    a_rmem = a_mode == "rmem"
    b_row = b_mode == "row"
    b_col = b_mode == "col"

    KS = 128 * 8 // ab_t.bits  # One swizzle (128 byte) row of K
    K = KT * KS
    # MN-major B requires N divisible by 64; otherwise test an N that isn't.
    N = 128 if b_row else 96
    N64 = N // 64

    # All staging arrays are 4D so one proc covers all modes;
    # after simplify, only the code for the chosen mode remains.
    if a_row:
        A_dims = (KT, 1, 64, KS)  # [kt, 1, m, k]
    elif a_col:
        A_dims = (KT, 1, KS, 64)  # [kt, 1, k, m]
    else:
        A_dims = (4, KT, 16, KS)  # [m_warp, kt, mi, k]
    A_mem = Sm90_TkRmemTileA(KS) if a_rmem else Sm90_SmemSwizzled(128)
    a0, a1, a2, a3 = A_dims
    if b_row:
        B_dims = (KT, N64, KS, 64)  # [kt, n // 64, k, n % 64]
    else:
        B_dims = (KT, 1, N, KS)  # [kt, 1, n, k]
    b0, b1, b2, b3 = B_dims
    instr = {
        "row_col": Sm90_tk_mma_row_col,
        "rmem_col": Sm90_tk_mma_rmem_col,
        "col_col": Sm90_tk_mma_col_col,
        "row_row": Sm90_tk_mma_row_row,
        "rmem_row": Sm90_tk_mma_rmem_row,
        "col_row": Sm90_tk_mma_col_row,
    }[f"{a_mode}_{b_mode}"]

    # fmt: off
    @proc
    def wgmma_accum(
        h_A: f32[64, K] @ DRAM,
        h_B: f32[N, K] @ DRAM,
        h_C: f32[64, N] @ DRAM,  # C_in on input, C_out on output
    ):
        A: f32[64, K] @ CudaGmemLinear
        B: f32[N, K] @ CudaGmemLinear
        C: f32[64, N] @ CudaGmemLinear
        cudaMemcpyAsync_htod_2d(64, K, A[:, :], h_A[:, :], dst=f32, src=f32)
        cudaMemcpyAsync_htod_2d(N, K, B[:, :], h_B[:, :], dst=f32, src=f32)
        cudaMemcpyAsync_htod_2d(64, N, C[:, :], h_C[:, :], dst=f32, src=f32)

        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                A_s: ab_t[a0, a1, a2, a3] @ A_mem
                B_s: ab_t[b0, b1, b2, b3] @ Sm90_SmemSwizzled(128)
                D: d_t[4, 16, N] @ Sm90_TkRmemTileD(N)
                cg: barrier @ Sm90_WgmmaCommitGroup

                # Stage A, B in SMEM (converting from f32) with plain CUDA code.
                for kt in seq(0, KT):
                    if a_row:
                        for m in seq(0, 64):
                            for k in cuda_threads(0, KS):
                                A_s[kt, 0, m, k] = A[m, kt * KS + k]
                    if a_col:
                        for k in seq(0, KS):
                            for m in cuda_threads(0, 64):
                                A_s[kt, 0, k, m] = A[m, kt * KS + k]
                    if b_col:
                        for n in seq(0, N):
                            for k in cuda_threads(0, KS):
                                B_s[kt, 0, n, k] = B[n, kt * KS + k]
                    if b_row:
                        for no in seq(0, N64):
                            for k in seq(0, KS):
                                for ni in cuda_threads(0, 64):
                                    B_s[kt, no, k, ni] = B[no * 64 + ni, kt * KS + k]

                # Load D (and A, if in registers).
                for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
                    for w in cuda_threads(0, 4, unit=cuda_warp):
                        for d_r in seq(0, 16):
                            for d_c in seq(0, N):
                                D[w, d_r, d_c] = C[16 * w + d_r, d_c]
                        if a_rmem:
                            for kt in seq(0, KT):
                                for a_r in seq(0, 16):
                                    for a_c in seq(0, KS):
                                        A_s[w, kt, a_r, a_c] = A[16 * w + a_r, kt * KS + a_c]

                Fence(cuda_in_order, cuda_generic_and_async_proxy)

                for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
                    Fence(wgmma_fence_1, wgmma_fence_2)
                    for kt in seq(0, KT):
                        # D += A @ B^T; to be replaced with accumulate-only wgmma.
                        if b_col:
                            for mw in seq(0, 4):
                                for mi in seq(0, 16):
                                    for n in seq(0, N):
                                        for k in seq(0, KS):
                                            if a_row:
                                                D[mw, mi, n] += A_s[kt, 0, mw * 16 + mi, k] * B_s[kt, 0, n, k]
                                            if a_col:
                                                D[mw, mi, n] += A_s[kt, 0, k, mw * 16 + mi] * B_s[kt, 0, n, k]
                                            if a_rmem:
                                                D[mw, mi, n] += A_s[mw, kt, mi, k] * B_s[kt, 0, n, k]
                        if b_row:
                            for mw in seq(0, 4):
                                for mi in seq(0, 16):
                                    for no in seq(0, N64):
                                        for ni in seq(0, 64):
                                            for k in seq(0, KS):
                                                if a_row:
                                                    D[mw, mi, no * 64 + ni] += A_s[kt, 0, mw * 16 + mi, k] * B_s[kt, no, k, ni]
                                                if a_col:
                                                    D[mw, mi, no * 64 + ni] += A_s[kt, 0, k, mw * 16 + mi] * B_s[kt, no, k, ni]
                                                if a_rmem:
                                                    D[mw, mi, no * 64 + ni] += A_s[mw, kt, mi, k] * B_s[kt, no, k, ni]
                    Arrive(wgmma_async) >> cg
                    Await(cg, cuda_in_order, 0)
                    for w in cuda_threads(0, 4, unit=cuda_warp):
                        for st_r in seq(0, 16):
                            for st_c in seq(0, N):
                                C[16 * w + st_r, st_c] = D[w, st_r, st_c]

        cudaMemcpyAsync_dtoh_2d(64, N, h_C[:, :], C[:, :], dst=f32, src=f32)
    # fmt: on

    p = simplify(wgmma_accum)  # Removes the if statements for other modes
    p = replace(p, p.find_loop("d_r"), cuda_tk_load_rg)
    if a_rmem:
        p = replace(p, p.find_loop("a_r"), cuda_tk_load_rg)
    p = replace(p, p.find_loop("mw"), instr)
    p = replace(p, p.find_loop("st_r"), cuda_tk_store_rg)
    p = simplify(p)
    p = rename(p, f"wgmma_accum_{a_mode}_{b_mode}_{ab_t}_{d_t}")
    return p


def mktest_golden(a_mode, b_mode, ab_t, d_t):
    def test(compiler, golden):
        p = mkproc_wgmma_accum(a_mode, b_mode, ab_t, d_t)
        p.sync_check()
        compiler.cuda_cpu_test(lambda: p, golden, sm="90a")

    return test


def mktest_run(a_mode, b_mode, ab_t, d_t):
    def test(compiler_Sm90a):
        p = mkproc_wgmma_accum(a_mode, b_mode, ab_t, d_t)
        K = KT * (128 * 8 // ab_t.bits)
        N = 128 if b_mode == "row" else 96

        # Small integers, so everything is exact.
        # f16 accumulator: keep all partial sums within +/-2048.
        lim = 2 if d_t == f16 else 8
        rand = random.Random(f"{a_mode}_{b_mode}_{ab_t}_{d_t}")
        A = np.array(
            [[rand.randint(-lim, lim) for k in range(K)] for m in range(64)],
            dtype=np.float32,
        )
        B = np.array(
            [[rand.randint(-lim, lim) for k in range(K)] for n in range(N)],
            dtype=np.float32,
        )
        C_in = np.array(
            [[rand.randint(-100, 100) for n in range(N)] for m in range(64)],
            dtype=np.float32,
        )
        C_expected = C_in + A @ B.T
        assert np.abs(C_expected).max() < 2048 or d_t != f16

        C = C_in.copy()
        cu = compiler_Sm90a.cuda_test_context(p)
        cu(None, A, B, C)
        assert np.array_equal(C, C_expected)

    return test


# (a_mode, b_mode, A/B type, D type)
# tf32 (f32 A/B) only supports K-major SMEM (row_col) and not registers.
# f16 D only supports f16 A/B.
test_golden_wgmma_accum_row_col_f32_f32 = mktest_golden("row", "col", f32, f32)
test_run_wgmma_accum_row_col_f32_f32 = mktest_run("row", "col", f32, f32)
test_golden_wgmma_accum_row_col_bf16_f32 = mktest_golden("row", "col", bf16, f32)
test_run_wgmma_accum_row_col_bf16_f32 = mktest_run("row", "col", bf16, f32)
test_golden_wgmma_accum_row_col_f16_f16 = mktest_golden("row", "col", f16, f16)
test_run_wgmma_accum_row_col_f16_f16 = mktest_run("row", "col", f16, f16)
test_golden_wgmma_accum_rmem_col_bf16_f32 = mktest_golden("rmem", "col", bf16, f32)
test_run_wgmma_accum_rmem_col_bf16_f32 = mktest_run("rmem", "col", bf16, f32)
test_golden_wgmma_accum_col_col_bf16_f32 = mktest_golden("col", "col", bf16, f32)
test_run_wgmma_accum_col_col_bf16_f32 = mktest_run("col", "col", bf16, f32)
test_golden_wgmma_accum_row_row_bf16_f32 = mktest_golden("row", "row", bf16, f32)
test_run_wgmma_accum_row_row_bf16_f32 = mktest_run("row", "row", bf16, f32)
test_golden_wgmma_accum_rmem_row_f16_f32 = mktest_golden("rmem", "row", f16, f32)
test_run_wgmma_accum_rmem_row_f16_f32 = mktest_run("rmem", "row", f16, f32)
test_golden_wgmma_accum_col_row_f16_f16 = mktest_golden("col", "row", f16, f16)
test_run_wgmma_accum_col_row_f16_f16 = mktest_run("col", "row", f16, f16)
