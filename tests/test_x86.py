from __future__ import annotations

import itertools

import numpy as np
import pytest

from exo import proc
from exo.platforms.x86 import *
from exo.stdlib.scheduling import *

old_split = repeat(divide_loop)
old_unroll = repeat(unroll_loop)


def old_fission_after(proc, stmt_pattern, n_lifts=1):
    def find_stmts(p):
        return [c.after() for c in p.find_all(stmt_pattern)]

    return loop_hack(autofission, find_stmts)(proc, n_lifts)


@pytest.mark.isa("AVX2")
def test_avx2_memcpy(compiler):
    """
    Compute dst = src
    """

    @proc
    def memcpy_avx2(n: size, dst: R[n] @ DRAM, src: R[n] @ DRAM):  # pragma: no cover
        for i in seq(0, (n + 7) / 8):
            if n - 8 * i >= 8:
                tmp: f32[8] @ AVX2
                mm256_loadu_ps(tmp, src[8 * i : 8 * i + 8])
                mm256_storeu_ps(dst[8 * i : 8 * i + 8], tmp)
            else:
                for j in seq(0, n - 8 * i):
                    dst[8 * i + j] = src[8 * i + j]

    # TODO: -march=skylake here is a hack. Such flags should be somehow handled
    #   automatically. Maybe this should be inferred by the use of AVX2, but
    #   "skylake" isn't right anyway. We might need a first-class notion of
    #   a Target, which has certain memories available. Then we can say that
    #   e.g. Skylake-X has AVX2, AVX512, etc.
    fn = compiler.compile(
        memcpy_avx2, skip_on_fail=True, CMAKE_C_FLAGS="-march=skylake"
    )

    for n in (7, 8, 9, 31, 32, 33, 127, 128, 129):
        inp = np.array([float(i) for i in range(n)], dtype=np.float32)
        out = np.array([float(0) for _ in range(n)], dtype=np.float32)
        fn(None, n, out, inp)

        assert np.array_equal(inp, out)


@pytest.mark.isa("AVX2")
def test_avx2_simple_math(compiler):
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_avx2(n: size, x: R[n] @ DRAM, y: R[n] @ DRAM):  # pragma: no cover
        assert n % 8 == 0
        for i in seq(0, n / 8):
            xVec: f32[8] @ AVX2
            yVec: f32[8] @ AVX2
            mm256_loadu_ps(xVec, x[8 * i : 8 * i + 8])
            mm256_loadu_ps(yVec, y[8 * i : 8 * i + 8])
            mm256_mul_ps(xVec, xVec, yVec)
            mm256_mul_ps(xVec, xVec, yVec)
            mm256_storeu_ps(x[8 * i : 8 * i + 8], xVec)

    fn = compiler.compile(
        simple_math_avx2, skip_on_fail=True, CMAKE_C_FLAGS="-march=skylake"
    )

    for n in (8, 16, 24, 32, 64, 128):
        x = np.array([float(i) for i in range(n)], dtype=np.float32)
        y = np.array([float(3 * i) for i in range(n)], dtype=np.float32)
        expected = x * y * y

        fn(None, n, x, y)
        np.testing.assert_almost_equal(x, expected)


@pytest.fixture
def simple_math_avx2_sched():
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_avx2_sched(
        n: size, x: R[n] @ DRAM, y: R[n] @ DRAM
    ):  # pragma: no cover
        for i in seq(0, n):
            x[i] = x[i] * y[i] * y[i]

    def sched_simple_math_avx2_sched(p=simple_math_avx2_sched):
        p = old_split(p, "i", 8, ["io", "ii"], tail="cut_and_guard")

        p = stage_mem(p, "for ii in _:_", "x[8 * io: 8 * io + 8]", "xVec")
        p = set_memory(p, "xVec", AVX2)

        p = replace(p, "for i0 in _:_ #0", mm256_loadu_ps)
        p = replace(p, "for i0 in _:_ #0", mm256_storeu_ps)

        p = bind_expr(p, "y[_]", "yVec", cse=True)
        p = autolift_alloc(p, "yVec: _", keep_dims=True)
        p = set_memory(p, "yVec", AVX2)
        p = old_fission_after(p, "yVec[_] = _")

        p = replace_all(p, mm256_loadu_ps)

        p = bind_expr(p, "xVec[_] * yVec[_]", "xy")
        p = autolift_alloc(p, "xy: _", keep_dims=True)
        p = set_memory(p, "xy", AVX2)
        p = old_fission_after(p, "xy[_] = _")

        p = replace_all(p, mm256_mul_ps)
        p = simplify(p)
        return p

    simple_math_avx2_sched = sched_simple_math_avx2_sched()

    return simple_math_avx2_sched


def test_gen_avx2_simple_math_scheduling(golden, simple_math_avx2_sched):
    assert str(simple_math_avx2_sched) == golden


@pytest.mark.isa("AVX2")
def test_exec_avx2_simple_math_scheduling(compiler, simple_math_avx2_sched):
    fn = compiler.compile(
        simple_math_avx2_sched, skip_on_fail=True, CMAKE_C_FLAGS="-march=skylake"
    )

    for n in (8, 16, 24, 32, 64, 128):
        x = np.array([float(i) for i in range(n)], dtype=np.float32)
        y = np.array([float(3 * i) for i in range(n)], dtype=np.float32)
        expected = x * y * y

        fn(None, n, x, y)
        np.testing.assert_almost_equal(x, expected)


def _sgemm_test_cases(fn, M, N, K):
    for m, n, k in itertools.product(M, N, K):
        A = np.random.rand(m, k).astype(np.float32)
        B = np.random.rand(k, n).astype(np.float32)
        C = A @ B

        C_out = np.zeros_like(C)

        fn(None, m, n, k, C_out, A, B)
        np.testing.assert_almost_equal(C, C_out, decimal=3)


@pytest.fixture
def sgemm_6x16():
    @proc
    def sgemm_6x16(
        K: size,
        C: [f32][6, 16] @ DRAM,
        A: [f32][6, K] @ DRAM,
        B: [f32][K, 16] @ DRAM,
    ):
        for i in seq(0, 6):
            for j in seq(0, 16):
                for k in seq(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    return sgemm_6x16


@pytest.fixture
def avx2_sgemm_6x16(sgemm_6x16):
    avx = rename(sgemm_6x16, "rank_k_reduce_6x16_scheduled")
    print(avx)
    avx = stage_mem(avx, "C[_] += _", "C[i, j]", "C_reg")
    avx = set_memory(avx, "C_reg", AVX2)
    avx = old_split(avx, "j", 8, ["jo", "ji"], perfect=True)
    avx = reorder_loops(avx, "ji k")
    avx = reorder_loops(avx, "jo k")
    avx = reorder_loops(avx, "i k")
    avx = autolift_alloc(avx, "C_reg:_", n_lifts=3, keep_dims=True)
    avx = old_fission_after(avx, "C_reg = _ #0", n_lifts=3)
    avx = old_fission_after(avx, "C_reg[_] += _ #0", n_lifts=3)
    avx = autolift_alloc(avx, "C_reg:_", n_lifts=1)
    avx = old_fission_after(avx, "for i in _:_#0", n_lifts=1)
    avx = old_fission_after(avx, "for i in _:_#1", n_lifts=1)
    avx = simplify(avx)

    avx2_sgemm_6x16 = avx

    def sched_avx2_sgemm_6x16(p=avx2_sgemm_6x16):
        p = bind_expr(p, "A[i, k]", "a_vec")
        p = set_memory(p, "a_vec", AVX2)
        p = expand_dim(p, "a_vec:_", "8", "ji")
        p = autolift_alloc(p, "a_vec:_")
        p = old_fission_after(p, "a_vec[_] = _")
        #
        p = bind_expr(p, "B[k, _]", "b_vec")
        p = set_memory(p, "b_vec", AVX2)
        p = expand_dim(p, "b_vec:_", "8", "ji")
        p = autolift_alloc(p, "b_vec:_")
        p = old_fission_after(p, "b_vec[_] = _")
        #
        p = replace_all(p, avx2_set0_ps)
        p = replace_all(p, mm256_broadcast_ss)
        p = replace_all(p, mm256_fmadd_ps)
        p = replace_all(p, avx2_fmadd_memu_ps)
        p = replace(p, "for ji in _:_ #0", mm256_loadu_ps)
        p = replace(p, "for ji in _:_ #0", mm256_loadu_ps)
        p = replace(p, "for ji in _:_ #0", mm256_storeu_ps)
        #
        p = old_unroll(p, "jo")
        p = old_unroll(p, "i")
        #
        p = simplify(p)
        return p

    avx2_sgemm_6x16 = sched_avx2_sgemm_6x16()

    return avx2_sgemm_6x16


# @pytest.mark.skip(reason='apparently unifying the broadcast is '
#                         'non-deterministic')
# Until this non-determinism can be removed, at least
# try to run the code here...
def test_gen_avx2_sgemm_kernel(avx2_sgemm_6x16):
    pass
    # assert str(avx2_sgemm_6x16) == golden


@pytest.fixture
def sgemm_full():
    @proc
    def sgemm_full(
        N: size,
        M: size,
        K: size,
        C: f32[N, M] @ DRAM,
        A: f32[N, K] @ DRAM,
        B: f32[K, M] @ DRAM,
    ):
        assert K > 0

        for i in seq(0, N):
            for j in seq(0, M):
                for k in seq(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    return sgemm_full


@pytest.fixture
def avx2_sgemm_full(sgemm_full, sgemm_6x16, avx2_sgemm_6x16):
    cache_i = 16
    cache_j = 4
    cache_k = 2

    def sched_avx_sgemm_full(p=sgemm_full):
        print(p)
        p = rename(p, "avx_sgemm_full")
        # initial i,j tiling
        p = old_split(p, "i", 6, ["io", "ii"], tail="cut")
        p = reorder_loops(p, "ii j #0")
        p = divide_loop(p, "j #0", 16, ["jo", "ji"], tail="cut")
        p = reorder_loops(p, "ji ii")
        # breaking off the main loop
        p = old_fission_after(p, "for jo in _: _")
        # introduce k-tiling for later
        p = divide_loop(p, "k #0", cache_k * 16, ["ko", "ki"], tail="cut")
        p = old_fission_after(p, "for ko in _: _", n_lifts=2)
        p = reorder_loops(p, "ji ko")
        p = reorder_loops(p, "ii ko")
        p = replace_all(p, sgemm_6x16)
        # insert uses of micro-kernel now
        p = call_eqv(p, "sgemm_6x16(_, _, _, _)", avx2_sgemm_6x16)
        p = call_eqv(p, "sgemm_6x16(_, _, _, _)", avx2_sgemm_6x16)
        # do outer tiling for cache-locality
        p = divide_loop(p, "io #0", cache_i, ["io", "im"], tail="cut")
        p = reorder_loops(p, "im jo")
        p = divide_loop(p, "jo #0", cache_j, ["jo", "jm"], tail="cut")
        p = divide_loop(p, "jo #1", cache_j, ["jo", "jm"], tail="cut")
        p = reorder_loops(p, "jm im")
        # move the ko loop up and out
        p = old_fission_after(p, "for ko in _: _", n_lifts=2)
        p = reorder_loops(p, "jm ko #0")
        p = reorder_loops(p, "im ko #0")
        return p

    avx_sgemm_full = sched_avx_sgemm_full()

    return avx_sgemm_full


# just make sure the scheduling works
def test_gen_avx2_sgemm_full(avx2_sgemm_full):
    pass


@pytest.mark.isa("AVX2")
def test_avx2_sgemm_full(compiler, avx2_sgemm_full):
    fn = compiler.compile(
        avx2_sgemm_full, skip_on_fail=True, CMAKE_C_FLAGS="-march=skylake"
    )

    _sgemm_test_cases(
        fn, M=range(10, 600, 200), N=range(20, 400, 120), K=range(1, 512, 160)
    )


@pytest.mark.isa("AVX2")
def test_avx2_sgemm_6x16(compiler, avx2_sgemm_6x16):
    @proc
    def avx2_sgemm_6x16_wrapper(
        M: size,
        N: size,
        K: size,
        C: f32[6, 16] @ DRAM,
        A: f32[6, K] @ DRAM,
        B: f32[K, 16] @ DRAM,
    ):
        avx2_sgemm_6x16(K, C, A, B)

    fn = compiler.compile(
        avx2_sgemm_6x16_wrapper, skip_on_fail=True, CMAKE_C_FLAGS="-march=skylake"
    )

    _sgemm_test_cases(fn, M=[6], N=[16], K=range(1, 512))


@pytest.fixture
def spec_kernel(sgemm_full, sgemm_6x16, avx2_sgemm_6x16):
    @proc
    def sgemm_micro_kernel_staged(
        M: size,
        N: size,
        K: size,
        A: f32[M, K],
        B: f32[K, 16 * ((N + 15) / 16)],
        C: [f32][M, N],
    ):
        assert M >= 1
        assert N >= 1
        assert K >= 1
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1

        C_reg: f32[M, ((N + 15) / 16), 16] @ AVX512
        for i in seq(0, M):
            for j in seq(0, N / 16):
                mm512_loadu_ps(C_reg[i, j, :], C[i, 16 * j : 16 * j + 16])
            if N % 16 > 0:
                mm512_maskz_loadu_ps(
                    N % 16,
                    C_reg[i, N / 16, :],
                    C[i, 16 * (N / 16) : 16 * (N / 16) + N % 16],
                )

        for k in seq(0, K):
            for i in seq(0, M):
                a_vec: f32[16] @ AVX512
                mm512_set1_ps(a_vec, A[i, k : k + 1])
                for j in seq(0, ((N + 15) / 16)):
                    b_vec: f32[16] @ AVX512
                    mm512_loadu_ps(b_vec, B[k, j * 16 : j * 16 + 16])
                    mm512_fmadd_ps(a_vec, b_vec, C_reg[i, j, :])

        for i in seq(0, M):
            for j in seq(0, N / 16):
                mm512_storeu_ps(C[i, 16 * j : 16 * j + 16], C_reg[i, j, :])
            if N % 16 > 0:
                mm512_mask_storeu_ps(
                    N % 16,
                    C[i, 16 * (N / 16) : 16 * (N / 16) + N % 16],
                    C_reg[i, N / 16, :],
                )

    def sched_spec_kernel(p=sgemm_micro_kernel_staged):
        p = p.partial_eval(6, 64)
        p = simplify(p)
        p = old_unroll(p, "j")
        p = old_unroll(p, "i")
        p = simplify(p)
        return p

    spec_kernel = sched_spec_kernel()

    return spec_kernel


# just make sure the scheduling works
def test_gen_avx512_sgemm_full(spec_kernel):
    pass


@pytest.mark.isa("AVX512f")
def test_avx512_sgemm_full(compiler, spec_kernel):

    spec_kernel.c_code_str()

    @proc
    def sgemm_full(
        N: size,
        M: size,
        K: size,
        C: f32[N, M] @ DRAM,
        A: f32[N, K] @ DRAM,
        B: f32[K, M] @ DRAM,
    ):
        for i in seq(0, N):
            for j in seq(0, M):
                for k in seq(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    fn = compiler.compile(
        sgemm_full, skip_on_fail=True, CMAKE_C_FLAGS="-march=skylake-avx512"
    )

    _sgemm_test_cases(
        fn, M=range(10, 600, 200), N=range(20, 400, 120), K=range(1, 512, 160)
    )
