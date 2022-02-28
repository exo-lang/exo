from __future__ import annotations

import itertools

import numpy as np
import pytest

from exo import proc
from exo.platforms.x86 import *


@pytest.mark.isa('AVX2')
def test_avx2_memcpy(compiler):
    """
    Compute dst = src
    """

    @proc
    def memcpy_avx2(n: size, dst: R[n] @ DRAM,
                    src: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, (n + 7) / 8):
            if n - 8 * i >= 8:
                tmp: f32[8] @ AVX2
                mm256_loadu_ps(tmp, src[8 * i:8 * i + 8])
                mm256_storeu_ps(dst[8 * i:8 * i + 8], tmp)
            else:
                for j in par(0, n - 8 * i):
                    dst[8 * i + j] = src[8 * i + j]

    # TODO: -march=skylake here is a hack. Such flags should be somehow handled
    #   automatically. Maybe this should be inferred by the use of AVX2, but
    #   "skylake" isn't right anyway. We might need a first-class notion of
    #   a Target, which has certain memories available. Then we can say that
    #   e.g. Skylake-X has AVX2, AVX512, etc.
    fn = compiler.compile(memcpy_avx2,
                          skip_on_fail=True,
                          CMAKE_C_FLAGS="-march=skylake")

    for n in (7, 8, 9, 31, 32, 33, 127, 128, 129):
        inp = np.array([float(i) for i in range(n)], dtype=np.float32)
        out = np.array([float(0) for _ in range(n)], dtype=np.float32)
        fn(None, n, out, inp)

        assert np.array_equal(inp, out)


@pytest.mark.isa('AVX2')
def test_avx2_simple_math(compiler):
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_avx2(n: size, x: R[n] @ DRAM,
                         y: R[n] @ DRAM):  # pragma: no cover
        assert n % 8 == 0
        for i in par(0, n / 8):
            xVec: f32[8] @ AVX2
            yVec: f32[8] @ AVX2
            mm256_loadu_ps(xVec, x[8 * i:8 * i + 8])
            mm256_loadu_ps(yVec, y[8 * i:8 * i + 8])
            mm256_mul_ps(xVec, xVec, yVec)
            mm256_mul_ps(xVec, xVec, yVec)
            mm256_storeu_ps(x[8 * i:8 * i + 8], xVec)

    fn = compiler.compile(simple_math_avx2,
                          skip_on_fail=True,
                          CMAKE_C_FLAGS="-march=skylake")

    for n in (8, 16, 24, 32, 64, 128):
        x = np.array([float(i) for i in range(n)], dtype=np.float32)
        y = np.array([float(3 * i) for i in range(n)], dtype=np.float32)
        expected = x * y * y

        fn(None, n, x, y)
        np.testing.assert_almost_equal(x, expected)


@pytest.mark.isa('AVX2')
def test_avx2_simple_math_scheduling(compiler):
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_avx2_sched(n: size, x: R[n] @ DRAM,
                               y: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, n):
            x[i] = x[i] * y[i] * y[i]

    simple_math_avx2_sched = (
        simple_math_avx2_sched
            .split('i', 8, ['io', 'ii'], tail='cut_and_guard')
            .stage_assn('xyy', 'x[_] = _ #0')
            .lift_alloc('xyy: _')
            .set_memory('xyy', AVX2)
            .fission_after('xyy[_] = _')

            .replace_all(mm256_storeu_ps)

            .bind_expr('xVec', 'x[_]')
            .lift_alloc('xVec: _')
            .set_memory('xVec', AVX2)
            .fission_after('xVec[_] = _')

            .bind_expr('yVec', 'y[_]', cse=True)
            .lift_alloc('yVec: _')
            .set_memory('yVec', AVX2)
            .fission_after('yVec[_] = _')

            .replace_all(mm256_loadu_ps)

            .bind_expr('xy', 'xVec[_] * yVec[_]')
            .lift_alloc('xy: _')
            .set_memory('xy', AVX2)
            .fission_after('xy[_] = _')

            .replace_all(mm256_mul_ps)
    )

    fn = compiler.compile(simple_math_avx2_sched,
                          skip_on_fail=True,
                          CMAKE_C_FLAGS="-march=skylake")

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


def gen_rank_k_reduce_6x16():
    @proc
    def rank_k_reduce_6x16(
            K: size,
            C: [f32][6, 16] @ DRAM,
            A: [f32][6, K] @ DRAM,
            B: [f32][K, 16] @ DRAM,
    ):
        for i in par(0, 6):
            for j in par(0, 16):
                for k in par(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    avx = rank_k_reduce_6x16.rename("rank_k_reduce_6x16_scheduled")
    avx = avx.stage_assn('C_reg', 'C[_] += _')
    avx = avx.set_memory('C_reg', AVX2)
    avx = avx.split('j', 8, ['jo', 'ji'], perfect=True)
    avx = avx.reorder('ji', 'k')
    avx = avx.reorder('jo', 'k')
    avx = avx.reorder('i', 'k')
    avx = avx.lift_alloc('C_reg:_', n_lifts=3)
    avx = avx.fission_after('C_reg = _ #0', n_lifts=3)
    avx = avx.fission_after('C_reg[_] += _ #0', n_lifts=3)
    avx = avx.par_to_seq('for k in _:_')
    avx = avx.lift_alloc('C_reg:_', n_lifts=1)
    avx = avx.fission_after('for i in _:_#0', n_lifts=1)
    avx = avx.fission_after('for i in _:_#1', n_lifts=1)
    avx = avx.simplify()

    return avx, rank_k_reduce_6x16


def gen_sgemm_6x16_avx():
    avx2_sgemm_6x16, rank_k_reduce_6x16 = gen_rank_k_reduce_6x16()

    avx2_sgemm_6x16 = (
        avx2_sgemm_6x16
            .bind_expr('a_vec', 'A[i, k]')
            .set_memory('a_vec', AVX2)
            .lift_alloc('a_vec:_', keep_dims=True)
            .fission_after('a_vec[_] = _')
            #
            .bind_expr('b_vec', 'B[k, _]')
            .set_memory('b_vec', AVX2)
            .lift_alloc('b_vec:_', keep_dims=True)
            .fission_after('b_vec[_] = _')
            #
            .replace_all(avx2_set0_ps)
            .replace_all(mm256_broadcast_ss)
            .replace_all(mm256_fmadd_ps)
            .replace_all(avx2_fmadd_memu_ps)
            .replace(mm256_loadu_ps, 'for ji in _:_ #0')
            .replace(mm256_loadu_ps, 'for ji in _:_ #0')
            .replace(mm256_storeu_ps, 'for ji in _:_ #0')
            #
            .unroll('jo')
            .unroll('i')
            #
            .simplify()
    )

    return rank_k_reduce_6x16, avx2_sgemm_6x16


@pytest.mark.skip(reason='apparently unifying the broadcast is '
                         'non-deterministic')
def test_print_avx2_sgemm_kernel(golden):
    _, avx2_sgemm_kernel = gen_sgemm_6x16_avx()
    assert str(avx2_sgemm_kernel) == golden


@pytest.mark.isa('AVX2')
def test_avx2_sgemm_full(compiler):
    sgemm_6x16, avx2_sgemm_6x16 = gen_sgemm_6x16_avx()

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

        for i in par(0, N):
            for j in par(0, M):
                for k in par(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    cache_i = 16
    cache_j = 4
    cache_k = 2

    avx_sgemm_full = (
        sgemm_full.rename('avx_sgemm_full')
            # initial i,j tiling
            .split('i', 6, ['io', 'ii'], tail='cut')
            .reorder('ii #0', 'j')
            .split('j #0', 16, ['jo', 'ji'], tail='cut')
            .reorder('ji', 'ii')
            # breaking off the main loop
            .fission_after('for jo in _: _')
            # introduce k-tiling for later
            .split('k #0', cache_k * 16, ['ko', 'ki'], tail='cut')
            .fission_after('for ko in _: _', n_lifts=2)
            .reorder('ji', 'ko')
            .reorder('ii', 'ko')
            .replace_all(sgemm_6x16)
            # insert uses of micro-kernel now
            .call_eqv(avx2_sgemm_6x16, 'rank_k_reduce_6x16(_, _, _, _)')
            .call_eqv(avx2_sgemm_6x16, 'rank_k_reduce_6x16(_, _, _, _)')
            # do outer tiling for cache-locality
            .split('io #0', cache_i, ['io', 'im'], tail='cut')
            .reorder('im', 'jo')
            .split('jo', cache_j, ['jo', 'jm'], tail='cut')
            .reorder('jm', 'im')
            # move the ko loop up and out
            .fission_after('for ko in _: _', n_lifts=2)
            .reorder('jm # 0', 'ko')
            .reorder('im # 0', 'ko')
    )

    fn = compiler.compile(avx_sgemm_full,
                          skip_on_fail=True,
                          CMAKE_C_FLAGS="-march=skylake")

    _sgemm_test_cases(fn,
                      M=range(10, 600, 200),
                      N=range(20, 400, 120),
                      K=range(1, 512, 160))


@pytest.mark.isa('AVX2')
def test_avx2_sgemm_6x16(compiler):
    _, avx2_sgemm_6x16 = gen_sgemm_6x16_avx()

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

    fn = compiler.compile(avx2_sgemm_6x16_wrapper,
                          skip_on_fail=True,
                          CMAKE_C_FLAGS="-march=skylake")

    _sgemm_test_cases(fn, M=[6], N=[16], K=range(1, 512))


@pytest.mark.isa('AVX512f')
def test_avx512_sgemm_full(compiler):
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
        for i in par(0, M):
            for j in par(0, N / 16):
                mm512_loadu_ps(C_reg[i, j, :], C[i, 16 * j:16 * j + 16])
            if N % 16 > 0:
                mm512_maskz_loadu_ps(
                    N % 16,
                    C_reg[i, N / 16, :],
                    C[i, 16 * (N / 16):16 * (N / 16) + N % 16]
                )

        for k in par(0, K):
            for i in par(0, M):
                a_vec: f32[16] @ AVX512
                mm512_set1_ps(a_vec, A[i, k:k + 1])
                for j in par(0, ((N + 15) / 16)):
                    b_vec: f32[16] @ AVX512
                    mm512_loadu_ps(b_vec, B[k, j * 16:j * 16 + 16])
                    mm512_fmadd_ps(a_vec, b_vec, C_reg[i, j, :])

        for i in par(0, M):
            for j in par(0, N / 16):
                mm512_storeu_ps(C[i, 16 * j:16 * j + 16], C_reg[i, j, :])
            if N % 16 > 0:
                mm512_mask_storeu_ps(
                    N % 16,
                    C[i, 16 * (N / 16):16 * (N / 16) + N % 16],
                    C_reg[i, N / 16, :]
                )

    spec_kernel = (
        sgemm_micro_kernel_staged
            .partial_eval(6, 64)
            .simplify()
            .unroll('j')
            .unroll('i')
            .simplify()
    )

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
        for i in par(0, N):
            for j in par(0, M):
                for k in par(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    fn = compiler.compile(sgemm_full,
                          skip_on_fail=True,
                          CMAKE_C_FLAGS="-march=skylake-avx512")

    _sgemm_test_cases(fn,
                      M=range(10, 600, 200),
                      N=range(20, 400, 120),
                      K=range(1, 512, 160))
