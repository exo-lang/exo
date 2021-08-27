from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.append(sys.path[0] + "/..")
from SYS_ATL import DRAM
from SYS_ATL.libs.memories import AVX2
from .x86 import *

sys.path.append(sys.path[0] + "/.")
from .helper import *

import platform
import pytest

if platform.system() == 'Darwin':
    pytest.skip("skipping x86 tests on Apple machines for now",
                allow_module_level=True)


def test_avx2_memcpy():
    """
    Compute dst = src
    """

    @proc
    def memcpy_avx2(n: size, dst: R[n] @ DRAM,
                    src: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, (n + 7) / 8):
            if n - 8 * i >= 8:
                tmp: f32[8] @ AVX2
                loadu(tmp, src[8 * i:8 * i + 8])
                storeu(dst[8 * i:8 * i + 8], tmp)
            else:
                for j in par(0, n - 8 * i):
                    dst[8 * i + j] = src[8 * i + j]

    basename = test_avx2_memcpy.__name__

    with open(os.path.join(TMP_DIR, f'{basename}_pretty.atl'), 'w') as f:
        f.write(str(memcpy_avx2))

    memcpy_avx2.compile_c(TMP_DIR, basename)

    # TODO: -march=skylake here is a hack. Such flags should be somehow handled
    #   automatically. Maybe this should be inferred by the use of AVX2, but
    #   "skylake" isn't right anyway. We might need a first-class notion of
    #   a Target, which has certain memories available. Then we can say that
    #   e.g. Skylake-X has AVX2, AVX512, etc.
    library = generate_lib(basename, extra_flags="-march=skylake")

    for n in (7, 8, 9, 31, 32, 33, 127, 128, 129):
        inp = nparray([float(i) for i in range(n)])
        out = nparray([float(0) for _ in range(n)])
        library.memcpy_avx2(POINTER(c_int)(), n, cvt_c(out), cvt_c(inp))

        assert np.array_equal(inp, out)


def test_avx2_simple_math():
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
            loadu(xVec, x[8 * i:8 * i + 8])
            loadu(yVec, y[8 * i:8 * i + 8])
            mul(xVec, xVec, yVec)
            mul(xVec, xVec, yVec)
            storeu(x[8 * i:8 * i + 8], xVec)

    basename = test_avx2_simple_math.__name__

    with open(os.path.join(TMP_DIR, f'{basename}_pretty.atl'), 'w') as f:
        f.write(str(simple_math_avx2))

    simple_math_avx2.compile_c(TMP_DIR, basename)
    library = generate_lib(basename, extra_flags="-march=skylake")

    for n in (8, 16, 24, 32, 64, 128):
        x = nparray([float(i) for i in range(n)])
        y = nparray([float(3 * i) for i in range(n)])
        expected = x * y * y

        library.simple_math_avx2(POINTER(c_int)(), n, cvt_c(x), cvt_c(y))
        assert np.allclose(x, expected)


def test_avx2_simple_math_scheduling():
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_avx2_sched(n: size, x: R[n] @ DRAM,
                               y: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, n):
            x[i] = x[i] * y[i] * y[i]

    print()
    print()

    simple_math_avx2_sched = (
        simple_math_avx2_sched
            .split('i', 8, ['io', 'ii'], tail='cut_and_guard')
            .stage_assn('xyy', 'x[_] = _ #0')
            .lift_alloc('xyy: _')
            .set_memory('xyy', AVX2)
            .fission_after('xyy[_] = _')

            .replace_all(storeu)

            .bind_expr('xVec', 'x[_]')
            .lift_alloc('xVec: _')
            .set_memory('xVec', AVX2)
            .fission_after('xVec[_] = _')

            .bind_expr('yVec', 'y[_]', cse=True)
            .lift_alloc('yVec: _')
            .set_memory('yVec', AVX2)
            .fission_after('yVec[_] = _')

            .replace_all(loadu)

            .bind_expr('xy', 'xVec[_] * yVec[_]')
            .lift_alloc('xy: _')
            .set_memory('xy', AVX2)
            .fission_after('xy[_] = _')

            .replace_all(mul)
    )

    print(simple_math_avx2_sched)

    basename = test_avx2_simple_math_scheduling.__name__

    with open(os.path.join(TMP_DIR, f'{basename}_pretty.atl'), 'w') as f:
        f.write(str(simple_math_avx2_sched))

    simple_math_avx2_sched.compile_c(TMP_DIR, basename)
    library = generate_lib(basename, extra_flags="-march=skylake")

    for n in (8, 16, 24, 32, 64, 128):
        x = nparray([float(i) for i in range(n)])
        y = nparray([float(3 * i) for i in range(n)])
        expected = x * y * y

        int_ptr = POINTER(c_int)()
        library.simple_math_avx2_sched(int_ptr, n, cvt_c(x), cvt_c(y))
        assert np.allclose(x, expected)


def sgemm_test_cases(proc, M, N, K):
    for m, n, k in itertools.product(M, N, K):
        A = np.random.rand(m, k).astype(np.float32)
        B = np.random.rand(k, n).astype(np.float32)
        C = A @ B

        C_out = np.zeros_like(C)

        ctxt = POINTER(c_int)()
        proc(ctxt, m, n, k, cvt_c(C_out), cvt_c(A), cvt_c(B))
        assert np.allclose(C, C_out), f"sgemm failed for m={m} n={n} k={k}"


def gen_sgemm_6x16_avx():
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

    @proc
    def avx2_sgemm_6x16(
        K: size,
        C: [f32][6, 16] @ DRAM,
        A: [f32][6, K] @ DRAM,
        B: [f32][K, 16] @ DRAM,
    ):
        assert K > 0
        if K < 1:
            unreachable()

        C_reg: f32[6, 2, 8] @ AVX2
        for i in par(0, 6):
            for jo in par(0, 2):
                for ji in par(0, 8):
                    C_reg[i, jo, ji] = 0.0
        for k in par(0, K):
            for i in par(0, 6):
                for jo in par(0, 2):
                    for ji in par(0, 8):
                        C_reg[i, jo, ji] += A[i, k] * B[k, jo * 8 + ji]
        for i in par(0, 6):
            for jo in par(0, 2):
                for ji in par(0, 8):
                    C[i, jo * 8 + ji] += C_reg[i, jo, ji]

    rank_k_reduce_6x16.unsafe_assert_eq(avx2_sgemm_6x16)
    """
    compute C += A*B (for m x n = 6 x 16)
    """

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
            .replace_all(clear_reg)
            .replace_all(broadcast)
            .replace_all(loadu)
            .replace_all(fma)
            .replace_all(mem_accum)
            #
            .unroll('jo')
            .unroll('i')
    )

    return rank_k_reduce_6x16, avx2_sgemm_6x16


def test_avx2_sgemm_full():
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
        for i in par(0, N):
            for j in par(0, M):
                for k in par(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    avx_sgemm_full = (
        sgemm_full.rename('avx_sgemm_full')
            .split('i', 6, ['io', 'ii'], tail='cut')
            .reorder('ii #0', 'j')
            .split('j #0', 16, ['jo', 'ji'], tail='cut')
            .reorder('ji', 'ii')
            .replace(sgemm_6x16, 'for ii in _: _ #0')
            .call_eqv(avx2_sgemm_6x16, 'rank_k_reduce_6x16(_,_,_,_)')
    )
    print()
    print(avx_sgemm_full)

    basename = test_avx2_sgemm_full.__name__

    avx_sgemm_full.compile_c(TMP_DIR, basename)
    library = generate_lib(basename, extra_flags="-march=skylake")

    sgemm_test_cases(library.avx_sgemm_full,
                     M=range(10, 600, 200),
                     N=range(20, 400, 120),
                     K=range(1, 512, 160))


def test_avx2_sgemm_6x16():
    _, avx2_sgemm_6x16 = gen_sgemm_6x16_avx()

    print()
    print(avx2_sgemm_6x16)

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

    basename = test_avx2_sgemm_6x16.__name__

    avx2_sgemm_6x16_wrapper.compile_c(TMP_DIR, basename)
    library = generate_lib(basename, extra_flags="-march=skylake")

    sgemm_test_cases(library.avx2_sgemm_6x16_wrapper, M=[6], N=[16],
                     K=range(1, 512))
