from __future__ import annotations

import sys

sys.path.append(sys.path[0] + "/..")
from SYS_ATL import DRAM
from SYS_ATL.libs.memories import AVX2
from .x86 import loadu, storeu, mul, fma, broadcast

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


def test_avx2_sgemm_6x16():
    @proc
    def avx2_sgemm_6x16(
            K: size,
            C: f32[6, 16] @ DRAM,
            A: f32[6, K] @ DRAM,
            B: f32[K, 16] @ DRAM,
    ):
        for i in par(0, 6):
            for j in par(0, 16):
                for k in par(0, K):
                    C[i, j] += A[i, k] * B[k, j]

    """
    compute C += A*B (for m x n = 6 x 16)
    """

    avx2_sgemm_6x16 = (
        avx2_sgemm_6x16
            .reorder('j', 'k')
            .reorder('i', 'k')
            .stage_assn('C_mem', 'C[_] += _')
            .set_memory('C_mem', AVX2)
            .split('j', 8, ['jo', 'ji'], perfect=True)
            .lift_alloc('C_mem: _')
            .fission_after('C_mem[_] = C[_]')
            .fission_after('C_mem[_] += A[_] * B[_]')
            .replace_all(loadu)
            .replace_all(storeu)
            .bind_expr('a_vec', 'A[i, k]')
            .set_memory('a_vec', AVX2)
            .lift_alloc('a_vec: _', keep_dims=True)
            .fission_after('a_vec = _')
            # TODO: unification should be able to handle this sequence
            .bind_expr('aik', 'A[i, k]')
            .lift_alloc('aik: _')
            .set_memory('aik', DRAM)
            .fission_after('aik = _')
            .replace_all(broadcast)
            # end
            .bind_expr('b_vec', 'B[_]')
            .lift_alloc('b_vec: _')
            .set_memory('b_vec', AVX2)
            .fission_after('b_vec[_] = _')
            .replace_all(loadu)
            .replace_all(fma)
        # .lift_alloc('C_mem: _', n_lifts=3, keep_dims=False)
        # .fission_after('loadu(_)')
    )

    print(avx2_sgemm_6x16)

    basename = test_avx2_sgemm_6x16.__name__

    with open(os.path.join(TMP_DIR, f'{basename}_pretty.atl'), 'w') as f:
        f.write(str(avx2_sgemm_6x16))

    avx2_sgemm_6x16.compile_c(TMP_DIR, basename)
    library = generate_lib(basename, extra_flags="-march=skylake")

    for K in (1, 2, 3, 4, 8, 12, 16, 31, 32, 33, 63, 64, 65, 47, 48, 49):
        A = np.random.rand(6, K).astype(np.float32)
        B = np.random.rand(K, 16).astype(np.float32)
        C = A @ B

        C_out = 0 * C

        ctxt = POINTER(c_int)()
        library.avx2_sgemm_6x16(ctxt, K, cvt_c(C_out), cvt_c(A), cvt_c(B))
        assert np.allclose(C, C_out)
