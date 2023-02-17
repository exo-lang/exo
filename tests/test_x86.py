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


@pytest.fixture
def simple_buffer_select():
    @proc
    def simple_buffer_select(
        N: size,
        out: f32[N] @ DRAM,
        x: f32[N] @ DRAM,
        v: f32[N] @ DRAM,
        y: f32[N] @ DRAM,
        z: f32[N] @ DRAM,
    ):
        assert N >= 1

        for i in seq(0, N):
            out[i] = select(x[i], v[i], y[i], z[i])

    VEC_W = 256 // 32
    outer_it = "io"
    inner_it = "ii"
    loop_fragment = lambda it, idx=0: f"for {it} in _:_ #{idx}"

    def sched_simple_buffer_select(proc):
        proc = divide_loop(
            proc, loop_fragment("i"), VEC_W, (outer_it, inner_it), tail="cut"
        )
        stage = lambda proc, buffer: set_memory(
            stage_mem(
                proc,
                loop_fragment(inner_it),
                f"{buffer}[{VEC_W} * {outer_it}:{VEC_W} * {outer_it} + {VEC_W}]",
                f"{buffer}Reg",
            ),
            f"{buffer}Reg",
            AVX2,
        )
        proc = stage(proc, "x")
        proc = stage(proc, "v")
        proc = stage(proc, "y")
        proc = stage(proc, "z")
        proc = stage(proc, "out")
        proc = replace(proc, loop_fragment("i0"), mm256_loadu_ps)
        proc = replace(proc, loop_fragment("i0"), mm256_loadu_ps)
        proc = replace(proc, loop_fragment("i0"), mm256_loadu_ps)
        proc = replace(proc, loop_fragment("i0"), mm256_loadu_ps)
        proc = replace(proc, loop_fragment(inner_it), avx2_select_ps)
        proc = replace(proc, loop_fragment("i0"), mm256_storeu_ps)
        proc = bind_expr(proc, "x[ii + N / 8 * 8]", "x_temp")
        proc = bind_expr(proc, "v[ii + N / 8 * 8]", "v_temp")
        proc = bind_expr(proc, "y[ii + N / 8 * 8]", "y_temp")
        proc = bind_expr(proc, "z[ii + N / 8 * 8]", "z_temp")
        proc = simplify(proc)

        return proc

    return sched_simple_buffer_select(simple_buffer_select)


@pytest.mark.isa("AVX2")
def test_avx2_select_ps(compiler, simple_buffer_select):
    fn = compiler.compile(
        simple_buffer_select, skip_on_fail=True, CMAKE_C_FLAGS="-march=skylake"
    )

    def run_and_check(N, x, v, y, z):
        expected = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if x[i] < v[i]:
                expected[i] = y[i]
            else:
                expected[i] = z[i]

        out = np.array(np.random.rand(N), dtype=np.float32)
        x_copy = x.copy()
        v_copy = v.copy()
        y_copy = y.copy()
        z_copy = z.copy()

        fn(None, N, out, x, v, y, z)

        np.testing.assert_almost_equal(out, expected)
        np.testing.assert_almost_equal(x, x_copy)
        np.testing.assert_almost_equal(v, v_copy)
        np.testing.assert_almost_equal(y, y_copy)
        np.testing.assert_almost_equal(z, z_copy)

    N = 50
    x = np.array([float(i) for i in range(N)], dtype=np.float32)
    v = np.array(
        [1, 0, 3, 4, 1, 1, 9, 0]
        + [100] * 8
        + [0] * 8
        + [float(i) for i in range(24, N)],
        dtype=np.float32,
    )
    y = np.array([float(i) for i in range(200, 200 + N)], dtype=np.float32)
    z = np.array([float(i) for i in range(300, 300 + N)], dtype=np.float32)

    # Correctness testing
    run_and_check(N, x, v, y, z)

    # Precision testing
    N = 111
    run_and_check(
        N,
        np.array(np.random.rand(N), dtype=np.float32),
        np.array(np.random.rand(N), dtype=np.float32),
        np.array(np.random.rand(N), dtype=np.float32),
        np.array(np.random.rand(N), dtype=np.float32),
    )


@pytest.mark.isa("AVX2")
def test_avx2_assoc_reduce_add_ps(compiler):
    @proc
    def accumulate_buffer(x: f32[8], result: [f32][1]):
        tmp_result: f32
        tmp_result = result[0]
        for i in seq(0, 8):
            tmp_result += x[i]
        result[0] = tmp_result

    accumulate_buffer = stage_mem(accumulate_buffer, "for i in _:_", "x[0:8]", "xReg")
    accumulate_buffer = set_memory(accumulate_buffer, "xReg", AVX2)
    accumulate_buffer = replace(accumulate_buffer, "for i0 in _:_", mm256_loadu_ps)
    accumulate_buffer = replace(
        accumulate_buffer, "for i in _:_", avx2_assoc_reduce_add_ps
    )
    accumulate_buffer = simplify(accumulate_buffer)

    fn = compiler.compile(
        accumulate_buffer, skip_on_fail=True, CMAKE_C_FLAGS="-march=skylake"
    )

    def run_and_check(x, result):
        expected = result.copy()
        for i in range(8):
            expected[0] += x[i]

        x_copy = x.copy()

        fn(None, x, result)

        # lower precision checking because we are assuming float addition is associative in the instruction
        np.testing.assert_almost_equal(result, expected, decimal=4)
        np.testing.assert_almost_equal(x, x_copy)

    result = np.array([0.0], dtype=np.float32)
    x = np.array([1, 2, 3, 1, 2, 3, 7, 2], dtype=np.float32)

    fn(None, x, result)

    run_and_check(x, result)
    run_and_check(
        np.array(
            [
                0.47299325,
                0.2869141,
                0.23663807,
                0.12012372,
                0.93651915,
                0.06829825,
                0.22391547,
                0.20829211,
            ],
            dtype=np.float32,
        ),
        np.array([0.11827405109974021], dtype=np.float32),
    )


def test_mm256_broadcast_ss_scalar(compiler):
    @proc
    def mm256_broadcast_ss_scalar_wrapper(out: f32[8] @ DRAM, val: f32[1] @ DRAM):
        tmp_val: f32
        tmp_val = val[0]
        tmp_buffer_0: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_0, out)
        mm256_broadcast_ss_scalar(tmp_buffer_0, tmp_val)
        mm256_storeu_ps(out, tmp_buffer_0)
        val[0] = tmp_val

    @proc
    def mm256_broadcast_ss_scalar_ref(out: f32[8] @ DRAM, val: f32[1] @ DRAM):
        # @instr {out_data} = _mm256_broadcast_ss(*{val});
        assert stride(out, 0) == 1
        for i in seq(0, 8):
            out[i] = val[0]

    fn = compiler.compile(
        [mm256_broadcast_ss_scalar_wrapper, mm256_broadcast_ss_scalar_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )

    out = np.array(
        [
            0.8352003,
            0.119042955,
            0.05500923,
            0.9115426,
            0.8030574,
            0.07465152,
            0.5667018,
            0.53294945,
        ],
        dtype=np.float32,
    )
    val = np.array([2.44])
    out_copy = out.copy()
    val_copy = val.copy()

    getattr(fn, "mm256_broadcast_ss_scalar_wrapper")(None, out, val)
    getattr(fn, "mm256_broadcast_ss_scalar_ref")(None, out_copy, val_copy)

    np.testing.assert_almost_equal(out, out_copy)
    np.testing.assert_almost_equal(val, val_copy)


def test_mm256_add_ps(compiler):
    @proc
    def mm256_add_ps_wrapper(out: f32[8] @ DRAM, x: f32[8] @ DRAM, y: f32[8] @ DRAM):
        tmp_buffer_0: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_0, out)
        tmp_buffer_1: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_1, x)
        tmp_buffer_2: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_2, y)
        mm256_add_ps(tmp_buffer_0, tmp_buffer_1, tmp_buffer_2)
        mm256_storeu_ps(out, tmp_buffer_0)
        mm256_storeu_ps(x, tmp_buffer_1)
        mm256_storeu_ps(y, tmp_buffer_2)

    @proc
    def mm256_add_ps_ref(out: f32[8] @ DRAM, x: f32[8] @ DRAM, y: f32[8] @ DRAM):
        # @instr {out_data} _mm256_add_ps ({x_data}, {y_data});
        assert stride(out, 0) == 1
        assert stride(x, 0) == 1
        assert stride(y, 0) == 1
        for i in seq(0, 8):
            out[i] = x[i] + y[i]

    fn = compiler.compile(
        [mm256_add_ps_wrapper, mm256_add_ps_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )

    out = np.array(
        [
            0.19168626,
            0.9292728,
            0.08044847,
            0.118411385,
            0.586027,
            0.9493457,
            0.91186064,
            0.97682995,
        ],
        dtype=np.float32,
    )
    x = np.array(
        [
            0.29922104,
            0.6485327,
            0.111039855,
            0.12260633,
            0.86832726,
            0.06105361,
            0.26866043,
            0.38205943,
        ],
        dtype=np.float32,
    )
    y = np.array(
        [
            0.30989048,
            0.3230521,
            0.70411354,
            0.117873766,
            0.5853253,
            0.08196206,
            0.31477037,
            0.8284393,
        ],
        dtype=np.float32,
    )
    out_copy = out.copy()
    x_copy = x.copy()
    y_copy = y.copy()

    getattr(fn, "mm256_add_ps_wrapper")(None, out, x, y)
    getattr(fn, "mm256_add_ps_ref")(None, out_copy, x_copy, y_copy)

    np.testing.assert_almost_equal(out, out_copy)
    np.testing.assert_almost_equal(x, x_copy)
    np.testing.assert_almost_equal(y, y_copy)


def test_mm256_reg_copy(compiler):
    @proc
    def mm256_reg_copy_wrapper(dst: f32[8] @ DRAM, src: f32[8] @ DRAM):
        tmp_buffer_0: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_0, dst)
        tmp_buffer_1: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_1, src)
        mm256_reg_copy(tmp_buffer_0, tmp_buffer_1)
        mm256_storeu_ps(dst, tmp_buffer_0)
        mm256_storeu_ps(src, tmp_buffer_1)

    @proc
    def mm256_reg_copy_ref(dst: f32[8] @ DRAM, src: f32[8] @ DRAM):
        # @instr {dst_data} = {src_data};
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 8):
            dst[i] = src[i]

    fn = compiler.compile(
        [mm256_reg_copy_wrapper, mm256_reg_copy_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )

    dst = np.array(
        [
            0.95774984,
            0.5476575,
            0.2301955,
            0.69115007,
            0.6172301,
            0.37313432,
            0.5556015,
            0.7339064,
        ],
        dtype=np.float32,
    )
    src = np.array(
        [
            0.29309168,
            0.034619242,
            0.0769644,
            0.9533431,
            0.11209598,
            0.5699761,
            0.36666384,
            0.32560244,
        ],
        dtype=np.float32,
    )
    dst_copy = dst.copy()
    src_copy = src.copy()

    getattr(fn, "mm256_reg_copy_wrapper")(None, dst, src)
    getattr(fn, "mm256_reg_copy_ref")(None, dst_copy, src_copy)

    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(src, src_copy)
