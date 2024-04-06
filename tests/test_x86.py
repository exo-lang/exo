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
            tmp: f32[8] @ AVX2
            yVec: f32[8] @ AVX2
            mm256_loadu_ps(xVec, x[8 * i : 8 * i + 8])
            mm256_loadu_ps(yVec, y[8 * i : 8 * i + 8])
            mm256_mul_ps(tmp, xVec, yVec)
            mm256_mul_ps(xVec, tmp, yVec)
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

        p = bind_expr(p, p.find("y[_]", many=True), "yVec")
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


@pytest.mark.isa("AVX2")
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


@pytest.mark.isa("AVX2")
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


@pytest.mark.isa("AVX2")
def test_avx2_reg_copy_ps(compiler):
    @proc
    def avx2_reg_copy_ps_wrapper(dst: f32[8] @ DRAM, src: f32[8] @ DRAM):
        tmp_buffer_0: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_0, dst)
        tmp_buffer_1: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_1, src)
        avx2_reg_copy_ps(tmp_buffer_0, tmp_buffer_1)
        mm256_storeu_ps(dst, tmp_buffer_0)
        mm256_storeu_ps(src, tmp_buffer_1)

    @proc
    def avx2_reg_copy_ps_ref(dst: f32[8] @ DRAM, src: f32[8] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 8):
            dst[i] = src[i]

    fn = compiler.compile(
        [avx2_reg_copy_ps_wrapper, avx2_reg_copy_ps_ref],
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

    getattr(fn, "avx2_reg_copy_ps_wrapper")(None, dst, src)
    getattr(fn, "avx2_reg_copy_ps_ref")(None, dst_copy, src_copy)

    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(src, src_copy)


@pytest.mark.isa("AVX2")
def test_avx2_sign_ps(compiler):
    @proc
    def avx2_sign_ps_wrapper(dst: f32[8] @ DRAM, src: f32[8] @ DRAM):
        tmp_buffer_0: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_0, dst)
        tmp_buffer_1: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_1, src)
        avx2_sign_ps(tmp_buffer_0, tmp_buffer_1)
        mm256_storeu_ps(dst, tmp_buffer_0)
        mm256_storeu_ps(src, tmp_buffer_1)

    @proc
    def avx2_sign_ps_ref(dst: f32[8] @ DRAM, src: f32[8] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 8):
            dst[i] = -src[i]

    fn = compiler.compile(
        [avx2_sign_ps_wrapper, avx2_sign_ps_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )

    dst = np.array(
        [
            0.61119187,
            0.57439226,
            0.63750356,
            0.01567109,
            0.7531479,
            0.80388564,
            0.6817162,
            0.3611551,
        ],
        dtype=np.float32,
    )
    src = np.array(
        [
            0.1869111,
            0.53224814,
            0.71396947,
            0.7539144,
            0.6865989,
            0.33050302,
            0.86975175,
            0.17079325,
        ],
        dtype=np.float32,
    )

    dst_copy = dst.copy()
    src_copy = src.copy()

    getattr(fn, "avx2_sign_ps_wrapper")(None, dst, src)
    getattr(fn, "avx2_sign_ps_ref")(None, dst_copy, src_copy)

    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(src, src_copy)


@pytest.mark.isa("AVX2")
def test_avx2_reduce_add_wide_ps(compiler):
    @proc
    def avx2_reduce_add_wide_ps_wrapper(dst: f32[8] @ DRAM, src: f32[8] @ DRAM):
        tmp_buffer_0: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_0, dst)
        tmp_buffer_1: f32[8] @ AVX2
        mm256_loadu_ps(tmp_buffer_1, src)
        avx2_reduce_add_wide_ps(tmp_buffer_0, tmp_buffer_1)
        mm256_storeu_ps(dst, tmp_buffer_0)
        mm256_storeu_ps(src, tmp_buffer_1)

    @proc
    def avx2_reduce_add_wide_ps_ref(dst: f32[8] @ DRAM, src: f32[8] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 8):
            dst[i] += src[i]

    fn = compiler.compile(
        [avx2_reduce_add_wide_ps_wrapper, avx2_reduce_add_wide_ps_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )

    dst = np.array(
        [
            0.82633126,
            0.18224466,
            0.8483131,
            0.85528636,
            0.9373481,
            0.87415653,
            0.619115,
            0.85448426,
        ],
        dtype=np.float32,
    )
    src = np.array(
        [
            0.3565467,
            0.25555333,
            0.8524338,
            0.33920884,
            0.5461596,
            0.93643206,
            0.7152863,
            0.7914703,
        ],
        dtype=np.float32,
    )
    dst_copy = dst.copy()
    src_copy = src.copy()

    getattr(fn, "avx2_reduce_add_wide_ps_wrapper")(None, dst, src)
    getattr(fn, "avx2_reduce_add_wide_ps_ref")(None, dst_copy, src_copy)

    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(src, src_copy)


@pytest.mark.isa("AVX2")
def test_mm256_setzero_pd(compiler):
    @proc
    def mm256_setzero_pd_wrapper(dst: f64[4] @ DRAM):
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, dst)
        mm256_setzero_pd(tmp_buffer_0)
        mm256_storeu_pd(dst, tmp_buffer_0)

    @proc
    def mm256_setzero_pd_ref(dst: f64[4] @ DRAM):
        assert stride(dst, 0) == 1
        for i in seq(0, 4):
            dst[i] = 0.0

    fn = compiler.compile(
        [mm256_setzero_pd_wrapper, mm256_setzero_pd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    dst = np.array(
        [
            0.7702997103754631,
            0.45626842944141055,
            0.7439364522403988,
            0.701510512624819,
        ],
        dtype=np.float64,
    )
    dst_copy = dst.copy()
    getattr(fn, "mm256_setzero_pd_wrapper")(None, dst)
    getattr(fn, "mm256_setzero_pd_ref")(None, dst_copy)
    np.testing.assert_almost_equal(dst, dst_copy)


@pytest.mark.isa("AVX2")
def test_mm256_fmadd_pd(compiler):
    @proc
    def mm256_fmadd_pd_wrapper(
        dst: f64[4] @ DRAM, src1: f64[4] @ DRAM, src2: f64[4] @ DRAM
    ):
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, dst)
        tmp_buffer_1: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_1, src1)
        tmp_buffer_2: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_2, src2)
        mm256_fmadd_pd(tmp_buffer_0, tmp_buffer_1, tmp_buffer_2)
        mm256_storeu_pd(dst, tmp_buffer_0)
        mm256_storeu_pd(src1, tmp_buffer_1)
        mm256_storeu_pd(src2, tmp_buffer_2)

    @proc
    def mm256_fmadd_pd_ref(
        dst: f64[4] @ DRAM, src1: f64[4] @ DRAM, src2: f64[4] @ DRAM
    ):
        assert stride(src1, 0) == 1
        assert stride(src2, 0) == 1
        assert stride(dst, 0) == 1
        for i in seq(0, 4):
            dst[i] += src1[i] * src2[i]

    fn = compiler.compile(
        [mm256_fmadd_pd_wrapper, mm256_fmadd_pd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    dst = np.array(
        [
            0.06690474560928039,
            0.0850078709985519,
            0.029640255722781395,
            0.7284983930187073,
        ],
        dtype=np.float64,
    )
    src1 = np.array(
        [
            0.16333810334859344,
            0.773775867984533,
            0.4243993631288452,
            0.03447281200342622,
        ],
        dtype=np.float64,
    )
    src2 = np.array(
        [
            0.6898950978728396,
            0.7565890349219402,
            0.9694891176666618,
            0.18128624734877385,
        ],
        dtype=np.float64,
    )
    dst_copy = dst.copy()
    src1_copy = src1.copy()
    src2_copy = src2.copy()
    getattr(fn, "mm256_fmadd_pd_wrapper")(None, dst, src1, src2)
    getattr(fn, "mm256_fmadd_pd_ref")(None, dst_copy, src1_copy, src2_copy)
    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(src1, src1_copy)
    np.testing.assert_almost_equal(src2, src2_copy)


@pytest.mark.isa("AVX2")
def test_mm256_broadcast_sd(compiler):
    @proc
    def mm256_broadcast_sd_wrapper(out: f64[4] @ DRAM, val: f64[1] @ DRAM):
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, out)
        mm256_broadcast_sd(tmp_buffer_0, val)
        mm256_storeu_pd(out, tmp_buffer_0)

    @proc
    def mm256_broadcast_sd_ref(out: f64[4] @ DRAM, val: f64[1] @ DRAM):
        assert stride(out, 0) == 1
        for i in seq(0, 4):
            out[i] = val[0]

    fn = compiler.compile(
        [mm256_broadcast_sd_wrapper, mm256_broadcast_sd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    out = np.array(
        [
            0.9992828001754692,
            0.6054592176163047,
            0.21423730227883409,
            0.06414175723973703,
        ],
        dtype=np.float64,
    )
    val = np.array([0.6716955229801979], dtype=np.float64)
    out_copy = out.copy()
    val_copy = val.copy()
    getattr(fn, "mm256_broadcast_sd_wrapper")(None, out, val)
    getattr(fn, "mm256_broadcast_sd_ref")(None, out_copy, val_copy)
    np.testing.assert_almost_equal(out, out_copy)
    np.testing.assert_almost_equal(val, val_copy)


@pytest.mark.isa("AVX2")
def test_mm256_broadcast_sd_scalar(compiler):
    @proc
    def mm256_broadcast_sd_scalar_wrapper(out: f64[4] @ DRAM, val: f64[1] @ DRAM):
        tmp_val: f64
        tmp_val = val[0]
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, out)
        mm256_broadcast_sd_scalar(tmp_buffer_0, tmp_val)
        mm256_storeu_pd(out, tmp_buffer_0)

    @proc
    def mm256_broadcast_sd_scalar_ref(out: f64[4] @ DRAM, val: f64[1] @ DRAM):
        assert stride(out, 0) == 1
        for i in seq(0, 4):
            out[i] = val[0]

    fn = compiler.compile(
        [mm256_broadcast_sd_scalar_wrapper, mm256_broadcast_sd_scalar_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    out = np.array(
        [
            0.08715297607238981,
            0.2554485029748307,
            0.8467682976066226,
            0.9005310811939783,
        ],
        dtype=np.float64,
    )
    val = np.array([3.51])
    out_copy = out.copy()
    val_copy = val.copy()
    getattr(fn, "mm256_broadcast_sd_scalar_wrapper")(None, out, val)
    getattr(fn, "mm256_broadcast_sd_scalar_ref")(None, out_copy, val_copy)
    np.testing.assert_almost_equal(out, out_copy)
    np.testing.assert_almost_equal(val, val_copy)


@pytest.mark.isa("AVX2")
def test_mm256_mul_pd(compiler):
    @proc
    def mm256_mul_pd_wrapper(out: f64[4] @ DRAM, x: f64[4] @ DRAM, y: f64[4] @ DRAM):
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, out)
        tmp_buffer_1: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_1, x)
        tmp_buffer_2: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_2, y)
        mm256_mul_pd(tmp_buffer_0, tmp_buffer_1, tmp_buffer_2)
        mm256_storeu_pd(out, tmp_buffer_0)
        mm256_storeu_pd(x, tmp_buffer_1)
        mm256_storeu_pd(y, tmp_buffer_2)

    @proc
    def mm256_mul_pd_ref(out: f64[4] @ DRAM, x: f64[4] @ DRAM, y: f64[4] @ DRAM):
        assert stride(out, 0) == 1
        assert stride(x, 0) == 1
        assert stride(y, 0) == 1
        for i in seq(0, 4):
            out[i] = x[i] * y[i]

    fn = compiler.compile(
        [mm256_mul_pd_wrapper, mm256_mul_pd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    out = np.array(
        [
            0.20493105389880395,
            0.9358746107512442,
            0.5513072892432449,
            0.1416622828015115,
        ],
        dtype=np.float64,
    )
    x = np.array(
        [
            0.6359108492392163,
            0.7391629379581788,
            0.8894695567639811,
            0.6009249216013818,
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            0.9102732096226585,
            0.6246662381685367,
            0.8297542680325571,
            0.7470065913040405,
        ],
        dtype=np.float64,
    )
    out_copy = out.copy()
    x_copy = x.copy()
    y_copy = y.copy()
    getattr(fn, "mm256_mul_pd_wrapper")(None, out, x, y)
    getattr(fn, "mm256_mul_pd_ref")(None, out_copy, x_copy, y_copy)
    np.testing.assert_almost_equal(out, out_copy)
    np.testing.assert_almost_equal(x, x_copy)
    np.testing.assert_almost_equal(y, y_copy)


@pytest.mark.isa("AVX2")
def test_mm256_add_pd(compiler):
    @proc
    def mm256_add_pd_wrapper(out: f64[4] @ DRAM, x: f64[4] @ DRAM, y: f64[4] @ DRAM):
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, out)
        tmp_buffer_1: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_1, x)
        tmp_buffer_2: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_2, y)
        mm256_add_pd(tmp_buffer_0, tmp_buffer_1, tmp_buffer_2)
        mm256_storeu_pd(out, tmp_buffer_0)
        mm256_storeu_pd(x, tmp_buffer_1)
        mm256_storeu_pd(y, tmp_buffer_2)

    @proc
    def mm256_add_pd_ref(out: f64[4] @ DRAM, x: f64[4] @ DRAM, y: f64[4] @ DRAM):
        assert stride(out, 0) == 1
        assert stride(x, 0) == 1
        assert stride(y, 0) == 1
        for i in seq(0, 4):
            out[i] = x[i] + y[i]

    fn = compiler.compile(
        [mm256_add_pd_wrapper, mm256_add_pd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    out = np.array(
        [
            0.8527647836251072,
            0.4656462939895032,
            0.35186540014581635,
            0.630313690114371,
        ],
        dtype=np.float64,
    )
    x = np.array(
        [
            0.24068467638706337,
            0.8464360008490506,
            0.783023994276535,
            0.7109937138465194,
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            0.9352666294024309,
            0.32475739858813224,
            0.15796304969982733,
            0.31789446356547857,
        ],
        dtype=np.float64,
    )
    out_copy = out.copy()
    x_copy = x.copy()
    y_copy = y.copy()
    getattr(fn, "mm256_add_pd_wrapper")(None, out, x, y)
    getattr(fn, "mm256_add_pd_ref")(None, out_copy, x_copy, y_copy)
    np.testing.assert_almost_equal(out, out_copy)
    np.testing.assert_almost_equal(x, x_copy)
    np.testing.assert_almost_equal(y, y_copy)


@pytest.mark.isa("AVX2")
def test_avx2_select_pd(compiler):
    @proc
    def avx2_select_pd_wrapper(
        out: f64[4] @ DRAM,
        x: f64[4] @ DRAM,
        v: f64[4] @ DRAM,
        y: f64[4] @ DRAM,
        z: f64[4] @ DRAM,
    ):
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, out)
        tmp_buffer_1: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_1, x)
        tmp_buffer_2: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_2, v)
        tmp_buffer_3: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_3, y)
        tmp_buffer_4: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_4, z)
        avx2_select_pd(
            tmp_buffer_0, tmp_buffer_1, tmp_buffer_2, tmp_buffer_3, tmp_buffer_4
        )
        mm256_storeu_pd(out, tmp_buffer_0)
        mm256_storeu_pd(x, tmp_buffer_1)
        mm256_storeu_pd(v, tmp_buffer_2)
        mm256_storeu_pd(y, tmp_buffer_3)
        mm256_storeu_pd(z, tmp_buffer_4)

    @proc
    def avx2_select_pd_ref(
        out: f64[4] @ DRAM,
        x: f64[4] @ DRAM,
        v: f64[4] @ DRAM,
        y: f64[4] @ DRAM,
        z: f64[4] @ DRAM,
    ):
        assert stride(out, 0) == 1
        assert stride(x, 0) == 1
        assert stride(v, 0) == 1
        assert stride(y, 0) == 1
        assert stride(z, 0) == 1
        for i in seq(0, 4):
            xTmp: f64
            xTmp = x[i]
            vTmp: f64
            vTmp = v[i]
            yTmp: f64
            yTmp = y[i]
            zTmp: f64
            zTmp = z[i]
            out[i] = select(xTmp, vTmp, yTmp, zTmp)

    fn = compiler.compile(
        [avx2_select_pd_wrapper, avx2_select_pd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    out = np.array(
        [
            0.15072544011156797,
            0.36126593942879226,
            0.2962307720690991,
            0.5533859089478346,
        ],
        dtype=np.float64,
    )
    x = np.array(
        [
            0.8735799415726602,
            0.29340630006863155,
            0.3845715188628306,
            0.04794077205280567,
        ],
        dtype=np.float64,
    )
    v = np.array(
        [
            0.47764878514866593,
            0.7121357194784308,
            0.08003544890580039,
            0.16599111414940448,
        ],
        dtype=np.float64,
    )
    y = np.array(
        [
            0.18912116677473167,
            0.012455292110266747,
            0.9069633225674659,
            0.5928029432231223,
        ],
        dtype=np.float64,
    )
    z = np.array(
        [
            0.4357337303927791,
            0.7667888372979041,
            0.6989032832260329,
            0.006324755999849274,
        ],
        dtype=np.float64,
    )
    out_copy = out.copy()
    x_copy = x.copy()
    v_copy = v.copy()
    y_copy = y.copy()
    z_copy = z.copy()
    getattr(fn, "avx2_select_pd_wrapper")(None, out, x, v, y, z)
    getattr(fn, "avx2_select_pd_ref")(None, out_copy, x_copy, v_copy, y_copy, z_copy)
    np.testing.assert_almost_equal(out, out_copy)
    np.testing.assert_almost_equal(x, x_copy)
    np.testing.assert_almost_equal(v, v_copy)
    np.testing.assert_almost_equal(y, y_copy)
    np.testing.assert_almost_equal(z, z_copy)


@pytest.mark.isa("AVX2")
def test_avx2_assoc_reduce_add_pd(compiler):
    @proc
    def avx2_assoc_reduce_add_pd_wrapper(x: f64[4] @ DRAM, result: f64[1] @ DRAM):
        tmp_result: f64
        tmp_result = result[0]
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, x)
        avx2_assoc_reduce_add_pd(tmp_buffer_0, tmp_result)
        mm256_storeu_pd(x, tmp_buffer_0)
        result[0] = tmp_result

    @proc
    def avx2_assoc_reduce_add_pd_ref(x: f64[4] @ DRAM, result: f64[1] @ DRAM):
        assert stride(x, 0) == 1
        for i in seq(0, 4):
            result[0] += x[i]

    fn = compiler.compile(
        [avx2_assoc_reduce_add_pd_wrapper, avx2_assoc_reduce_add_pd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    x = np.array(
        [
            0.24349706300899643,
            0.44304522832871585,
            0.7845959927912908,
            0.27109035501352363,
        ],
        dtype=np.float64,
    )
    result = np.array([-1.253])
    x_copy = x.copy()
    result_copy = result.copy()
    getattr(fn, "avx2_assoc_reduce_add_pd_wrapper")(None, x, result)
    getattr(fn, "avx2_assoc_reduce_add_pd_ref")(None, x_copy, result_copy)
    np.testing.assert_almost_equal(x, x_copy)
    np.testing.assert_almost_equal(result, result_copy)


@pytest.mark.isa("AVX2")
def test_avx2_sign_pd(compiler):
    @proc
    def avx2_sign_pd_wrapper(dst: f64[4] @ DRAM, src: f64[4] @ DRAM):
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, dst)
        tmp_buffer_1: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_1, src)
        avx2_sign_pd(tmp_buffer_0, tmp_buffer_1)
        mm256_storeu_pd(dst, tmp_buffer_0)
        mm256_storeu_pd(src, tmp_buffer_1)

    @proc
    def avx2_sign_pd_ref(dst: f64[4] @ DRAM, src: f64[4] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 4):
            dst[i] = -src[i]

    fn = compiler.compile(
        [avx2_sign_pd_wrapper, avx2_sign_pd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    dst = np.array(
        [
            0.066282086666972,
            0.23414881487540895,
            0.4922961364730185,
            0.11810813857973845,
        ],
        dtype=np.float64,
    )
    src = np.array(
        [
            0.06436446994101086,
            0.637838437168881,
            0.6797374431407577,
            0.040744738211098475,
        ],
        dtype=np.float64,
    )
    dst_copy = dst.copy()
    src_copy = src.copy()
    getattr(fn, "avx2_sign_pd_wrapper")(None, dst, src)
    getattr(fn, "avx2_sign_pd_ref")(None, dst_copy, src_copy)
    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(src, src_copy)


@pytest.mark.isa("AVX2")
def test_avx2_reduce_add_wide_pd(compiler):
    @proc
    def avx2_reduce_add_wide_pd_wrapper(dst: f64[4] @ DRAM, src: f64[4] @ DRAM):
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, dst)
        tmp_buffer_1: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_1, src)
        avx2_reduce_add_wide_pd(tmp_buffer_0, tmp_buffer_1)
        mm256_storeu_pd(dst, tmp_buffer_0)
        mm256_storeu_pd(src, tmp_buffer_1)

    @proc
    def avx2_reduce_add_wide_pd_ref(dst: f64[4] @ DRAM, src: f64[4] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 4):
            dst[i] += src[i]

    fn = compiler.compile(
        [avx2_reduce_add_wide_pd_wrapper, avx2_reduce_add_wide_pd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    dst = np.array(
        [
            0.15640257715569084,
            0.6783560522800346,
            0.3288580779472433,
            0.05681854279921417,
        ],
        dtype=np.float64,
    )
    src = np.array(
        [
            0.9117144753906365,
            0.6016906956921138,
            0.3095551646830016,
            0.5330758995820801,
        ],
        dtype=np.float64,
    )
    dst_copy = dst.copy()
    src_copy = src.copy()
    getattr(fn, "avx2_reduce_add_wide_pd_wrapper")(None, dst, src)
    getattr(fn, "avx2_reduce_add_wide_pd_ref")(None, dst_copy, src_copy)
    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(src, src_copy)


@pytest.mark.isa("AVX2")
def test_avx2_reg_copy_pd(compiler):
    @proc
    def avx2_reg_copy_pd_wrapper(dst: f64[4] @ DRAM, src: f64[4] @ DRAM):
        tmp_buffer_0: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_0, dst)
        tmp_buffer_1: f64[4] @ AVX2
        mm256_loadu_pd(tmp_buffer_1, src)
        avx2_reg_copy_pd(tmp_buffer_0, tmp_buffer_1)
        mm256_storeu_pd(dst, tmp_buffer_0)
        mm256_storeu_pd(src, tmp_buffer_1)

    @proc
    def avx2_reg_copy_pd_ref(dst: f64[4] @ DRAM, src: f64[4] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 4):
            dst[i] = src[i]

    fn = compiler.compile(
        [avx2_reg_copy_pd_wrapper, avx2_reg_copy_pd_ref],
        skip_on_fail=True,
        CMAKE_C_FLAGS="-march=skylake",
    )
    dst = np.array(
        [
            0.8230485448185485,
            0.400123708278312,
            0.49373866892708396,
            0.7150382338404423,
        ],
        dtype=np.float64,
    )
    src = np.array(
        [
            0.15711822012981025,
            0.9644952348521756,
            0.770817397283775,
            0.9775115541309777,
        ],
        dtype=np.float64,
    )
    dst_copy = dst.copy()
    src_copy = src.copy()
    getattr(fn, "avx2_reg_copy_pd_wrapper")(None, dst, src)
    getattr(fn, "avx2_reg_copy_pd_ref")(None, dst_copy, src_copy)
    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(src, src_copy)


def test_avx2_divide_by_3(golden):
    @proc
    def foo():
        out: ui16[16] @ AVX2
        x: ui16[16] @ AVX2

        for i in seq(0, 16):
            out[i] = x[i] / 3.0

    foo = replace_all(foo, [avx2_ui16_divide_by_3])
    assert str(foo) == golden
