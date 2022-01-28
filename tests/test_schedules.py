from __future__ import annotations

import pytest

from SYS_ATL import proc, DRAM, SchedulingError
from SYS_ATL.libs.memories import GEMM_SCRATCH
from SYS_ATL.parse_fragment import ParseFragmentError


def test_fission_after_simple(golden):
    @proc
    def foo(n: size, m: size):
        for i in par(0, n):
            for j in par(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

    @proc
    def bar(n: size, m: size):
        for i in par(0, n):
            for j in par(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

        for k in par(0, 30):
            for l in par(0, 100):
                x: i8
                x = 4.0
                y: f32
                y = 1.1

    cases = [
        foo.fission_after_simple('x = _', n_lifts=2),
        bar.fission_after_simple('x = _', n_lifts=2),
    ]

    assert '\n'.join(map(str, cases)) == golden


def test_rearrange_dim(golden):
    @proc
    def foo(N: size, M: size, K: size, x: i8[N, M, K]):
        a: i8[N, M, K]
        for n in seq(0, N):
            for m in seq(0, M):
                for k in seq(0, K):
                    a[n, m, k] = x[n, m, k]

    @proc
    def bar(N: size, M: size, K: size, x: i8[N, M, K]):
        a: i8[N, M, K]
        for n in seq(0, N):
            for m in seq(0, M):
                for k in seq(0, K):
                    a[n, m, k] = x[n, m, k]

        a: i8[M, K, N]
        for n in seq(0, N):
            for m in seq(0, M):
                for k in seq(0, K):
                    a[m, k, n] = x[n, m, k]

    cases = [
        foo.rearrange_dim('a : i8[_]', [1, 2, 0]),
        bar.rearrange_dim('a : i8[_]', [1, 0, 2]),
    ]

    assert '\n'.join(map(str, cases)) == golden


def test_remove_loop(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        a: i8
        for i in seq(0, n):
            for j in seq(0, m):
                x = a

    @proc
    def bar(n: size, m: size, x: i8):
        a: i8
        for i in seq(0, n):
            for j in seq(0, m):
                x = a

        for i in seq(0, n):
            for j in seq(0, m):
                pass

    cases = [
        foo.remove_loop('for i in _:_'),
        bar.remove_loop('for i in _:_'),
    ]

    assert '\n'.join(map(str, cases)) == golden


def test_lift_alloc_simple(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

    bar = bar.lift_alloc_simple('tmp_a : _', n_lifts=2)
    assert str(bar) == golden


def test_lift_alloc_simple2(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

    bar = bar.lift_alloc_simple('tmp_a : _', n_lifts=2)
    assert str(bar) == golden


def test_lift_alloc_simple_error():
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

    with pytest.raises(SchedulingError, match='specified lift level'):
        bar.lift_alloc_simple('tmp_a : _', n_lifts=3)


def test_expand_dim(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        a: i8
        for i in seq(0, n):
            for j in seq(0, m):
                x = a

    foo = foo.expand_dim('a : i8', 'n', 'i')
    assert str(foo) == golden


def test_expand_dim2():
    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            a: i8
            for j in seq(0, m):
                x = a

        for i in seq(0, n):
            for k in seq(0, m):
                pass

    with pytest.raises(ParseFragmentError, match='k not found in'):
        foo.expand_dim('a : i8', 'n', 'k')  # should be error


def test_expand_dim3(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            for j in seq(0, m):
                pass

        for i in seq(0, n):
            a: i8
            for j in seq(0, m):
                x = a

        for i in seq(0, n):
            for j in seq(0, m):
                pass

    foo = foo.expand_dim('a : i8', 'n', 'i')  # did it pick the right i?
    assert foo.c_code_str() == golden


def test_expand_dim4(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            for j in seq(0, m):
                pass

        for q in seq(0, 30):
            a: i8
            for i in seq(0, n):
                for j in seq(0, m):
                    x = a

        for i in seq(0, n):
            for j in seq(0, m):
                pass

    with pytest.raises(TypeError, match='effect checking'):
        foo.expand_dim('a : i8', '10-20', '10')  # this is not fine

    with pytest.raises(TypeError, match='effect checking'):
        foo.expand_dim('a : i8', 'n - m', 'i')  # out of bounds

    with pytest.raises(ParseFragmentError, match='not found in'):
        foo.expand_dim('a : i8', 'hoge', 'i')  # does not exist

    with pytest.raises(TypeError, match='effect checking'):
        foo.expand_dim('a : i8', 'n', 'i-j')  # bound check should fail

    cases = [
        foo.expand_dim('a : i8', 'n', 'i'),  # did it pick the right i?
        foo.expand_dim('a : i8', '40 + 1', '10'),  # this is fine
        foo.expand_dim('a : i8', 'n + m', 'i'),  # fine
        foo.expand_dim('a : i8', 'n', 'n-1'),
    ]

    assert '\n'.join(map(str, cases)) == golden


def test_expand_dim5(golden):
    @proc
    def foo(n: size, x: i8):
        a: i8
        for i in seq(0, n):
            a = x

    foo = foo.expand_dim('a : i8', 'n', 'i')
    assert str(foo) == golden


def test_double_fission(golden):
    @proc
    def foo(N: size, a: f32[N], b: f32[N], out: f32[N]):
        for i in par(0, N):
            res: f32
            res = 0.0

            res += a[i] * b[i]

            out[i] = res

    foo = foo.lift_alloc('res : _')
    foo = foo.double_fission('res = _ #0', 'res += _ #0')
    assert str(foo) == golden


def test_data_reuse(golden):
    @proc
    def foo(a: f32 @ DRAM, b: f32 @ DRAM):
        aa: f32
        bb: f32
        aa = a
        bb = b

        c: f32
        c = aa + bb
        b = c

    foo = foo.data_reuse('bb:_', 'c:_')
    assert str(foo) == golden


def test_bind_lhs(golden):
    @proc
    def myfunc_cpu(inp: i32[1, 1, 16] @ DRAM, out: i32[1, 1, 16] @ DRAM):
        for ii in par(0, 1):
            for jj in par(0, 1):
                for kk in par(0, 16):
                    out[ii, jj, kk] = out[ii, jj, kk] + inp[ii, jj, kk]
                    out[ii, jj, kk] = out[ii, jj, kk] * inp[ii, jj, kk]

    myfunc_cpu = myfunc_cpu.bind_expr('inp_ram', 'inp[_]', cse=True)
    myfunc_cpu = myfunc_cpu.bind_expr('out_ram', 'out[_]', cse=True)
    assert str(myfunc_cpu) == golden


def test_simple_split(golden):
    @proc
    def bar(n: size, A: i8[n]):
        tmp: i8[n]
        for i in par(0, n):
            tmp[i] = A[i]

    bar = bar.split('i', 4, ['io', 'ii'], tail='guard')
    assert str(bar) == golden


def test_simple_reorder(golden):
    @proc
    def bar(n: size, m: size, A: i8[n, m]):
        tmp: i8[n, m]
        for i in par(0, n):
            for j in par(0, m):
                tmp[i, j] = A[i, j]

    bar = bar.reorder('i', 'j')
    assert str(bar) == golden


def test_simple_unroll(golden):
    @proc
    def bar(A: i8[10]):
        tmp: i8[10]
        for i in par(0, 10):
            tmp[i] = A[i]

    bar = bar.unroll('i')
    assert str(bar) == golden


def test_simple_inline(golden):
    @proc
    def foo(x: i8, y: i8, z: i8):
        z = x + y

    @proc
    def bar(n: size, src: i8[n], dst: i8[n]):
        for i in par(0, n):
            tmp_src: i8
            tmp_dst: i8
            tmp_src = src[i]
            tmp_dst = dst[i]
            foo(tmp_src, tmp_src, tmp_dst)

    # TODO: these should fail
    # with pytest.raises(SchedulingError, match='blah'):
    #     bar.inline('foo(_)')
    #
    # with pytest.raises(SchedulingError, match='blah'):
    #     bar.inline('foo(io, i1, i2)')

    bar = bar.inline('foo(_, _, _)')
    assert str(bar) == golden


def test_simple_partial_eval(golden):
    @proc
    def bar(n: size, A: i8[n]):
        tmp: i8[n]
        for i in par(0, n):
            tmp[i] = A[i]

    bar = bar.partial_eval(10)
    assert str(bar) == golden


def test_bool_partial_eval(golden):
    @proc
    def bar(b: bool, n: size, A: i8[n]):
        tmp: i8[n]
        for i in par(0, n):
            if b == True:
                tmp[i] = A[i]

    bar = bar.partial_eval(False)
    assert str(bar) == golden


def test_simple_typ_and_mem(golden):
    @proc
    def bar(n: size, A: R[n]):
        pass

    bar = (bar.set_precision('A', 'i32')
           .set_memory('A', GEMM_SCRATCH))
    assert str(bar) == golden


def test_simple_bind_expr(golden):
    @proc
    def bar(n: size, x: i8[n], y: i8[n], z: i8[n]):
        for i in par(0, n):
            z[i] = x[i] + y[i]

    bar = bar.bind_expr('z_tmp', 'x[_] + y[_]')
    assert str(bar) == golden


def test_simple_lift_alloc(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in par(0, n):
            tmp_a: i8
            tmp_a = A[i]

    bar = bar.lift_alloc('tmp_a : _', n_lifts=1)
    assert str(bar) == golden


def test_simple_fission(golden):
    @proc
    def bar(n: size, A: i8[n], B: i8[n], C: i8[n]):
        for i in par(0, n):
            C[i] += A[i]
            C[i] += B[i]

    bar = bar.fission_after('C[_] += A[_]')
    assert str(bar) == golden


@pytest.mark.skip()
def test_partition():
    @proc
    def bar(n: size, A: i8[n], pad: size):
        assert n > pad
        for i in par(0, n):
            tmp = A[i]

        for i in par(0, pad):
            tmp = A[i]
        for i in par(pad, n - pad):
            tmp = A[i]
        for i in par(n - pad, n):
            tmp = A[i]


def test_fission(golden):
    @proc
    def bar(n: size, m: size):
        for i in par(0, n):
            for j in par(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

    bar = bar.fission_after('x = _', n_lifts=2)
    assert str(bar) == golden


def test_fission2():
    @proc
    def bar(n: size, m: size):
        for i in par(0, n):
            for j in par(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1
                y = x

    with pytest.raises(Exception,
                       match='Will not fission here'):
        bar.fission_after('x = _', n_lifts=2)


def test_lift(golden):
    @proc
    def bar(A: i8[16, 10]):
        for i in par(0, 10):
            a: i8[16]
            for k in par(0, 16):
                a[k] = A[k, i]

    bar = bar.lift_alloc('a: i8[_]', n_lifts=1, mode='col', size=20)
    assert str(bar) == golden


def test_unify1(golden):
    @proc
    def bar(n: size, src: R[n, n], dst: R[n, n]):
        for i in par(0, n):
            for j in par(0, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in par(0, 5):
            for j in par(0, 5):
                x[i, j] = y[i, j]

    foo = foo.replace(bar, "for i in _ : _")
    assert str(foo) == golden


def test_unify2(golden):
    @proc
    def bar(n: size, src: [R][n, n], dst: [R][n, n]):
        for i in par(0, n):
            for j in par(0, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[12, 12], y: R[12, 12]):
        for i in par(0, 5):
            for j in par(0, 5):
                x[i + 3, j + 1] = y[i + 5, j + 2]

    foo = foo.replace(bar, "for i in _ : _")
    assert str(foo) == golden


def test_unify3(golden):
    @proc
    def simd_add4(dst: [R][4], a: [R][4], b: [R][4]):
        for i in par(0, 4):
            dst[i] = a[i] + b[i]

    @proc
    def foo(n: size, z: R[n], x: R[n], y: R[n]):
        assert n % 4 == 0

        for i in par(0, n / 4):
            for j in par(0, 4):
                z[4 * i + j] = x[4 * i + j] + y[4 * i + j]

    foo = foo.replace(simd_add4, "for j in _ : _")
    assert str(foo) == golden


def test_unify4(golden):
    @proc
    def bar(n: size, src: [R][n], dst: [R][n]):
        for i in par(0, n):
            if i < n - 2:
                dst[i] = src[i] + src[i + 1]

    @proc
    def foo(x: R[50, 2]):
        for j in par(0, 50):
            if j < 48:
                x[j, 1] = x[j, 0] + x[j + 1, 0]

    foo = foo.replace(bar, "for j in _ : _")
    assert str(foo) == golden


def test_unify5(golden):
    @proc
    def bar(n: size, src: R[n, n], dst: R[n, n]):
        for i in par(0, n):
            for j in par(0, n):
                tmp: f32
                tmp = src[i, j]
                dst[i, j] = tmp

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in par(0, 5):
            for j in par(0, 5):
                c: f32
                c = y[i, j]
                x[i, j] = c

    foo = foo.replace(bar, "for i in _ : _")
    assert str(foo) == golden


def test_unify6(golden):
    @proc
    def load(
            n: size,
            m: size,
            src: [i8][n, m],
            dst: [i8][n, 16],
    ):
        assert n <= 16
        assert m <= 16

        for i in par(0, n):
            for j in par(0, m):
                dst[i, j] = src[i, j]

    @proc
    def bar(K: size, A: [i8][16, K] @ DRAM):

        for k in par(0, K / 16):
            a: i8[16, 16] @ DRAM
            for i in par(0, 16):
                for k_in in par(0, 16):
                    a[i, k_in] = A[i, 16 * k + k_in]

    bar = bar.replace(load, "for i in _:_")
    assert str(bar) == golden


# Unused arguments
def test_unify7(golden):
    @proc
    def bar(unused_b: bool, n: size, src: R[n, n], dst: R[n, n],
            unused_m: index):
        for i in par(0, n):
            for j in par(0, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in par(0, 5):
            for j in par(0, 5):
                x[i, j] = y[i, j]

    foo = foo.replace(bar, "for i in _ : _")
    assert str(foo) == golden


def test_inline_window(golden):
    @proc
    def foo(n: size, m: size, x: R[n, m]):
        assert n > 4
        assert m > 4
        y = x[2:n - 2, 1:m - 3]

        for i in seq(0, n - 4):
            for j in seq(0, m - 4):
                a: R
                a = x[i, j] * y[i, j]
                y[i, j] = a + x[i + 1, j + 1]

    foo = foo.inline_window('y = _')
    assert str(foo) == golden
