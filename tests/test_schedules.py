from __future__ import annotations

import pytest

from exo import proc, DRAM, SchedulingError, Procedure
from exo.libs.memories import GEMM_SCRATCH
from exo import ParseFragmentError
from exo.stdlib.scheduling import *

def test_commute(golden):
    @proc
    def foo(x : R[3], y : R[3], z : R):
        z = x[0]*y[2]
    assert str(commute_expr(foo, 'x[0] * y[_]')) == golden

def test_commute2():
    @proc
    def foo(x : R[3], y : R[3], z : R):
        z = x[0] + y[0] + x[1] + y[1]

    with pytest.raises(SchedulingError, match='failed to find matches'):
        # TODO: Currently, expression pattern matching fails to find
        # 'y[0]+x[1]' because LoopIR.BinOp is structured as (x[0], (y[0], (x[1], y[1]))).
        # I think pattern matching should be powerful to find this.
        commute_expr(foo, 'y[0] + x[1]')

def test_commute3(golden):
    @proc
    def foo(x : R[3], y : R[3], z : R):
        z = (x[0] + y[0]) * (x[1] + y[1] + y[2])
    assert str(commute_expr(foo, '(x[_] + y[_]) * (x[_] + y[_] + y[_])')) == golden

def test_commute4():
    @proc
    def foo(x : R[3], y : R[3], z : R):
        z = x[0] - y[2]

    with pytest.raises(TypeError, match="can commute"):
        commute_expr(foo, 'x[0] - y[_]')


def test_product_loop(golden):
    @proc
    def foo(n : size):
        x : R[n, 30]
        for i in seq(0, n):
            for j in seq(0, 30):
                x[i,j] = 0.0

    assert str(mult_loops(foo, 'i j', 'ij')) == golden

def test_product_loop2(golden):
    @proc
    def foo(n : size, x : R[n, 30]):
        for i in seq(0, n):
            for j in seq(0, 30):
                x[i,j] = 0.0

    assert str(mult_loops(foo, 'i j', 'ij')) == golden

def test_product_loop3():
    @proc
    def foo(n : size, m : size):
        x : R[n, m]
        for i in seq(0, n):
            for j in seq(0, m):
                x[i,j] = 0.0

    with pytest.raises(SchedulingError,
            match='expected the inner loop to have a constant bound'):
        mult_loops(foo, 'i j', 'ij')

def test_product_loop4(golden):
    @proc
    def foo(n : size, x : R[n]):
        for i in seq(0, n):
            for j in seq(0, 30):
                x[i] = 0.0

    assert str(mult_loops(foo, 'i j', 'ij')) == golden

def test_product_loop5(golden):
    @proc
    def foo(n : size, m : size, x : R[n, 100]):
        assert m < n
        x2 = x[0:m, 0:30]
        for i in seq(0, m):
            for j in seq(0, 30):
                x2[i,j] = 0.0

    assert str(mult_loops(foo, 'i j', 'ij')) == golden



def test_delete_pass(golden):
    @proc
    def foo(x : R):
        pass
        x = 0.0

    assert str(delete_pass(foo)) == golden

    @proc
    def foo(x : R):
        for i in seq(0, 16):
            for j in seq(0, 2):
                pass
        x = 0.0

    assert str(delete_pass(foo)) == golden

def test_add_loop1(golden):
    @proc
    def foo():
        x : R
        x = 0.0

    assert str(add_loop(foo, 'x = _', 'i', 10)) == golden

def test_add_loop2(golden):
    @proc
    def foo():
        x : R
        x = 0.0

    assert str(add_loop(foo, 'x = _', 'i', 10, guard=True)) == golden

def test_add_loop3():
    @proc
    def foo():
        x : R
        x = 0.0

    with pytest.raises(TypeError, match='expected a bool'):
        add_loop(foo, 'x = _', 'i', 10, guard=100)

def test_add_loop4(golden):
    @proc
    def foo(n : size, m : size):
        x : R
        x = 0.0

    assert str(add_loop(foo, 'x = _', 'i', 'n+m', guard=True)) == golden

# Should fix this test with program analysis
@pytest.mark.skip
def test_add_loop5():
    @proc
    def foo(n : size, m : size):
        x : R
        x = 0.0

    with pytest.raises(Exception, match='bound expression should be positive'):
        add_loop(foo, 'x = _', 'i', 'n-m', guard=True)


def test_proc_equal():
    def make_foo():
        @proc
        def foo(n: size, m: size):
            assert m == 1 and n == 1
            y: R[10]
            y[10 * m - 10 * n + 2 * n] = 2.0

        return foo

    foo = make_foo()
    foo2 = Procedure(foo.INTERNAL_proc())

    assert foo != 3  # check that wrong type doesn't crash, but returns false
    assert foo == foo  # reflexivity
    assert foo == foo2  # same underlying LoopIR.proc
    assert foo2 == foo  # symmetric

    # Coincidentally identical procs created from scratch should not be
    # considered equal.
    assert foo != make_foo()

def test_simplify3(golden):
    @proc
    def foo(n : size, m : size):
        assert m == 1 and n == 1
        y : R[10]
        y[10*m - 10*n + 2*n] = 2.0

    assert str(simplify(foo)) == golden

def test_simplify2(golden):
    @proc
    def foo(A: i8[32, 64] @ DRAM, B: i8[16, 128] @ DRAM,
            C: i32[32, 32] @ DRAM, ko : size, ji_unroll : size, ii_unroll : size):
        for io in seq(0, 1):
            for jo in seq(0, 1):
                Btile1: i8[16 * (ko + 1) - 16 * ko, 128 * jo + 64 *
                           (ji_unroll + 1 + 1) - (128 * jo + 64 *
                                                  (ji_unroll + 1))] @ DRAM
                Btile0: i8[16 * (ko + 1) - 16 * ko, 128 * jo + 64 *
                           (ji_unroll + 1) -
                           (128 * jo + 64 * ji_unroll)] @ DRAM
                Atile0: i8[32 * io + 16 * (ii_unroll + 1) -
                           (32 * io + 16 * ii_unroll), 64 *
                           (ko + 1) - 64 * ko] @ DRAM
                Atile1: i8[32 * io + 16 * (ii_unroll + 1 + 1) -
                           (32 * io + 16 *
                            (ii_unroll + 1)), 64 * (ko + 1) - 64 * ko] @ DRAM

    assert str(simplify(foo)) == golden

def test_simplify(golden):
    @proc
    def foo(n : size, m : size):
        x : R[n, 16 * (n + 1) - n * 16, (10 + 2) * m - m * 12 + 10]
        for i in seq(0, 4 * (n + 2) - n * 4 + n * 5):
            pass
        y : R[10]
        y[n*4 - n*4 + 1] = 0.0

    assert str(simplify(foo)) == golden

def test_pattern_match():
    @proc
    def foo(N1 : size, M1 : size, K1 : size, N2: size, M2: size, K2: size):
        x : R[N1, M1, K1]
        x : R[N2, M2, K2]

    res1 = rearrange_dim(foo, 'x : _', [2,1,0])
    res1 = rearrange_dim(res1, 'x : _ #1', [2,1,0])
    res2 = rearrange_dim(foo, 'x : R[N1, M1, K1]', [2,1,0])

    assert str(res1) != str(res2)

def test_fission_after_simple(golden):
    @proc
    def foo(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

    @proc
    def bar(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

        for k in seq(0, 30):
            for l in seq(0, 100):
                x: i8
                x = 4.0
                y: f32
                y = 1.1

    cases = [
        fission(foo, foo.find('x = _').after(), n_lifts=2),
        fission(bar, bar.find('x = _').after(), n_lifts=2),
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

    foo = rearrange_dim(foo, 'a : i8[_]', [1,2,0])
    bar = rearrange_dim(bar, 'a : i8[_]', [1, 0, 2])
    bar = rearrange_dim(bar, 'a : i8[_] #1', [1, 0, 2])
    cases = [ foo, bar ]

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
        remove_loop(foo, 'for i in _:_'),
        remove_loop(bar, 'for i in _:_'),
    ]

    assert '\n'.join(map(str, cases)) == golden


def test_lift_alloc_simple(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

    bar = lift_alloc(bar, 'tmp_a : _', n_lifts=2)
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

    bar = lift_alloc(bar, 'tmp_a : _', n_lifts=2)
    assert str(bar) == golden


def test_lift_alloc_simple_error():
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

    with pytest.raises(SchedulingError, match='specified lift level'):
        lift_alloc(bar, 'tmp_a : _', n_lifts=3)


def test_expand_dim(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            for j in seq(0, m):
                a: i8
                x = a

    foo = expand_dim(foo, 'a : i8', 'n', 'i')
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
        foo = expand_dim(foo, 'a : i8', 'n', 'k')  # should be error


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

    foo = expand_dim(foo, 'a : i8', 'n', 'i')  # did it pick the right i?
    assert foo.c_code_str() == golden


def test_expand_dim4(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            for j in seq(0, m):
                pass

        for q in seq(0, 30):
            for i in seq(0, n):
                for j in seq(0, m):
                    a: i8
                    x = a

        for i in seq(0, n):
            for j in seq(0, m):
                pass

    with pytest.raises(TypeError, match='effect checking'):
        expand_dim(foo, 'a : i8', '10-20', '10')  # this is not fine

    with pytest.raises(TypeError, match='effect checking'):
        expand_dim(foo, 'a : i8', 'n - m', 'i')  # out of bounds

    with pytest.raises(ParseFragmentError, match='not found in'):
        expand_dim(foo, 'a : i8', 'hoge', 'i')  # does not exist

    with pytest.raises(TypeError, match='effect checking'):
        expand_dim(foo, 'a : i8', 'n', 'i-j')  # bound check should fail

    cases = [
        expand_dim(foo, 'a : i8', 'n', 'i'),  # did it pick the right i?
        expand_dim(foo, 'a : i8', '40 + 1', '10'),  # this is fine
        expand_dim(foo, 'a : i8', 'n + m', 'i'),  # fine
        expand_dim(foo, 'a : i8', 'n', 'n-1'),
    ]

    assert '\n'.join(map(str, cases)) == golden


def test_expand_dim5(golden):
    @proc
    def foo(n: size, x: i8):
        for i in seq(0, n):
            a : i8
            a = x

    foo = expand_dim(foo, 'a : i8', 'n', 'i')
    assert str(foo) == golden

def test_divide_dim_1(golden):
    @proc
    def foo(n: size, m: size, A : R[n + m + 12]):
        x : R[n, 12, m]
        for i in seq(0, n):
            for j in seq(0, 12):
                for k in seq(0, m):
                    x[i, j, k] = A[i + j + k]

    foo = divide_dim(foo, 'x', 1, 4)
    assert str(foo) == golden


def test_divide_dim_fail_1():
    @proc
    def foo(n: size, m: size, A : R[n + m + 12]):
        x : R[n, 12, m]
        for i in seq(0, n):
            for j in seq(0, 12):
                for k in seq(0, m):
                    x[i, j, k] = A[i + j + k]

    with pytest.raises(ValueError, match='out-of-bounds'):
        divide_dim(foo, 'x', 3, 4)

    with pytest.raises(SchedulingError, match='Cannot divide 12 evenly'):
        divide_dim(foo, 'x', 1, 5)

def test_mult_dim_1(golden):
    @proc
    def foo(n: size, m: size, A : R[n + m + 12]):
        x : R[n, m, 4]
        for i in seq(0, n):
            for j in seq(0, m):
                for k in seq(0, 4):
                    x[i, j, k] = A[i + j + k]

    foo = mult_dim(foo, 'x', 0, 2)
    assert str(foo) == golden

def test_mult_dim_fail_1():
    @proc
    def foo(n: size, m: size, A : R[n + m + 12]):
        x : R[n, m, 4]
        for i in seq(0, n):
            for j in seq(0, m):
                for k in seq(0, 4):
                    x[i, j, k] = A[i + j + k]

    with pytest.raises(ValueError, match='out-of-bounds'):
        mult_dim(foo, 'x', 3, 4)

    with pytest.raises(ValueError, match='by itself'):
        mult_dim(foo, 'x', 2, 2)

    with pytest.raises(SchedulingError,
                       match='Cannot multiply with non-literal'):
        mult_dim(foo, 'x', 0, 1)

def test_double_fission(golden):
    @proc
    def foo(N: size, a: f32[N], b: f32[N], out: f32[N]):
        for i in seq(0, N):
            res: f32
            res = 0.0

            res += a[i] * b[i]

            out[i] = res

    foo = autolift_alloc(foo, 'res : _', keep_dims=True)
    foo = double_fission(foo, 'res = _ #0', 'res += _ #0')
    assert str(foo) == golden


def test_reuse_buffer(golden):
    @proc
    def foo(a: f32 @ DRAM, b: f32 @ DRAM):
        aa: f32
        bb: f32
        aa = a
        bb = b

        c: f32
        c = aa + bb
        b = c

    foo = reuse_buffer(foo, 'bb:_', 'c:_')
    assert str(foo) == golden


def test_bind_lhs(golden):
    @proc
    def myfunc_cpu(inp: i32[1, 1, 16] @ DRAM, out: i32[1, 1, 16] @ DRAM):
        for ii in seq(0, 1):
            for jj in seq(0, 1):
                for kk in seq(0, 16):
                    out[ii, jj, kk] = out[ii, jj, kk] + inp[ii, jj, kk]
                    out[ii, jj, kk] = out[ii, jj, kk] * inp[ii, jj, kk]

    myfunc_cpu = bind_expr(myfunc_cpu, 'inp[_]', 'inp_ram', cse=True)
    myfunc_cpu = bind_expr(myfunc_cpu, 'out[_]', 'out_ram', cse=True)
    assert str(myfunc_cpu) == golden


def test_simple_divide_loop(golden):
    @proc
    def bar(n: size, A: i8[n]):
        tmp: i8[n]
        for i in seq(0, n):
            tmp[i] = A[i]

    bar = divide_loop(bar, 'i', 4, ['io', 'ii'], tail='guard')
    assert str(bar) == golden


def test_simple_reorder(golden):
    @proc
    def bar(n: size, m: size, A: i8[n, m]):
        tmp: i8[n, m]
        for i in seq(0, n):
            for j in seq(0, m):
                tmp[i, j] = A[i, j]

    bar = reorder_loops(bar, 'i j')
    assert str(bar) == golden


def test_merge_reduce_1(golden):
    @proc
    def bar(x : R[3], y : R[3], z : R):
        z = x[0]
        z += y[2]

    bar = merge_reduce(bar, bar.find('z = x[0]').after())
    assert str(bar) == golden

def test_merge_reduce_2(golden):
    @proc
    def bar(w : R, x : R, y : R, z : R):
        z = w
        z += x
        z += y
        w = x

    bar = merge_reduce(bar, bar.find('z += x').before())
    bar = merge_reduce(bar, bar.find('z += y').before())
    assert str(bar) == golden


def test_simple_unroll(golden):
    @proc
    def bar(A: i8[10]):
        tmp: i8[10]
        for i in seq(0, 10):
            tmp[i] = A[i]

    bar = unroll_loop(bar, 'i')
    assert str(bar) == golden


def test_simple_inline(golden):
    @proc
    def foo(x: i8, y: i8, z: i8):
        z = x + y

    @proc
    def bar(n: size, src: i8[n], dst: i8[n]):
        for i in seq(0, n):
            tmp_src: i8
            tmp_dst: i8
            tmp_src = src[i]
            tmp_dst = dst[i]
            foo(tmp_src, tmp_src, tmp_dst)

    # TODO: these should fail
    # with pytest.raises(SchedulingError, match='blah'):
    #     inline(bar, 'foo(_)')
    #
    # with pytest.raises(SchedulingError, match='blah'):
    #     inline(bar, 'foo(io, i1, i2)')

    bar = inline(bar, 'foo(_, _, _)')
    assert str(bar) == golden


def test_simple_partial_eval(golden):
    @proc
    def bar(n: size, A: i8[n]):
        tmp: i8[n]
        for i in seq(0, n):
            tmp[i] = A[i]

    bar = bar.partial_eval(10)
    assert str(bar) == golden


def test_bool_partial_eval(golden):
    @proc
    def bar(b: bool, n: size, A: i8[n]):
        tmp: i8[n]
        for i in seq(0, n):
            if b == True:
                tmp[i] = A[i]

    bar = bar.partial_eval(False)
    assert str(bar) == golden


def test_simple_typ_and_mem(golden):
    @proc
    def bar(n: size, A: R[n]):
        pass

    bar = set_precision(bar, 'A', 'i32')
    bar = set_memory(bar, 'A', GEMM_SCRATCH)
    assert str(bar) == golden


def test_simple_bind_expr(golden):
    @proc
    def bar(n: size, x: i8[n], y: i8[n], z: i8[n]):
        for i in seq(0, n):
            z[i] = x[i] + y[i]

    bar = bind_expr(bar, 'x[_] + y[_]', 'z_tmp')
    assert str(bar) == golden


def test_simple_lift_alloc(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            tmp_a: i8
            tmp_a = A[i]

    bar = autolift_alloc(bar, 'tmp_a : _', n_lifts=1, keep_dims=True)
    assert str(bar) == golden


def test_simple_fission(golden):
    @proc
    def bar(n: size, A: i8[n], B: i8[n], C: i8[n]):
        for i in seq(0, n):
            C[i] += A[i]
            C[i] += B[i]

    bar = autofission(bar, bar.find('C[_] += A[_]').after())
    assert str(bar) == golden


@pytest.mark.skip()
def test_partition():
    @proc
    def bar(n: size, A: i8[n], pad: size):
        assert n > pad
        for i in seq(0, n):
            tmp = A[i]

        for i in seq(0, pad):
            tmp = A[i]
        for i in seq(pad, n - pad):
            tmp = A[i]
        for i in seq(n - pad, n):
            tmp = A[i]


def test_fission(golden):
    @proc
    def bar(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

    bar = autofission(bar, bar.find('x = _').after(), n_lifts=2)
    assert str(bar) == golden


def test_fission2():
    @proc
    def bar(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1
                y = x

    with pytest.raises(SchedulingError,
                       match='Will not fission here'):
        autofission(bar, bar.find('x = _').after(), n_lifts=2)


def test_lift(golden):
    @proc
    def bar(A: i8[16, 10]):
        for i in seq(0, 10):
            a: i8[16]
            for k in seq(0, 16):
                a[k] = A[k, i]

    bar = autolift_alloc(bar, 'a: i8[_]', n_lifts=1, mode='col', size=20, keep_dims=True)
    assert str(bar) == golden


def test_unify1(golden):
    @proc
    def bar(n: size, src: R[n, n], dst: R[n, n]):
        for i in seq(0, n):
            for j in seq(0, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                x[i, j] = y[i, j]

    foo = replace(foo, "for i in _ : _", bar)
    assert str(foo) == golden


def test_unify2(golden):
    @proc
    def bar(n: size, src: [R][n, n], dst: [R][n, n]):
        for i in seq(0, n):
            for j in seq(0, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[12, 12], y: R[12, 12]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                x[i + 3, j + 1] = y[i + 5, j + 2]

    foo = replace(foo, "for i in _ : _", bar)
    assert str(foo) == golden


def test_unify3(golden):
    @proc
    def simd_add4(dst: [R][4], a: [R][4], b: [R][4]):
        for i in seq(0, 4):
            dst[i] = a[i] + b[i]

    @proc
    def foo(n: size, z: R[n], x: R[n], y: R[n]):
        assert n % 4 == 0

        for i in seq(0, n / 4):
            for j in seq(0, 4):
                z[4 * i + j] = x[4 * i + j] + y[4 * i + j]

    foo = replace(foo, "for j in _ : _", simd_add4)
    assert str(foo) == golden


def test_unify4(golden):
    @proc
    def bar(n: size, src: [R][n], dst: [R][n]):
        for i in seq(0, n):
            if i < n - 2:
                dst[i] = src[i] + src[i + 1]

    @proc
    def foo(x: R[50, 2]):
        for j in seq(0, 50):
            if j < 48:
                x[j, 1] = x[j, 0] + x[j + 1, 0]

    foo = replace(foo, "for j in _ : _", bar)
    assert str(foo) == golden


def test_unify5(golden):
    @proc
    def bar(n: size, src: R[n, n], dst: R[n, n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp: f32
                tmp = src[i, j]
                dst[i, j] = tmp

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                c: f32
                c = y[i, j]
                x[i, j] = c

    foo = replace(foo, "for i in _ : _", bar)
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

        for i in seq(0, n):
            for j in seq(0, m):
                dst[i, j] = src[i, j]

    @proc
    def bar(K: size, A: [i8][16, K] @ DRAM):

        for k in seq(0, K / 16):
            a: i8[16, 16] @ DRAM
            for i in seq(0, 16):
                for k_in in seq(0, 16):
                    a[i, k_in] = A[i, 16 * k + k_in]

    bar = replace(bar, "for i in _:_", load)
    assert str(bar) == golden


# Unused arguments
def test_unify7(golden):
    @proc
    def bar(unused_b: bool, n: size, src: R[n, n], dst: R[n, n],
            unused_m: index):
        for i in seq(0, n):
            for j in seq(0, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                x[i, j] = y[i, j]

    foo = replace(foo, "for i in _ : _", bar)
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

    foo = inline_window(foo, 'y = _')
    assert str(foo) == golden


def test_lift_if_second_statement_in_then_error():
    @proc
    def foo(m: size, x: R[m]):
        for i in seq(0, m):
            if m > 12:
                x[0] = 1.0
                if i < 10:
                    x[i] = 2.0

    with pytest.raises(SchedulingError,
                       match='expected if statement to be directly nested in '
                             'parents'):
        foo = lift_if(foo, 'if i < 10: _')
        print(foo)


def test_lift_if_second_statement_in_else_error():
    @proc
    def foo(m: size, x: R[m]):
        for i in seq(0, m):
            if m > 12:
                pass
            else:
                x[0] = 1.0
                if i < 10:
                    x[i] = 2.0

    with pytest.raises(SchedulingError,
                       match='expected if statement to be directly nested in '
                             'parents'):
        foo = lift_if(foo, 'if i < 10: _')
        print(foo)


def test_lift_if_second_statement_in_for_error():
    @proc
    def foo(m: size, x: R[m]):
        for i in seq(0, m):
            x[0] = 1.0
            if m > 12:
                pass

    with pytest.raises(SchedulingError,
                       match='expected if statement to be directly nested in '
                             'parents'):
        foo = lift_if(foo, 'if m > 12: _')
        print(foo)


def test_lift_if_too_high_error():
    @proc
    def foo(m: size, x: R[m], j: size):
        for i in seq(0, m):
            if j < 10:
                x[i] = 2.0

    with pytest.raises(SchedulingError, match=r'1 lift\(s\) remain!'):
        foo = lift_if(foo, 'if j < 10: _', n_lifts=2)
        print(foo)


def test_lift_if_dependency_error():
    @proc
    def foo(m: size, x: R[m]):
        for i in seq(0, m):
            if i < 10:
                x[i] = 2.0

    with pytest.raises(SchedulingError,
                       match=r'if statement depends on iteration variable'):
        foo = lift_if(foo, 'if i < 10: _')
        print(foo)


def test_lift_if_past_if(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        assert i > 0
        if i < n:
            if i < 10:
                x[i] = 1.0

    foo = lift_if(foo, 'if i < 10: _')
    assert str(foo) == golden


def test_lift_if_past_for(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        for j in seq(0, n):
            if i < 10:
                x[j] = 1.0

    foo = lift_if(foo, 'if i < 10: _')
    assert str(foo) == golden


def test_lift_if_halfway(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        for j in seq(0, n):
            if n > 20:
                if i < 10:
                    x[j] = 1.0

    foo = lift_if(foo, 'if i < 10: _')
    assert str(foo) == golden


def test_lift_if_past_if_then_for(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        for j in seq(0, n):
            if n > 20:
                if i < 10:
                    x[j] = 1.0

    foo = lift_if(foo, 'if i < 10: _', n_lifts=2)
    assert str(foo) == golden


def test_lift_if_middle(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        for j in seq(0, n):
            if n > 20:
                if i < 10:
                    x[j] = 1.0

    foo = lift_if(foo, 'if n > 20: _')
    assert str(foo) == golden


def test_lift_if_with_else_past_if(golden):
    @proc
    def foo(n: size, x: R[n], i: size):
        assert n > 10
        if i < 10:
            if n > 20:
                x[i] = 1.0
            else:
                x[i] = 2.0

    foo = lift_if(foo, 'if n > 20: _')
    assert str(foo) == golden


def test_lift_if_with_else_past_if_with_else(golden):
    @proc
    def foo(n: size, x: R[n], i: size):
        assert n > 10
        assert i < n
        if i < 10:
            if n > 20:
                x[i] = 1.0
            else:
                x[i] = 2.0
        else:
            x[i] = 3.0

    foo = lift_if(foo, 'if n > 20: _')
    assert str(foo) == golden


def test_lift_if_with_pass_body(golden):
    @proc
    def foo(n: size):
        if 10 < n:
            if n < 20:
                pass

    foo = lift_if(foo, 'if n < 20: _')
    assert str(foo) == golden


def test_lift_if_with_pass_body_and_else(golden):
    @proc
    def foo(n: size):
        if 10 < n:
            if n < 20:
                pass
            else:
                pass

    foo = lift_if(foo, 'if n < 20: _')
    assert str(foo) == golden


def test_lift_if_in_else_branch_of_parent(golden):
    @proc
    def foo(n: size, x: R[n]):
        if 10 < n:
            x[0] = 1.0
        else:
            if n < 20:
                x[0] = 2.0
            else:
                x[0] = 3.0

    foo = lift_if(foo, 'if n < 20: _')
    assert str(foo) == golden


def test_lift_if_in_full_nest(golden):
    @proc
    def foo(n: size, x: R[n]):
        if 10 < n:
            if n < 15:
                x[0] = 1.0
            else:
                x[0] = 2.0
        else:
            if n < 20:
                x[0] = 3.0
            else:
                x[0] = 4.0

    foo = lift_if(foo, 'if n < 20: _')
    assert str(foo) == golden
                

def test_stage_mem(golden):
    # This test stages a buffer being accumulated in
    # on a per-tile basis
    @proc
    def sqmat(n : size, A : R[n,n], B : R[n,n]):
        assert n % 4 == 0
        for i in seq(0,n/4):
            for j in seq(0,n/4):
                for k in seq(0,n/4):
                    for ii in seq(0,4):
                        for jj in seq(0,4):
                            for kk in seq(0,4):
                                A[4*i+ii,4*j+jj] += ( B[4*i+ii,4*k+kk] *
                                                      B[4*k+kk,4*j+jj] )

    sqmat = stage_mem(sqmat, 'for k in _: _',
                             'A[4*i:4*i+4, 4*j:4*j+4]', 'Atile')
    assert str(simplify(sqmat)) == golden

def test_stage_mem_point(golden):
    @proc
    def matmul(n : size, A : R[n,n], B : R[n,n], C : R[n,n]):
        for i in seq(0, n):
            for j in seq(0, n):
                for k in seq(0, n):
                    C[i,j] += A[i,k] * B[k,j]

    matmul = stage_mem(matmul, 'for k in _:_', 'C[i, j]', 'res')
    assert str(simplify(matmul)) == golden

def test_fail_stage_mem():
    # This test fails to stage the buffer B
    # because it's not just being read in a single way
    # therefore the bounds check will fail
    @proc
    def sqmat(n : size, A : R[n,n], B : R[n,n]):
        assert n % 4 == 0
        for i in seq(0,n/4):
            for j in seq(0,n/4):
                for k in seq(0,n/4):
                    for ii in seq(0,4):
                        for jj in seq(0,4):
                            for kk in seq(0,4):
                                A[4*i+ii,4*j+jj] += ( B[4*i+ii,4*k+kk] *
                                                      B[4*k+kk,4*j+jj] )

    with pytest.raises(SchedulingError,
                       match='accessed out-of-bounds'):
        sqmat = stage_mem(sqmat, 'for ii in _: _',
                                 'B[4*i:4*i+4, 4*k:4*k+4]', 'Btile')

def test_stage_mem_twice(golden):
    # This test now finds a way to stage the buffer B twice
    @proc
    def sqmat(n : size, A : R[n,n], B : R[n,n]):
        assert n % 4 == 0
        for i in seq(0,n/4):
            for j in seq(0,n/4):
                for k in seq(0,n/4):
                    for ii in seq(0,4):
                        for jj in seq(0,4):
                            for kk in seq(0,4):
                                A[4*i+ii,4*j+jj] += ( B[4*i+ii,4*k+kk] *
                                                      B[4*k+kk,4*j+jj] )

    sqmat = bind_expr(sqmat, 'B[4*i+ii,4*k+kk]', 'B1')
    sqmat = expand_dim(sqmat, 'B1 : _', '4', 'kk')
    sqmat = expand_dim(sqmat, 'B1 : _', '4', 'ii')
    sqmat = autolift_alloc(sqmat, 'B1 : _', n_lifts=3)
    sqmat = autofission(sqmat, sqmat.find('B1[_] = _').after(), n_lifts=3)
    sqmat = stage_mem(sqmat, 'for ii in _: _ #1',
                             'B[4*k:4*k+4, 4*j:4*j+4]', 'B2')
    assert str(simplify(sqmat)) == golden


def test_stage_mem_accum(golden):
    # This test stages a buffer being accumulated in
    # on a per-tile basis
    @proc
    def sqmat(n : size, A : R[n,n], B : R[n,n]):
        assert n % 4 == 0
        for i in seq(0,n/4):
            for j in seq(0,n/4):
                for k in seq(0,n/4):
                    for ii in seq(0,4):
                        for jj in seq(0,4):
                            for kk in seq(0,4):
                                A[4*i+ii,4*j+jj] += ( B[4*i+ii,4*k+kk] *
                                                      B[4*k+kk,4*j+jj] )

    sqmat = stage_mem(sqmat, 'for k in _: _', 'A[4*i:4*i+4, 4*j:4*j+4]',
                             'Atile',  accum=True)
    assert str(simplify(sqmat)) == golden

def test_stage_mem_accum2(golden):
    @proc
    def accum(out : R[4, 16, 16], w : R[16], im : R[16]):
        for k in seq(0, 4):
            for i in seq(0, 16):
                for j in seq(0, 16):
                    out[k, i, j] += w[j] * im[i]

    accum = stage_mem(accum, 'for i in _:_', 'out[k, 0:16, 0:16]', 'o')

    assert str(simplify(accum)) == golden
