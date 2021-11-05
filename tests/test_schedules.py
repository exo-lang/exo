from __future__ import annotations

import pytest

from SYS_ATL import proc, DRAM
from SYS_ATL.libs.memories import GEMM_SCRATCH


def test_data_reuse():
    @proc
    def foo(a : f32 @DRAM, b : f32 @DRAM):
        aa : f32
        bb : f32
        aa = a
        bb = b

        c : f32
        c = aa + bb
        b = c

    foo = foo.data_reuse('bb:_', 'c:_')
    print(foo)


def test_bind_lhs():
    @proc
    def myfunc_cpu(inp: i32[1, 1, 16] @ DRAM, out: i32[1, 1, 16] @ DRAM):
        for ii in par(0, 1):
            for jj in par(0, 1):
                for kk in par(0, 16):
                    out[ii, jj, kk] = out[ii, jj, kk] + inp[ii, jj, kk]
                    out[ii, jj, kk] = out[ii, jj, kk] * inp[ii, jj, kk]

    myfunc_cpu = myfunc_cpu.bind_expr('inp_ram', 'inp[_]', cse=True) 
    myfunc_cpu = myfunc_cpu.bind_expr('out_ram', 'out[_]', cse=True) 
    print(myfunc_cpu)

def test_simple_split():
    @proc
    def bar(n : size, A : i8[n]):
        tmp : i8[n]
        for i in par(0, n):
            tmp[i] = A[i]

    print("old\n", bar)
    bar = bar.split('i', 4, ['io', 'ii'], tail='guard')
    print("new\n", bar)

def test_simple_reorder():
    @proc
    def bar(n : size, m : size, A : i8[n, m]):
        tmp : i8[n, m]
        for i in par(0, n):
            for j in par(0, m):
                tmp[i,j] = A[i,j]

    print("old\n", bar)
    bar = bar.reorder('i', 'j')
    print("new\n", bar)

def test_simple_unroll():
    @proc
    def bar(A : i8[10]):
        tmp : i8[10]
        for i in par(0, 10):
            tmp[i] = A[i]

    print("old\n", bar)
    bar = bar.unroll('i')
    print("new\n", bar)

def test_simple_inline():
    @proc
    def foo(x : i8, y : i8, z : i8):
        z = x + y

    @proc
    def bar(n : size, src : i8[n], dst : i8[n]):
        for i in par(0, n):
            tmp_src : i8
            tmp_dst : i8
            tmp_src = src[i]
            tmp_dst = dst[i]
            foo(tmp_src, tmp_src, tmp_dst)

    print("old\n", bar)
    # TODO: This should fail
    # bar = bar.inline('foo(_)')
    # TODO: This should fail
    # bar = bar.inline('foo(io, i1, i2)')
    bar = bar.inline('foo(_, _, _)')
    print("new\n", bar)

def test_simple_partial_eval():
    @proc
    def bar(n : size, A : i8[n]):
        tmp : i8[n]
        for i in par(0, n):
            tmp[i] = A[i]

    print("old\n", bar)
    N = 10
    bar = bar.partial_eval(N)
    print("new\n", bar)

def test_simple_typ_and_mem():
    @proc
    def bar(n : size, A : R[n]):
        pass

    print("old\n", bar)
    bar = (bar.set_precision('A', 'i32')
              .set_memory('A', GEMM_SCRATCH))
    print("new\n", bar)

def test_simple_bind_expr():
    @proc
    def bar(n : size, x : i8[n], y : i8[n], z : i8[n]):
        for i in par(0, n):
            z[i] = x[i] + y[i]

    print("old\n", bar)
    bar = bar.bind_expr('z_tmp', 'x[_] + y[_]')
    print("new\n", bar)

def test_simple_lift_alloc():
    @proc
    def bar(n : size, A : i8[n]):
        for i in par(0, n):
            tmp_a : i8
            tmp_a = A[i]

    print("old\n", bar)
    bar = bar.lift_alloc('tmp_a : _', n_lifts=1)
    print("new\n", bar)

def test_simple_fission():
    @proc
    def bar(n : size, A : i8[n], B : i8[n], C : i8[n]):
        for i in par(0, n):
            C[i] += A[i]
            C[i] += B[i]

    print("old\n", bar)
    bar = bar.fission_after('C[_] += A[_]')
    print("new\n", bar)




@pytest.mark.skip()
def test_partition():
    @proc
    def bar(n : size, A : i8[n], pad : size):
        assert n > pad
        for i in par(0,n):
            tmp = A[i]

        for i in par(0,pad):
            tmp = A[i]
        for i in par(pad,n-pad):
            tmp = A[i]
        for i in par(n-pad,n):
            tmp = A[i]


def test_fission():
    @proc
    def bar(n : size, m : size):
        for i in par(0,n):
            for j in par(0,m):
                x : f32
                x = 0.0
                y : f32
                y = 1.1

    bar = bar.fission_after('x = _', n_lifts=2)


def test_fission2():
    with pytest.raises(Exception,
                       match='Will not fission here'):
        @proc
        def bar(n : size, m : size):
            for i in par(0,n):
                for j in par(0,m):
                    x : f32
                    x = 0.0
                    y : f32
                    y = 1.1
                    y = x

        bar = bar.fission_after('x = _', n_lifts=2)


def test_lift():
    @proc
    def bar(A : i8[16, 10]):
        for i in par(0, 10):
            a : i8[16]
            for k in par(0, 16):
                a[k] = A[k,i]

    bar = bar.lift_alloc('a: i8[_]', n_lifts=1, mode='col', size=20)


def test_unify1():
    @proc
    def bar(n : size, src : R[n,n], dst : R[n,n]):
        for i in par(0,n):
            for j in par(0,n):
                dst[i,j] = src[i,j]

    @proc
    def foo(x : R[5,5], y : R[5,5]):
        for i in par(0,5):
            for j in par(0,5):
                x[i,j] = y[i,j]

    foo = foo.replace(bar, "for i in _ : _")
    assert 'bar(5, y, x)' in str(foo)


def test_unify2():
    @proc
    def bar(n : size, src : [R][n,n], dst : [R][n,n]):
        for i in par(0,n):
            for j in par(0,n):
                dst[i,j] = src[i,j]

    @proc
    def foo(x : R[12,12], y : R[12,12]):
        for i in par(0,5):
            for j in par(0,5):
                x[i+3,j+1] = y[i+5,j+2]

    foo = foo.replace(bar, "for i in _ : _")
    assert 'bar(5, y[5:10, 2:7], x[3:8, 1:6])' in str(foo)


def test_unify3():
    @proc
    def simd_add4(dst : [R][4], a : [R][4], b : [R][4]):
        for i in par(0,4):
            dst[i] = a[i] + b[i]

    @proc
    def foo(n : size, z : R[n], x : R[n], y : R[n]):
        assert n % 4 == 0

        for i in par(0,n/4):
            for j in par(0,4):
                z[4*i + j] = x[4*i + j] + y[4*i + j]

    foo = foo.replace(simd_add4, "for j in _ : _")

    expected = '''
        simd_add4(z[4 * i + 0:4 * i + 4], x[4 * i + 0:4 * i + 4],
                  y[4 * i + 0:4 * i + 4])
'''
    assert expected in str(foo)


def test_unify4():
    @proc
    def bar(n : size, src : [R][n], dst : [R][n]):
        for i in par(0,n):
            if i < n-2:
                dst[i] = src[i] + src[i+1]

    @proc
    def foo(x : R[50, 2]):
        for j in par(0,50):
            if j < 48:
                x[j,1] = x[j,0] + x[j+1,0]

    foo = foo.replace(bar, "for j in _ : _")
    assert 'bar(50, x[0:50, 0], x[0:50, 1])' in str(foo)


def test_unify5():
    @proc
    def bar(n : size, src : R[n,n], dst : R[n,n]):
        for i in par(0,n):
            for j in par(0,n):
                tmp : f32
                tmp = src[i,j]
                dst[i,j] = tmp

    @proc
    def foo(x : R[5,5], y : R[5,5]):
        for i in par(0,5):
            for j in par(0,5):
                c : f32
                c = y[i,j]
                x[i,j] = c

    foo = foo.replace(bar, "for i in _ : _")
    assert 'bar(5, y, x)' in str(foo)


def test_unify6():
    @proc
    def load(
        n     : size,
        m     : size,
        src   : [i8][n, m],
        dst   : [i8][n, 16],
    ):
        assert n <= 16
        assert m <= 16

        for i in par(0, n):
            for j in par(0, m):
                dst[i,j] = src[i,j]

    @proc
    def bar(K: size, A: [i8][16, K] @ DRAM):

        for k in par(0, K / 16):
            a: i8[16, 16] @ DRAM
            for i in par(0, 16):
                for k_in in par(0, 16):
                    a[i, k_in] = A[i, 16 * k + k_in]

    bar = bar.replace(load, "for i in _:_")
    assert 'load(16, 16, A[0:16, 16 * k + 0:16 * k + 16], a[0:16, 0:16])' in str(bar)

