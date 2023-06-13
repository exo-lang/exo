from __future__ import annotations

import pytest

from exo import proc, DRAM
from exo.libs.memories import GEMM_SCRATCH
from exo.stdlib.scheduling import SchedulingError


# ------- Effect check tests ---------


def test_seq_write1(golden):
    @proc
    def foo(n: size, A: i8[n]):
        a: i8
        for i in seq(0, n):
            a = A[i]

    assert foo.show_effects() == golden


def test_new_stride1(golden):
    @proc
    def foo(s: stride, scale: f32):
        assert s == 1
        pass

    @proc
    def bar(n: size, A: i8[n]):
        scale: f32
        scale = 0.0
        foo(stride(A, 0), scale)

    assert bar.show_effects() == golden


def test_write_write3(golden):
    @proc
    def foo(n: size, A: i8[n]):
        a: i8
        for i in seq(0, n):
            a = 3.0

    assert foo.show_effects() == golden


def test_different_id(golden):
    @proc
    def foo(n: size):
        for ihi in seq(0, n):
            a: i8 @ DRAM
            for ilo in seq(0, 16):
                a = 0.0
        if False:
            b: i8 @ DRAM
            b = 0.0

    assert foo.show_effects() == golden


# This is fine
@pytest.mark.skip()
def test_reduce_write1():
    @proc
    def foo(n: size, A: i8[n]):
        a: i8
        a = 4.0
        for i in seq(0, n):
            a += A[i]
            a = 0.0


# This is fine
@pytest.mark.skip()
def test_reduce_write2():
    @proc
    def foo(n: size, A: i8[n]):
        a: i8
        a = 4.0
        for i in seq(0, n):
            a = 0.0
            a += A[i]


def test_index1():
    with pytest.raises(
        TypeError, match="expected expression to always be non-negative"
    ):

        @proc
        def foo(n: index, m: index, A: i8[n, m]):
            assert n > 0 and m > 0
            for i in seq(0, n):
                for j in seq(0, m):
                    for k in seq(0, i - j):
                        a: i8
                        a = 0.0


def test_index2():
    with pytest.raises(TypeError, match="A is read out-of-bounds"):

        @proc
        def foo(n: index, m: index, A: i8[n, m]):
            assert n > 0 and m > 0
            for i in seq(0, n):
                for j in seq(0, m):
                    a: i8
                    a = A[i, j - 1]


def test_index3(golden):
    @proc
    def foo():
        for i in seq(0, 0):
            a: i8
            a = 0.0

    assert foo.show_effects() == golden


def test_index4():
    with pytest.raises(
        TypeError, match="expected expression to always be non-negative"
    ):

        @proc
        def foo(n: index):
            for i in seq(0, n):
                a: i8
                a = 0.0


# For-loop bound non-negative check tests
def test_good_bound1(golden):
    @proc
    def good_bound1(n: size, dst: R[n] @ DRAM, src: R[n] @ DRAM):
        for i in seq(0, (n + 7) / 8):
            if n - 8 * i >= 8:
                pass
            else:
                for j in seq(0, n - 8 * i):
                    dst[8 * i + j] = src[8 * i + j]

    assert good_bound1.show_effects() == golden


def test_bad_bound1():
    with pytest.raises(TypeError, match="Errors occurred during effect checking"):

        @proc
        def bar():
            for i in seq(0, -2):
                pass


def test_bad_bound2():
    with pytest.raises(TypeError, match="Errors occurred during effect checking"):

        @proc
        def bar(n: size):
            for i in seq(0, n):
                for j in seq(0, i - 1):
                    pass


def test_bad_bound3():
    with pytest.raises(TypeError, match="Errors occurred during effect checking"):

        @proc
        def bar(n: size):
            for i in seq(0, n):
                for j in seq(0, i - n):
                    pass


def test_bad_bound4():
    with pytest.raises(TypeError, match="Errors occurred during effect checking"):

        @proc
        def bar(n: size, dst: R[n] @ DRAM, src: R[n] @ DRAM):
            for i in seq(0, (n + 7) / 8):
                if n - 8 * i >= 8:
                    pass
                else:
                    for j in seq(0, n - 9 * i):
                        dst[8 * i + j] = src[8 * i + j]


def test_bad_access1():
    with pytest.raises(TypeError, match="Errors occurred during effect checking"):

        @proc
        def bad_access1(n: size, m: size, x: R[n, m], y: R[n, m], res: R[n, m]):
            rloc: R[m]
            for i in seq(0, m):
                xloc: R[m]
                yloc: R[m]
                for j in seq(0, n):
                    xloc[j] = x[i, j]
                for j in seq(0, m):
                    yloc[j] = y[i, j]
                for j in seq(0, m):
                    rloc[j] = xloc[j] + yloc[j]
                for j in seq(0, m):
                    res[i, j] = rloc[j]


def test_bad_access2():
    with pytest.raises(TypeError, match="Errors occurred during effect checking"):

        @proc
        def bad_access2(
            n: size, m: size, x: R[n, m], y: R[n, m] @ DRAM, res: R[n, m] @ DRAM
        ):
            rloc: R[m]
            for i in seq(0, n):
                xloc: R[m]
                yloc: R[m]
                for j in seq(0, m):
                    xloc[j] = x[i + 1, j]
                for j in seq(0, m):
                    yloc[j] = y[i, j]
                for j in seq(0, m):
                    rloc[j] = xloc[j] + yloc[j - 1]
                for j in seq(0, m):
                    res[i, j] = rloc[j]


def test_bad_access3():
    with pytest.raises(TypeError, match="x2 is read out-of-bounds"):

        @proc
        def foo():
            x2: R[1]
            huga: R
            huga = x2[100]


def test_assert1():
    with pytest.raises(TypeError, match="Could not verify assertion"):

        @proc
        def foo(n: size, x: i8[n, n]):
            assert n == 1
            pass

        @proc
        def bar():
            z: i8[3, 3]
            foo(3, z)


def test_size1():
    with pytest.raises(TypeError, match="type-shape of calling argument"):

        @proc
        def foo(x: i8[3, 3]):
            pass

        @proc
        def bar():
            z: i8[3, 4]
            foo(z)


# Data Race? No
def test_race2(golden):
    @proc
    def foo(n: size, x: R[n, n]):
        for i in seq(0, n):
            if i + 1 < n:
                x[i, i] = x[i + 1, i]

    assert foo.show_effects() == golden


# Data Race? No
def test_race3(golden):
    @proc
    def foo(n: size, x: R[n, n]):
        y = x[1:, :]
        for i in seq(0, n):
            if i + 1 < n:
                x[i, i] = y[i, i]

    assert foo.show_effects() == golden


# one big issue is aliasing in sub-procedure arguments
def test_race4():
    @proc
    def foo(n: size, x: [R][n, n], y: [R][n, n]):
        for i in seq(0, n):
            if i + 1 < n:
                x[i, i] = y[i, i]

    with pytest.raises(SchedulingError, match="Cannot Pass the same buffer"):

        @proc
        def bar(n: size, z: R[n, n]):
            foo(n, z, z)


def test_div1(golden):
    @proc
    def foo(n: size):
        assert n == 3
        pass

    @proc
    def bar():
        foo(10 / 3)

    assert bar.show_effects() == golden


def test_mod1(golden):
    @proc
    def foo(n: size):
        assert n == 1
        pass

    @proc
    def bar():
        foo(10 % 3)

    assert bar.show_effects() == golden


# Callee has a window but caller has a tensor case
def test_stride_assert1(golden):
    @proc
    def foo(
        n: size,
        m: size,
        src: [i8][n, m] @ DRAM,
        dst: [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass

    @proc
    def bar(x: i8[30, 10] @ DRAM, y: i8[30, 16] @ GEMM_SCRATCH):
        foo(30, 10, x, y)

    assert bar.show_effects() == golden


# Both callee and caller has a window case
def test_stride_assert2():
    with pytest.raises(TypeError, match="Could not verify assertion"):

        @proc
        def foo(
            n: size,
            m: size,
            src: [i8][n, m] @ DRAM,
            dst: [i8][n, 16] @ GEMM_SCRATCH,
        ):
            assert stride(src, 1) == 1
            assert stride(dst, 0) == 16
            assert stride(dst, 1) == 1
            pass

        @proc
        def bar(x: i8[30, 10] @ DRAM, y: [i8][30, 16] @ GEMM_SCRATCH):
            foo(30, 10, x, y)


# Both callee and caller has a window case, but with top level assert
def test_stride_assert3(golden):
    @proc
    def foo(
        n: size,
        m: size,
        src: [i8][n, m] @ DRAM,
        dst: [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass

    @proc
    def bar(x: i8[30, 10] @ DRAM, y: [i8][30, 16] @ GEMM_SCRATCH):
        assert stride(y, 0) == 16
        assert stride(y, 1) == 1
        foo(30, 10, x, y)

    assert bar.show_effects() == golden


# callee is Tensor case and caller is a window case.
# this will be an error because we don't know anything about incoming stride
def test_stride_assert4():
    with pytest.raises(TypeError, match="Could not verify assertion"):

        @proc
        def foo(
            n: size,
            m: size,
            src: i8[n, m] @ DRAM,
            dst: i8[n, 16] @ GEMM_SCRATCH,
        ):
            assert stride(src, 1) == 1
            assert stride(dst, 0) == 16
            assert stride(dst, 1) == 1
            pass

        @proc
        def bar(x: [i8][30, 10] @ DRAM, y: [i8][30, 16] @ GEMM_SCRATCH):
            foo(30, 10, x, y)


# Tensor with wrong size and stride
def test_stride_assert5():
    with pytest.raises(TypeError, match="is always unsatisfiable"):

        @proc
        def bar(x: i8[30, 10] @ DRAM, y: i8[30, 16] @ GEMM_SCRATCH):
            assert stride(x, 0) == 9
            pass


# Tensor asserting last dimension is fine
def test_stride_assert6(golden):
    @proc
    def bar(n: size, m: size, x: i8[n, m] @ DRAM):
        assert stride(x, 1) == 1
        pass

    assert bar.show_effects() == golden


# Test Tensor having insufficient information (sizes)
def test_stride_assert7(golden):
    # with changes, this should not trigger an error
    # since it might be true, even if it is highly unlikely
    # i.e. this would have to be always called with m == 10
    @proc
    def bar(n: size, m: size, x: i8[n, m] @ DRAM):
        assert stride(x, 0) == 10
        pass

    assert bar.show_effects() == golden


# Test Windowstmt
def test_stride_assert8(golden):
    @proc
    def foo(
        n: size,
        m: size,
        src: [i8][n, m] @ DRAM,
        dst: [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass

    @proc
    def bar(x: i8[8, 30, 10] @ DRAM, y: i8[50, 4, 100, 16] @ GEMM_SCRATCH):
        xx = x[0, :, :]
        yy = y[3, 1, 3:33, :]

        foo(30, 10, xx, yy)

    assert bar.show_effects() == golden


# Test Windowexpr within call arg
def test_stride_assert9(golden):
    @proc
    def foo(
        n: size,
        m: size,
        src: [i8][n, m] @ DRAM,
        dst: [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass

    @proc
    def bar(x: i8[8, 30, 10] @ DRAM, y: i8[50, 4, 100, 16] @ GEMM_SCRATCH):
        foo(30, 10, x[0, :, :], y[3, 1, 3:33, :])

    assert bar.show_effects() == golden


# Test Alloc
def test_stride_assert10(golden):
    @proc
    def foo(
        n: size,
        m: size,
        src: [i8][n, m] @ DRAM,
        dst: [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass

    @proc
    def bar():
        x: i8[8, 30, 10] @ DRAM
        y: i8[50, 4, 100, 16] @ GEMM_SCRATCH

        foo(30, 10, x[0, :, :], y[3, 1, 3:33, :])

    assert bar.show_effects() == golden


# Test stride arguments
def test_stride_assert11(golden):
    @proc
    def foo(
        n: size,
        m: size,
        s: stride,
        src: [i8][n, m] @ DRAM,
        dst: [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 0) == s
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass

    @proc
    def bar():
        x: i8[8, 30, 10] @ DRAM
        y: i8[50, 4, 100, 16] @ GEMM_SCRATCH

        foo(30, 10, stride(x, 1), x[0, :, :], y[3, 1, 3:33, :])

    assert bar.show_effects() == golden


def test_infereffects(golden):
    @proc
    def foo(n: size, m: size, A: f32[n, m]):
        for i in seq(0, n):
            for j in seq(0, m):
                A[i, j] = 0.0

    foo = foo.partial_eval(10, 20)

    @proc
    def bar(A: f32[10, 20]):
        foo(A)

    assert bar.show_effects() == golden


def test_infereffects2(golden):
    @proc
    def foo(n: size, m: size, A: f32[n, m]):
        for i in seq(0, n):
            for j in seq(0, m):
                A[i, j] = 0.0

    foo = foo.partial_eval(10, 20)
    assert foo.show_effects() == golden


# are we testing a case of an else branch?

# what if effset.pred is None?

# size positivity checks

# making sure asserts are checked
# making sure asserts are used to prove other things

# make sure that effects are getting translated through windowing
# correctly

# check that windowing is always in-bounds
#   note checking above translation is maybe better done for data-races

# stride assert

# test basic commutativity properties exhaustively in combinations
# R,W,+  R,W,+
#
# def foo():
#   x : R
#   for i in seq(0,2):
#       x = 3
#       y = x

# for i in seq(0, n):
#   x[2*i] = x[2*i+1]

# https://en.wikipedia.org/wiki/Jacobi_method
# red black gauss seidel
# https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec14.pdf
# wavefront parallel
"""
# These are write shadow read tests

# This should work!
def test_nest_loop1():
    @proc
    def foo(n : size, m : size, A : i8[n]):
        a : i8
        a = 4.0
        b : i8
        for i in seq(0, n):
            for j in seq(0, m):
                a    = 0.0
            b = a

# This is fine
def test_read_write2():
    @proc
    def foo(n : size, A : i8[n]):
        a : i8[n]
        a[0] = 4.0
        for i in seq(0, n):
            a[0]    = 0.0
            A[i] = a[0]


"""
