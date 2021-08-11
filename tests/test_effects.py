from __future__ import annotations
import ctypes
from ctypes import *
import os
import sys
import subprocess
import numpy as np
import scipy.stats as st
import pytest
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, Procedure, DRAM
from SYS_ATL.libs.memories import GEMM_SCRATCH
sys.path.append(sys.path[0]+"/.")
from .helper import *

# ------- Effect check tests ---------

# Effects + schedule tests
# Handle false dependency.
def test_different_id1():
    @proc
    def foo(n : size, A : i8[n]):
        a : i8
        for i in par(0, n):
            a = 0.0

    foo.check_effects()

def test_different_id2():
    @proc
    def foo(n : size):
        for ihi in par(0, n):
            a: i8 @ DRAM
            for ilo in par(0, 16):
                a = 0.0
        if False:
            b: i8 @ DRAM
            b = 0.0

    foo.check_effects()

# For-loop bound non-negative check tests
@proc
def test_good_bound1(n: size, dst: R[n] @ DRAM, src: R[n] @ DRAM):
    for i in par(0, (n + 7) / 8):
        if n - 8 * i >= 8:
            pass
        else:
            for j in par(0, n - 8 * i):
                dst[8 * i + j] = src[8 * i + j]

def test_bad_bound1():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
        @proc
        def bar():
            for i in par(0, -2):
                pass

def test_bad_bound2():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
        @proc
        def bar(n : size):
            for i in par(0, n):
                for j in par(0, i-1):
                    pass

def test_bad_bound3():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
        @proc
        def bar(n : size):
            for i in par(0, n):
                for j in par(0, i-n):
                    pass

def test_bad_bound4():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
        @proc
        def bar(n: size, dst: R[n] @ DRAM, src: R[n] @ DRAM):
            for i in par(0, (n + 7) / 8):
                if n - 8 * i >= 8:
                    pass
                else:
                    for j in par(0, n - 9 * i):
                        dst[8 * i + j] = src[8 * i + j]



def test_bad_access1():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
        @proc
        def bad_access1(n : size, m : size,
                        x : R[n,m], y: R[n,m], res : R[n,m]):
            rloc : R[m]
            for i in par(0,m):
                xloc : R[m]
                yloc : R[m]
                for j in par(0,n):
                    xloc[j] = x[i,j]
                for j in par(0,m):
                    yloc[j] = y[i,j]
                for j in par(0,m):
                    rloc[j] = xloc[j] + yloc[j]
                for j in par(0,m):
                    res[i,j] = rloc[j]

def test_bad_access2():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
        @proc
        def bad_access2(n : size, m : size,
                       x : R[n,m], y: R[n,m] @ DRAM, res : R[n,m] @ DRAM):
            rloc : R[m]
            for i in par(0,n):
                xloc : R[m]
                yloc : R[m]
                for j in par(0,m):
                    xloc[j] = x[i+1,j]
                for j in par(0,m):
                    yloc[j] = y[i,j]
                for j in par(0,m):
                    rloc[j] = xloc[j] + yloc[j-1]
                for j in par(0,m):
                    res[i,j] = rloc[j]

def test_bad_access3():
    with pytest.raises(TypeError,
                       match='x2 is read out-of-bounds'):
        @proc
        def foo():
            x2 : R[1]
            huga : R
            huga = x2[100]

def test_assert1():
    with pytest.raises(TypeError,
                       match='Could not verify assertion'):
        @proc
        def foo(n :size, x : i8[n,n]):
            assert n == 1
            pass
        @proc
        def bar():
            z : i8[3, 3]
            foo(3, z)

def test_size1():
    with pytest.raises(TypeError,
                       match='type-shape of calling argument'):
        @proc
        def foo(x : i8[3,3]):
            pass
        @proc
        def bar():
            z : i8[3, 4]
            foo(z)

# Data Race? Yes
def test_race1():
    with pytest.raises(TypeError,
                       match='data race conflict with statement'):
        @proc
        def foo(n : size, x : R[n,n]):
            for i in par(0,n):
                if i+1 < n:
                    x[i,i] = x[i+1,i+1]

# Data Race? No
def test_race2():
    @proc
    def foo(n : size, x : R[n,n]):
        for i in par(0,n):
            if i+1 < n:
                x[i,i] = x[i+1,i]

# Data Race? No
def test_race3():
    @proc
    def foo(n : size, x : R[n,n]):
        y = x[1:,:]
        for i in par(0,n):
            if i+1 < n:
                x[i,i] = y[i,i]

# one big issue is aliasing in sub-procedure arguments
# TODO: Think about this behaviour
def test_race4():
    @proc
    def foo(n : size, x : [R][n,n], y : [R][n,n]):
        for i in par(0,n):
            if i+1 < n:
                x[i,i] = y[i,i]
                
    @proc
    def bar(n : size, z : R[n,n]):
        foo(n, z, z)

def test_div1():
    @proc
    def foo(n : size):
        assert n == 3
        pass

    @proc
    def bar():
        foo(10/3)

def test_mod1():
    @proc
    def foo(n : size):
        assert n == 1
        pass

    @proc
    def bar():
        foo(10%3)

# Callee has a window but caller has a tensor case
def test_stride_assert1():
    @proc
    def foo(
        n   : size,
        m   : size,
        src : [i8][n, m]  @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass
    @proc
    def bar(x : i8[30,10] @ DRAM, y : i8[30,16] @ GEMM_SCRATCH):
        foo(30,10, x, y)

# Both callee and caller has a window case
def test_stride_assert2():
    with pytest.raises(TypeError,
                       match='Could not verify assertion'):
        @proc
        def foo(
            n   : size,
            m   : size,
            src : [i8][n, m]  @ DRAM,
            dst : [i8][n, 16] @ GEMM_SCRATCH,
        ):
            assert stride(src, 1) == 1
            assert stride(dst, 0) == 16
            assert stride(dst, 1) == 1
            pass
        @proc
        def bar(x : i8[30,10] @ DRAM, y : [i8][30,16] @ GEMM_SCRATCH):
            foo(30,10, x, y)

# Both callee and caller has a window case, but with top level assert
def test_stride_assert3():
    @proc
    def foo(
        n   : size,
        m   : size,
        src : [i8][n, m]  @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass
    @proc
    def bar(x : i8[30,10] @ DRAM, y : [i8][30,16] @ GEMM_SCRATCH):
        assert stride(y, 0) == 16
        assert stride(y, 1) == 1
        foo(30,10, x, y)

# callee is Tensor case and caller is a window case.
# this will be an error because we don't know anything about incoming stride
def test_stride_assert4():
    with pytest.raises(TypeError,
                       match='Could not verify assertion'):
        @proc
        def foo(
            n   : size,
            m   : size,
            src : i8[n, m]  @ DRAM,
            dst : i8[n, 16] @ GEMM_SCRATCH,
        ):
            assert stride(src, 1) == 1
            assert stride(dst, 0) == 16
            assert stride(dst, 1) == 1
            pass
        @proc
        def bar(x : [i8][30,10] @ DRAM, y : [i8][30,16] @ GEMM_SCRATCH):
            foo(30,10, x, y)

# Tensor with wrong size and stride
def test_stride_assert5():
    with pytest.raises(TypeError,
                       match='is always unsatisfiable'):
        @proc
        def bar(x : i8[30,10] @ DRAM, y : i8[30,16] @ GEMM_SCRATCH):
            assert stride(x, 0) == 9
            pass

# Tensor asserting last dimension is fine
def test_stride_assert6():
    @proc
    def bar(n : size, m : size, x : i8[n,m] @ DRAM):
        assert stride(x, 1) == 1
        pass

# Test Tensor having insufficient information (sizes)
def test_stride_assert7():
    # with changes, this should not trigger an error
    # since it might be true, even if it is highly unlikely
    # i.e. this would have to be always called with m == 10
    @proc
    def bar(n : size, m : size, x : i8[n,m] @ DRAM):
        assert stride(x, 0) == 10
        pass

# Test Windowstmt
def test_stride_assert8():
    @proc
    def foo(
        n   : size,
        m   : size,
        src : [i8][n, m]  @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass
    @proc
    def bar(x : i8[8,30,10] @ DRAM, y : i8[50, 4, 100,16] @ GEMM_SCRATCH):
        xx = x[0,:,:]
        yy = y[3,1, 3:33,:]

        foo(30,10, xx, yy)

# Test Windowexpr within call arg
def test_stride_assert9():
    @proc
    def foo(
        n   : size,
        m   : size,
        src : [i8][n, m]  @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass
    @proc
    def bar(x : i8[8,30,10] @ DRAM, y : i8[50, 4, 100,16] @ GEMM_SCRATCH):
        foo(30,10, x[0,:,:], y[3,1, 3:33,:])

# Test Alloc
def test_stride_assert10():
    @proc
    def foo(
        n   : size,
        m   : size,
        src : [i8][n, m]  @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass
    @proc
    def bar():
        x : i8[8,30,10] @ DRAM
        y : i8[50, 4, 100,16] @ GEMM_SCRATCH

        foo(30,10, x[0,:,:], y[3,1, 3:33,:])

# Test stride arguments
def test_stride_assert11():
    @proc
    def foo(
        n   : size,
        m   : size,
        s   : stride,
        src : [i8][n, m]  @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert stride(src, 0) == s
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1
        pass
    @proc
    def bar():
        x : i8[8,30,10] @ DRAM
        y : i8[50, 4, 100,16] @ GEMM_SCRATCH

        foo(30,10, stride(x,1), x[0,:,:], y[3,1, 3:33,:])


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
#   for i in par(0,2):
#       x = 3
#       y = x

# for i in par(0, n):
#   x[2*i] = x[2*i+1]

# https://en.wikipedia.org/wiki/Jacobi_method
# red black gauss seidel
# https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec14.pdf
# wavefront parallel

