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

# This should work
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
    def bar():
        x : i8[30,10] @ DRAM
        y : i8[30,16] @ GEMM_SCRATCH
        foo(30,10, x, y)

# This should not work
def test_stride_assert2():
    with pytest.raises(TypeError,
                       match='Could not verify stride assert in foo'):
        @proc
        def foo(
            n   : size,
            m   : size,
            src : [i8][n, m]  @ DRAM,
            dst : [i8][n, 16] @ GEMM_SCRATCH,
        ):
            assert stride(src, 1) == 1
            assert stride(dst, 0) == 3
            assert stride(dst, 1) == 1
            pass
        @proc
        def bar():
            x : i8[30,10] @ DRAM
            y : i8[30,16] @ GEMM_SCRATCH
            foo(30,10, x, y)

# This should work
def test_stride_assert3():
    @proc
    def foo(n : size, m : size,
            x : [i8][n, m] @ DRAM, y : [i8][10, 9]):
        assert stride(x, 1) == 1
        assert stride(y, 0) == 9
        assert stride(y, 1) == 1
        pass

# This should not work
def test_stride_assert4():
    with pytest.raises(TypeError,
                       match='Could not verify stride assert in foo'):
        @proc
        def foo(n : size, m : size,
                x : [i8][n, m] @ DRAM, y : [i8][10, 9]):
            assert stride(x, 1) == 1
            assert stride(y, 0) == 5
            assert stride(y, 1) == 1
            pass

# do sizes match

# something that works but is on the edge of working

# we added complex stuff about division and modulo
# is that working?

# are we testing a case of an else branch?

# what if effset.pred is None?

# size positivity checks

# making sure asserts are checked
# making sure asserts are used to prove other things

# make sure that effects are getting translated through windowing
# correctly

# check that windowing is always in-bounds
#   note checking above translation is maybe better done for data-races

# Data Race? Yes
# for i in par(0,n):
#     if i+1 < n:
#         x[i,i] = x[i+1,i+1]

# Data Race? No
# for i in par(0,n):
#     if i+1 < n:
#         x[i,i] = x[i+1,i]

# Data Race? No
# y = x[1:,:]
# for i in par(0,n):
#     if i+1 < n:
#         x[i,i] = y[i,i]

# one big issue is aliasing in sub-procedure arguments
# def foo(n : size, x : [R][n,n], y : [R][n,n]):
#   for i in par(0,n):
#     if i+1 < n:
#         x[i,i] = y[i,i]

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

