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
    print(bar)


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
    print(foo)
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
    print(foo)
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
    # should be simd_add4(z[4*i:4*i+4], x[4*i:4*i+4], y[4*i:4*i+4])
    print(foo)

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
    # should be bar(50, x[:, 0], x[:, 1])
    print(foo)

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
    print(foo)
    assert 'bar(5, y, x)' in str(foo)


#@pytest.mark.skip()
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
    # should be load(16, 16, A[:, 16*k : 16*k + 16], a[:,:])
    print(bar)
