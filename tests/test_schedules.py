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

@pytest.mark.skip
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
    # should be bar(5, y, x)
    print(foo)

@pytest.mark.skip
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
    # should be bar(5, y[5:10,2:7], x[3:8,1:6])
    print(foo)

@pytest.mark.skip
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

    foo = foo.replace(bar, "for j in _ : _")
    # should be simd_add4(z[i:i+4], x[i:i+4], y[i:i+4])
    print(foo)
