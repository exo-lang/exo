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
sys.path.append(sys.path[0]+"/.")
from .helper import *

# ------- Window related tests ---------

def test_window():
    @proc
    def window(
        n   : size,
        m   : size,
        src : [i8][n, m] @ DRAM,
        dst : [i8][n, 16] @ DRAM,
    ):
        assert n <= 16
        assert m <= 16

        for i in par(0, n):
            for j in par(0, m):
                dst[i,j] = src[i,j]

    assert type(window) is Procedure

    filename = "test_window_window"
    window.compile_c(TMP_DIR, filename)

def test_stride_assert():
    @proc
    def stride_assert(
        n   : size,
        m   : size,
        src : [i8][n, m] @ DRAM,
        dst : [i8][n, 16] @ DRAM,
    ):
        assert n <= 16
        assert m <= 16
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1

        for i in par(0, n):
            for j in par(0, m):
                dst[i,j] = src[i,j]

    assert type(stride_assert) is Procedure

    filename = "test_window_stride_assert"
    stride_assert.compile_c(TMP_DIR, filename)

def test_window_stmt():
    @proc
    def window_stmt(n : size, m : size, x : f32[n, m]):
        y = x[:, 0]
        z : f32[n]
        for i in par(0, n):
            z[i] = y[i]

    assert type(window_stmt) is Procedure

    filename = "test_window_stmt"
    window_stmt.compile_c(TMP_DIR, filename)

def test_normalize():
    @proc
    def dot(m: size, x : [f32][m] , y : [f32][m] , r : f32 ):
        r = 0.0
        for i in par(0, m):
            r += x[i]*y[i]

    @proc
    def proj(n : size, m : size, x : f32[n,m], y : f32[m,n]):
        assert n > 4
        assert m > 4
        xy : f32
        y2 : f32
        dot(m, x[1,:], y[:,2], xy)
        dot(m, y[:,3], y[:,3], y2)

    assert type(dot) is Procedure
    assert type(proj) is Procedure
    filename = "test_window_proj"
    proj.compile_c(TMP_DIR, filename)


