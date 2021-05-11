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
from SYS_ATL import proc, instr, Procedure, DRAM
from SYS_ATL.libs.memories import GEMM_SCRATCH, MDRAM
sys.path.append(sys.path[0]+"/.")
from .helper import *

def gen_window():
    @proc
    def window(
        n   : size,
        m   : size,
        src : [i8][n, m] @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert n <= 16
        assert m <= 16

        for i in par(0, n):
            for j in par(0, m):
                dst[i,j] = src[i,j]

    return window
def test_window():
    win = gen_window()
    assert type(win) is Procedure


def gen_stride_assert():
    @proc
    def stride_assert(
        n   : size,
        m   : size,
        src : [i8][n, m] @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert n <= 16
        assert m <= 16
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1

        for i in par(0, n):
            for j in par(0, m):
                dst[i,j] = src[i,j]

    return stride_assert
def test_stride_assert():
    sa = gen_stride_assert()
    assert type(sa) is Procedure


def gen_window_stmt():
    @proc
    def window_stmt(n : size, m : size, x : f32[n, m]):
        y = x[:, 0]
        z : f32[n]
        for i in par(0, n):
            z[i] = y[i]

    return window_stmt
def test_window_stmt():
    ws = gen_window_stmt()
    assert type(ws) is Procedure


def gen_dot():
    @proc
    def dot(m: size, x : [f32][m] , y : [f32][m] , r : f32 ):
        r = 0.0
        for i in par(0, m):
            r += x[i]*y[i]
    return dot
def gen_proj(dot):
    @proc
    def proj(n : size, m : size, x : f32[n,m], y : f32[m,n]):
        assert n > 4
        assert m > 4
        xy : f32
        y2 : f32
        dot(m, x[1,:], y[:,2], xy)
        dot(m, y[:,3], y[:,3], y2)
    return proj
def test_normalize():
    dot  = gen_dot()
    proj = gen_proj(dot)
    assert type(dot) is Procedure
    assert type(proj) is Procedure


def gen_gemmini_ld():
    @instr("gemmini_extended3_config_ld({dst.strides[0]}, 1.0f, 0, 0);\n"+
           "gemmini_extended_mvin( {src.data}, ((uint32_t) {dst.data}),"+
                                  "{m}, {n} );")
    def gemmini_ld(
        n   : size,
        m   : size,
        src : [i8][n, m] @ DRAM,
        dst : [i8][n, 16] @ GEMM_SCRATCH,
    ):
        assert n <= 16
        assert m <= 16
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1

        for i in par(0, n):
            for j in par(0, m):
                dst[i,j] = src[i,j]

    return gemmini_ld
def gen_ld_2d(gemmini_ld):
    @proc
    def ld_2d(n : size, m : size, x : f32[n, m] @DRAM,
                                  y : f32[n, (m+15)/16, 16] @ GEMM_SCRATCH):
        # handle all full tile-rows
        for i in par(0, n/16):
            for j in par(0, m/16):
                xx = x[ i*16:i*16+16, j*16:j*16+16 ]
                yy = y[ i*16:i*16+16, j, : ]
                gemmini_ld(16, 16, xx, yy)
        # handle last tile row
        if n%16 > 0:
            for j in par(0, m/16):
                xx = x[ n - n%16:n, j*16:j*16+16 ]
                yy = y[ n - n%16:n, j, : ]
                gemmini_ld(n%16, 16, xx, yy)
        # handle last tile column
        if m%16 > 0:
            for i in par(0, n/16):
                xx = x[ i*16:i*16+16, m - m%16:m ]
                yy = y[ i*16:i*16+16, m/16, : ]
                gemmini_ld(16, m%16, xx, yy)
        # handle last corner
        if n%16 > 0 and m%16 > 0:
            gemmini_ld(n%16, m%16, x[n - n%16:n, m - m%16:m],
                                   y[n - n%16:n, m/16, :])

        #gemmini_ld(n%16, m%16, x[n-n%16: , m-m%16: ], y[n-n%16: , m-m%16, :])
    return ld_2d
def test_ld():
    gemmini_ld = gen_gemmini_ld()
    ld_2d  = gen_ld_2d(gemmini_ld)
    assert type(gemmini_ld) is Procedure
    assert type(ld_2d) is Procedure
