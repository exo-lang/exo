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
        dst : [i8][n, 16] @ DRAM,
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

    filename = "test_window_window"
    win.compile_c(TMP_DIR, filename)


def gen_stride_assert():
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

    return stride_assert
def test_stride_assert():
    sa = gen_stride_assert()
    assert type(sa) is Procedure

    filename = "test_window_stride_assert"
    sa.compile_c(TMP_DIR, filename)

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

    filename = "test_window_stmt"
    ws.compile_c(TMP_DIR, filename)

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
    filename = "test_window_proj"
    proj.compile_c(TMP_DIR, filename)


def gen_gemmini_ld():
    @instr("gemmini_extended3_config_ld({dst}.strides[0]*sizeof(float), 1.0f, 0, 0);\n"+
           "gemmini_extended_mvin( {src}.data, ((uint64_t) {dst}.data),"+
                                  "16, {n} );")
    def gemmini_ld(
        n   : size,
        m   : size,
        src : [f32][n, m] @ DRAM,
        dst : [f32][n, 16] @ GEMM_SCRATCH,
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
            xx = x[n - n%16:n, m - m%16:m]
            yy = y[n - n%16:n, m/16, :]
            gemmini_ld(n%16, m%16, xx, yy)

    return ld_2d
def test_ld():
    gemmini_ld = gen_gemmini_ld()
    ld_2d  = gen_ld_2d(gemmini_ld)
    assert type(gemmini_ld) is Procedure
    assert type(ld_2d) is Procedure

    filename = "test_window_ld_2d"
    ld_2d.compile_c(TMP_DIR, filename)


def gen_gemmini_store():
    @instr("gemmini_config_st({src}.strides[0]*sizeof(float));\n"+
           "gemmini_extended_mvout( "+
                "((uint64_t) {dst}.data), {src}.data, 16, {n} );")
    def gemmini_st(
        n   : size,
        m   : size,
        src : [f32][n, 16] @ GEMM_SCRATCH,
        dst : [f32][n, m]  @ DRAM
    ):
        assert n <= 16
        assert m <= 16
        assert stride(dst, 1) == 1
        assert stride(src, 0) == 16
        assert stride(src, 1) == 1

        for i in par(0, n):
            for j in par(0, m):
                dst[i, j] = src[i, j]

    return gemmini_st

def gen_st_2d(gemmini_st):
    @proc
    def st_2d(n : size, m : size, x : f32[n, (m+15)/16, 16] @ GEMM_SCRATCH,
                                  y : f32[n, m] @DRAM):
        # handle all full tile-rows
        for i in par(0, n/16):
            for j in par(0, m/16):
                xx = x[ i*16:i*16+16, j, : ]
                yy = y[ i*16:i*16+16, j*16:j*16+16 ]
                gemmini_st(16, 16, xx, yy)
        # handle last tile row
        if n%16 > 0:
            for j in par(0, m/16):
                xx = x[ n - n%16:n, j, : ]
                yy = y[ n - n%16:n, j*16:j*16+16 ]
                gemmini_st(n%16, 16, xx, yy)
        # handle last tile column
        if m%16 > 0:
            for i in par(0, n/16):
                xx = x[ i*16:i*16+16, m/16, : ]
                yy = y[ i*16:i*16+16, m - m%16:m ]
                gemmini_st(16, m%16, xx, yy)
        # handle last corner
        if n%16 > 0 and m%16 > 0:
            xx = x[n - n%16:n, m/16, :]
            yy = y[n - n%16:n, m - m%16:m]
            gemmini_st(n%16, m%16, xx, yy)

    return st_2d
def test_st():
    gemmini_st = gen_gemmini_store()
    st_2d  = gen_st_2d(gemmini_st)

    filename = "test_window_st_2d"
    st_2d.compile_c(TMP_DIR, filename)

def gen_ld_st_2d(ld_2d, st_2d):
    @proc
    def ld_st_2d(
            N : size,
            M : size,
            A      : f32[N, M]             @ DRAM,
            A_GEMM : f32[N, (M+15)/16, 16] @ GEMM_SCRATCH,
            B      : f32[N, M]             @ DRAM,
            ):

        ld_2d(N, M, A, A_GEMM)
        st_2d(N, M, A_GEMM, B)

    return ld_st_2d
def test_ld_st_2d():
    gemmini_ld = gen_gemmini_ld()
    ld_2d  = gen_ld_2d(gemmini_ld)
    gemmini_st = gen_gemmini_store()
    st_2d  = gen_st_2d(gemmini_st)
    ld_st_2d = gen_ld_st_2d(ld_2d, st_2d)

    filename = "test_window_ld_st_2d"
    ld_st_2d.compile_c(TMP_DIR, filename)


def gen_gemmini_matmul():
    @instr("gemmini_config_ex(WS, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 0);\n"+
           "gemmini_extended_preload("+
                "(uint64_t)({B}.data), (uint64_t)({C}.data), "+
                "{M}, {K}, "+
                "{M}, {N}"+
           ");\n"+
           "gemmini_extended_compute_preloaded("+
                "(uint64_t)({A}.data), ~((uint64_t)0), "+
                "{K}, {N}, "+
                "16, 16"+
           ");")
    def gemmini_matmul(
        N : size,
        M : size,
        K : size,
        A : [f32][N, 16] @ GEMM_SCRATCH,
        B : [f32][K, 16] @ GEMM_SCRATCH,
        C : [f32][N, 16] @ GEMM_SCRATCH
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                for k in par(0,K):
                    C[i, j] += A[i, k] * B[k, j]

    return gemmini_matmul
# C = A * B
def gen_matmul_2d(ld_2d, st_2d, gemmini_matmul):
    @proc
    def matmul_2d(
            N : size,
            M : size,
            K : size,
            A      : f32[N, K]             @ DRAM,
            A_GEMM : f32[N, (K+15)/16, 16] @ GEMM_SCRATCH,
            B      : f32[K, M]             @ DRAM,
            B_GEMM : f32[K, (M+15)/16, 16] @ GEMM_SCRATCH,
            C      : f32[N, M]             @ DRAM,
            C_GEMM : f32[N, (M+15)/16, 16] @ GEMM_SCRATCH):

        # Load A and B to scratchpad
        ld_2d(N, K, A, A_GEMM)
        ld_2d(K, M, B, B_GEMM)

        # handle all full tile-rows
        for i in par(0, N/16):
            for j in par(0, M/16):
                for k in par(0, K/16):
                    aa = A_GEMM[ i*16:i*16+16, k, : ]
                    bb = B_GEMM[ k*16:k*16+16, j, : ]
                    cc = C_GEMM[ i*16:i*16+16, j, : ]

                    gemmini_matmul(16, 16, 16, aa, bb, cc)

        if N%16 > 0:
            for j in par(0, M/16):
                for k in par(0, K/16):
                    aa = A_GEMM[ N-N%16:N, k, : ]
                    bb = B_GEMM[ k*16:k*16+16, j, : ]
                    cc = C_GEMM[ N-N%16:N, j, : ]

                    gemmini_matmul(N%16, 16, 16, aa, bb, cc)

        if M%16 > 0:
            for i in par(0, N/16):
                for k in par(0, K/16):
                    aa = A_GEMM[ i*16:i*16+16, k, : ]
                    bb = B_GEMM[ k*16:k*16+16, M/16, : ]
                    cc = C_GEMM[ i*16:i*16+16, M/16, : ]

                    gemmini_matmul(16, M%16, 16, aa, bb, cc)

        if K%16 > 0:
            for i in par(0, N/16):
                for j in par(0, M/16):
                    aa = A_GEMM[ i*16:i*16+16, K/16, : ]
                    bb = B_GEMM[ K-K%16:K, j, : ]
                    cc = C_GEMM[ i*16:i*16+16, j, : ]

                    gemmini_matmul(16, 16, K%16, aa, bb, cc)

        if N%16 > 0 and K%16 > 0:
            for j in par(0, M/16):
                aa = A_GEMM[ N-N%16:N, K/16, : ]
                bb = B_GEMM[ K-K%16:K, j, : ]
                cc = C_GEMM[ N-N%16:N, j, : ]

                gemmini_matmul(N%16, 16, K%16, aa, bb, cc)

        if N%16 > 0 and M%16 > 0:
            for k in par(0, K/16):
                aa = A_GEMM[ N-N%16:N, k, : ]
                bb = B_GEMM[ k*16:k*16+16, M/16, : ]
                cc = C_GEMM[ N-N%16:N, M/16, : ]

                gemmini_matmul(N%16, M%16, 16, aa, bb, cc)

        if M%16 > 0 and K%16 > 0:
            for i in par(0, N/16):
                aa = A_GEMM[ i*16:i*16+16, K/16, : ]
                bb = B_GEMM[ K-K%16:K, M/16, : ]
                cc = C_GEMM[ i*16:i*16+16, M/16, : ]

                gemmini_matmul(16, M%16, K%16, aa, bb, cc)

        if N%16 > 0 and M%16 > 0 and K%16 > 0:
            aa = A_GEMM[ N-N%16:N, K/16, : ]
            bb = B_GEMM[ K-K%16:K, M/16, : ]
            cc = C_GEMM[ N-N%16:N, M/16, : ]

            gemmini_matmul(N%16, M%16, K%16, aa, bb, cc)

        # Store C_GEMM to C
        st_2d(N, M, C_GEMM, C)

    return matmul_2d

def test_matmul_2d():
    gemmini_ld = gen_gemmini_ld()
    ld_2d  = gen_ld_2d(gemmini_ld)
    gemmini_st = gen_gemmini_store()
    st_2d  = gen_st_2d(gemmini_st)
    gemmini_matmul = gen_gemmini_matmul()
    matmul_2d = gen_matmul_2d(ld_2d, st_2d, gemmini_matmul)

    filename = "test_window_matmul_2d"
    matmul_2d.compile_c(TMP_DIR, filename)
