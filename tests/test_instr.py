from __future__ import annotations
import subprocess
import os
import ctypes
from ctypes import *
import numpy as np
import sys
from PIL import Image
import scipy.stats as st
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, instr, Procedure, DRAM
from SYS_ATL.libs.memories import GEMM_SCRATCH
sys.path.append(sys.path[0]+"/.")
from .helper import *
import pytest

#--------------------- GEMMINI MVIN ----------------------
def gen_gemmini_ld():
    @instr("gemmini_extended3_config_ld(1 * {src_m}, 1.0f, 0, 0);\n"+
           "gemmini_extended_mvin( "+
                "{src} + {src_r}*{src_m} + {src_c},"+
                "((uint32_t) {dst}) + {dst_r}, {col_dim}, {row_dim} );")
    def gemmini_ld(
        src_n : size,
        src_m : size,
        src_r : index,
        src_c : index,
        dst_n : size,
        dst_r : index,
        col_dim : size,
        row_dim : size,
        src : INT8[src_n, src_m] @ DRAM,
        dst : INT8[dst_n, 16]    @ GEMM_SCRATCH,
    ):
        assert row_dim <= 16
        assert col_dim <= 16
        assert 0 <= src_r < src_n
        assert 0 <= src_c < src_m
        assert 0 <= src_r + row_dim <= src_n
        assert 0 <= src_c + col_dim <= src_m
        assert 0 <= dst_r < dst_n
        assert 0 <= dst_r + row_dim <= dst_n

        for i in par(0, row_dim):
            for j in par(0, col_dim):
                dst[dst_r + i, j] = src[src_r + i, src_c + j]
        
    return gemmini_ld

#--------------------- GEMMINI MVOUT ----------------------
def gen_gemmini_store():
    @instr("gemmini_config_st(1 * {dst_m});\n"+
           "gemmini_extended_mvout( "+
                "((uint32_t) {dst}) + {dst_r}*{dst_m} + {dst_c},"+
                "{src} + {src_r} , {col_dim}, {row_dim} );")
    def gemmini_st(
        src_n : size,
        src_r : index,
        dst_n : size,
        dst_m : size,
        dst_r : index,
        dst_c : index,
        col_dim : size,
        row_dim : size,
        src : INT8[src_n,16]    @ GEMM_SCRATCH,
        dst : INT8[dst_n,dst_m] @ DRAM
    ):
        assert row_dim <= 16
        assert col_dim <= 16
        assert 0 <= src_r < src_n
        assert 0 <= src_r + row_dim <= src_n
        assert 0 <= dst_r < dst_n
        assert 0 <= dst_c < dst_m
        assert 0 <= dst_r + row_dim <= dst_n
        assert 0 <= dst_c + col_dim <= dst_m

        for i in par(0,row_dim):
            for j in par(0,col_dim):
                dst[dst_r + i, dst_c + j] = src[src_r + i, j]

    return gemmini_st


#--------------------- GEMMINI MATMUL ----------------------
def gen_gemmini_matmul():
    @instr("gemmini_extended_preload("+
                "(uint32_t)({B} + {B_row_off}), (uint32_t)({C} + {C_row_off}), "+
                "{M}, {K}, "+
                "{M}, {N}"+
           ");\n"+
           "gemmini_extended_compute_preloaded("+
                "(uint32_t)({A} + {A_row_off}), ~((uint32_t)0), "+
                "{K}, {N}, "+
                "16, 16"+
           ");")
    def gemmini_matmul(
        N : size,
        M : size,
        K : size,
        A_row_off : index,
        B_row_off : index,
        C_row_off : index,
        nA : size,
        nB : size,
        nC : size,
        A : INT8[nA,16] @ GEMM_SCRATCH,
        B : INT8[nB,16] @ GEMM_SCRATCH,
        C : INT8[nC,16] @ GEMM_SCRATCH
    ):
        assert 1 <= N <= 16
        assert 1 <= M <= 16
        assert 1 <= K <= 16
        assert 0 <= A_row_off < nA
        assert 0 <= B_row_off < nB
        assert 0 <= C_row_off < nC
        assert 0 <= A_row_off + N <= nA
        assert 0 <= B_row_off + K <= nB
        assert 0 <= C_row_off + N <= nC
        assert N <= nC
        assert N <= nA
        assert K <= nB

        for i in par(0,N):
            for j in par(0,M):
                C[C_row_off+i, j] = 0.0
                for k in par(0,K):
                    C[C_row_off+i, j] += A[A_row_off+i, k] * B[B_row_off+k, j]

    return gemmini_matmul



def gen_matmul_16(gemmini_ld, gemmini_st, gemmini_matmul):
    @proc
    def matmul_16(A      : INT8[16, 16] @ DRAM,
                  A_GEMM : INT8[16, 16] @ GEMM_SCRATCH,
                  B      : INT8[16, 16] @ DRAM,
                  B_GEMM : INT8[16, 16] @ GEMM_SCRATCH,
                  C      : INT8[16, 16] @ DRAM,
                  C_GEMM : INT8[16, 16] @ GEMM_SCRATCH):
        # Load A and B to scratchpad
        gemmini_ld(16, 16, 0, 0, 16, 0, 16, 16, A, A_GEMM)
        gemmini_ld(16, 16, 0, 0, 16, 0, 16, 16, B, B_GEMM)

        gemmini_matmul(16, 16, 16, 0, 0, 0, 16, 16, 16, A_GEMM, B_GEMM, C_GEMM)

        # Store C_GEMM to C
        gemmini_st(16, 0, 16, 16, 0, 0, 16, 16, C_GEMM, C)

    return matmul_16
def test_matmul_16():
    gemm_ld = gen_gemmini_ld()
    gemm_st = gen_gemmini_store()
    gemm_matmul = gen_gemmini_matmul()
    matmul = gen_matmul_16(gemm_ld, gemm_st, gemm_matmul)

    assert type(gemm_ld) is Procedure
    assert type(gemm_st) is Procedure
    assert type(gemm_matmul) is Procedure
    assert type(matmul) is Procedure

    filename = "test_matmul_16"

    matmul.compile_c(directory, filename)



def gen_ld_st_16(gemmini_ld, gemmini_st):
    @proc
    def ld_st_16(x : INT8[16, 16] @ DRAM,
                 y : INT8[16, 16] @ GEMM_SCRATCH,
                 z : INT8[16, 16] @ DRAM):
        gemmini_ld(16, 16, 0, 0, 16, 0, 16, 16, x, y)
        gemmini_st(16, 0, 16, 16, 0, 0, 16, 16, y, z)

    return ld_st_16
def test_ld_st_16():
    gemm_ld = gen_gemmini_ld()
    gemm_st = gen_gemmini_store()
    ld_st_16 = gen_ld_st_16(gemm_ld, gemm_st)

    assert type(gemm_ld) is Procedure
    assert type(gemm_st) is Procedure
    assert type(ld_st_16) is Procedure

    filename = "test_ld_st_16"

    ld_st_16.compile_c(directory, filename)



def gen_st_16(gemmini_st):
    @proc
    def st_16(x : INT8[16, 16] @ GEMM_SCRATCH, y : INT8[16, 16] @ DRAM):
        gemmini_st(16, 0, 16, 16, 0, 0, 16, 16, x, y)

    return st_16
def test_store_16():
    gemm_st = gen_gemmini_store()
    st_16 = gen_st_16(gemm_st)

    filename = "test_store_16"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(st_16))
    f_pretty.close()

    st_16.compile_c(directory, filename)




def gen_ld_16(gemmini_ld):
    @proc
    def ld_16(x : INT8[16, 16] @ DRAM, y : INT8[16, 16] @ GEMM_SCRATCH):
        gemmini_ld(16, 16, 0, 0, 16, 0, 16, 16, x, y)

    return ld_16
def test_load_16():
    gemm_ld = gen_gemmini_ld()
    ld_16 = gen_ld_16(gemm_ld)

    assert type(gemm_ld) is Procedure
    assert type(ld_16) is Procedure

    filename = "test_load_16"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(ld_16))
    f_pretty.close()

    ld_16.compile_c(directory, filename)


#----------------- arbitrary size matrix multiply --------------------
# Assume n%16 == 0 and m%16 == 0
# r = n*m/16
# w = (i+1)*j*16 #TODO: How to handle windowing?
def gen_ld_2d(gemmini_ld):
    @proc
    def ld_2d(n : size, m : size, r : size, w : index, x : F32[n, m], y : F32[r, 16]):
        for i in par(0, n/16):
            for j in par(0, m/16):
                gemmini_ld(n, m, i*16, j*16, r, w, 16, 16, x, y)

    return ld_2d

@pytest.mark.skip
def test_load():
    # TODO: How to inline the instruction?
    # LoopIR.Call? Or add scheduling directive?
    gemm_ld = gen_gemmini_ld()
    ld_2d = gen_ld_2d(gemm_ld)
    #ld_2d = ld_2d.inline("gemmini_ld(_,_,_,_,_,_,_,_,_,_)")

    assert type(gemm_ld) is Procedure
    assert type(ld_2d) is Procedure

    filename = "test_load"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(ld_2d))
    f_pretty.close()

    ld_2d.compile_c(directory, filename)
