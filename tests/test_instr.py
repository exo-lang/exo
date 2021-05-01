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
from SYS_ATL.libs.memories import GEMM_SCRATCH, MDRAM
sys.path.append(sys.path[0]+"/.")
from .helper import *
import pytest

#--------------------- GEMMINI MVIN ----------------------
def gen_gemmini_ld():
    @instr("gemmini_extended3_config_ld(4 * {src_m}, 1.0f, 0, 0);\n"+
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
        src : f32[src_n, src_m] @ DRAM,
        dst : f32[dst_n, 16]    @ GEMM_SCRATCH,
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

    """
    { P } S { Q }

    # if x is an order-m tensor (m=1 vector, m=2 matrix, etc.)
    # then let 0 <= i < m index the i-th stride
    # let n be an arbitrary value for the stride in the i-th dimension
    # stride(x, 0, n1) /\ stride(x, 1, n2) /\ .... /\ stride(x, i, ni)
    ------- { stride(x, i, n) }
    xx = x[...] # (if k indices before the ith index are point indices)
    ------- { stride(xx, i-k, n) }

    # e.g.
    ------- { stride(x, 1, n) }
    xx = x[31,:]
    ------- { stride(xx, 0, n) }

    # other basic axiom about strides
    ------- {}
    x : R[n_0,n_1,...,n_m-1]
    ------- { stride(x, 0, n_1*n_2*...) /\ ... /\ stride(x, m-1, 1) }
    return [x[-1], 1]


    x : R[n,m/16,16] @ GEMM_SCRATCH
    def get_alloc_strides(dims=[None, None, 16]):
        strides = [ (len(dims)-1, 1)) ]
        sz      = 1
        for i,d in reversed(enumerate(dims)):
            if d is not None:
                sz = sz * d
                strides.push_front((i-1,d))
            else:
                break

        return strides

    #return [(len(x)-2,16),(2,1)]
        #   (d0, d1, d2)     --> strides = (d1*d2, d2, 1)
        #   [:,1,:]
        #                    --> strides = (d1*d2,1)


    gemmini_ld(n,m,xx,yy)
    """



#--------------------- GEMMINI MVOUT ----------------------
def gen_gemmini_store():
    @instr("gemmini_config_st(4 * {dst_m});\n"+
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
        src : f32[src_n,16]    @ GEMM_SCRATCH,
        dst : f32[dst_n,dst_m] @ DRAM
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
    @instr("gemmini_config_ex(WS, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 0);\n"+
           "gemmini_extended_preload("+
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
        A : f32[nA,16] @ GEMM_SCRATCH,
        B : f32[nB,16] @ GEMM_SCRATCH,
        C : f32[nC,16] @ GEMM_SCRATCH
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


# Matmul test with custom mallocs (DRAM & GEMM)
def gen_matmul_16_malloc(gemmini_ld, gemmini_st, gemmini_matmul):
    @proc
    def matmul_16_malloc(C : f32[16, 16] @ DRAM):
        A : f32[16,16] @ MDRAM
        B : f32[16,16] @ MDRAM
        A_GEMM : f32[16,16] @ GEMM_SCRATCH
        B_GEMM : f32[16,16] @ GEMM_SCRATCH
        C_GEMM : f32[16,16] @ GEMM_SCRATCH

        for i in par(0, 16):
            for j in par(0, 16):
                A[i,j] = 3.0
                B[i,j] = 5.0

        # Load A and B to scratchpad
        gemmini_ld(16, 16, 0, 0, 16, 0, 16, 16, A, A_GEMM)
        gemmini_ld(16, 16, 0, 0, 16, 0, 16, 16, B, B_GEMM)

        gemmini_matmul(16, 16, 16, 0, 0, 0, 16, 16, 16, A_GEMM, B_GEMM, C_GEMM)

        # Store C_GEMM to C
        gemmini_st(16, 0, 16, 16, 0, 0, 16, 16, C_GEMM, C)

    return matmul_16_malloc
def test_matmul_16_malloc():
    gemm_ld = gen_gemmini_ld()
    gemm_st = gen_gemmini_store()
    gemm_matmul = gen_gemmini_matmul()
    matmul_malloc = gen_matmul_16_malloc(gemm_ld, gemm_st, gemm_matmul)

    assert type(gemm_ld) is Procedure
    assert type(gemm_st) is Procedure
    assert type(gemm_matmul) is Procedure
    assert type(matmul_malloc) is Procedure

    filename = "test_matmul_16_malloc"

    matmul_malloc.compile_c(directory, filename)


# matmul test
def gen_matmul_16(gemmini_ld, gemmini_st, gemmini_matmul):
    @proc
    def matmul_16(A      : f32[16, 16] @ DRAM,
                  A_GEMM : f32[16, 16] @ GEMM_SCRATCH,
                  B      : f32[16, 16] @ DRAM,
                  B_GEMM : f32[16, 16] @ GEMM_SCRATCH,
                  C      : f32[16, 16] @ DRAM,
                  C_GEMM : f32[16, 16] @ GEMM_SCRATCH):
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
    def ld_st_16(x : f32[16, 16] @ DRAM,
                 y : f32[16, 16] @ GEMM_SCRATCH,
                 z : f32[16, 16] @ DRAM):
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
    def st_16(x : f32[16, 16] @ GEMM_SCRATCH, y : f32[16, 16] @ DRAM):
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
    def ld_16(x : f32[16, 16] @ DRAM, y : f32[16, 16] @ GEMM_SCRATCH):
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


#----------------- arbitrary size matrix load --------------------
# Assume n%16 == 0 and m%16 == 0
def gen_ld_2d(gemmini_ld):
    @proc
    def ld_2d(n : size, m : size, x : f32[n, m] @DRAM, y : f32[n, m/16, 16] @ GEMM_SCRATCH):
        for i in par(0, n/16):
            for j in par(0, m/16):
                gemmini_ld(16, 16, x[i:i+16, j:j+16], y[i:i+16, j, :])
                #gemmini_ld(n, m, x[i:i+16, j:j+16], y[i:i+16, j])

    return ld_2d

@pytest.mark.skip
def test_load():
    gemm_ld = gen_gemmini_ld()
    ld_2d = gen_ld_2d(gemm_ld)

    assert type(gemm_ld) is Procedure
    assert type(ld_2d) is Procedure

    filename = "test_load"

    ld_2d.compile_c(directory, filename)

#
def gen_ld_2d_2(gemmini_ld):
    @proc
    def ld_2d(n : size, m : size, x : f32[n, m] @DRAM, y : f32[n, m/16, 16] @ GEMM_SCRATCH):
        for i in par(0, n/16):
            for j in par(0, m/16):
                gemmini_ld(16, 16, x[i:i+16, j:j+16], y[i:i+16, j, :])
        gemmini_ld(n%16, m%16, x[n-n%16: , m-m%16: ], y[n-n%16: ,])

    return ld_2d
