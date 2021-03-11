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
from SYS_ATL import proc, instr, Procedure, DRAM, GEMM_SCRATCH
sys.path.append(sys.path[0]+"/.")
from .helper import *
import pytest

# TODO: Add predicate to args
# Otherwise effectcheck is not happy about dst and src windowing
# depending on arbitrary src_r and dst_r
# Effect on src READ:
#   0 <= i < row_dim, 0 <= j < col_dim
#   0 <= src_r + i < src_n, 0 <= src_c + j < src_m
# This is not valid..
def gen_gemmini_ld():
    @instr("gemmini_extended3_config_ld(4 * {src_m}, 1.0f, 0, 0);\n"+
           "gemmini_extended_mvin( "+
                "({src}) + ({src_r})*({src_m}) + ({src_c}),"+
                "({dst}) + ({dst_r}) );")
    def gemmini_ld(
        src_n : size,
        src_m : size,
        src_r : index,
        src_c : index,
        dst_n : size,
        dst_r : index,
        col_dim : size,
        row_dim : size,
        src : F32[src_n, src_m] @ DRAM,
        dst : F32[dst_n, 16]    @ GEMM_SCRATCH,
    ):
        for i in par(0, row_dim):
            for j in par(0, col_dim):
                dst[dst_r + i, j] = src[src_r + i, src_c + j]
        
    return gemmini_ld

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
