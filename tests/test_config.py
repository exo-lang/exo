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
from SYS_ATL import proc, Procedure, DRAM, config
from SYS_ATL.libs.memories import GEMM_SCRATCH
sys.path.append(sys.path[0]+"/.")
from .helper import *

# ------- Configuration tests ---------

def new_config_f32():
  @config
  class ConfigAB:
    a : f32
    b : f32

  return ConfigAB

def test_basic_config():
  ConfigAB = new_config_f32()

  @proc
  def foo(x : f32):
    ConfigAB.a  = 32.0
    x           = ConfigAB.a


"""
def new_CONFIG_2f32():



@proc
def test_make_config(n: size, dst: R[n] @ DRAM, src: R[n] @ DRAM):
    for i in par(0, (n + 7) / 8):
        if n - 8 * i >= 8:
            pass
        else:
            for j in par(0, n - 8 * i):
                dst[8 * i + j] = src[8 * i + j]





_gemm_config_ld_i8   = ("gemmini_extended3_config_ld({src_stride}, "+
                        "{scale}[0], 0, 0);\n"+
@instr(_gemm_config_ld_i8)
def config_ld_i8(
    scale : f32,
    src_stride : stride
):
    CONFIG.scale = scale
    CONFIG.src_stride = src_stride

_gemm_do_ld_i8   = ("gemmini_extended_mvin( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_do_ld_i8)
def do_ld_i8(
    n     : size,
    m     : size,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(src, 0) == CONFIG.src_stride
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1
    #assert gemmini.stride == stride(src, 0)

    for i in par(0, n):
        for j in par(0, m):
            tmp : f32
            tmp      = src[i,j]
            tmp      = tmp * CONFIG.scale
            dst[i,j] = tmp #no clamping

_gemm_ld_i8   = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                 "{scale}[0], 0, 0);\n"+
                 "gemmini_extended_mvin( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_ld_i8)
def ld_i8(
    n     : size,
    m     : size,
    scale : f32,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1
    #assert gemmini.stride == stride(src, 0)

    for i in par(0, n):
        for j in par(0, m):
            tmp : f32
            tmp      = src[i,j]
            tmp      = tmp * scale
            dst[i,j] = tmp #no clamping

@proc
def ld_i8_v2(
    n     : size,
    m     : size,
    scale : f32,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    config_ld_i8(scale, stride(src, 0))
    do_ld_i8(n, m, src, dst)







@instr("write_config({a}, {b})")
def set_real_config( a : f32, b : f32):
    REAL_CONFIG.a = a 
    REAL_CONFIG.b = b 

# st_config_ex
@proc
def set_config_a( a : f32 ):
    DUMMY_CONFIG.a = a 
    set_real_config(DUMMY_CONFIG.a, DUMMY_CONFIG.b )

# matmul_config_ex
@proc
def set_config_b( b : f32 ):
    DUMMY_CONFIG.b = b 
    set_real_config(DUMMY_CONFIG.a, DUMMY_CONFIG.b )

@instr("mvin...")
def st_i8_v2( ..., stride : stride, ...)
    set_config_a(stride)
    do_st_i8(...)
"""

