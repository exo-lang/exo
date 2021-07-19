from __future__ import annotations

#from ctypes import *
#import os
#import subprocess
#import numpy as np
#import scipy.stats as st
#import os

import sys
sys.path.append(sys.path[0]+"/..")
sys.path.append(sys.path[0]+"/.")
from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import GEMM_SCRATCH, GEMM_ACCUM, MDRAM
from .gemmini import *
from .harness_gemmini import ENV, GemmTestBuilder
import pytest



# --------------------------------------------------------------------------- #
#   Individual Load / Store / Zero Tests
# --------------------------------------------------------------------------- #


def test_ldst_i8_16():
  T = GemmTestBuilder('ldst_i8_16')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])

  @proc
  def ldst_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
    tmp : i8[16,16] @ GEMM_SCRATCH
    scale : f32
    scale = 1.0
    acc   : i8
    acc   = 1.0
    ld_i8(16,16, scale, x, tmp)
    st_i8(16,16, scale, acc, tmp, y)
  T.add_proc(ldst_i8_16)

  T.alloc_dram_2i8('x', 16, 16, 'i+j')
  T.alloc_dram_2i8('y', 16, 16, '0')

  T.add_body(['ldst_i8_16(x, y);',
              '',
              'gemmini_fence();',
              '',
              'if(check_eq_2i8(16,16, x, y)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (x):\\n");',
              '    print_2i8(16,16, x);',
              '    printf("Computed Roundtrip (y):\\n");',
              '    print_2i8(16,16, y);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()


def test_ldst_acc_i8_16():
  T = GemmTestBuilder('ldst_acc_i8_16')
  T.add_body(['gemm_acc_init_mem();',
              'gemmini_flush(0);',
              ''])

  @proc
  def ldst_acc_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
    tmp : i32[16,16] @ GEMM_ACCUM
    scale : f32
    scale = 1.0
    acc   : i8
    acc   = 1.0
    ld_acc_i8(16,16, scale, x, tmp)
    st_acc_i8(16,16, scale, acc, tmp, y)
  T.add_proc(ldst_acc_i8_16)

  T.alloc_dram_2i8('x', 16, 16, 'i+j')
  T.alloc_dram_2i8('y', 16, 16, '0')

  T.add_body(['ldst_acc_i8_16(x, y);',
              '',
              'gemmini_fence();',
              '',
              'if(check_eq_2i8(16,16, x, y)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (x):\\n");',
              '    print_2i8(16,16, x);',
              '    printf("Computed Roundtrip (y):\\n");',
              '    print_2i8(16,16, y);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()


def test_ldst_i8_odd():
  T = GemmTestBuilder('ldst_i8_odd')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])

  @proc
  def ldst_i8_odd( x : i8[15,7] @ DRAM, y : i8[15,7] @ DRAM ):
    tmp : i8[15,16] @ GEMM_SCRATCH
    scale : f32
    scale = 1.0
    acc   : i8
    acc   = 1.0
    ld_i8(15,7, scale, x, tmp)
    st_i8(15,7, scale, acc, tmp, y)
  T.add_proc(ldst_i8_odd)

  T.alloc_dram_2i8('x', 15, 7, 'i+j')
  T.alloc_dram_2i8('y', 15, 7, '0')

  T.add_body(['ldst_i8_odd(x, y);',
              '',
              'gemmini_fence();',
              '',
              'if(check_eq_2i8(15,7, x, y)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (x):\\n");',
              '    print_2i8(15,7, x);',
              '    printf("Computed Roundtrip (y):\\n");',
              '    print_2i8(15,7, y);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()


def test_ldst_acc_i8_acc():
  T = GemmTestBuilder('ldst_acc_i8_acc')
  T.add_body(['gemm_acc_init_mem();',
              'gemmini_flush(0);',
              ''])

  @proc
  def ldst_acc_i8_acc( x : i8[7,13] @ DRAM, y : i8[7,13] @ DRAM ):
    tmp : i32[7,16] @ GEMM_ACCUM
    scale : f32
    scale = 1.0
    acc   : i8
    acc   = 1.0
    ld_acc_i8(7,13, scale, x, tmp)
    st_acc_i8(7,13, scale, acc, tmp, y)
  T.add_proc(ldst_acc_i8_acc)

  T.alloc_dram_2i8('x', 7, 13, 'i+j')
  T.alloc_dram_2i8('y', 7, 13, '0')

  T.add_body(['ldst_acc_i8_acc(x, y);',
              '',
              'gemmini_fence();',
              '',
              'if(check_eq_2i8(7,13, x, y)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (x):\\n");',
              '    print_2i8(7,13, x);',
              '    printf("Computed Roundtrip (y):\\n");',
              '    print_2i8(7,13, y);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()


def test_ldzerost_i8_16():
  T = GemmTestBuilder('ldzerost_i8_16')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])

  @proc
  def ldzerost_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
    tmp : i8[16,16] @ GEMM_SCRATCH
    scale : f32
    scale = 1.0
    acc   : i8
    acc   = 1.0
    ld_i8(16,16, scale, x, tmp)
    zero_i8(8,8, tmp[4:12,:])
    st_i8(16,16, scale, acc, tmp, y)
  T.add_proc(ldzerost_i8_16)

  T.alloc_dram_2i8('x', 16, 16, 'i+j')
  T.alloc_dram_2i8('y', 16, 16, '0')

  T.add_body(['ldzerost_i8_16(x, y);',
              '',
              'gemmini_fence();',
              '',
              '// zero out the same region of x',
              'for(int i=4; i<12; i++) {',
              '    for(int j=0; j<8; j++) {'
              '        x[i*16 + j] = 0;'
              '}}',
              '',
              'if(check_eq_2i8(16,16, x, y)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (x):\\n");',
              '    print_2i8(16,16, x);',
              '    printf("Computed Roundtrip (y):\\n");',
              '    print_2i8(16,16, y);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()


def test_ldzerost_acc_i8_16():
  T = GemmTestBuilder('ldzerost_acc_i8_16')
  T.add_body(['gemm_acc_init_mem();',
              'gemmini_flush(0);',
              ''])

  @proc
  def ldzerost_acc_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
    tmp : i32[16,16] @ GEMM_ACCUM
    scale : f32
    scale = 1.0
    acc   : i8
    acc   = 1.0
    ld_acc_i8(16,16, scale, x, tmp)
    zero_acc_i32(8,8, tmp[4:12,:])
    st_acc_i8(16,16, scale, acc, tmp, y)
  T.add_proc(ldzerost_acc_i8_16)

  T.alloc_dram_2i8('x', 16, 16, 'i+j')
  T.alloc_dram_2i8('y', 16, 16, '0')

  T.add_body(['ldzerost_acc_i8_16(x, y);',
              '',
              'gemmini_fence();',
              '',
              '// zero out the same region of x',
              'for(int i=4; i<12; i++) {',
              '    for(int j=0; j<8; j++) {'
              '        x[i*16 + j] = 0;'
              '}}',
              '',
              'if(check_eq_2i8(16,16, x, y)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (x):\\n");',
              '    print_2i8(16,16, x);',
              '    printf("Computed Roundtrip (y):\\n");',
              '    print_2i8(16,16, y);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()


# --------------------------------------------------------------------------- #
#   Individual MatMul Tests
# --------------------------------------------------------------------------- #



def test_matmul_i8_ones_16():
  T = GemmTestBuilder('matmul_i8_ones_16')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])


  T.alloc_dram_2i8('x', 16, 16, '1')
  T.alloc_dram_2i8('y', 16, 16, '1')
  T.alloc_dram_2i8('z', 16, 16, '16') # expected result
  T.alloc_dram_2i8('res', 16, 16, '0')

  @proc
  def matmul_i8_ones_16(
    x : i8[16,16] @ DRAM,
    y : i8[16,16] @ DRAM,
    res : i8[16,16] @ DRAM,
  ):
    A : i8[16,16] @ GEMM_SCRATCH
    B : i8[16,16] @ GEMM_SCRATCH
    C : i32[16,16] @ GEMM_ACCUM
    scale : f32
    scale = 1.0
    ld_i8(16,16, scale, x, A)
    ld_i8(16,16, scale, y, B)
    zero_acc_i32(16,16, C)

    trans_a : i8
    trans_b : i8
    trans_a = 0.0
    trans_b = 0.0
    matmul_i8(16,16,16, trans_a, trans_b, A, B, C)

    act : i8
    act = 0.0
    st_acc_i8(16,16, scale, act, C, res)
  T.add_proc(matmul_i8_ones_16)


  T.add_body(['matmul_i8_ones_16(x, y, res);',
              '',
              'gemmini_fence();',
              '',
              'if(check_eq_2i8(16,16, z, res)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (res):\\n");',
              '    print_2i8(16,16, res);',
              '    printf("Computed Roundtrip (z):\\n");',
              '    print_2i8(16,16, z);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()


def test_matmul_i8_ones_odd():
  T = GemmTestBuilder('matmul_i8_ones_odd')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])


  # 15 x 9 x 13
  T.alloc_dram_2i8('x', 15, 9, '1')
  T.alloc_dram_2i8('y', 9, 13, '1')
  T.alloc_dram_2i8('z', 15, 13, '9') # expected result
  T.alloc_dram_2i8('res', 15, 13, '0')

  @proc
  def matmul_i8_ones_odd(
    x : i8[15,9] @ DRAM,
    y : i8[9,13] @ DRAM,
    res : i8[15,13] @ DRAM,
  ):
    A : i8[15,16] @ GEMM_SCRATCH
    B : i8[9,16] @ GEMM_SCRATCH
    C : i32[15,16] @ GEMM_ACCUM
    scale : f32
    scale = 1.0
    ld_i8(15,9, scale, x, A)
    ld_i8(9,13, scale, y, B)
    zero_acc_i32(15,13, C)

    trans_a : i8
    trans_b : i8
    trans_a = 0.0
    trans_b = 0.0
    matmul_i8(15,13,9, trans_a, trans_b, A, B, C)

    act : i8
    act = 0.0
    st_acc_i8(15,13, scale, act, C, res)
  T.add_proc(matmul_i8_ones_odd)


  T.add_body(['matmul_i8_ones_odd(x, y, res);',
              '',
              'gemmini_fence();',
              '',
              'if(check_eq_2i8(15,13, z, res)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (res):\\n");',
              '    print_2i8(15,13, res);',
              '    printf("Computed Roundtrip (z):\\n");',
              '    print_2i8(15,13, z);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()

def test_ldst_acc_i32_16():
  T = GemmTestBuilder('ldst_acc_i32_16')
  T.add_body(['gemm_acc_init_mem();',
              'gemmini_flush(0);',
              ''])

  @proc
  def ldst_acc_i32_16( x : i32[16,16] @ DRAM, y : i32[16,16] @ DRAM ):
    tmp : i32[16,16] @ GEMM_ACCUM
    scale : f32
    scale = 4.0
    ld_acc_i32(16,16, scale, x, tmp)
    act : i8
    act = 0.0
    st_acc_i32(16,16, act, tmp, y)
  T.add_proc(ldst_acc_i32_16)

  T.alloc_dram_2i32('x', 16, 16, '1')
  T.alloc_dram_2i32('y', 16, 16, '0')
  T.alloc_dram_2i32('res', 16, 16, '4')

  T.add_body(['ldst_acc_i32_16(x, y);',
              '',
              'gemmini_fence();',
              '',
              'if(check_eq_2i32(16,16, y, res)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (res):\\n");',
              '    print_2i32(16,16, res);',
              '    printf("Computed Roundtrip (y):\\n");',
              '    print_2i32(16,16, y);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()



def test_matmul_i8_ones_odd():
  T = GemmTestBuilder('matmul_i8_ones_odd')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])


  # 15 x 9 x 13
  T.alloc_dram_2i8('x', 15, 9, '1')
  T.alloc_dram_2i8('y', 9, 13, '1')
  T.alloc_dram_2i8('z', 15, 13, '9') # expected result
  T.alloc_dram_2i8('res', 15, 13, '0')

  @proc
  def matmul_i8_ones_odd(
    x : i8[15,9] @ DRAM,
    y : i8[9,13] @ DRAM,
    res : i8[15,13] @ DRAM,
  ):
    A : i8[15,16] @ GEMM_SCRATCH
    B : i8[9,16] @ GEMM_SCRATCH
    C : i32[15,16] @ GEMM_ACCUM
    scale : f32
    scale = 1.0
    ld_i8(15,9, scale, x, A)
    ld_i8(9,13, scale, y, B)
    zero_acc_i32(15,13, C)

    act : i8
    trans_a : i8
    trans_b : i8
    act = 0.0
    trans_a = 0.0
    trans_b = 0.0
    matmul_i8(15,13,9, trans_a, trans_b, A, B, C)

    st_acc_i8(15,13, scale, act, C, res)
  T.add_proc(matmul_i8_ones_odd)


  T.add_body(['matmul_i8_ones_odd(x, y, res);',
              '',
              'gemmini_fence();',
              '',
              'if(check_eq_2i8(15,13, z, res)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (res):\\n");',
              '    print_2i8(15,13, res);',
              '    printf("Computed Roundtrip (z):\\n");',
              '    print_2i8(15,13, z);',
              '    exit(1);',
              '}',
              ''])

  T.compile().run()
