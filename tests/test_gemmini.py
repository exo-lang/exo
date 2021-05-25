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
    ld_i8(16,16, x, tmp)
    st_i8(16,16, tmp, y)
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
    ld_acc_i8(16,16, x, tmp)
    st_acc_i8(16,16, tmp, y)
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
    ld_i8(15,7, x, tmp)
    st_i8(15,7, tmp, y)
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
    ld_acc_i8(7,13, x, tmp)
    st_acc_i8(7,13, tmp, y)
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
    ld_i8(16,16, x, tmp)
    zero_i8(8,8, tmp[4:12,:])
    st_i8(16,16, tmp, y)
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
    ld_acc_i8(16,16, x, tmp)
    zero_acc_i8(8,8, tmp[4:12,:])
    st_acc_i8(16,16, tmp, y)
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
