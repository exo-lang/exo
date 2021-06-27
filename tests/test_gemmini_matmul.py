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
#   Basic MatMul Test
# --------------------------------------------------------------------------- #

def test_matmul_basic():
  T = GemmTestBuilder('matmul_basic')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])

  NN = 64
  MM = 32
  KK = 128

  T.alloc_dram_2i8('x', NN, KK, '1')
  T.alloc_dram_2i8('y', KK, MM, '1')
  T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
  T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

  @proc
  def matmul_on_cpu(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i8[N,M] @ DRAM,
  ):
    for i in par(0,N):
      for j in par(0,M):
        res : i32
        res = 0.0
        for k in par(0,K):
          a : i32
          b : i32
          a = A[i,k]
          b = B[k,j]
          res += a*b
        C[i,j] = res
  T.add_proc(matmul_on_cpu)

  @proc
  def matmul_on_gemmini(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i8[N,M] @ DRAM,
  ):
    assert N % 16 == 0
    assert M % 16 == 0
    assert K % 16 == 0
    
    for i in par(0,N/16):
      for j in par(0,M/16):
        res : i32[16,16] @ GEMM_ACCUM
        zero_acc_i32(16,16,res)

        for k in par(0,K/16):
          Ablock : i8[16,16] @ GEMM_SCRATCH
          Bblock : i8[16,16] @ GEMM_SCRATCH
          ld_i8(16,16, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
          ld_i8(16,16, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)
          
          matmul_acc_i8(16,16,16, Ablock, Bblock, res)

        st_acc_i8(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])
  
  T.add_proc(matmul_on_gemmini)

  T.start_timer('cpu')
  T.add_body([f'matmul_on_cpu({NN}, {MM}, {KK}, x, y, z_cpu);',
              f'gemmini_fence();'])
  T.stop_timer('cpu', 'Cycles for CPU version')

  T.start_timer('gemmini')
  T.add_body([f'matmul_on_gemmini({NN}, {MM}, {KK}, x, y, z_gemmini);',
              f'gemmini_fence();',
              f''])
  T.stop_timer('gemmini', 'Cycles for GEMMINI version')
  
  T.add_body([f'if(check_eq_2i8({NN},{MM}, z_cpu, z_gemmini)) {{',
               '    printf("Correct\\n");',
               '} else {',
               '    printf("Results Don\'t Match\\n");',
               '    printf("Correct Result (z_cpu):\\n");',
              f'    print_2i8({NN},{MM}, z_cpu);',
               '    printf("Computed Roundtrip (z_gemmini):\\n");',
              f'    print_2i8({NN},{MM}, z_gemmini);',
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
    ld_i8(15,9, x, A)
    ld_i8(9,13, y, B)
    zero_acc_i32(15,13, C)

    matmul_i8(15,13,9, A, B, C)

    st_acc_i8(15,13, C, res)
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


def test_matmul_basic_i32():
  T = GemmTestBuilder('matmul_basic_i32')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])

  NN = 64
  MM = 32
  KK = 128

  T.alloc_dram_2i8('x', NN, KK, '1')
  T.alloc_dram_2i8('y', KK, MM, '1')
  T.alloc_dram_2i32('z_cpu', NN, MM, '0') # expected result
  T.alloc_dram_2i32('z_gemmini', NN, MM, '0')

  @proc
  def matmul_on_cpu(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i32[N,M] @ DRAM,
  ):
    for i in par(0,N):
      for j in par(0,M):
        res : i32
        res = 0.0
        for k in par(0,K):
          a : i32
          b : i32
          a = A[i,k]
          b = B[k,j]
          res += a*b
        C[i,j] = res
  T.add_proc(matmul_on_cpu)

  @proc
  def matmul_on_gemmini(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i32[N,M] @ DRAM,
  ):
    assert N % 16 == 0
    assert M % 16 == 0
    assert K % 16 == 0
    
    for i in par(0,N/16):
      for j in par(0,M/16):
        res : i32[16,16] @ GEMM_ACCUM
        zero_acc_i32(16,16,res)

        for k in par(0,K/16):
          Ablock : i8[16,16] @ GEMM_SCRATCH
          Bblock : i8[16,16] @ GEMM_SCRATCH
          ld_i8(16,16, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
          ld_i8(16,16, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)
          
          matmul_acc_i8(16,16,16, Ablock, Bblock, res)

        st_acc_i32(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])
  
  T.add_proc(matmul_on_gemmini)

  T.start_timer('cpu')
  T.add_body([f'matmul_on_cpu({NN}, {MM}, {KK}, x, y, z_cpu);',
              f'gemmini_fence();'])
  T.stop_timer('cpu', 'Cycles for CPU version')

  T.start_timer('gemmini')
  T.add_body([f'matmul_on_gemmini({NN}, {MM}, {KK}, x, y, z_gemmini);',
              f'gemmini_fence();',
              f''])
  T.stop_timer('gemmini', 'Cycles for GEMMINI version')
  
  T.add_body([f'if(check_eq_2i32({NN},{MM}, z_cpu, z_gemmini)) {{',
               '    printf("Correct\\n");',
               '} else {',
               '    printf("Results Don\'t Match\\n");',
               '    printf("Correct Result (z_cpu):\\n");',
              f'    print_2i32({NN},{MM}, z_cpu);',
               '    printf("Computed Roundtrip (z_gemmini):\\n");',
              f'    print_2i32({NN},{MM}, z_gemmini);',
               '    exit(1);',
               '}',
               ''])
  

  T.compile().run()



def test_matmul_c_i8_d_i8():
  T = GemmTestBuilder('matmul_c_i8_d_i8')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])

  NN = 64
  MM = 32
  KK = 128

  T.alloc_dram_2i8('x', NN, KK, '1')
  T.alloc_dram_2i8('y', KK, MM, '1')
  T.alloc_dram_2i8('d', NN, MM, '2')
  T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
  T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

  @proc
  def matmul_on_cpu(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i8[N,M] @ DRAM,
    D : i8[N,M] @ DRAM
  ):
    for i in par(0,N):
      for j in par(0,M):
        res : i32
        res = D[i,j]
        for k in par(0,K):
          a : i32
          b : i32
          a = A[i,k]
          b = B[k,j]
          res += a*b
        C[i,j] = res
  T.add_proc(matmul_on_cpu)

  @proc
  def matmul_c_i8_d_i8(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i8[N,M] @ DRAM,
    D : i8[N,M] @ DRAM
  ):
    assert N % 16 == 0
    assert M % 16 == 0
    assert K % 16 == 0
    
    for i in par(0,N/16):
      for j in par(0,M/16):
        res : i32[16,16] @ GEMM_ACCUM
        ld_acc_i8(16, 16, D[ 16*i:16*(i+1), 16*j:16*(j+1) ], res)

        for k in par(0,K/16):
          Ablock : i8[16,16] @ GEMM_SCRATCH
          Bblock : i8[16,16] @ GEMM_SCRATCH
          ld_i8(16,16, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
          ld_i8(16,16, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)
          
          matmul_acc_i8(16,16,16, Ablock, Bblock, res)

        st_acc_i8(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])
  
  T.add_proc(matmul_c_i8_d_i8)

  T.start_timer('cpu')
  T.add_body([f'matmul_on_cpu({NN}, {MM}, {KK}, x, y, z_cpu, d);',
              f'gemmini_fence();'])
  T.stop_timer('cpu', 'Cycles for CPU version')

  T.start_timer('gemmini')
  T.add_body([f'matmul_c_i8_d_i8({NN}, {MM}, {KK}, x, y, z_gemmini, d);',
              f'gemmini_fence();',
              f''])
  T.stop_timer('gemmini', 'Cycles for GEMMINI version')
  
  T.add_body([f'if(check_eq_2i8({NN},{MM}, z_cpu, z_gemmini)) {{',
               '    printf("Correct\\n");',
               '} else {',
               '    printf("Results Don\'t Match\\n");',
               '    printf("Correct Result (z_cpu):\\n");',
              f'    print_2i8({NN},{MM}, z_cpu);',
               '    printf("Computed Roundtrip (z_gemmini):\\n");',
              f'    print_2i8({NN},{MM}, z_gemmini);',
               '    exit(1);',
               '}',
               ''])
  

  T.compile().run()



def test_matmul_c_i8_d_i32():
  T = GemmTestBuilder('matmul_c_i8_d_i32')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])

  NN = 64
  MM = 32
  KK = 128

  T.alloc_dram_2i8('x', NN, KK, '1')
  T.alloc_dram_2i8('y', KK, MM, '1')
  T.alloc_dram_2i32('d', NN, MM, '2')
  T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
  T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

  @proc
  def matmul_on_cpu(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i8[N,M] @ DRAM,
    D : i32[N,M] @ DRAM
  ):
    for i in par(0,N):
      for j in par(0,M):
        res : i32
        res = D[i,j]
        for k in par(0,K):
          a : i32
          b : i32
          a = A[i,k]
          b = B[k,j]
          res += a*b
        C[i,j] = res
  T.add_proc(matmul_on_cpu)

  @proc
  def matmul_c_i8_d_i32(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i8[N,M] @ DRAM,
    D : i32[N,M] @ DRAM
  ):
    assert N % 16 == 0
    assert M % 16 == 0
    assert K % 16 == 0
    
    for i in par(0,N/16):
      for j in par(0,M/16):
        res : i32[16,16] @ GEMM_ACCUM
        ld_acc_i32(16, 16, D[ 16*i:16*(i+1), 16*j:16*(j+1) ], res)

        for k in par(0,K/16):
          Ablock : i8[16,16] @ GEMM_SCRATCH
          Bblock : i8[16,16] @ GEMM_SCRATCH
          ld_i8(16,16, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
          ld_i8(16,16, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)
          
          matmul_acc_i8(16,16,16, Ablock, Bblock, res)

        st_acc_i8(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])
  
  T.add_proc(matmul_c_i8_d_i32)

  T.start_timer('cpu')
  T.add_body([f'matmul_on_cpu({NN}, {MM}, {KK}, x, y, z_cpu, d);',
              f'gemmini_fence();'])
  T.stop_timer('cpu', 'Cycles for CPU version')

  T.start_timer('gemmini')
  T.add_body([f'matmul_c_i8_d_i32({NN}, {MM}, {KK}, x, y, z_gemmini, d);',
              f'gemmini_fence();',
              f''])
  T.stop_timer('gemmini', 'Cycles for GEMMINI version')
  
  T.add_body([f'if(check_eq_2i8({NN},{MM}, z_cpu, z_gemmini)) {{',
               '    printf("Correct\\n");',
               '} else {',
               '    printf("Results Don\'t Match\\n");',
               '    printf("Correct Result (z_cpu):\\n");',
              f'    print_2i8({NN},{MM}, z_cpu);',
               '    printf("Computed Roundtrip (z_gemmini):\\n");',
              f'    print_2i8({NN},{MM}, z_gemmini);',
               '    exit(1);',
               '}',
               ''])
  

  T.compile().run()



def test_matmul_c_i32_d_i8():
  T = GemmTestBuilder('matmul_c_i32_d_i8')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])

  NN = 64
  MM = 32
  KK = 128

  T.alloc_dram_2i8('x', NN, KK, '1')
  T.alloc_dram_2i8('y', KK, MM, '1')
  T.alloc_dram_2i8('d', NN, MM, '2')
  T.alloc_dram_2i32('z_cpu', NN, MM, '0') # expected result
  T.alloc_dram_2i32('z_gemmini', NN, MM, '0')

  @proc
  def matmul_on_cpu(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i32[N,M] @ DRAM,
    D : i8[N,M] @ DRAM
  ):
    for i in par(0,N):
      for j in par(0,M):
        res : i32
        res = D[i,j]
        for k in par(0,K):
          a : i32
          b : i32
          a = A[i,k]
          b = B[k,j]
          res += a*b
        C[i,j] = res
  T.add_proc(matmul_on_cpu)

  @proc
  def matmul_c_i32_d_i8(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i32[N,M] @ DRAM,
    D : i8[N,M] @ DRAM
  ):
    assert N % 16 == 0
    assert M % 16 == 0
    assert K % 16 == 0
    
    for i in par(0,N/16):
      for j in par(0,M/16):
        res : i32[16,16] @ GEMM_ACCUM
        ld_acc_i8(16, 16, D[ 16*i:16*(i+1), 16*j:16*(j+1) ], res)

        for k in par(0,K/16):
          Ablock : i8[16,16] @ GEMM_SCRATCH
          Bblock : i8[16,16] @ GEMM_SCRATCH
          ld_i8(16,16, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
          ld_i8(16,16, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)
          
          matmul_acc_i8(16,16,16, Ablock, Bblock, res)

        st_acc_i32(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])
  
  T.add_proc(matmul_c_i32_d_i8)

  T.start_timer('cpu')
  T.add_body([f'matmul_on_cpu({NN}, {MM}, {KK}, x, y, z_cpu, d);',
              f'gemmini_fence();'])
  T.stop_timer('cpu', 'Cycles for CPU version')

  T.start_timer('gemmini')
  T.add_body([f'matmul_c_i32_d_i8({NN}, {MM}, {KK}, x, y, z_gemmini, d);',
              f'gemmini_fence();',
              f''])
  T.stop_timer('gemmini', 'Cycles for GEMMINI version')
  
  T.add_body([f'if(check_eq_2i8({NN},{MM}, z_cpu, z_gemmini)) {{',
               '    printf("Correct\\n");',
               '} else {',
               '    printf("Results Don\'t Match\\n");',
               '    printf("Correct Result (z_cpu):\\n");',
              f'    print_2i8({NN},{MM}, z_cpu);',
               '    printf("Computed Roundtrip (z_gemmini):\\n");',
              f'    print_2i8({NN},{MM}, z_gemmini);',
               '    exit(1);',
               '}',
               ''])
  

  T.compile().run()


def test_matmul_c_i32_d_i32():
  T = GemmTestBuilder('matmul_c_i32_d_i32')
  T.add_body(['gemm_init_mem();',
              'gemmini_flush(0);',
              ''])

  NN = 64
  MM = 32
  KK = 128

  T.alloc_dram_2i8('x', NN, KK, '1')
  T.alloc_dram_2i8('y', KK, MM, '1')
  T.alloc_dram_2i32('d', NN, MM, '2')
  T.alloc_dram_2i32('z_cpu', NN, MM, '0') # expected result
  T.alloc_dram_2i32('z_gemmini', NN, MM, '0')

  @proc
  def matmul_on_cpu(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i32[N,M] @ DRAM,
    D : i32[N,M] @ DRAM
  ):
    for i in par(0,N):
      for j in par(0,M):
        res : i32
        res = D[i,j]
        for k in par(0,K):
          a : i32
          b : i32
          a = A[i,k]
          b = B[k,j]
          res += a*b
        C[i,j] = res
  T.add_proc(matmul_on_cpu)

  @proc
  def matmul_c_i32_d_i32(
    N : size,
    M : size,
    K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i32[N,M] @ DRAM,
    D : i32[N,M] @ DRAM
  ):
    assert N % 16 == 0
    assert M % 16 == 0
    assert K % 16 == 0
    
    for i in par(0,N/16):
      for j in par(0,M/16):
        res : i32[16,16] @ GEMM_ACCUM
        ld_acc_i32(16, 16, D[ 16*i:16*(i+1), 16*j:16*(j+1) ], res)

        for k in par(0,K/16):
          Ablock : i8[16,16] @ GEMM_SCRATCH
          Bblock : i8[16,16] @ GEMM_SCRATCH
          ld_i8(16,16, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
          ld_i8(16,16, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)
          
          matmul_acc_i8(16,16,16, Ablock, Bblock, res)

        st_acc_i32(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])
  
  T.add_proc(matmul_c_i32_d_i32)

  T.start_timer('cpu')
  T.add_body([f'matmul_on_cpu({NN}, {MM}, {KK}, x, y, z_cpu, d);',
              f'gemmini_fence();'])
  T.stop_timer('cpu', 'Cycles for CPU version')

  T.start_timer('gemmini')
  T.add_body([f'matmul_c_i32_d_i32({NN}, {MM}, {KK}, x, y, z_gemmini, d);',
              f'gemmini_fence();',
              f''])
  T.stop_timer('gemmini', 'Cycles for GEMMINI version')
  
  T.add_body([f'if(check_eq_2i8({NN},{MM}, z_cpu, z_gemmini)) {{',
               '    printf("Correct\\n");',
               '} else {',
               '    printf("Results Don\'t Match\\n");',
               '    printf("Correct Result (z_cpu):\\n");',
              f'    print_2i8({NN},{MM}, z_cpu);',
               '    printf("Computed Roundtrip (z_gemmini):\\n");',
              f'    print_2i8({NN},{MM}, z_gemmini);',
               '    exit(1);',
               '}',
               ''])
  

  T.compile().run()


