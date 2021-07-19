from __future__ import annotations

#from ctypes import *
#import os
#import subprocess
#import numpy as np
#import scipy.stats as st
#import os

import os
import sys
_HERE_ = os.path.dirname(os.path.abspath(__file__))
print(sys.path[0])
sys.path.append(sys.path[0]+"/..")
sys.path.append(sys.path[0]+"/.")
from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import GEMM_SCRATCH, GEMM_ACCUM, MDRAM
from .gemmini import *
from .harness_gemmini import ENV, GemmTestBuilder
import pytest


# --------------------------------------------------------------------------- #
#   MatMul Demo
# --------------------------------------------------------------------------- #

@pytest.mark.skip()
def test_matmul_demo():
  T = GemmTestBuilder('matmul_demo')
  do_init(T)

  NN = 64
  MM = 64
  KK = 64

  T.alloc_dram_2i8('x', NN, KK, '1')
  T.alloc_dram_2i8('y', KK, MM, '1')
  T.alloc_dram_2i8('z', NN, MM, '0')


  @proc
  def matmul2d(
    N : size, M : size, K : size,
    A : i8[N,K] @ DRAM,
    B : i8[K,M] @ DRAM,
    C : i8[N,M] @ DRAM,
  ):
    for i in par(0,N):
      for j in par(0,M):
        C[i,j] = 0.0
        for k in par(0,K):
          C[i,j] += A[i,k]*B[k,j]


  matmul2d = matmul2d.partial_eval(NN, MM, KK)

  matmul2d = (matmul2d.split('k',16,['k','k_in'], perfect=True)
                      .split('j',16,['j','j_in'], perfect=True)
                      .split('i',16,['i','i_in'], perfect=True))

  matmul2d = (matmul2d.reorder('i_in','j')
                      .fission_after('C[_] = 0.0', n_lifts=2)
                      .reorder('j_in #1','k')
                      .reorder('i_in #1','k'))

  matmul2d = pre_bake_stage_C(matmul2d, "for i_in in _: _\n"+
                                        "for k in _: _", 'C', 'CG')

  matmul2d = pre_bake_stage_A_and_B(matmul2d)

  matmul2d = pre_bake_abstract_A(matmul2d, 'for i_in in _: _ #1', ld_i8)

  matmul2d = pre_bake_abstract_BC_and_mmul(matmul2d)

  matmul2d = matmul2d.set_precision('CG','i32')

  matmul2d = (matmul2d.set_memory('CG',GEMM_ACCUM)
                      .set_memory('BG',GEMM_SCRATCH)
                      .set_memory('AG',GEMM_SCRATCH))

  matmul2d = (matmul2d.lift_alloc('AG : _')
                      .lift_alloc('BG : _'))

  matmul2d = (matmul2d.lift_alloc('AG : _', n_lifts=2)
                      .lift_alloc('BG : _', n_lifts=2)
                      .lift_alloc('CG : _', n_lifts=2))


  matmul2d = (matmul2d.unroll('k'))
  
  matmul2d = (matmul2d.unroll('j').unroll('i'))

  orig_matmul = matmul2d  

  print()
  print(matmul2d)
  T.add_proc(matmul2d)

  T.start_timer('gemmini')
  T.add_body([f'matmul2d(x, y, z);',
              f'gemmini_fence();',
              f''])
  T.stop_timer('gemmini', 'Instruction Count for GEMMINI version')
  
  T.compile().run()































def pre_bake_stage_C(p, pattern, name_in, name='CG'):
  @proc
  def matmul2d(
    A: i8[64, 64] @ DRAM,
    B: i8[64, 64] @ DRAM,
    C: i8[64, 64] @ DRAM
  ):
    for i in par(0, 4):
        for j in par(0, 4):
            CG : i8[16,16] @ MDRAM
            for i_in in par(0, 16):
                for j_in in par(0, 16):
                    CG[i_in,j_in] = 0.0
            for k in par(0, 4):
                for i_in in par(0, 16):
                    for j_in in par(0, 16):
                        for k_in in par(0, 16):
                            CG[i_in,j_in] += (
                                A[16 * i + i_in, 16 * k + k_in] *
                                B[16 * k + k_in, 16 * j + j_in] )
            for i_in in par(0, 16):
                for j_in in par(0, 16):
                    C[16 * i + i_in, 16 * j + j_in] = CG[i_in,j_in]
  return matmul2d


def pre_bake_stage_A_and_B(p):
  @proc
  def matmul2d(
    A: i8[64, 64] @ DRAM,
    B: i8[64, 64] @ DRAM,
    C: i8[64, 64] @ DRAM
  ):
    for i in par(0, 4):
        for j in par(0, 4):
            CG : i8[16,16] @ MDRAM
            for i_in in par(0, 16):
                for j_in in par(0, 16):
                    CG[i_in,j_in] = 0.0
            for k in par(0, 4):
                AG : i8[16,16] @ MDRAM
                BG : i8[16,16] @ MDRAM
                for i_in in par(0, 16):
                    for k_in in par(0, 16):
                        AG[i_in,k_in] = A[16 * i + i_in, 16 * k + k_in]
                for k_in in par(0, 16):
                    for j_in in par(0, 16):
                        BG[k_in,j_in] = B[16 * k + k_in, 16 * j + j_in]
                for i_in in par(0, 16):
                    for j_in in par(0, 16):
                        for k_in in par(0, 16):
                            CG[i_in,j_in] += AG[i_in,k_in] * BG[k_in,j_in]
            for i_in in par(0, 16):
                for j_in in par(0, 16):
                    C[16 * i + i_in, 16 * j + j_in] = CG[i_in,j_in]
  return matmul2d



def pre_bake_abstract_A(p, pattern, instr):
  @proc
  def matmul2d(
    A: i8[64, 64] @ DRAM,
    B: i8[64, 64] @ DRAM,
    C: i8[64, 64] @ DRAM
  ):
    for i in par(0, 4):
        for j in par(0, 4):
            CG : i8[16,16] @ MDRAM
            for i_in in par(0, 16):
                for j_in in par(0, 16):
                    CG[i_in,j_in] = 0.0
            for k in par(0, 4):
                AG : i8[16,16] @ MDRAM
                BG : i8[16,16] @ MDRAM
                scale : f32
                scale = 1.0
                ld_i8(16, 16, scale, A[16*i:16*i+16, 16*k:16*k+16], AG)
                for k_in in par(0, 16):
                    for j_in in par(0, 16):
                        BG[k_in,j_in] = B[16 * k + k_in, 16 * j + j_in]
                for i_in in par(0, 16):
                    for j_in in par(0, 16):
                        for k_in in par(0, 16):
                            CG[i_in,j_in] += AG[i_in,k_in] * BG[k_in,j_in]
            for i_in in par(0, 16):
                for j_in in par(0, 16):
                    C[16 * i + i_in, 16 * j + j_in] = CG[i_in,j_in]
  return matmul2d


def pre_bake_abstract_BC_and_mmul(p):
  @proc
  def matmul2d(
    A: i8[64, 64] @ DRAM,
    B: i8[64, 64] @ DRAM,
    C: i8[64, 64] @ DRAM
  ):
    scale : f32
    scale = 1.0
    for i in par(0, 4):
        for j in par(0, 4):
            CG : i8[16,16] @ MDRAM
            zero_acc_i32(16,16, CG)
            for k in par(0, 4):
                AG : i8[16,16] @ MDRAM
                BG : i8[16,16] @ MDRAM
                ld_i8(16, 16, scale, A[16*i:16*i+16, 16*k:16*k+16], AG)
                ld_i8(16, 16, scale, B[16*k:16*k+16, 16*j:16*j+16], BG)
                matmul_acc_i8(16,16,16, AG, BG, CG)
            st_acc_i8(16,16, scale, CG, C[16*i:16*i+16, 16*j:16*j+16])
  return matmul2d



def do_init(T):
  T.add_body(['gemm_init_mem();',
              'gemm_acc_init_mem();',
              'init_mem();',
              'gemmini_flush(0);',
              ''])

  T.add_proc(ld_i8)
  T.add_proc(ld_acc_i8)

  @proc
  def mdram_dummy():
    x : i8 @ MDRAM
  T.add_proc(mdram_dummy)


