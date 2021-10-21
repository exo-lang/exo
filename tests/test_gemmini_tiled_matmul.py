from __future__ import annotations

#from ctypes import *
#import os
#import subprocess
#import numpy as np
#import scipy.stats as st
#import os

import os
import sys
from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import GEMM_SCRATCH, GEMM_ACCUM, MDRAM
from .gemmini import *
from .harness_gemmini import ENV, GemmTestBuilder
import pytest


# --------------------------------------------------------------------------- #
#   MatMul Demo
# --------------------------------------------------------------------------- #

@pytest.mark.skip()
def test_matmul_c_i8():
    T = GemmTestBuilder('matmul_c_i8')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i8_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    @proc
    def matmul_c_i8(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i8][N,M] @ DRAM,
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1

        for i in par(0,N):
            for j in par(0,M):
                res : i32 @ GEMM_ACCUM
                res = 0.0
                for k in par(0,K):
                    tmp_a : f32
                    tmp_a = A[i,k]
                    tmp_a = tmp_a * a_scale
                    a : i8 @ GEMM_SCRATCH
                    a = tmp_a

                    tmp_b : f32
                    tmp_b = B[k,j]
                    tmp_b = tmp_b * b_scale
                    b : i8 @ GEMM_SCRATCH
                    b = tmp_b

                    a2 : i32
                    b2 : i32
                    a2 = a
                    b2 = b
                    res += a2*b2

                tmp_res : i8
                if acc == True:
                    tmp_res = relu(res)
                else:
                    tmp_res = res

                tmp_res2 : f32
                tmp_res2 = tmp_res
                tmp_res2 = tmp_res2 * c_scale
                clamp(tmp_res2, tmp_res)
                C[i,j] = tmp_res


    matmul_c_i8 = matmul_c_i8.split('i',128,['io','i'], tail='cut')
    matmul_c_i8 = matmul_c_i8.split('j',128,['jo','j'], tail='cut')
    matmul_c_i8 = matmul_c_i8.fission_after('for jo in _:_', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.reorder('i #0','jo')

# main block
    matmul_c_i8 = matmul_c_i8.split('i #0',16,['i','i_in'], perfect=True)
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','j')
    matmul_c_i8 = matmul_c_i8.split('j #0',16,['j','j_in'], perfect=True)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #0', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #0', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('res[_] = 0.0 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('for k in _:_ #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','k')
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','k')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.split('k #0',16,['k','k_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #0', n_lifts=1, mode='col')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #1', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #0', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #1', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(st_acc_i8, "for i_in in _:_ #0")
# main block basic tiling done

# next block
    matmul_c_i8 = matmul_c_i8.split('i #1',16,['i','i_in'], perfect=True)
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','j')
    matmul_c_i8 = matmul_c_i8.split('j #1',16,['j','j_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #1', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #1', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #1', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('res[_] = 0.0 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('for k in _:_ #1', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','k')
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','k')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #2', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #2', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.split('k #1',16,['k','k_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #2', n_lifts=1, mode='col')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #3', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #2', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #3', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(st_acc_i8, "for i_in in _:_ #0")

# if M % 128 % 16 > 0: block
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #2', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #2', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.fission_after('res[_] = 0.0 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('for k in _:_ #2', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','k')
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','k')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #4', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #4', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #4', n_lifts=1, size=16)
    matmul_c_i8 = matmul_c_i8.split('k #2',16,['k','k_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #4', n_lifts=1, mode='col')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #5', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #4', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #5', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(st_acc_i8, "for i_in in _:_ #0")

# next....
    matmul_c_i8 = matmul_c_i8.split('i #2',16,['i','i_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.fission_after('for jo in _:_ #1', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','jo')
    matmul_c_i8 = matmul_c_i8.split('j #2',16,['j','j_in'], perfect=True)
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','j')
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #3', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #3', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #3', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('res[_] = 0.0 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('for k in _:_ #3', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.reorder('j_in','k')
    matmul_c_i8 = matmul_c_i8.reorder('i_in','k')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #6', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #6', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.split('k #3',16,['k','k_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #6', n_lifts=1, mode='col')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #7', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #6', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #7', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(st_acc_i8, "for i_in in _:_ #0")


# next..
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','j')
    matmul_c_i8 = matmul_c_i8.split('j #3',16,['j','j_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #4', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #4', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #4', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('res[_] = 0.0 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('for k in _:_ #4', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','k')
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','k')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #8', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #8', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.split('k #4',16,['k','k_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #8', n_lifts=1, mode='col')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #9', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #8', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #9', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(st_acc_i8, "for i_in in _:_ #0")

# next!
    matmul_c_i8 = matmul_c_i8.reorder('j_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #5', n_lifts=1, size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #5', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('res[_] = 0.0 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('for k in _:_ #5', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.reorder('j_in','k')
    matmul_c_i8 = matmul_c_i8.reorder('i_in','k')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #10', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #10', n_lifts=2, size=16)
    matmul_c_i8 = matmul_c_i8.split('k #5',16,['k','k_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #10', n_lifts=1, mode='col')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #11', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #10', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #11', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(st_acc_i8, "for i_in in _:_ #0")

# almost last!!
    matmul_c_i8 = matmul_c_i8.fission_after('for jo in _:_ #2', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.reorder('i_in','jo')
    matmul_c_i8 = matmul_c_i8.split('j #4',16,['j','j_in'], perfect=True)
    matmul_c_i8 = matmul_c_i8.reorder('i_in #0','j')
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #6', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('res[_] = 0.0 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('for k in _:_ #6', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.reorder('j_in','k')
    matmul_c_i8 = matmul_c_i8.reorder('i_in','k')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #12', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #12', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.split('k #6',16,['k','k_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #12', n_lifts=1, mode='col')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #13', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #12', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #13', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(st_acc_i8, "for i_in in _:_ #0")


# Last!
    matmul_c_i8 = matmul_c_i8.reorder('i_in','j')
    matmul_c_i8 = matmul_c_i8.split('j #5',16,['j','j_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.reorder('j_in','i_in')
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #7', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('res[_] = 0.0 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('for k in _:_ #7', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.reorder('j_in','k')
    matmul_c_i8 = matmul_c_i8.reorder('i_in','k')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #14', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #14', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.split('k #7',16,['k','k_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #14', n_lifts=1, mode='col')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #15', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #14', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #15', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(st_acc_i8, "for i_in in _:_ #0")

#last!!!
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #8', n_lifts=1, size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('res : _ #8', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('res[_] = 0.0 #0', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.fission_after('for k in _:_ #8', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.reorder('j_in','k')
    matmul_c_i8 = matmul_c_i8.reorder('i_in','k')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #16', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #16', n_lifts=1, size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #16', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.split('k #8',16,['k','k_in'], tail='cut_and_guard')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #16', n_lifts=1, mode='col')
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : _ #17', n_lifts=1, mode='col', size=16)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #16', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : _ #17', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('a[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #0', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.fission_after('b[_] = _ #1', n_lifts=3)
    matmul_c_i8 = matmul_c_i8.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','i_in')
    matmul_c_i8 = matmul_c_i8.reorder('k_in #0','j_in')
    matmul_c_i8 = matmul_c_i8.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8 = matmul_c_i8.replace(st_acc_i8, "for i_in in _:_ #0")

   # Optimization
    matmul_c_i8 = matmul_c_i8.par_to_seq("for k in _:_")
    matmul_c_i8 = matmul_c_i8.par_to_seq("for i in _:_")
    matmul_c_i8 = matmul_c_i8.par_to_seq("for j in _:_")
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8', n_lifts=2)
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #0', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #0', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('a : i8 #1', n_lifts=1)
    matmul_c_i8 = matmul_c_i8.lift_alloc('b : i8 #1', n_lifts=1)

    print(matmul_c_i8)
    matmul_c_i8.check_effects()

    T.add_proc(matmul_c_i8)
    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i8(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_cpu, {NN}, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('gemmini', 'Cycles for GEMMINI version')
    T.compile().run()





def test_matmul_c_i8_perfect():
    T = GemmTestBuilder('matmul_c_i8_perfect')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i8_perfect_lib_Context *ctxt;"])

    NN = 512
    MM = 512
    KK = 512

    T.alloc_dram_2i8('x', NN, KK, '4')
    T.alloc_dram_2i8('y', KK, MM, '6')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    @proc
    def matmul_c_i8_perfect(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      acc     : bool,
      A : i8[N,K] @ DRAM,
      B : i8[K,M] @ DRAM,
      C : i8[N,M] @ DRAM,
    ):
        assert N == 512
        assert M == 512
        assert K == 512

        for i in par(0,512):
            for j in par(0,512):
                res : i32 @ GEMM_ACCUM
                res = 0.0
                for k in par(0,512):
                    tmp_a : f32
                    tmp_a = A[i,k]
                    tmp_a = tmp_a * a_scale
                    a : i8 @ GEMM_SCRATCH
                    a = tmp_a

                    tmp_b : f32
                    tmp_b = B[k,j]
                    tmp_b = tmp_b * b_scale
                    b : i8 @ GEMM_SCRATCH
                    b = tmp_b

                    a2 : i32
                    b2 : i32
                    a2 = a
                    b2 = b
                    res += a2*b2

                tmp_res : i8
                if acc == True:
                    tmp_res = relu(res)
                else:
                    tmp_res = res

                tmp_res2 : f32
                tmp_res2 = tmp_res
                tmp_res2 = tmp_res2 * c_scale
                clamp(tmp_res2, tmp_res)
                C[i,j] = tmp_res

    matmul_c_i8_cpu = matmul_c_i8_perfect.rename("matmul_c_i8_cpu")
    matmul_c_i8_cpu = (matmul_c_i8_cpu.set_memory('res', DRAM)
                                     .set_memory('a', DRAM)
                                     .set_memory('b', DRAM))

    matmul_c_i8_perfect = matmul_c_i8_perfect.split('i',128,['io','i'], perfect=True)
    matmul_c_i8_perfect = matmul_c_i8_perfect.split('j',128,['jo','j'], perfect=True)
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('i','jo')

    matmul_c_i8_perfect = matmul_c_i8_perfect.split('i',16,['i','i_in'], perfect=True)
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('i_in','j')
    matmul_c_i8_perfect = matmul_c_i8_perfect.split('j',16,['j','j_in'], perfect=True)

    matmul_c_i8_perfect = matmul_c_i8_perfect.lift_alloc('res : _ #0', n_lifts=1)
    matmul_c_i8_perfect = matmul_c_i8_perfect.lift_alloc('res : _ #0', n_lifts=1, mode='col', size=16)
    matmul_c_i8_perfect = matmul_c_i8_perfect.lift_alloc('res : _ #0', n_lifts=2)
    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after('res[_] = 0.0 #0', n_lifts=2)

    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after('for k in _:_ #0', n_lifts=2)

    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('i_in','k')
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('j_in','k')

    matmul_c_i8_perfect = matmul_c_i8_perfect.lift_alloc('a : i8', n_lifts=2)
    matmul_c_i8_perfect = matmul_c_i8_perfect.lift_alloc('b : i8', n_lifts=2)

    matmul_c_i8_perfect = matmul_c_i8_perfect.split('k',16,['k','k_in'], perfect=True)

    matmul_c_i8_perfect = matmul_c_i8_perfect.lift_alloc('a : _ #0', n_lifts=1, mode='col')
    matmul_c_i8_perfect = matmul_c_i8_perfect.lift_alloc('b : _', n_lifts=1)

    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after('a[_] = _', n_lifts=3)
    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after('b[_] = _', n_lifts=3)

    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('j_in','i_in')
    matmul_c_i8_perfect = matmul_c_i8_perfect.replace(zero_acc_i32, "for i_in in _:_ #0")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('k_in','i_in')
    matmul_c_i8_perfect = matmul_c_i8_perfect.replace(ld_i8, "for i_in in _:_ #0")
    matmul_c_i8_perfect = matmul_c_i8_perfect.replace(ld_i8, "for k_in in _:_ #0")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('k_in','j_in')
    matmul_c_i8_perfect = matmul_c_i8_perfect.replace(matmul_acc_i8, "for i_in in _:_ #0")
    matmul_c_i8_perfect = matmul_c_i8_perfect.replace(st_acc_i8, "for i_in in _:_ #0")

    matmul_c_i8_perfect = matmul_c_i8_perfect.call_eqv(ld_i8_v2, "ld_i8(_, _, _, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline("ld_i8_v2(_, _, _, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("src = A[_]")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("dst = a[_]")

    matmul_c_i8_perfect = matmul_c_i8_perfect.call_eqv(ld_i8_v2, "ld_i8(_, _, _, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline("ld_i8_v2(_, _, _, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("src = B[_]")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("dst = b[_]")

    matmul_c_i8_perfect = matmul_c_i8_perfect.call_eqv(st_acc_i8_v2, "st_acc_i8(_, _, _, _, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline("st_acc_i8_v2(_, _, _, _, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("src = res[_]")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("dst = C[_]")

    matmul_c_i8_perfect = matmul_c_i8_perfect.call_eqv(zero_acc_i32_v2, "zero_acc_i32(_, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline("zero_acc_i32_v2(_, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("dst = res[_]")

    matmul_c_i8_perfect = matmul_c_i8_perfect.call_eqv(matmul_acc_i8_v2, "matmul_acc_i8(_, _, _, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline("matmul_acc_i8_v2(_, _, _, _, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("A = a[_]")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("B = b[_]")
    matmul_c_i8_perfect = matmul_c_i8_perfect.inline_window("C = res[_]")


    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("for k in _:_", "config_st_acc_i8(_, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("do_zero_acc_i32(_, _, _)", "config_st_acc_i8(_, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("config_zero()", "config_st_acc_i8(_, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after("config_st_acc_i8(_, _)", n_lifts=2)
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("res : _", "config_st_acc_i8(_, _)")
    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after("config_st_acc_i8(_, _)", n_lifts=2)


    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("do_ld_i8(_,_,_,_) #1", "config_matmul()")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("config_ld_i8(_,_) #1", "config_matmul()")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("do_ld_i8(_,_,_,_) #0", "config_matmul()")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("config_ld_i8(_,_) #0", "config_matmul()")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("b : _", "config_matmul()")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("a : _", "config_matmul()")
    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after("config_matmul()", n_lifts=1)
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("do_zero_acc_i32(_, _, _)", "config_matmul()")
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("config_zero()", "config_matmul()")
    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after("config_matmul()", n_lifts=2)
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder_stmts("res : _", "config_matmul()")
    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after("config_matmul()", n_lifts=2)


    matmul_c_i8_perfect = matmul_c_i8_perfect.lift_alloc('a : i8', n_lifts=3)
    matmul_c_i8_perfect = matmul_c_i8_perfect.lift_alloc('b : i8', n_lifts=3)


    # Real optimization
    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after('do_zero_acc_i32(_)', n_lifts=2)
    matmul_c_i8_perfect = matmul_c_i8_perfect.fission_after('for k in _:_', n_lifts=2)
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('j','k')
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('i','k')
    matmul_c_i8_perfect = matmul_c_i8_perfect.reorder('i','j')

    matmul_c_i8_perfect = matmul_c_i8_perfect.par_to_seq('for i in _:_ #1')
    matmul_c_i8_perfect = matmul_c_i8_perfect.par_to_seq('for j in _:_ #1')
    matmul_c_i8_perfect = matmul_c_i8_perfect.add_guard('config_ld_i8(_,_) #0', 'j #1', 0)
    matmul_c_i8_perfect = matmul_c_i8_perfect.add_guard('do_ld_i8(_) #0', 'j #1', 0)
    matmul_c_i8_perfect = matmul_c_i8_perfect.merge_guard('if j == 0:_ #0', 'if j == 0:_ #1')

    matmul_c_i8_perfect = matmul_c_i8_perfect.add_guard('config_ld_i8(_,_) #1', 'i #1', 0)
    matmul_c_i8_perfect = matmul_c_i8_perfect.add_guard('do_ld_i8(_) #1', 'i #1', 0)
    matmul_c_i8_perfect = matmul_c_i8_perfect.merge_guard('if i == 0:_ #0', 'if i == 0:_ #1')

    matmul_c_i8_perfect = matmul_c_i8_perfect.add_loop('for j in _:_ #0', 'k', 32)
    matmul_c_i8_perfect = matmul_c_i8_perfect.fuse_loop('for k in _:_ #0', 'for k in _:_ #1')
    matmul_c_i8_perfect = matmul_c_i8_perfect.par_to_seq('for j in _:_ #0')
    matmul_c_i8_perfect = matmul_c_i8_perfect.par_to_seq('for i in _:_ #0')
    matmul_c_i8_perfect = matmul_c_i8_perfect.fuse_loop('for j in _:_ #0', 'for j in _:_ #1')
    matmul_c_i8_perfect = matmul_c_i8_perfect.fuse_loop('for i in _:_ #0', 'for i in _:_ #1')

    matmul_c_i8_perfect = matmul_c_i8_perfect.par_to_seq('for k in _:_ #0')
    matmul_c_i8_perfect = matmul_c_i8_perfect.add_guard('config_zero(_) #0', 'k #0', 0)
    matmul_c_i8_perfect = matmul_c_i8_perfect.add_guard('do_zero_acc_i32(_) #0', 'k #0', 0)
    matmul_c_i8_perfect = matmul_c_i8_perfect.merge_guard('if k == 0:_ #0', 'if k == 0:_ #1')

    matmul_c_i8_perfect = matmul_c_i8_perfect.unroll('i')
    #matmul_c_i8_perfect = matmul_c_i8_perfect.unroll('j')
    matmul_c_i8_perfect = matmul_c_i8_perfect.simplify()

    T.add_proc(matmul_c_i8_perfect)
    T.add_proc(matmul_c_i8_cpu)

    T.start_timer('cpu')
    T.add_body([f'matmul_c_i8_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, false, x, y, z_cpu);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i8_perfect(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, false, x, y, z_gemmini);',
                f'gemmini_fence();'])
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


    print(matmul_c_i8_perfect)
"""
    #matmul_c_i8_perfect.check_effects()
"""
