from __future__ import annotations

import sys
sys.path.append(sys.path[0]+"/..")
sys.path.append(sys.path[0]+"/.")
from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import GEMM_SCRATCH, GEMM_ACCUM, MDRAM
from .gemmini import *
from .harness_gemmini import ENV, GemmTestBuilder
import pytest

# --------------------------------------------------------------------------- #
#   Basic Conv Test
# --------------------------------------------------------------------------- #


def test_conv_stride_1():
  T = GemmTestBuilder('conv_on_cpu_stride_1')
#  T.add_body(['gemm_init_mem();',
#              'init_mem();',
#              'gemmini_flush(0);',
#              ''])

  @proc
  def conv_on_cpu_stride_1(
      batch_size : size,
      out_dim    : size,
      out_channel: size,
      kernel_dim : size,
      in_channel : size,
      in_dim     : size,
      padding    : size,
      output     : i8[batch_size, out_dim, out_dim, out_channel],
      bias       : i32[out_channel],
      inp        : i8[batch_size, in_dim, in_dim, in_channel],
      weights    : i8[kernel_dim, kernel_dim, in_channel, out_channel],
      act        : bool,
      scale      : f32
      ):
      
      for b in par(0, batch_size):
          for orow in par(0, out_dim):
              for ocol in par(0, out_dim):
                  for och in par(0, out_channel):

                      res : i32
                      res = bias[och]
                      for krow in par(0, kernel_dim):
                          for kcol in par(0, kernel_dim):
                              for kch in par(0, in_channel):

                                  if ((orow+krow-padding) < 0 or (orow+krow-padding) >= in_dim
                                          or (ocol+kcol-padding) < 0 or (ocol+kcol-padding) >= in_dim):
                                      pass
                                  else:
                                      res += weights[krow,kcol,kch,och] * inp[b,orow+krow-padding,ocol+kcol-padding,kch]

                      if act == True:
                          res = relu(res)
                      
                      tmp_res1 : f32
                      tmp_res1 = res
                      tmp_res1 = tmp_res1 * scale
                      tmp_res2 : i8
                      clamp(tmp_res1, tmp_res2)
                      output[b,orow,ocol,och] = tmp_res2


  T.add_proc(conv_on_cpu_stride_1)
  T.compile().run()



def test_conv_stride_2():
  T = GemmTestBuilder('conv_on_cpu_stride_2')

  @proc
  def conv_on_cpu_stride_2(
      batch_size : size,
      out_dim    : size,
      out_channel: size,
      kernel_dim : size,
      in_channel : size,
      in_dim     : size,
      padding    : size,
      output     : i8[batch_size, out_dim, out_dim, out_channel],
      bias       : i32[out_channel],
      inp        : i8[batch_size, in_dim, in_dim, in_channel],
      weights    : i8[kernel_dim, kernel_dim, in_channel, out_channel],
      act        : bool,
      scale      : f32
      ):
      
      for b in par(0, batch_size):
          for orow in par(0, out_dim):
              for ocol in par(0, out_dim):
                  for och in par(0, out_channel):

                      res : i32
                      res = bias[och]
                      for krow in par(0, kernel_dim):
                          for kcol in par(0, kernel_dim):
                              for kch in par(0, in_channel):

                                  if ((orow*2+krow-padding) < 0 or (orow*2+krow-padding) >= in_dim
                                          or (ocol*2+kcol-padding) < 0 or (ocol*2+kcol-padding) >= in_dim):
                                      pass
                                  else:
                                      res += weights[krow,kcol,kch,och] * inp[b,orow*2+krow-padding,ocol*2+kcol-padding,kch]

                      if act == True:
                          res = relu(res)
                      
                      tmp_res1 : f32
                      tmp_res1 = res
                      tmp_res1 = tmp_res1 * scale
                      tmp_res2 : i8
                      clamp(tmp_res1, tmp_res2)
                      output[b,orow,ocol,och] = tmp_res2

  T.add_proc(conv_on_cpu_stride_2)
  T.compile().run()



