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


@pytest.mark.skip()
def test_conv_stride_1_gemmini():
  T = GemmTestBuilder('conv_on_cpu_stride_1_gemmini')
  T.add_body(['gemm_init_mem();',
#              'init_mem();',
              'gemm_acc_init_mem();',
              'gemmini_flush(0);',
              ''])
  T.add_body(["conv_stride_1_gemmini_lib_Context *ctxt;"])

  batch_size = 1
  out_dim    = 16
  out_channel= 16
  kernel_dim = 1
  in_channel = 16
  in_dim     = 31
  padding    = 1
  out_dim    = int((in_dim + 2 * padding - kernel_dim) / 2)
  T.alloc_dram_f32('scale', '1.0f')

  T.alloc_dram_2i32('bias', 1, out_channel, '0')
  T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
  T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
  T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, '1')
  T.alloc_dram_4i8('weights', kernel_dim, kernel_dim, in_channel, out_channel, '1')

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
      bias       : i32[1,out_channel],
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
                      res = bias[0,och]
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

  @proc
  def conv_on_cpu_stride_1_gemmini(
      batch_size : size,
      out_dim    : size,
      out_channel: size,
      kernel_dim : size,
      in_channel : size,
      in_dim     : size,
      padding    : size,
      output     : i8[batch_size, out_dim, out_dim, out_channel],
      bias       : i32[1, out_channel],
      inp        : i8[batch_size, in_dim, in_dim, in_channel],
      weights    : i8[kernel_dim, kernel_dim, in_channel, out_channel],
      act        : bool,
      scale      : f32
      ):
      
      assert 2*out_dim == in_dim + 2*padding - kernel_dim
      assert in_dim > 4*out_dim + 32*kernel_dim + 32 # really??
      
      one : f32
      one = 1.0
      for b in par(0, batch_size):
          for orow in par(0, out_dim):
              for ocol in par(0, out_dim/16):
                  for och in par(0, out_channel/16):

                      res : i32[16,16] @ GEMM_ACCUM
                      zero_acc_i32(16,16,res)
                      #for l in par(0, 16):
                      #    ld_acc_i32(1, 16, one, bias[ 0:1, 16*och:16*(och+1) ], res[l:l+1, :])

                      for krow in par(0, kernel_dim):
                          for kcol in par(0, kernel_dim):
                              for kch in par(0, in_channel/16):

                                  if ((orow+krow-padding) < 0 or (orow+krow-padding) >= in_dim
                                          or (ocol+kcol-padding) < 0 or 16*(ocol+kcol-padding+1) >= in_dim):
                                      pass
                                  else:
                                      in_scratch : i8[16,16] @ GEMM_SCRATCH
                                      weight_scratch : i8[16,16] @ GEMM_SCRATCH

                                      ld_i8(16, 16, one, inp[ b, orow+krow-padding, 16*(ocol+kcol-padding):16*(ocol+kcol-padding+1), 16*kch:16*(kch+1)], in_scratch)
                                      ld_i8(16, 16, one, weights[ krow, kcol, 16*kch:16*(kch+1), 16*och:16*(och+1)], weight_scratch)

                                      matmul_acc_i8(16,16,16,in_scratch,weight_scratch,res)

                      st_acc_i8(16,16, scale, act, res, output[b, orow, 16*ocol:16*(ocol+1), 16*och:16*(och+1)])


  T.add_proc(conv_on_cpu_stride_1)
  T.add_proc(conv_on_cpu_stride_1_gemmini)

  T.start_timer('cpu')
  T.add_body([f'conv_on_cpu_stride_1(ctxt, {batch_size}, {out_dim}, {out_channel}, {kernel_dim},',
              f'{in_channel}, {in_dim}, {padding}, output_cpu, bias, inp, weights, false, scale);',
              f'gemmini_fence();'])
  T.stop_timer('cpu', 'Cycles for CPU version')

  T.start_timer('gemmini')
  T.add_body([f'conv_on_cpu_stride_1_gemmini(ctxt, {batch_size}, {out_dim}, {out_channel}, {kernel_dim},',
              f'{in_channel}, {in_dim}, {padding}, output_gemmini, bias, inp, weights, false, scale);',
              f'gemmini_fence();'])
  T.stop_timer('gemmini', 'Cycles for GEMMINI version')
  
  T.add_body([f'if(check_eq_4i8({batch_size},{out_dim},{out_dim},{out_channel}, output_cpu, output_gemmini)) {{',
               '    printf("Correct\\n");',
               '} else {',
               '    printf("Results Don\'t Match\\n");',
               '    printf("Correct Result (output_cpu):\\n");',
              f'    print_4i8({batch_size},{out_dim},{out_dim},{out_channel}, output_cpu);',
               '    printf("Computed Roundtrip (output_gemmini):\\n");',
              f'    print_4i8({batch_size},{out_dim},{out_dim},{out_channel}, output_gemmini);',
               '    exit(1);',
               '}',
               ''])
  
  T.compile().run()




@pytest.mark.skip()
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
      bias       : i32[1,out_channel],
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
                      res = bias[0,och]
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





@pytest.mark.skip()
def test_conv_stride_2_gemmini():
  T = GemmTestBuilder('conv_on_cpu_stride_2_gemmini')

  @proc
  def conv_on_cpu_stride_2_gemmini(
      batch_size : size,
      out_dim    : size,
      out_channel: size,
      kernel_dim : size,
      in_channel : size,
      in_dim     : size,
      padding    : size,
      output     : i8[batch_size, out_dim, out_dim, out_channel],
      bias       : i32[1, out_channel],
      inp        : i8[batch_size, in_dim, in_dim, in_channel],
      weights    : i8[kernel_dim, kernel_dim, in_channel, out_channel],
      act        : bool,
      scale      : f32
      ):
      
      assert 2*out_dim == in_dim + 2*padding - kernel_dim
      assert in_dim > 4*out_dim + 32*kernel_dim + 32 # really??
      
      one : f32
      one = 1.0
      for b in par(0, batch_size):
          for orow in par(0, out_dim):
              for ocol in par(0, out_dim/16):
                  for och in par(0, out_channel/16):

                      res : i32[16,16] @ GEMM_ACCUM
                      for l in par(0, 16):
                          ld_acc_i32(1, 16, one, bias[ 0:1, 16*och:16*(och+1) ], res[l:l+1, :])

                      for krow in par(0, kernel_dim):
                          for kcol in par(0, kernel_dim):
                              for kch in par(0, in_channel/16):

                                  in_scratch : i8[16,16] @ GEMM_SCRATCH
                                  weight_scratch : i8[16,16] @ GEMM_SCRATCH

                                  ld_i8_stride_2(16, 16, one,
                                          inp[ b, orow*2+krow, 16*(ocol*2+kcol):16*2*(ocol*2+kcol+1) , 16*kch:16*(kch+1)], in_scratch)
                                  ld_i8(16, 16, one, weights[ krow, kcol, 16*kch:16*(kch+1), 16*och:16*(och+1)], weight_scratch)

                                  matmul_acc_i8(16,16,16,in_scratch,weight_scratch,res)

                      st_acc_i8(16,16, scale, act, res, output[b, orow, 16*ocol:16*(ocol+1), 16*och:16*(och+1)])


  T.add_proc(conv_on_cpu_stride_2_gemmini)
  T.compile().run()


