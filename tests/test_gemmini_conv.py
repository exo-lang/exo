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
def test_conv():
  T = GemmTestBuilder('conv')
  T.add_body(['gemm_init_mem();',
              'init_mem();',
              'gemmini_flush(0);',
              ''])

  """
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
  """

  @proc
  def conv_on_cpu(
      batch_size : size,
      out_dim    : size,
      out_channel: size,
      kernel_dim : size,
      in_channel : size,
      in_dim     : size,
      stride     : size,
      dilation   : size,
      padding    : size,
      output     : i8[batch_size, out_dim, out_dim, out_channel],
      bias       : i32[out_channel],
      inp        : i8[batch_size, in_dim, in_dim, in_channel],
      weights    : i8[out_channel, kernel_dim, kernel_dim, in_channel]
      ):
      
      for b in par(0, batch_size):
          for orow in par(0, out_dim):
              for ocol in par(0, out_dim):
                  for och in par(0, out_channel):

                      output[b,orow,ocol,och] = bias[och]
                      for krow in par(0, kernel_dim):
                          for kcol in par(0, kernel_dim):
                              for kch in par(0, in_channel):

                                  irow = orow * stride + krow * dilation - padding
                                  icol = ocol * stride + kcol * dilation - padding

                                  pixel : i8
                                  if irow < 0 or irow >= in_dim or icol < 0 or icol >= in_dim:
                                      pixel = 0
                                  else:
                                      pixel = inp[b,irow,icol,kch]

                                  output[b,orow,ocol,och] += weights[och,krow,kcol,kch] * pixel


"""
  T.add_proc(conv_on_cpu)

  T.start_timer('cpu')
  T.add_body([f'conv_on_cpu({NN}, {MM}, {KK}, a_scale, b_scale, c_scale, false, true, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_cpu, {NN}, 1 }});',
              f'gemmini_fence();'])
  T.stop_timer('cpu', 'Cycles for CPU version')

  
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
"""


