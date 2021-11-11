from __future__ import annotations

import pytest

from .gemmini import *
from .harness_gemmini import GemmTestBuilder


def conv_on_cpu():
    @proc
    def conv_max_pool(
        batch_size : size,
        out_dim    : size,
        out_channel: size,
        kernel_dim : size,
        in_channel : size,
        in_dim     : size,
        padding    : size,
        pool_size  : size,
        pool_padding : size,
        output     : i8[batch_size, (out_dim+2*pool_padding-pool_size)/2+1, (out_dim+2*pool_padding-pool_size)/2+1, out_channel],
        bias       : i32[1,out_channel],
        inp        : i8[batch_size, in_dim, in_dim, in_channel],
        weights    : i8[kernel_dim, kernel_dim, in_channel, out_channel],
        act        : bool,
        scale      : f32
        ):

        assert out_dim == (in_dim + 2*padding - kernel_dim)/2 + 1
        assert 0 <= padding < 16
        assert padding < out_dim

        zero : i8
        zero = 0.0
        for b in par(0, batch_size):
            for porow in par(0, (out_dim+2*pool_padding-pool_size)/2+1):
                for pocol in par(0, (out_dim+2*pool_padding-pool_size)/2+1):
                    for poch in par(0, out_channel):

                        running_max : i8
                        running_max = -128.0

                        for pwrow in par(0, pool_size):
                            for pwcol in par(0, pool_size):
                                if (porow*2+pwrow-pool_padding) < 0 or (porow*2+pwrow-pool_padding) >= out_dim or (pocol*2+pwcol-pool_padding) < 0 or (pocol*2+pwcol-pool_padding) >= out_dim:
                                    running_max = select(running_max, zero, zero, running_max)
                                else:
                                    res : i32
                                    res = bias[0,poch]

                                    for krow in par(0, kernel_dim):
                                        for kcol in par(0, kernel_dim):
                                            for kch in par(0, in_channel):
                                                if (0 <= (porow*2+pwrow-pool_padding)*2+krow-padding  and (porow*2+pwrow-pool_padding)*2+krow-padding < in_dim and
                                                        0 <= (pocol*2+pwcol-pool_padding)*2+kcol-padding and (pocol*2+pwcol-pool_padding)*2+kcol-padding < in_dim):
                                                    res += weights[krow,kcol,kch,poch] * inp[b,(porow*2+pwrow-pool_padding)*2+krow-padding,(pocol*2+pwcol-pool_padding)*2+kcol-padding,kch]

                                    tmp_res1 : f32
                                    acc_scale(res, tmp_res1, scale)
                                    tmp_res2 : i8
                                    clamp(tmp_res1, tmp_res2)
                                    if act == True:
                                        tmp_res2 = relu(tmp_res2)

                                    running_max = select(running_max, tmp_res2, tmp_res2, running_max)

                                if pwrow == pool_size - 1 and pwcol == pool_size - 1:
                                    output[b,porow*2+pwrow-pool_padding,pocol*2+pwcol-pool_padding,poch] = running_max

    return conv_max_pool


def test_conv_1():
    T = GemmTestBuilder('conv_1')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
#                'gemm_acc_init_mem();',
#                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_1_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 64
    kernel_dim = 7
    in_channel = 3
    padding    = 3
    in_dim     = 224
    pool_size  = 3
    pool_stride = 2
    pool_padding = 1
    out_dim    = 112
    assert 0 <= padding < 16
    assert padding < out_dim

    T.alloc_dram_f32('scale', '0.33')
    T.alloc_dram_2i32('bias', 1, out_channel, 'i')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, 'i')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, 'j')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'k+r')
    T.alloc_dram_4i8('weights', kernel_dim, kernel_dim, in_channel, out_channel, 'k+r')

    @proc
    def gemmini():
        x : i8 @GEMM_SCRATCH
        pass

    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, pool_size, pool_stride)
    T.add_proc(cpu)
    T.add_proc(gemmini)

    T.start_timer('cpu')
    T.add_body([f'conv_max_pool(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.add_body([f'gemmini(ctxt);'])

    T.compile().run()


"""
"""
