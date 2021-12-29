from __future__ import annotations

import pytest

from .gemmini import *
from .harness_gemmini import GemmTestBuilder

pytest.skip("skipping gemmini tests that are bitrotted",
            allow_module_level=True)

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

        assert out_channel == 64
        assert kernel_dim == 7
        assert in_channel == 3
        assert padding == 3
        assert in_dim == 224
        assert pool_size == 3
        assert pool_padding == 1
        assert out_dim == 112

        zero : i8
        zero = 0.0
        min_ : i8
        min_ = -128.0
        for b in par(0, batch_size):
            for porow in par(0, (out_dim+2*pool_padding-pool_size)/2+1):
                for pocol in par(0, (out_dim+2*pool_padding-pool_size)/2+1):
                    for poch in par(0, out_channel):

                        running_max : i8
                        running_max = min_

                        for pwrow in seq(0, pool_size):
                            for pwcol in seq(0, pool_size):
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

                        output[b,porow,pocol,poch] = running_max

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
    def gemmini(
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
        scale      : f32):

        assert out_channel == 64
        assert kernel_dim == 7
        assert in_channel == 3
        assert padding == 3
        assert in_dim == 224
        assert pool_size == 3
        assert pool_padding == 1
        assert out_dim == 112

        config_st_acc_i8(scale, stride(output, 2), act)
        config_ld_i8(one, stride(bias, 0))
        config_ld_i8_s2_id1(one, stride(inp, 2))
        config_ld_i8_id2(one, stride(weights, 2))
        config_matmul()
        zero : i8
        zero = 0.0
        min_ : i8
        min_ = -128.0
        min_array : i8[16,16] @ DRAM
        for i in par(0, 16):
            for j in par(0, 16):
                min_array[i,j] = min_

        for b in par(0, batch_size):
            for porow in par(0, (out_dim+2*pool_padding-pool_size)/2+1):
                for pocol in par(0, ((out_dim+2*pool_padding-pool_size)/2+1)/16):
                    for poch in par(0, out_channel/16):

                        running_max : i8[16, 16] @ GEMM_ACCUM
                        do_ld_acc_i32(16, 16, min_array, running_max)

                        for pwrow in seq(0, pool_size):
                            for pwcol in seq(0, pool_size):
                                if (porow*2+pwrow-pool_padding) < 0 or (porow*2+pwrow-pool_padding) >= out_dim or (pocol*16*2+pwcol-pool_padding) < 0 or (pocol*16*2+pwcol-pool_padding) >= out_dim:
                                    if (porow*2+pwrow-pool_padding) < 0:
                                        zero_acc_i32(-(porow*2+pwrow-pool_padding), running_max)
                                    if (porow*2+pwrow-pool_padding) >= out_dim:
                                        zero_acc_i32((porow*2+pwrow-pool_padding) - out_dim)
                                    if (pocol*16*2+pwcol-pool_padding) < 0:
                                        pass
                                    if (pocol*16*2+pwcol-pool_padding) >= out_dim:
                                        pass


                                    running_max = select(running_max, zero, zero, running_max)
                                else:
                                    res : i32[16,16] @ GEMM_ACCUM
                                    for l in par(0, 16):
                                        do_ld_acc_i32(1, 16, bias[ 0:1, 16*poch:16*(poch+1) ], res[l:l+1, :])

                                    for kcol in par(0, kernel_dim):
                                        for krow in par(0, kernel_dim):
                                            for kch in par(0, in_channel/16):
                                                if 0 <= orow*2+krow-padding and orow*2+krow-padding < in_dim:
                                                    in_scratch : i8[DIM_SIZE,16] @ GEMM_SCRATCH
                                                    weight_scratch : i8[16,16] @ GEMM_SCRATCH

                                                    if DIM_LO*2+kcol-padding < 0 and DIM_HI*2+kcol-padding <= in_dim:
                                                        do_zero_i8(1, 16, in_scratch[0:1, :])
                                                        do_ld_i8_s2_id1(DIM_SIZE-1, 16,
                                                            inp[ b, orow*2+krow-1, 1:(DIM_SIZE-1)*2, 16*kch:16*(kch+1)],
                                                            in_scratch[1:DIM_SIZE, :])
                                                    if DIM_LO*2+kcol-1 >= 0 and DIM_HI*2+kcol-1 > in_dim and DIM_LO*2+kcol-1 < in_dim and (in_dim-(DIM_LO*2+kcol-1))%2 == 0:
                                                        do_ld_i8_s2_id1((in_dim-(DIM_LO*2+kcol-1))/2, 16,
                                                            inp[ b, orow*2+krow-1, DIM_LO*2+kcol-1:in_dim-1, 16*kch:16*(kch+1)],
                                                            in_scratch[0:(in_dim-(DIM_LO*2+kcol-1))/2, :])
                                                        do_zero_i8(DIM_SIZE-((in_dim-(DIM_LO*2+kcol-1))/2), 16, in_scratch[(in_dim-(DIM_LO*2+kcol-1))/2:, :])
                                                    if DIM_LO*2+kcol-1 >= 0 and DIM_HI*2+kcol-1 > in_dim and DIM_LO*2+kcol-1 < in_dim and (in_dim-(DIM_LO*2+kcol-1))%2 == 1:
                                                        do_ld_i8_s2_id1((in_dim-(DIM_LO*2+kcol-1))/2+1, 16,
                                                            inp[ b, orow*2+krow-1, DIM_LO*2+kcol-1:, 16*kch:16*(kch+1)],
                                                            in_scratch[0:(in_dim-(DIM_LO*2+kcol-1))/2+1, :])
                                                        if DIM_SIZE-((in_dim-(DIM_LO*2+kcol-1))/2+1) > 0:
                                                            do_zero_i8(DIM_SIZE-((in_dim-(DIM_LO*2+kcol-1))/2+1), 16, in_scratch[(in_dim-(DIM_LO*2+kcol-1))/2+1:, :])
                                                    if DIM_LO*2+kcol-1 >= 0 and DIM_HI*2+kcol-1 <= in_dim:
                                                        do_ld_i8_s2_id1(DIM_SIZE, 16,
                                                        inp[ b, orow*2+krow-1, DIM_LO*2+kcol-1:DIM_HI*2+kcol-2, 16*kch:16*(kch+1)],
                                                        in_scratch)


                                                    do_ld_i8_id2(16, 16, weights[ krow, kcol, 16*kch:16*(kch+1), 16*och:16*(och+1)], weight_scratch)
                                                    do_matmul_acc_i8(DIM_SIZE,16,16,in_scratch,weight_scratch,res)

                                    running_max = select(running_max, tmp_res2, tmp_res2, running_max)

                        do_st_acc_i8(16,16, running_max, output[b, porow, pocol*16:(pocol+1)*16, 16*poch:16*(poch+1)])


    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, pool_size, pool_padding)

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
