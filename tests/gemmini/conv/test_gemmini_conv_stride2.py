from __future__ import annotations

import pytest

from exo.platforms.gemmini import *
from ..harness_gemmini import GemmTestBuilder

pytest.skip("skipping gemmini tests that are bitrotted",
            allow_module_level=True)

def conv_on_cpu():
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

        assert out_dim == (in_dim + 2*padding - kernel_dim)/2 + 1
        assert 0 <= padding < 16
        assert padding < out_dim

        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim):
                    for och in par(0, out_channel):

                        res : i32
                        res = bias[0,och]
                        for krow in par(0, kernel_dim):
                            for kcol in par(0, kernel_dim):
                                for kch in par(0, in_channel):
                                    if (0 <= orow*2+krow-padding  and orow*2+krow-padding < in_dim):
                                        w_s : i8 @ DRAM
                                        i_s : i8 @ DRAM

                                        w_s = weights[krow,kcol,kch,och]

                                        if (0 <= ocol*2+kcol-padding):
                                            if (ocol*2+kcol-padding < in_dim):
                                                i_s = inp[b,orow*2+krow-padding,ocol*2+kcol-padding,kch]
                                            else:
                                                i_s = 0.0
                                        else:
                                            i_s = 0.0


                                        a2 : i32
                                        b2 : i32
                                        a2 = i_s
                                        b2 = w_s

                                        res += a2 * b2

                        src_tmp : i32
                        src_tmp = res
                        tmp_res1 : f32
                        acc_scale(src_tmp, tmp_res1, scale)
                        tmp_res2 : i8
                        clamp(tmp_res1, tmp_res2)
                        if act == True:
                            tmp_res2 = relu(tmp_res2)

                        output[b,orow,ocol,och] = tmp_res2

    return conv_on_cpu_stride_2

"""
@proc
def orig_conv_partial_padding(
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
    scale      : f32,
    b          : index,
    orow       : index,
    one        : f32,
    DIM_SIZE   : size,
    DIM_LO     : index,
    DIM_HI     : index
    ):

    assert out_dim == (in_dim + 2*padding - kernel_dim)/2 + 1
    assert 0 <= padding < 16
    assert padding < out_dim
    assert 0 <= b and b < batch_size
    assert 0 <= orow and orow < out_dim
    assert DIM_LO < DIM_HI
    assert DIM_HI - DIM_LO <= 16
    assert DIM_HI <= out_dim
    assert DIM_HI - DIM_LO == DIM_SIZE
    assert 0 <= DIM_LO
    assert DIM_HI-padding > 0
    assert in_channel%16 == 0
    assert out_channel%16 == 0
    assert padding == 1
    assert DIM_SIZE > 2

    for och in par(0, out_channel/16):

        res : i32[DIM_SIZE,16] @ GEMM_ACCUM
        for l in par(0, DIM_SIZE):
            ld_acc_i32(1, 16, one, bias[ 0:1, 16*och:16*(och+1) ], res[l:l+1, :])

        for kcol in par(0, kernel_dim):
            for krow in par(0, kernel_dim):
                for kch in par(0, in_channel/16):
                    if 0 <= orow*2+krow-padding and orow*2+krow-padding < in_dim:
                        in_scratch : i8[DIM_SIZE,16] @ GEMM_SCRATCH
                        weight_scratch : i8[16,16] @ GEMM_SCRATCH

                        if DIM_LO*2+kcol-padding < 0 and DIM_HI*2+kcol-padding <= in_dim:
                            zero_i8(1, 16, in_scratch[0:1, :])
                            ld_i8_s2(DIM_SIZE-1, 16, one,
                                inp[ b, orow*2+krow-1, 1:(DIM_SIZE-1)*2, 16*kch:16*(kch+1)],
                                in_scratch[1:DIM_SIZE, :])
                        if DIM_LO*2+kcol-1 >= 0 and DIM_HI*2+kcol-1 > in_dim and DIM_LO*2+kcol-1 < in_dim and (in_dim-(DIM_LO*2+kcol-1))%2 == 0:
                            ld_i8_s2((in_dim-(DIM_LO*2+kcol-1))/2, 16, one,
                                inp[ b, orow*2+krow-1, DIM_LO*2+kcol-1:in_dim-1, 16*kch:16*(kch+1)],
                                in_scratch[0:(in_dim-(DIM_LO*2+kcol-1))/2, :])
                            zero_i8(DIM_SIZE-((in_dim-(DIM_LO*2+kcol-1))/2), 16, in_scratch[(in_dim-(DIM_LO*2+kcol-1))/2:, :])
                        if DIM_LO*2+kcol-1 >= 0 and DIM_HI*2+kcol-1 > in_dim and DIM_LO*2+kcol-1 < in_dim and (in_dim-(DIM_LO*2+kcol-1))%2 == 1:
                            ld_i8_s2((in_dim-(DIM_LO*2+kcol-1))/2+1, 16, one,
                                inp[ b, orow*2+krow-1, DIM_LO*2+kcol-1:, 16*kch:16*(kch+1)],
                                in_scratch[0:(in_dim-(DIM_LO*2+kcol-1))/2+1, :])
                            if DIM_SIZE-((in_dim-(DIM_LO*2+kcol-1))/2+1) > 0:
                                zero_i8(DIM_SIZE-((in_dim-(DIM_LO*2+kcol-1))/2+1), 16, in_scratch[(in_dim-(DIM_LO*2+kcol-1))/2+1:, :])
                        if DIM_LO*2+kcol-1 >= 0 and DIM_HI*2+kcol-1 <= in_dim:
                            ld_i8_s2(DIM_SIZE, 16, one,
                            inp[ b, orow*2+krow-1, DIM_LO*2+kcol-1:DIM_HI*2+kcol-2, 16*kch:16*(kch+1)],
                            in_scratch)

                        ld_i8(16, 16, one, weights[ krow, kcol, 16*kch:16*(kch+1), 16*och:16*(och+1)], weight_scratch)
                        matmul_acc_i8(DIM_SIZE,16,16,in_scratch,weight_scratch,res)

        st_acc_i8(DIM_SIZE,16, scale, act, res, output[b, orow, DIM_LO:DIM_HI, 16*och:16*(och+1)])


@proc
def conv_partial_padding(
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
    scale      : f32,
    b          : index,
    orow       : index,
    one        : f32,
    DIM_SIZE   : size,
    DIM_LO     : index,
    DIM_HI     : index
    ):
    assert out_dim == (in_dim + 2*padding - kernel_dim)/2 + 1
    assert 0 <= padding < 16
    assert padding < out_dim
    assert 0 <= b and b < batch_size
    assert 0 <= orow and orow < out_dim
    assert DIM_LO < DIM_HI
    assert DIM_HI - DIM_LO <= 16
    assert DIM_HI <= out_dim
    assert DIM_HI - DIM_LO == DIM_SIZE
    assert 0 <= DIM_LO
    assert DIM_HI-padding > 0
    assert in_channel%16 == 0
    assert out_channel%16 == 0
    assert padding == 1

    config_st_acc_i8(scale, stride(output, 2), act)
    config_ld_i8(one, stride(bias, 0))
    config_ld_i8_s2_id1(one, stride(inp, 2))
    config_ld_i8_id2(one, stride(weights, 2))
    config_matmul()
    for och in par(0, out_channel/16):

        res : i32[DIM_SIZE,16] @ GEMM_ACCUM
        for l in par(0, DIM_SIZE):
            do_ld_acc_i32(1, 16, bias[ 0:1, 16*och:16*(och+1) ], res[l:l+1, :])

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

        do_st_acc_i8(DIM_SIZE,16, res, output[b, orow, DIM_LO:DIM_HI, 16*och:16*(och+1)])
"""

def test_conv_13():
    T = GemmTestBuilder('conv_13')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_13_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 128
    kernel_dim = 3
    in_channel = 128
    padding    = 1
    in_dim     = 56
    out_dim    = int((in_dim + 2*padding - kernel_dim)/2 + 1)
    assert 0 <= padding < 16
    assert padding < out_dim
    assert out_dim == 28

    T.alloc_dram_f32('scale', '0.33')
    T.alloc_dram_2i32('bias', 1, out_channel, 'i')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, 'i')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, 'j')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'k+r')
    T.alloc_dram_4i8('weights', kernel_dim, kernel_dim, in_channel, out_channel, 'k+r')

    conv = conv_on_cpu().rename("conv_13")
    conv = conv.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    conv = conv_basic_opt(conv, 'if 0 <= orow*2 + krow - 1 and orow*2 + krow - 1 < 56: _', 'if 0 <= (16 * ocol_o + ocol_i)*2 + kcol - 1: _',
            'if 0 <= (ocol_i + 28 / 16 * 16)*2 + kcol - 1: _', 'if 0 <= (16 * ocol_o + 0)*2 + kcol - 1: _ #0', 'if (16 * ocol_o + 0)*2 + kcol - 1 < 56: _ #0')

    # Size specific asserts
    conv = conv.unroll('ocol_o')
    conv = conv.assert_if('if _:_ #2', True)
    conv = conv.assert_if('if _:_ #2', True)
    conv = conv.assert_if('if _:_ #2', True)
    conv = conv.assert_if('if 0 <= (ocol_i + 28 / 16 * 16)*2 + kcol - 1:_', True)
    conv = conv.assert_if('if (ocol_i + 28 / 16 * 16)*2 + kcol - 1 < 56 :_', True)

    # Now start replacing
    conv = conv.replace(ld_acc_i32_vector, 'for och_i in _:_ #0')
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.replace(ld_i8, 'for kch_i in _:_ #0')
    conv = conv.replace(ld_i8_vector, 'for kch_i in _:_ #0')
    conv = conv.replace(zero_i8_vector, 'for kch_i in _:_ #0')
    conv = conv.replace(ld_i8_s2, 'for ocol_i in _:_ #1')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #1')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #1')

    conv = conv.replace(ld_acc_i32_vector, 'for och_i in _:_ #0')
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.replace(ld_i8, 'for kch_i in _:_ #0')
    conv = conv.replace(ld_i8_s2, 'for ocol_i in _:_ #2')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #2')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #2')

    conv = conv.set_memory('res', GEMM_ACCUM)
    conv = conv.set_memory('i_s', GEMM_SCRATCH)
    conv = conv.set_memory('w_s', GEMM_SCRATCH)


    cpu = conv_on_cpu().partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    T.add_proc(cpu)
    T.add_proc(conv)

    T.start_timer('cpu')
    T.add_body([f'conv_on_cpu_stride_2(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_13(ctxt, output_gemmini, bias, inp, weights, false, scale);',
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


    print(conv)
"""

    @proc
    def conv_13(
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

        assert out_dim == (in_dim + 2*padding - kernel_dim)/2 + 1
        assert 0 <= padding < 16
        assert padding < out_dim
        assert in_channel%16 == 0
        assert out_channel%16 == 0
        assert padding == 1

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding,
                                            output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))
                if out_dim%16 > 0:
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding,
                                            output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)


    gemmini = conv_13.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.simplify()

    gemmini = gemmini.unroll('ocol')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_s2_id1(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id2(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_matmul() #0', n_lifts=2)
    gemmini = gemmini.reorder_stmts('for och in _:_ #0', 'config_st_acc_i8(_) #1')
    gemmini = gemmini.reorder_stmts('for och in _:_ #0', 'config_ld_i8(_) #1')
    gemmini = gemmini.reorder_stmts('for och in _:_ #0', 'config_ld_i8_s2_id1(_) #1')
    gemmini = gemmini.reorder_stmts('for och in _:_ #0', 'config_ld_i8_id2(_) #1')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #1', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #1', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_s2_id1(_) #1', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id2(_) #1', n_lifts=2)

    gemmini = gemmini.lift_alloc('res:_', n_lifts=1)
    gemmini = gemmini.lift_alloc('in_scratch:_', n_lifts=5)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=4)

    gemmini = gemmini.par_to_seq('for b in _:_')
    gemmini = gemmini.par_to_seq('for orow in _:_')
    gemmini = gemmini.par_to_seq('for och in _:_')

    gemmini = gemmini.lift_alloc('res:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('in_scratch:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=3)

    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #0', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #1', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #2', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #3', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #4', 'och #1', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #5', 'och #1', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #6', 'och #1', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #7', 'och #1', 0)

    gemmini = gemmini.simplify()




def test_conv_26():
    T = GemmTestBuilder('conv_26')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_26_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 256
    kernel_dim = 3
    in_channel = 256
    padding    = 1
    in_dim     = 28
    out_dim    = int((in_dim + 2*padding - kernel_dim)/2 + 1)
    assert 0 <= padding < 16
    assert padding < out_dim
    assert out_dim == 14

    T.alloc_dram_f32('scale', '1.0f')
    T.alloc_dram_2i32('bias', 1, out_channel, 'i')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, 'i')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, 'j')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'k+r')
    T.alloc_dram_4i8('weights', kernel_dim, kernel_dim, in_channel, out_channel, 'k+r')

    @proc
    def conv_26(
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

        assert out_dim == (in_dim + 2*padding - kernel_dim)/2 + 1
        assert 0 <= padding < 16
        assert padding < out_dim
        assert in_channel%16 == 0
        assert out_channel%16 == 0
        assert padding == 1

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding,
                                            output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))

                if out_dim%16 > 0:
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding,
                                            output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)

    gemmini = conv_26.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.simplify()

    gemmini = gemmini.unroll('ocol')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_s2_id1(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id2(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_matmul() #0', n_lifts=2)

    gemmini = gemmini.lift_alloc('res:_', n_lifts=1)
    gemmini = gemmini.lift_alloc('in_scratch:_', n_lifts=5)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=4)

    gemmini = gemmini.par_to_seq('for b in _:_')
    gemmini = gemmini.par_to_seq('for orow in _:_')
    gemmini = gemmini.par_to_seq('for och in _:_')

    gemmini = gemmini.lift_alloc('res:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('in_scratch:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=3)

    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #0', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #1', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #2', 'och #0', 0)

    gemmini = gemmini.simplify()

    cpu = conv_on_cpu().partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    T.add_proc(cpu)
    T.add_proc(gemmini)

    T.start_timer('cpu')
    T.add_body([f'conv_on_cpu_stride_2(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_26(ctxt, output_gemmini, bias, inp, weights, false, scale);',
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



def test_conv_45():
    T = GemmTestBuilder('conv_45')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_45_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 512
    kernel_dim = 3
    in_channel = 512
    padding    = 1
    in_dim     = 14
    out_dim    = int((in_dim + 2*padding - kernel_dim)/2 + 1)
    assert 0 <= padding < 16
    assert padding < out_dim
    assert out_dim == 7

    T.alloc_dram_f32('scale', '1.0f')
    T.alloc_dram_2i32('bias', 1, out_channel, 'i')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, 'i')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, 'j')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'k+r')
    T.alloc_dram_4i8('weights', kernel_dim, kernel_dim, in_channel, out_channel, 'k+r')

    @proc
    def conv_45(
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

        assert out_dim == (in_dim + 2*padding - kernel_dim)/2 + 1
        assert 0 <= padding < 16
        assert padding < out_dim
        assert in_channel%16 == 0
        assert out_channel%16 == 0
        assert padding == 1

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding,
                                            output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))

                if out_dim%16 > 0:
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding,
                                            output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)

    gemmini = conv_45.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.simplify()

    gemmini = gemmini.unroll('ocol')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_s2_id1(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id2(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_matmul() #0', n_lifts=2)

    gemmini = gemmini.lift_alloc('res:_', n_lifts=1)
    gemmini = gemmini.lift_alloc('in_scratch:_', n_lifts=5)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=4)

    gemmini = gemmini.par_to_seq('for b in _:_')
    gemmini = gemmini.par_to_seq('for orow in _:_')
    gemmini = gemmini.par_to_seq('for och in _:_')

    gemmini = gemmini.lift_alloc('res:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('in_scratch:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=3)

    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #0', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #1', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_s2_id1(_) #2', 'och #0', 0)

    gemmini = gemmini.simplify()

    cpu = conv_on_cpu().partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    T.add_proc(cpu)
    T.add_proc(gemmini)

    T.start_timer('cpu')
    T.add_body([f'conv_on_cpu_stride_2(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_45(ctxt, output_gemmini, bias, inp, weights, false, scale);',
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





@proc
def conv_partial_no_padding(
    batch_size : size,
    out_dim    : size,
    out_channel: size,
    kernel_dim : size,
    in_channel : size,
    in_dim     : size,
    output     : i8[batch_size, out_dim, out_dim, out_channel],
    bias       : i32[1, out_channel],
    inp        : i8[batch_size, in_dim, in_dim, in_channel],
    weights    : i8[kernel_dim, kernel_dim, in_channel, out_channel],
    act        : bool,
    scale      : f32,
    b          : index,
    orow       : index,
    one        : f32,
    DIM_SIZE   : size,
    DIM_LO     : index,
    DIM_HI     : index
    ):
    assert out_dim == (in_dim - kernel_dim)/2 + 1
    assert 0 <= b and b < batch_size
    assert 0 <= orow and orow < out_dim
    assert DIM_LO < DIM_HI
    assert DIM_HI - DIM_LO <= 16
    assert DIM_HI <= out_dim
    assert DIM_HI - DIM_LO == DIM_SIZE
    assert 0 <= DIM_LO
    assert in_channel%16 == 0
    assert out_channel%16 == 0

    config_st_acc_i8(scale, stride(output, 2), act)
    config_ld_i8(one, stride(bias, 0))
    config_ld_i8_s2_id1(one, stride(inp, 2))
    config_ld_i8_id2(one, stride(weights, 2))
    config_matmul()
    for och in par(0, out_channel/16):

        res : i32[DIM_SIZE,16] @ GEMM_ACCUM
        for l in par(0, DIM_SIZE):
            do_ld_acc_i32(1, 16, bias[ 0:1, 16*och:16*(och+1) ], res[l:l+1, :])

        for kcol in par(0, kernel_dim):
            for krow in par(0, kernel_dim):
                for kch in par(0, in_channel/16):
                    in_scratch : i8[DIM_SIZE,16] @ GEMM_SCRATCH
                    weight_scratch : i8[16,16] @ GEMM_SCRATCH

                    do_ld_i8_s2_id1(DIM_SIZE, 16,
                    inp[ b, orow*2+krow, DIM_LO*2+kcol:DIM_HI*2+kcol-1, 16*kch:16*(kch+1)],
                        in_scratch)

                    do_ld_i8_id2(16, 16, weights[ krow, kcol, 16*kch:16*(kch+1), 16*och:16*(och+1)], weight_scratch)
                    do_matmul_acc_i8(DIM_SIZE,16,16,in_scratch,weight_scratch,res)

        do_st_acc_i8(DIM_SIZE,16, res, output[b, orow, DIM_LO:DIM_HI, 16*och:16*(och+1)])



def test_conv_47():
    T = GemmTestBuilder('conv_47')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_47_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 2048
    kernel_dim = 1
    in_channel = 1024
    padding    = 0
    in_dim     = 14
    out_dim    = int((in_dim + 2*padding - kernel_dim)/2 + 1)
    assert 0 <= padding < 16
    assert padding < out_dim
    assert out_dim == 7

    T.alloc_dram_f32('scale', '1.0f')
    T.alloc_dram_2i32('bias', 1, out_channel, 'i')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, 'i')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, 'j')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'k')
    T.alloc_dram_4i8('weights', kernel_dim, kernel_dim, in_channel, out_channel, 'r')

    @proc
    def conv_47(
        batch_size : size,
        out_dim    : size,
        out_channel: size,
        kernel_dim : size,
        in_channel : size,
        in_dim     : size,
        padding    : index,
        output     : i8[batch_size, out_dim, out_dim, out_channel],
        bias       : i32[1, out_channel],
        inp        : i8[batch_size, in_dim, in_dim, in_channel],
        weights    : i8[kernel_dim, kernel_dim, in_channel, out_channel],
        act        : bool,
        scale      : f32
        ):

        assert out_dim == (in_dim + 2*padding - kernel_dim)/2 + 1
        assert 0 <= padding < 16
        assert padding < out_dim
        assert padding == 0
        assert in_channel%16 == 0
        assert out_channel%16 == 0

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial_no_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim,
                                            output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))

                if out_dim%16 > 0:
                    conv_partial_no_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim,
                                            output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)

    gemmini = conv_47.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    gemmini = gemmini.inline('conv_partial_no_padding(_) #0')
    gemmini = gemmini.inline('conv_partial_no_padding(_) #0')
    gemmini = gemmini.simplify()

    gemmini = gemmini.unroll('ocol')
    gemmini = gemmini.unroll('kcol')
    gemmini = gemmini.unroll('krow')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_s2_id1(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id2(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_matmul() #0', n_lifts=2)

    gemmini = gemmini.lift_alloc('res:_', n_lifts=1)
    gemmini = gemmini.lift_alloc('in_scratch:_', n_lifts=5)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=1)

    gemmini = gemmini.par_to_seq('for b in _:_')
    gemmini = gemmini.par_to_seq('for orow in _:_')
    gemmini = gemmini.par_to_seq('for och in _:_')

    gemmini = gemmini.lift_alloc('res:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=3)

    gemmini = gemmini.simplify()

    cpu = conv_on_cpu().partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    T.add_proc(cpu)
    T.add_proc(gemmini)

    T.start_timer('cpu')
    T.add_body([f'conv_on_cpu_stride_2(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_47(ctxt, output_gemmini, bias, inp, weights, false, scale);',
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
    print(gemmini)
"""
