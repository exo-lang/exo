from __future__ import annotations

import pytest

from .gemmini import *
from .harness_gemmini import GemmTestBuilder


def conv_on_cpu():
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

        assert out_dim == in_dim + 2*padding - kernel_dim + 1

        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim):
                    for och in par(0, out_channel):

                        res : i32
                        res = bias[0,och]
                        for krow in par(0, kernel_dim):
                            for kcol in par(0, kernel_dim):
                                for kch in par(0, in_channel):
                                    if (0 <= orow+krow-padding  and orow+krow-padding < in_dim and
                                            0 <= ocol+kcol-padding and ocol+kcol-padding < in_dim):
                                        res += weights[krow,kcol,kch,och] * inp[b,orow+krow-padding,ocol+kcol-padding,kch]

                        tmp_res1 : f32
                        #tmp_res1 = res
                        #tmp_res1 = tmp_res1 * scale
                        acc_scale(res, tmp_res1, scale)
                        tmp_res2 : i8
                        clamp(tmp_res1, tmp_res2)
                        if act == True:
                            tmp_res2 = relu(tmp_res2)

                        output[b,orow,ocol,och] = tmp_res2

    return conv_on_cpu_stride_1

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

    assert out_dim == in_dim + 2*padding - kernel_dim + 1
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

    config_st_acc_i8(scale, stride(output, 2), act)
    config_ld_i8(one, stride(bias, 0))
    config_ld_i8_id1(one, stride(inp, 2))
    config_ld_i8_id2(one, stride(weights, 2))
    config_matmul()
    for och in par(0, out_channel/16):

        res : i32[DIM_SIZE,16] @ GEMM_ACCUM
        for l in par(0, DIM_SIZE):
            do_ld_acc_i32(1, 16, bias[ 0:1, 16*och:16*(och+1) ], res[l:l+1, :])

        for kcol in par(0, kernel_dim):
            for krow in par(0, kernel_dim):
                for kch in par(0, in_channel/16):
                    if 0 <= orow+krow-padding and orow+krow-padding < in_dim:
                        in_scratch : i8[DIM_SIZE,16] @ GEMM_SCRATCH
                        weight_scratch : i8[16,16] @ GEMM_SCRATCH

                        if DIM_LO+kcol-padding < 0 and DIM_HI+kcol-padding <= in_dim:
                            do_zero_i8(-(DIM_LO+kcol-padding), 16, in_scratch[0:-(DIM_LO+kcol-padding), :])
                            do_ld_i8_id1(DIM_SIZE+(DIM_LO+kcol-padding), 16,
                                inp[ b, orow+krow-padding, 0:DIM_HI+kcol-padding, 16*kch:16*(kch+1)],
                                in_scratch[-(DIM_LO+kcol-padding):, :])
                        if DIM_LO+kcol-padding >= 0 and DIM_HI+kcol-padding > in_dim and DIM_LO+kcol-padding < in_dim:
                            do_ld_i8_id1(DIM_SIZE-(DIM_HI+kcol-padding-in_dim), 16,
                                inp[ b, orow+krow-padding, DIM_LO+kcol-padding:, 16*kch:16*(kch+1)],
                                in_scratch[0:DIM_SIZE-(DIM_HI+kcol-padding-in_dim), :])
                            do_zero_i8((DIM_HI+kcol-padding-in_dim), 16, in_scratch[DIM_SIZE-(DIM_HI+kcol-padding-in_dim):, :])
                        if DIM_LO+kcol-padding < 0 and DIM_HI+kcol-padding > in_dim:
                            do_zero_i8(-(DIM_LO+kcol-padding), 16, in_scratch[0:-(DIM_LO+kcol-padding), :])
                            do_ld_i8_id1(in_dim, 16,
                                inp[ b, orow+krow-padding, :, 16*kch:16*(kch+1)],
                                in_scratch[-(DIM_LO+kcol-padding):DIM_SIZE-(DIM_HI+kcol-padding-in_dim), :])
                            do_zero_i8((DIM_HI+kcol-padding-in_dim), 16, in_scratch[DIM_SIZE-(DIM_HI+kcol-padding-in_dim):, :])
                        if DIM_LO+kcol-padding >= 0 and DIM_HI+kcol-padding <= in_dim:
                            do_ld_i8_id1(DIM_SIZE, 16,
                            inp[ b, orow+krow-padding, DIM_LO+kcol-padding:DIM_HI+kcol-padding, 16*kch:16*(kch+1)],
                            in_scratch)

                        do_ld_i8_id2(16, 16, weights[ krow, kcol, 16*kch:16*(kch+1), 16*och:16*(och+1)], weight_scratch)
                        do_matmul_acc_i8(DIM_SIZE,16,16,in_scratch,weight_scratch,res)

        do_st_acc_i8(DIM_SIZE,16, res, output[b, orow, DIM_LO:DIM_HI, 16*och:16*(och+1)])

@pytest.mark.skip()
def test_conv_3():
    T = GemmTestBuilder('conv_3')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_3_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 64
    kernel_dim = 3
    in_channel = 64
    padding    = 1
    in_dim     = 56
    out_dim    = int((in_dim + 2*padding - kernel_dim)/1 + 1)
    assert 0 <= padding < 16
    assert padding < out_dim
    assert out_dim == 56

    T.alloc_dram_f32('scale', '1.0')
    T.alloc_dram_2i32('bias', 1, out_channel, '-10000')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, '1')
    T.alloc_dram_4i8('weights', out_channel, kernel_dim, kernel_dim, in_channel, '1')

    @proc
    def conv_3(
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

        assert out_dim == in_dim + 2*padding - kernel_dim + 1
        assert 0 <= padding < 16
        assert padding < out_dim

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))

                if out_dim%16 > 0:
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)


    gemmini = conv_3
    gemmini = gemmini.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.simplify()

    gemmini = gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=3)
    gemmini = gemmini.fission_after('config_ld_i8(_) #0', n_lifts=3)
    gemmini = gemmini.fission_after('config_ld_i8_id1(_) #0', n_lifts=3)
    gemmini = gemmini.fission_after('config_ld_i8_id2(_) #0', n_lifts=3)
    gemmini = gemmini.fission_after('config_matmul() #0', n_lifts=3)
    gemmini = gemmini.reorder_stmts('for ocol in _:_ #0', 'config_st_acc_i8(_) #1')
    gemmini = gemmini.reorder_stmts('for ocol in _:_ #0', 'config_ld_i8(_) #1')
    gemmini = gemmini.reorder_stmts('for ocol in _:_ #0', 'config_ld_i8_id1(_) #1')
    gemmini = gemmini.reorder_stmts('for ocol in _:_ #0', 'config_ld_i8_id2(_) #1')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #1', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #1', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id1(_) #1', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id2(_) #1', n_lifts=2)

    gemmini = gemmini.lift_alloc('res:_', n_lifts=1)
    gemmini = gemmini.lift_alloc('res:_ #0', n_lifts=1)
    gemmini = gemmini.lift_alloc('in_scratch:_', n_lifts=5)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=5)
    gemmini = gemmini.lift_alloc('in_scratch:_ #0', n_lifts=1)
    gemmini = gemmini.lift_alloc('weight_scratch:_ #0', n_lifts=1)

    gemmini = gemmini.par_to_seq('for b in _:_')
    gemmini = gemmini.par_to_seq('for orow in _:_')

    gemmini = gemmini.lift_alloc('res:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('in_scratch:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('weight_scratch:_', n_lifts=3)

    gemmini = gemmini.par_to_seq('for och in _:_')
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #0', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #1', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #2', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #3', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #4', 'och #1', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #5', 'och #1', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #6', 'och #1', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #7', 'och #1', 0)
    gemmini = gemmini.par_to_seq('for ocol in _:_')
    gemmini = gemmini.add_guard('do_ld_i8_id2(_) #0', 'ocol #0', 0)

    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    T.add_proc(cpu)
    T.add_proc(gemmini)

    T.start_timer('cpu')

    T.add_body([f'conv_on_cpu_stride_1(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_3(ctxt, output_gemmini, bias, inp, weights, false, scale);',
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
def test_conv_17():
    T = GemmTestBuilder('conv_17')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_17_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 128
    kernel_dim = 3
    in_channel = 128
    padding    = 1
    in_dim     = 28
    out_dim    = int((in_dim + 2*padding - kernel_dim)/1 + 1)
    assert 0 <= padding < 16
    assert padding < out_dim
    assert out_dim == 28

    T.alloc_dram_f32('scale', '1.0')
    T.alloc_dram_2i32('bias', 1, out_channel, '-10000')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'i')
    T.alloc_dram_4i8('weights', out_channel, kernel_dim, kernel_dim, in_channel, 'j*10')

    @proc
    def conv_17(
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

        assert out_dim == in_dim + 2*padding - kernel_dim + 1
        assert 0 <= padding < 16
        assert padding < out_dim

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))

                if out_dim%16 > 0:
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)


    gemmini = conv_17
    gemmini = gemmini.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.simplify()

    gemmini = gemmini.unroll('ocol')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id1(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id2(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_matmul() #0', n_lifts=2)
    gemmini = gemmini.reorder_stmts('for och in _:_ #0', 'config_st_acc_i8(_) #1')
    gemmini = gemmini.reorder_stmts('for och in _:_ #0', 'config_ld_i8(_) #1')
    gemmini = gemmini.reorder_stmts('for och in _:_ #0', 'config_ld_i8_id1(_) #1')
    gemmini = gemmini.reorder_stmts('for och in _:_ #0', 'config_ld_i8_id2(_) #1')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #1', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #1', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id1(_) #1', n_lifts=2)
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

    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #0', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #1', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #2', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #3', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #4', 'och #1', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #5', 'och #1', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #6', 'och #1', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #7', 'och #1', 0)

    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    T.add_proc(cpu)
    T.add_proc(gemmini)

    T.start_timer('cpu')

    T.add_body([f'conv_on_cpu_stride_1(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_17(ctxt, output_gemmini, bias, inp, weights, false, scale);',
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
def test_conv_30():
    T = GemmTestBuilder('conv_30')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_30_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 256
    kernel_dim = 3
    in_channel = 256
    padding    = 1
    in_dim     = 14
    out_dim    = int((in_dim + 2*padding - kernel_dim)/1 + 1)
    assert 0 <= padding < 16
    assert padding < out_dim
    assert out_dim == 14

    T.alloc_dram_f32('scale', '1.0')
    T.alloc_dram_2i32('bias', 1, out_channel, '-10000')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'i')
    T.alloc_dram_4i8('weights', out_channel, kernel_dim, kernel_dim, in_channel, 'j*10')

    @proc
    def conv_30(
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

        assert out_dim == in_dim + 2*padding - kernel_dim + 1
        assert 0 <= padding < 16
        assert padding < out_dim

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))

                if out_dim%16 > 0:
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)


    gemmini = conv_30
    gemmini = gemmini.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.simplify()

    gemmini = gemmini.unroll('ocol')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id1(_) #0', n_lifts=2)
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

    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #0', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #1', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #2', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #3', 'och #0', 0)

    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    T.add_proc(cpu)
    T.add_proc(gemmini)

    T.start_timer('cpu')

    T.add_body([f'conv_on_cpu_stride_1(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_30(ctxt, output_gemmini, bias, inp, weights, false, scale);',
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


def test_conv_49():
    T = GemmTestBuilder('conv_49')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_49_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 512
    kernel_dim = 3
    in_channel = 512
    padding    = 1
    in_dim     = 7
    out_dim    = int((in_dim + 2*padding - kernel_dim)/1 + 1)
    assert 0 <= padding < 16
    assert padding < out_dim
    assert out_dim == 7

    T.alloc_dram_f32('scale', '1.0')
    T.alloc_dram_2i32('bias', 1, out_channel, '-10000')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'i')
    T.alloc_dram_4i8('weights', out_channel, kernel_dim, kernel_dim, in_channel, 'j*10')

    @proc
    def conv_49(
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

        assert out_dim == in_dim + 2*padding - kernel_dim + 1
        assert 0 <= padding < 16
        assert padding < out_dim

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))

                if out_dim%16 > 0:
                    conv_partial_padding(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding, output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)


    gemmini = conv_49
    gemmini = gemmini.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.inline('conv_partial_padding(_) #0')
    gemmini = gemmini.simplify()

    gemmini = gemmini.unroll('ocol')
    gemmini = gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8(_) #0', n_lifts=2)
    gemmini = gemmini.fission_after('config_ld_i8_id1(_) #0', n_lifts=2)
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

    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #0', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #1', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #2', 'och #0', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id1(_) #3', 'och #0', 0)

    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    T.add_proc(cpu)
    T.add_proc(gemmini)

    T.start_timer('cpu')

    T.add_body([f'conv_on_cpu_stride_1(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_49(ctxt, output_gemmini, bias, inp, weights, false, scale);',
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
"""
