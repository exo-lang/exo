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
                                    w_s : i8 @ DRAM
                                    w_s = weights[krow,kcol,kch,och]

                                    i_s : i8 @ DRAM
                                    if (0 <= orow+krow-padding  and orow+krow-padding < in_dim):
                                        if (0 <= ocol+kcol-padding):
                                            if (ocol+kcol-padding < in_dim):
                                                i_s = inp[b,orow+krow-padding,ocol+kcol-padding,kch]
                                            else:
                                                i_s = 0.0
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

    return conv_on_cpu_stride_1


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
    T.alloc_dram_2i32('bias', 1, out_channel, '-1*j')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'j+k')
    T.alloc_dram_4i8('weights', out_channel, kernel_dim, kernel_dim, in_channel, 'i+k*3')

    conv = conv_on_cpu().rename("conv_3")
    conv = conv.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)

    conv = conv.split('ocol', 16, ['ocol_o', 'ocol_i'], tail='cut_and_guard')
    conv = conv.split('och', 16, ['och_o', 'och_i'], perfect=True)
    conv = conv.split('kch', 16, ['kch_o', 'kch_i'], perfect=True)
    print(conv)
    conv = conv.reorder('ocol_i', 'och_o')
    conv = conv.lift_alloc('res : _', n_lifts=3)
    conv = conv.fission_after('res[_] = _', n_lifts=3)
    conv = conv.fission_after('for krow in _:_', n_lifts=3)
    conv = conv.reorder('och_i', 'krow')
    conv = conv.reorder('och_i', 'kcol')
    conv = conv.reorder('och_i', 'kch_o')
    conv = conv.reorder('ocol_i', 'krow')
    conv = conv.reorder('ocol_i', 'kcol')
    conv = conv.reorder('ocol_i', 'kch_o')
    conv = conv.reorder('och_o', 'krow')
    conv = conv.simplify()
    conv = conv.lift_alloc('i_s : _', n_lifts=6)
    conv = conv.lift_alloc('w_s : _', n_lifts=1)
    conv = conv.lift_alloc('w_s : _', n_lifts=1, mode='col')
    conv = conv.reorder('och_o', 'kcol')
    conv = conv.reorder('och_o', 'kch_o')
    conv = conv.lift_alloc('w_s : _', n_lifts=5)
    conv = conv.fission_after('w_s = _', n_lifts=5)
    conv = conv.fission_after('if 0 <= orow + krow - 1 and orow + krow - 1 < 56: _', n_lifts=5)
    conv = conv.lift_if('if 0 <= orow + krow - 1 and orow + krow - 1 < 56: _', n_lifts=3)
    conv = conv.reorder('kch_o', 'ocol_i')
    conv = conv.add_ifelse('for ocol_i in _:_ #1', 'ocol_o == 0 and kcol == 0')
    conv = conv.partition_loop('ocol_i #1', 1)
    conv = conv.unroll('ocol_i #1')
    conv = conv.simplify()
    conv = conv.assert_if('if _:_ #2', False)
    conv = conv.assert_if('if _:_ #2', True)
    conv = conv.assert_if('if _:_ #2', True)
    conv = conv.assert_if('if _:_ #2', True)
    conv = conv.assert_if('if _:_ #2', True)
    conv = conv.assert_if('if 0 <= ocol_i + 48 + kcol - 1:_', True)
    conv = conv.add_ifelse('for ocol_i in _:_ #7', 'kcol == 2')
    conv = conv.partition_loop('ocol_i #7', 7)
    conv = conv.assert_if('if ocol_i + 48 + kcol - 1 < 56:_', True)
    conv = conv.unroll('ocol_i #8')
    conv = conv.assert_if('if 0 + 7 + 48 + kcol - 1 < 56:_', False)

    conv = conv.replace(ld_acc_i32_vector, 'for ocol_i in _:_ #0')
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.reorder('och_o', 'kch_i')
    conv = conv.replace(ld_i8_block_id1, 'for kch_i in _:_ #0')
    conv = conv.replace(do_zero_i8, 'for kch_o in _:_ #1')
    conv = conv.replace(ld_i8_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.replace(ld_i8_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.replace(zero_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #0')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #0')

    conv = conv.replace(ld_acc_i32_vector, 'for ocol_i in _:_ #0')
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.replace(ld_i8_block_id1, 'for kch_i in _:_ #0')
    conv = conv.replace(ld_i8_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.replace(do_zero_i8, 'for kch_o in _:_ #3')
    conv = conv.replace(ld_i8_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.replace(zero_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #0')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #0')

    conv = conv.set_memory('res', GEMM_ACCUM)
    conv = conv.set_memory('i_s', GEMM_SCRATCH)
    conv = conv.set_memory('w_s', GEMM_SCRATCH)

    conv = inline_vector(conv)
    conv = lift_config(conv, 'config_ld_acc_i32_vector(_)')

    conv = inline_ld_id1(conv)
    conv = lift_config(conv, 'config_ld_i8_id1(_)')

    conv = inline_matmul(conv)
    conv = lift_config(conv, 'config_matmul(_)')

    conv = inline_st(conv)
    conv = lift_config(conv, 'config_st_acc_i8(_)')

    conv = inline_vector(conv)
    conv = inline_ld_id1(conv)
    conv = inline_matmul(conv)
    conv = inline_st(conv)
    conv = conv.delete_config("config_ld_acc_i32_vector(_) #1")
    conv = conv.delete_config("config_ld_i8_id1(_) #1")
    conv = conv.delete_config("config_matmul(_) #1")
    conv = conv.delete_config("config_st_acc_i8(_) #1")
    conv = conv.simplify()

    # Real optimization
    conv = conv.fission_after('for ocol_o in _:_ #0')
    conv = conv.reorder('orow', 'ocol_o')
    conv = conv.split('orow', 28, ['orow_o', 'orow_i'], perfect=True)
    conv = conv.expand_dim('i_s: i8[_]', '30', 'krow + orow_i')
    conv = conv.par_to_seq('for krow in _:_')
    conv = conv.par_to_seq('for b in _:_')
    conv = conv.par_to_seq('for orow_o in _:_')
    conv = conv.par_to_seq('for orow_i in _:_')
    conv = conv.par_to_seq('for ocol_o in _:_')
    conv = conv.lift_alloc('i_s : _', n_lifts=5)
    conv = conv.lift_alloc('w_s : _', n_lifts=4)

    conv = conv.add_guard('for kch_o in _:_', 'ocol_o', 0)
    conv = conv.add_guard('for kch_o in _:_', 'b', 0)
    conv = conv.add_guard('for kch_o in _:_ #2', 'b', 0)
    conv = conv.add_guard('for kch_o in _:_', 'orow_o', 0)
    conv = conv.add_guard('for kch_o in _:_', 'orow_i', 0)
    conv = conv.add_guard('for kch_o in _:_ #2', 'orow_o #1', 0)
    conv = conv.add_guard('for kch_o in _:_ #2', 'orow_i #1', 0)
    conv = conv.add_unsafe_guard('ld_i8_block_id2(_) #0', 'orow_i == 0 or krow == 2')
    conv = conv.add_unsafe_guard('ld_i8_block_id2(_) #1', 'orow_i == 0 or krow == 2')
    conv = conv.add_unsafe_guard('ld_i8_block_id2(_) #2', 'orow_i == 0 or krow == 2')
    conv = conv.add_unsafe_guard('ld_i8_block_id2(_) #3', 'orow_i == 0 or krow == 2')


    conv = conv.split('orow_i', 7, ['orow_io', 'orow_ii'], perfect=True)
    conv = conv.lift_alloc('res : _', n_lifts=1)
    conv = conv.par_to_seq('for orow_io in _:_')
    conv = conv.lift_alloc('res : _', n_lifts=4)
    #conv = conv.unroll('ocol_i')
    #conv = conv.unroll('och_o')
    #conv = conv.unroll('kch_o')
    conv = conv.simplify()

    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, padding)
    T.add_proc(cpu)
    T.add_proc(conv)

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


    print(conv)
"""








    conv = conv.fission_after('if 0 <= 16 * ocol_o + ocol_i + kcol - 1 and 16 * ocol_o + ocol_i + kcol - 1 < 56 : _', n_lifts=1)


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
"""

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

@pytest.mark.skip()
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
