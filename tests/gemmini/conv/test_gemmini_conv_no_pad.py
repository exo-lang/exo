from __future__ import annotations

import pytest

from ..gemmini import *
from ..harness_gemmini import GemmTestBuilder


def conv_on_cpu():
    @proc
    def conv_on_cpu_stride_1(
        batch_size : size,
        out_dim    : size,
        out_channel: size,
        kernel_dim : size,
        in_channel : size,
        in_dim     : size,
        output     : i8[batch_size, out_dim, out_dim, out_channel],
        bias       : i32[1,out_channel],
        inp        : i8[batch_size, in_dim, in_dim, in_channel],
        weights    : i8[kernel_dim, kernel_dim, in_channel, out_channel],
        act        : bool,
        scale      : f32
        ):

        assert out_dim == in_dim - kernel_dim + 1

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
                                    i_s = inp[b,orow+krow,ocol+kcol,kch]

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


def split_fission_dim(conv):
    conv = conv.split('ocol', 16, ['ocol_o', 'ocol_i'], tail='cut_and_guard')
    conv = conv.split('och', 16, ['och_o', 'och_i'], perfect=True)
    conv = conv.split('kch', 16, ['kch_o', 'kch_i'], perfect=True)
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
    conv = conv.lift_alloc('w_s : _', n_lifts=3)
    conv = conv.fission_after('w_s = _', n_lifts=5)
    conv = conv.fission_after('i_s = _', n_lifts=5)

    return conv

def replace_div_part(conv):
    conv = conv.replace(ld_acc_i32_vector, 'for ocol_i in _:_ #0')
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.reorder('och_o', 'kch_i')
    conv = conv.replace(ld_i8_block_id1, 'for kch_i in _:_ #0')
    conv = conv.reorder('kch_o', 'ocol_i')
    conv = conv.replace(ld_i8_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #0')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #0')

    return conv

def replace_mod_part(conv):
    conv = conv.replace(ld_acc_i32_vector, 'for ocol_i in _:_ #0')
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.replace(ld_i8_block_id1, 'for kch_i in _:_ #0')
    conv = conv.replace(ld_i8_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #0')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #0')

    return conv

def inline_div_part(conv):
    conv = inline_vector(conv)
    conv = lift_config(conv, 'config_ld_acc_i32_vector(_)')
    conv = inline_ld_id1(conv)
    conv = lift_config(conv, 'config_ld_i8_id1(_)')
    conv = inline_matmul(conv)
    conv = lift_config(conv, 'config_matmul(_)')
    conv = inline_st(conv)
    conv = lift_config(conv, 'config_st_acc_i8(_)')

    return conv

def inline_mod_part(conv):
    conv = inline_vector(conv)
    conv = inline_ld_id1(conv)
    conv = inline_matmul(conv)
    conv = inline_st(conv)
    conv = conv.delete_config("config_ld_acc_i32_vector(_) #1")
    conv = conv.delete_config("config_ld_i8_id1(_) #1")
    conv = conv.delete_config("config_matmul(_) #1")
    conv = conv.delete_config("config_st_acc_i8(_) #1")
    conv = conv.simplify()

    return conv

def set_memory(conv):
    conv = conv.set_memory('res', GEMM_ACCUM)
    conv = conv.set_memory('i_s', GEMM_SCRATCH)
    conv = conv.set_memory('w_s', GEMM_SCRATCH)

    return conv


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
    in_dim     = 58
    out_dim    = int((in_dim - kernel_dim)/1 + 1)
    assert out_dim == 56

    T.alloc_dram_f32('scale', '1.0')
    T.alloc_dram_2i32('bias', 1, out_channel, '-1*j')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'j+k+r*3')
    T.alloc_dram_4i8('weights', out_channel, kernel_dim, kernel_dim, in_channel, 'i+k*3+r')

    conv = conv_on_cpu().rename("conv_3")
    conv = conv.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)

    conv = split_fission_dim(conv)

    conv = replace_div_part(conv)
    conv = replace_mod_part(conv)

    conv = set_memory(conv)

    conv = inline_div_part(conv)
    conv = inline_mod_part(conv)

    # Real optimization
    conv = conv.lift_alloc('w_s : _', n_lifts=2)
    conv = conv.fission_after('for ocol_o in _:_ #0')
    conv = conv.reorder('orow', 'ocol_o')
    conv = conv.split('orow', 28, ['orow_o', 'orow_i'], perfect=True)
    conv = conv.expand_dim('i_s: i8[_]', '30', 'krow + orow_i')
    #conv = conv.par_to_seq('for krow in _:_')
    #conv = conv.par_to_seq('for b in _:_')
    #conv = conv.par_to_seq('for orow_o in _:_')
    #conv = conv.par_to_seq('for orow_i in _:_')
    #conv = conv.par_to_seq('for ocol_o in _:_')
    [ (conv := conv.par_to_seq(s)) for s in ['for krow in _:_', 'for b in _:_', 'for orow_o in _:_', 'for orow_i in _:_', 'for ocol_o in _:_'] ]
    conv = conv.lift_alloc('i_s : _', n_lifts=5)
    conv = conv.lift_alloc('w_s : _', n_lifts=4)

    [ (conv := conv.add_guard(s, i, 0)) for (s,i) in [('for kch_o in _:_', 'ocol_o'), ('for kch_o in _:_', 'b'), ('for kch_o in _:_ #2', 'b'), ('for kch_o in _:_', 'orow_o'), ('for kch_o in _:_', 'orow_i'), ('for kch_o in _:_ #2', 'orow_o #1'), ('for kch_o in _:_ #2', 'orow_i #1')] ]
    #conv = conv.add_guard('for kch_o in _:_', 'ocol_o', 0)
    #conv = conv.add_guard('for kch_o in _:_', 'b', 0)
    #conv = conv.add_guard('for kch_o in _:_ #2', 'b', 0)
    #conv = conv.add_guard('for kch_o in _:_', 'orow_o', 0)
    #conv = conv.add_guard('for kch_o in _:_', 'orow_i', 0)
    #conv = conv.add_guard('for kch_o in _:_ #2', 'orow_o #1', 0)
    #conv = conv.add_guard('for kch_o in _:_ #2', 'orow_i #1', 0)
    conv = conv.add_unsafe_guard('ld_i8_block_id2(_) #0', 'orow_i == 0 or krow == 2')
    conv = conv.add_unsafe_guard('ld_i8_block_id2(_) #1', 'orow_i == 0 or krow == 2')

    conv = conv.split('orow_i', 7, ['orow_io', 'orow_ii'], perfect=True)
    conv = conv.lift_alloc('res : _', n_lifts=1)
    conv = conv.par_to_seq('for orow_io in _:_')
    conv = conv.lift_alloc('res : _', n_lifts=4)
    conv = conv.unroll('och_o')
    #conv = conv.unroll('kch_o')
    #conv = conv.unroll('kcol')
    conv = conv.simplify()

    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)
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
    in_dim     = 30
    out_dim    = int((in_dim - kernel_dim)/1 + 1)
    assert out_dim == 28

    T.alloc_dram_f32('scale', '1.0')
    T.alloc_dram_2i32('bias', 1, out_channel, '-1*j')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'j+k+r*3')
    T.alloc_dram_4i8('weights', out_channel, kernel_dim, kernel_dim, in_channel, 'i+k*3+r')

    conv = conv_on_cpu().rename("conv_17")
    conv = conv.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)

    conv = split_fission_dim(conv)
    conv = conv.replace(ld_acc_i32_vector, 'for ocol_i in _:_ #0')
    conv = conv.split('och_o #1', 4, ['och_o_o', 'och_o_i'], perfect=True)
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.reorder('och_o_i', 'kch_i')
    conv = conv.replace(ld_i8_block_id1, 'for kch_i in _:_ #0')
    conv = conv.split('kch_o #1', 4, ['kch_o_o', 'kch_o_i'], perfect=True)
    conv = conv.reorder('kch_o_i', 'ocol_i')
    conv = conv.replace(ld_i8_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #0')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #0')

    conv = conv.replace(ld_acc_i32_vector, 'for ocol_i in _:_ #0')
    conv = conv.split('och_o #4', 4, ['och_o_o', 'och_o_i'], perfect=True)
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.reorder('och_o_i', 'kch_i')
    conv = conv.replace(ld_i8_block_id1, 'for kch_i in _:_ #0')
    conv = conv.split('kch_o #3', 4, ['kch_o_o', 'kch_o_i'], perfect=True)
    conv = conv.reorder('kch_o_i', 'ocol_i')
    conv = conv.replace(ld_i8_block_id2, 'for ocol_i in _:_ #0')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #0')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #0')

    conv = set_memory(conv)
    conv = inline_div_part(conv)
    conv = inline_mod_part(conv)

    # Real optimization
    conv = conv.unroll('ocol_o')
    conv = conv.fission_after('for och_o in _:_ #2', n_lifts=2)
    conv = conv.split('orow', 14, ['orow_o', 'orow_i'], perfect=True)
    conv = conv.expand_dim('i_s: i8[_]', '16', 'krow + orow_i')
    conv = conv.lift_alloc('w_s : _', n_lifts=2)
    conv = conv.split('b', 4, ['bo', 'bi'], perfect=True)
    conv = conv.par_to_seq('for krow in _:_')
    conv = conv.par_to_seq('for bi in _:_')
    conv = conv.par_to_seq('for orow_o in _:_')
    conv = conv.par_to_seq('for orow_i in _:_')
    conv = conv.lift_alloc('i_s : _', n_lifts=4)
    conv = conv.lift_alloc('w_s : _', n_lifts=3)
    conv = conv.add_guard('for kch_o in _:_', 'bi', 0)
    conv = conv.add_guard('for kch_o in _:_', 'orow_o', 0)
    conv = conv.add_guard('for kch_o in _:_', 'orow_i', 0)
    conv = conv.add_guard('for kch_o in _:_ #2', 'bi #1', 0)
    conv = conv.add_guard('for kch_o in _:_ #2', 'orow_o #1', 0)
    conv = conv.add_guard('for kch_o in _:_ #2', 'orow_i #1', 0)
    conv = conv.add_unsafe_guard('ld_i8_block_id2(_) #0', 'orow_i == 0 or krow == 2')
    conv = conv.add_unsafe_guard('ld_i8_block_id2(_) #1', 'orow_i == 0 or krow == 2')
    conv = conv.lift_alloc('res : _', n_lifts=3)

    conv = conv.unroll('och_o')
    conv = conv.unroll('kch_o_o')
    conv = conv.unroll('kch_o')
    conv = conv.unroll('kcol')
    conv = conv.simplify()

    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)
    T.add_proc(cpu)
    T.add_proc(conv)

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
    in_dim     = 16
    out_dim    = int((in_dim - kernel_dim)/1 + 1)
    assert out_dim == 14

    T.alloc_dram_f32('scale', '2.0')
    T.alloc_dram_2i32('bias', 1, out_channel, '-1*j')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'j+i+k*2+r*3')
    T.alloc_dram_4i8('weights', out_channel, kernel_dim, kernel_dim, in_channel, 'i+k*3+r')

    conv = conv_on_cpu().rename("conv_30")
    conv = conv.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)

    conv = conv.split('och', 64, ['och_out', 'och'], perfect=True)
    conv = conv.reorder('ocol', 'och_out')
    conv = conv.reorder('orow', 'och_out')
    conv = conv.reorder('b', 'och_out')
    conv = split_fission_dim(conv)
    conv = replace_div_part(conv)
    conv = set_memory(conv)
    conv = inline_div_part(conv)

    conv = conv.lift_alloc('w_s : _', n_lifts=2)
    conv = conv.split('orow', 7, ['orow_o', 'orow_i'], perfect=True)
    conv = conv.expand_dim('i_s: i8[_]', '9', 'krow + orow_i')
    conv = conv.lift_alloc('res : _', n_lifts=1)
    conv = conv.par_to_seq('for krow in _:_')
    conv = conv.par_to_seq('for b in _:_')
    conv = conv.par_to_seq('for orow_o in _:_')
    conv = conv.par_to_seq('for orow_i in _:_')
    conv = conv.par_to_seq('for och_out in _:_')
    conv = conv.lift_alloc('i_s : _', n_lifts=5)
    conv = conv.lift_alloc('w_s : _', n_lifts=4)
    conv = conv.lift_alloc('res : _', n_lifts=3)
    conv = conv.add_guard('for kch_o in _:_', 'b', 0)
    conv = conv.add_guard('for kch_o in _:_', 'orow_o', 0)
    conv = conv.add_guard('for kch_o in _:_', 'orow_i', 0)
    conv = conv.add_unsafe_guard('ld_i8_block_id2(_) #0', 'orow_i == 0 or krow == 2')

    conv = conv.unroll('och_o')
    #conv = conv.unroll('kch_o')
    #conv = conv.unroll('kcol')
    conv = conv.simplify()


    cpu = conv_on_cpu()
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)
    T.add_proc(cpu)
    T.add_proc(conv)

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



