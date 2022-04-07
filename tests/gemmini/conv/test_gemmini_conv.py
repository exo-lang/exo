from __future__ import annotations

import pytest

from exo.platforms.gemmini import *
from ..harness_gemmini import GemmTestBuilder

pytest.skip("skipping gemmini tests that are bitrotted",
            allow_module_level=True)
# --------------------------------------------------------------------------- #
#   Basic Conv Test
# --------------------------------------------------------------------------- #

@proc
def orig_conv_partial(
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
    DIM_HI     : size
    ):

    assert in_dim == out_dim + kernel_dim - 1
    assert 0 <= b and b < batch_size
    assert 0 <= orow and orow < out_dim
    assert DIM_LO < DIM_HI
    assert DIM_HI - DIM_LO <= 16
    assert DIM_HI <= out_dim
    assert DIM_HI - DIM_LO == DIM_SIZE
    assert 0 <= DIM_LO

    for och in par(0, out_channel/16):

        res : i32[DIM_SIZE,16] @ GEMM_ACCUM
        for l in par(0, DIM_SIZE):
            ld_acc_i32(1, 16, one, bias[ 0:1, 16*och:16*(och+1) ], res[l:l+1, :])

        for kcol in par(0, kernel_dim):
            for krow in par(0, kernel_dim):
                for kch in par(0, in_channel/16):
                    in_scratch : i8[DIM_SIZE,16] @ GEMM_SCRATCH
                    weight_scratch : i8[16,16] @ GEMM_SCRATCH

                    #config_ld_i8(one, stride(inp, 0))
                    ld_i8(DIM_SIZE, 16, one,
                            inp[ b, orow+krow, kcol+DIM_LO:kcol+DIM_HI, 16*kch:16*(kch+1)],
                            in_scratch)
                    #config_ld_i8(one, stride(weights, 0))
                    ld_i8(16, 16, one, weights[ krow, kcol, 16*kch:16*(kch+1), 16*och:16*(och+1)], weight_scratch)

                    matmul_acc_i8(DIM_SIZE,16,16,in_scratch,weight_scratch,res)

                if in_channel%16 > 0:
                    in_scratch : i8[DIM_SIZE,16] @ GEMM_SCRATCH
                    weight_scratch : i8[in_channel%16,16] @ GEMM_SCRATCH

                    ld_i8(DIM_SIZE, in_channel%16, one,
                            inp[b,orow+krow, kcol+DIM_LO:kcol+DIM_HI, in_channel-in_channel%16:],
                            in_scratch)
                    ld_i8(in_channel%16, 16, one,
                            weights[ krow, kcol, in_channel-in_channel%16:, 16*och:16*(och+1)],
                            weight_scratch)

                    #config_matmul()
                    matmul_acc_i8(DIM_SIZE,16,in_channel%16,in_scratch,weight_scratch,res)

        #config_st_acc_i8(scale, stride(dst, 0))
        st_acc_i8(DIM_SIZE,16, scale, act, res, output[b, orow, DIM_LO:DIM_HI, 16*och:16*(och+1)])


@proc
def conv_partial(
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
    DIM_HI     : size
    ):

    assert in_dim == out_dim + kernel_dim - 1
    assert 0 <= b and b < batch_size
    assert 0 <= orow and orow < out_dim
    assert DIM_LO < DIM_HI
    assert DIM_HI - DIM_LO <= 16
    assert DIM_HI <= out_dim
    assert DIM_HI - DIM_LO == DIM_SIZE
    assert 0 <= DIM_LO

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
                    in_scratch : i8[DIM_SIZE,16] @ GEMM_SCRATCH
                    weight_scratch : i8[16,16] @ GEMM_SCRATCH

                    do_ld_i8_id1(DIM_SIZE, 16,
                            inp[ b, orow+krow, kcol+DIM_LO:kcol+DIM_HI, 16*kch:16*(kch+1)],
                            in_scratch)
                    do_ld_i8_id2(16, 16, weights[ krow, kcol, 16*kch:16*(kch+1), 16*och:16*(och+1)], weight_scratch)

                    do_matmul_acc_i8(DIM_SIZE,16,16,in_scratch,weight_scratch,res)

                if in_channel%16 > 0:
                    in_scratch : i8[DIM_SIZE,16] @ GEMM_SCRATCH
                    weight_scratch : i8[in_channel%16,16] @ GEMM_SCRATCH

                    do_ld_i8_id1(DIM_SIZE, in_channel%16,
                            inp[b,orow+krow, kcol+DIM_LO:kcol+DIM_HI, in_channel-in_channel%16:],
                            in_scratch)
                    do_ld_i8_id2(in_channel%16, 16,
                            weights[ krow, kcol, in_channel-in_channel%16:, 16*och:16*(och+1)],
                            weight_scratch)

                    do_matmul_acc_i8(DIM_SIZE,16,in_channel%16,in_scratch,weight_scratch,res)

        do_st_acc_i8(DIM_SIZE, 16, res, output[b, orow, DIM_LO:DIM_HI, 16*och:16*(och+1)])


# TODO: Config optimization is super buggy... handwrinting for now..
#print(conv_partial)

def conv_cpu():
    @proc
    def conv_on_cpu(
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
                                    res += weights[krow,kcol,kch,och] * inp[b,orow+krow,ocol+kcol,kch]

                        tmp_res1 : f32
                        #tmp_res1 = res
                        #tmp_res1 = tmp_res1 * scale
                        acc_scale(res, tmp_res1, scale)
                        tmp_res2 : i8
                        clamp(tmp_res1, tmp_res2)
                        if act == True:
                            tmp_res2 = relu(tmp_res2)

                        output[b,orow,ocol,och] = tmp_res2

    return conv_on_cpu


# Padding = 0
# out_channel should be divisible by 16
def test_conv_2():
    T = GemmTestBuilder('conv_2')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_2_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 32
    kernel_dim = 3
    in_channel = 3
    in_dim     = 225
    out_dim    = int(in_dim - kernel_dim + 1)
    assert out_dim > 0
    assert out_dim == 223

    T.alloc_dram_f32('scale', '0.0001')
    T.alloc_dram_2i32('bias', 1, out_channel, '-10000')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, 'i')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, 'j')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, '100*i-k')
    T.alloc_dram_4i8('weights', kernel_dim, kernel_dim, in_channel, out_channel, '10*k')

    @proc
    def conv_on_gemmini(
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
        scale      : f32
        ):

        assert in_dim == out_dim + kernel_dim - 1

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))

                if out_dim%16 > 0:
                    conv_partial(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)


    conv_on_gemmini = conv_on_gemmini.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)
    conv_on_gemmini = conv_on_gemmini.inline('conv_partial(_) #0')
    conv_on_gemmini = conv_on_gemmini.inline('conv_partial(_) #0')
    conv_on_gemmini = conv_on_gemmini.simplify()
    conv_on_gemmini = conv_on_gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_ld_i8(_) #0', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_ld_i8_id1(_) #0', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_ld_i8_id2(_) #0', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_matmul() #0', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.reorder_stmts('for ocol in _:_ #0', 'config_st_acc_i8(_) #1')
    conv_on_gemmini = conv_on_gemmini.reorder_stmts('for ocol in _:_ #0', 'config_ld_i8(_) #1')
    conv_on_gemmini = conv_on_gemmini.reorder_stmts('for ocol in _:_ #0', 'config_ld_i8_id1(_) #1')
    conv_on_gemmini = conv_on_gemmini.reorder_stmts('for ocol in _:_ #0', 'config_ld_i8_id2(_) #1')
    conv_on_gemmini = conv_on_gemmini.fission_after('config_st_acc_i8(_) #1', n_lifts=2)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_ld_i8(_) #1', n_lifts=2)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_ld_i8_id1(_) #1', n_lifts=2)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_ld_i8_id2(_) #1', n_lifts=2)

    conv_on_gemmini = conv_on_gemmini.unroll('kch')

    conv_on_gemmini = conv_on_gemmini.lift_alloc('res:_', n_lifts=1)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('res:_ #0', n_lifts=1)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('in_scratch:_', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('weight_scratch:_', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('in_scratch:_ #0', n_lifts=1)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('weight_scratch:_', n_lifts=3)

    conv_on_gemmini = conv_on_gemmini.par_to_seq('for b in _:_')
    conv_on_gemmini = conv_on_gemmini.par_to_seq('for orow in _:_')

    conv_on_gemmini = conv_on_gemmini.lift_alloc('res:_', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('in_scratch:_', n_lifts=3)

    conv_on_gemmini = conv_on_gemmini.par_to_seq('for och in _:_')
    conv_on_gemmini = conv_on_gemmini.add_guard('do_ld_i8_id1(_) #0', 'och #0', 0)
    conv_on_gemmini = conv_on_gemmini.add_guard('do_ld_i8_id1(_) #1', 'och #1', 0)
    conv_on_gemmini = conv_on_gemmini.par_to_seq('for ocol in _:_')
    conv_on_gemmini = conv_on_gemmini.add_guard('do_ld_i8_id2(_) #0', 'ocol #0', 0)
    conv_on_gemmini = conv_on_gemmini.add_guard('do_ld_i8_id2(_) #0', 'orow', 0)
    conv_on_gemmini = conv_on_gemmini.add_guard('do_ld_i8_id2(_) #1', 'orow', 0)
    conv_on_gemmini = conv_on_gemmini.add_guard('do_ld_i8_id2(_) #0', 'b', 0)
    conv_on_gemmini = conv_on_gemmini.add_guard('do_ld_i8_id2(_) #1', 'b', 0)

    conv_on_gemmini = conv_on_gemmini.unroll('krow')

    T.add_proc(conv_on_gemmini)
    conv_on_cpu = conv_cpu()
    conv_on_cpu = conv_on_cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)
    T.add_proc(conv_on_cpu)

    T.start_timer('cpu')
    T.add_body([f'conv_on_cpu(ctxt,  output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_on_gemmini(ctxt,  output_gemmini, bias, inp, weights, false, scale);',
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

    print(conv_on_gemmini)


def test_conv_3():
    T = GemmTestBuilder('conv_2')
    T.add_body(['gemm_init_mem();',
  #              'init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_2_lib_Context *ctxt;"])

    batch_size = 4
    out_channel= 256
    kernel_dim = 3
    in_channel = 256
    in_dim     = 50
    out_dim    = int(in_dim - kernel_dim + 1)
    assert out_dim > 0

    T.alloc_dram_f32('scale', '1.0f')
    T.alloc_dram_2i32('bias', 1, out_channel, 'i+j')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'k+r')
    T.alloc_dram_4i8('weights', kernel_dim, kernel_dim, in_channel, out_channel, 'i+r')

    @proc
    def conv_on_gemmini(
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
        scale      : f32
        ):

        assert in_dim == out_dim + kernel_dim - 1

        one : f32
        one = 1.0
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim/16):
                    conv_partial(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, output, bias, inp, weights, act, scale, b, orow, one, 16, 16*ocol, 16*(ocol+1))

                if out_dim%16 > 0:

                    conv_partial(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim, output, bias, inp, weights, act, scale, b, orow, one, out_dim%16, out_dim-out_dim%16, out_dim)


    conv_on_gemmini = conv_on_gemmini.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)
    conv_on_gemmini = conv_on_gemmini.inline('conv_partial(_) #0')
    conv_on_gemmini = conv_on_gemmini.inline('conv_partial(_) #0')
    conv_on_gemmini = conv_on_gemmini.simplify()
    conv_on_gemmini = conv_on_gemmini.fission_after('config_st_acc_i8(_) #0', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_ld_i8(_) #0', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_ld_i8_id1(_) #0', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_ld_i8_id2(_) #0', n_lifts=3)
    conv_on_gemmini = conv_on_gemmini.fission_after('config_matmul() #0', n_lifts=3)

    conv_on_gemmini = conv_on_gemmini.lift_alloc('res:_', n_lifts=2)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('in_scratch:_', n_lifts=5)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('weight_scratch:_', n_lifts=3)

    conv_on_gemmini = conv_on_gemmini.par_to_seq('for b in _:_')
    conv_on_gemmini = conv_on_gemmini.par_to_seq('for orow in _:_')
    conv_on_gemmini = conv_on_gemmini.par_to_seq('for ocol in _:_')
    conv_on_gemmini = conv_on_gemmini.par_to_seq('for och in _:_')

    conv_on_gemmini = conv_on_gemmini.lift_alloc('res:_', n_lifts=2)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('in_scratch:_', n_lifts=2)
    conv_on_gemmini = conv_on_gemmini.lift_alloc('weight_scratch:_', n_lifts=4)

    conv_on_gemmini = conv_on_gemmini.add_guard('do_ld_i8_id1(_)', 'och', 0)

    conv_on_gemmini = conv_on_gemmini.unroll('kch')

    T.add_proc(conv_on_gemmini)
    conv_on_cpu = conv_cpu()
    conv_on_cpu = conv_on_cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)
    T.add_proc(conv_on_cpu)

    T.start_timer('cpu')
    T.add_body([f'conv_on_cpu(ctxt,  output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_on_gemmini(ctxt,  output_gemmini, bias, inp, weights, false, scale);',
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


    print(conv_on_gemmini)

