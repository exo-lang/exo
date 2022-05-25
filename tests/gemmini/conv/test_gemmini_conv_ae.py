from __future__ import annotations
import pytest
from exo.platforms.gemmini import *
from ..harness_gemmini import GemmTestBuilder


def conv_algorithm():
    @proc
    def conv(
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

        # Algorithm starts here
        for b in par(0, batch_size):
            for orow in par(0, out_dim):
                for ocol in par(0, out_dim):
                    for och in par(0, out_channel):
                        # initialize bias for the output
                        output[b,orow,ocol,och] = bias[0,och]

                        # do accumulations
                        for krow in par(0, kernel_dim):
                            for kcol in par(0, kernel_dim):
                                for kch in par(0, in_channel):
                                    output[b,orow,ocol,och] += (
                                        inp[b,orow+krow,ocol+kcol,kch] *
                                        weights[krow,kcol,kch,och])

                        if act == True:
                            output[b,orow,ocol,och] = relu(
                                output[b,orow,ocol,och])

    return conv

class Scale_and_Clamp_is_Identity:
    def __init__(self):
        # form for matching
        @proc
        def scale_id(src : R, dst : R, scale : R):
            match : R
            match = scale
            dst = src

        # expansion with types
        @proc
        def scale_and_clamp(src : i32, dst : i8, scale : f32):
            mid : f32
            acc_scale(src, mid, scale)
            clamp(mid, dst)

        scale_and_clamp.unsafe_assert_eq(scale_id)

        self._scale_id          = scale_id
        self._scale_and_clamp   = scale_and_clamp

    def replace(self, proc, stmt_pat, scale_expr):
        stmtblock_pat = (f"m_a_t_c_h : _ ; "
                         f"m_a_t_c_h = {scale_expr} ; "
                         f"{stmt_pat}")
        proc = (proc
            .assign_new_var_before(stmt_pat, 'm_a_t_c_h', scale_expr)
            .replace(self._scale_id, stmtblock_pat)
            .call_eqv(self._scale_and_clamp, 'scale_id(_)')
            .inline('scale_and_clamp(_)')
        )

        return proc


def bind_prec(proc, name, prec, expr):
    return (proc.bind_expr(name, expr)
                .set_precision(name, prec))

def test_make_simple_algorithm_precise(conv=None):
    conv = conv or conv_algorithm()
    conv = (conv.rename('complex_conv')
        .stage_mem('output[b,orow,ocol,och]', 'res',
                   'output[_]=_','if _:_')
        .set_precision('res','i32')
        .stage_mem('res','tmp_res2','if act==True:_','output[_]=_')
        .set_precision('tmp_res2','i8')
        .delete_dead_code('res = tmp_res2')
    )
    conv = bind_prec(conv, 'i_s', 'i8', 'inp[b,orow+krow,ocol+kcol,kch]')
    conv = bind_prec(conv, 'w_s', 'i8', 'weights[krow,kcol,kch,och]')
    conv = bind_prec(conv, 'a', 'i32', 'i_s')
    conv = bind_prec(conv, 'b', 'i32', 'w_s')

    scale_clamp = Scale_and_Clamp_is_Identity()
    conv = scale_clamp.replace(conv, 'tmp_res2 = res', 'scale')

    print(conv)

#test_make_simple_algorithm_precise()

def conv_algorithm_precise():
    @proc
    def conv(
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

        # Algorithm starts here
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
        # Algorithm ends here. 26 lines excluding newlines

    return conv


# Conv test for the artifact evaluation. The same algorithm and schedule
# was used for Table 2 (first row) and Table 3 (code size)
def test_conv_ae():
    batch_size = 4
    out_channel= 64
    kernel_dim = 3
    in_channel = 64
    in_dim     = 58
    out_dim    = int((in_dim - kernel_dim)/1 + 1)
    assert out_dim == 56

    # These lines are relevant if you have GEMMINI environment set up
    T = GemmTestBuilder('conv_ae')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["conv_ae_lib_Context *ctxt;"])

    T.alloc_dram_f32('scale', '1.0')
    T.alloc_dram_2i32('bias', 1, out_channel, '-1*j')
    T.alloc_dram_4i8('output_cpu', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('output_gemmini', batch_size, out_dim, out_dim, out_channel, '0')
    T.alloc_dram_4i8('inp', batch_size, in_dim, in_dim, in_channel, 'j+k+r*3')
    T.alloc_dram_4i8('weights', out_channel, kernel_dim, kernel_dim, in_channel, 'i+k*3+r')

    # Rename the conv algorithm to "conv_on_gemmini"
    gemmini = conv_algorithm_precise().rename("conv_on_gemmini")

    print("")
    print("===== THIS IS THE CONV ALGORITHM BEFORE SCHEDULING ====")
    print(gemmini)
    print("===== THIS IS THE CONV ALGORITHM BEFORE SCHEDULING ====")
    print("")

    # Schedule starts here. Below schedule partially evaluates the proc with conv parameters
    gemmini = gemmini.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)

    # Split the outer dimension and replace the code with gemmini instructions
    gemmini = split_fission_dim(gemmini)
    gemmini = replace_div_part(gemmini)
    gemmini = replace_mod_part(gemmini)

    # Set buffers to use gemmini memories
    gemmini = gemmini.set_memory('res', GEMM_ACCUM)
    gemmini = gemmini.set_memory('i_s', GEMM_SCRATCH)
    gemmini = gemmini.set_memory('w_s', GEMM_SCRATCH)

    # Inline and lift the configuration as high as possible for the "div" part
    gemmini = inline_vector(gemmini)
    gemmini = lift_config(gemmini, 'config_ld_acc_i32_vector(_)')
    gemmini = inline_ld_id1(gemmini)
    gemmini = lift_config(gemmini, 'config_ld_i8_id1(_)')
    gemmini = inline_matmul(gemmini)
    gemmini = lift_config(gemmini, 'config_matmul(_)')
    gemmini = inline_st(gemmini)
    gemmini = lift_config(gemmini, 'config_st_acc_i8(_)')

    # Inline and lift the configuration as high as possible for the "mod" part (tail case)
    gemmini = inline_vector(gemmini)
    gemmini = inline_ld_id1(gemmini)
    gemmini = inline_matmul(gemmini)
    gemmini = inline_st(gemmini)
    gemmini = gemmini.delete_config("config_ld_acc_i32_vector(_) #1")
    gemmini = gemmini.delete_config("config_ld_i8_id1(_) #1")
    gemmini = gemmini.delete_config("config_matmul(_) #1")
    gemmini = gemmini.delete_config("config_st_acc_i8(_) #1")
    gemmini = gemmini.simplify()

    # Real optimization
    gemmini = gemmini.lift_alloc('w_s : _', n_lifts=2)
    gemmini = gemmini.fission_after('for ocol_o in _:_ #0')
    gemmini = gemmini.reorder('orow', 'ocol_o')
    gemmini = gemmini.split('orow', 28, ['orow_o', 'orow_i'], perfect=True)
    gemmini = gemmini.expand_dim('i_s: i8[_]', '30', 'krow + orow_i', unsafe_disable_checks=True)
    [ (gemmini := gemmini.par_to_seq(s)) for s in ['for krow in _:_', 'for b in _:_', 'for orow_o in _:_', 'for orow_i in _:_', 'for ocol_o in _:_'] ]
    gemmini = gemmini.lift_alloc('i_s : _', n_lifts=5)
    gemmini = gemmini.lift_alloc('w_s : _', n_lifts=4)

    gemmini = gemmini.lift_alloc('res : _', n_lifts=4)

    gemmini = gemmini.fission_after('for kch_o in _:_ #0', n_lifts=6)
    gemmini = gemmini.fission_after('for kch_o in _:_ #2', n_lifts=5)
    gemmini = gemmini.fission_after('for och_o in _:_ #3')
    gemmini = gemmini.add_loop('for kch_o in _:_ #0', 'orow_i', 28, guard=True)
    gemmini = gemmini.add_loop('if orow_i == 0:_', 'orow_o', 2, guard=True)
    gemmini = gemmini.add_loop('if orow_o == 0:_', 'b', 4, guard=True)
    gemmini = gemmini.add_loop('if b == 0:_', 'ocol_o', 3, guard=True)
    gemmini = gemmini.add_loop('for kch_o in _:_ #2', 'orow_i', 28, guard=True)
    gemmini = gemmini.add_loop('if orow_i == 0:_ #1', 'orow_o', 2, guard=True)
    gemmini = gemmini.add_loop('if orow_o == 0:_ #1', 'b', 4, guard=True)
    # Start fissioning loops
    gemmini = gemmini.add_loop('for och_o in _:_ #0', 'b', 4)
    gemmini = gemmini.reorder('orow_o','b').reorder('orow_i','b').reorder('kcol','b').reorder('krow','b')
    gemmini = gemmini.fuse_loop('for b in _:_ #0', 'for b in _:_ #1')
    gemmini = gemmini.fuse_loop('for b in _:_ #0', 'for b in _:_ #1')
    gemmini = gemmini.fuse_loop('for b in _:_ #0', 'for b in _:_ #1')
    gemmini = gemmini.fuse_loop('for b in _:_ #0', 'for b in _:_ #1')
    gemmini = gemmini.add_loop('for och_o in _:_ #0', 'ocol_o', 3)
    gemmini = gemmini.reorder('orow_o','ocol_o').reorder('orow_i','ocol_o').reorder('kcol','ocol_o').reorder('krow','ocol_o')
    gemmini = gemmini.fuse_loop('for ocol_o in _:_ #0', 'for ocol_o in _:_ #1')
    gemmini = gemmini.fuse_loop('for ocol_o in _:_ #0', 'for ocol_o in _:_ #1')
    gemmini = gemmini.add_loop('for och_o in _:_ #0', 'orow_o', 2)
    gemmini = gemmini.reorder('orow_i','orow_o').reorder('kcol','orow_o').reorder('krow','orow_o')
    gemmini = gemmini.fuse_loop('for orow_o in _:_ #0', 'for orow_o in _:_ #1')
    gemmini = gemmini.fuse_loop('for orow_o in _:_ #0', 'for orow_o in _:_ #1')
    gemmini = gemmini.add_loop('for och_o in _:_ #0', 'orow_i', 28)
    gemmini = gemmini.reorder('kcol','orow_i').reorder('krow','orow_i')
    gemmini = gemmini.fuse_loop('for orow_i in _:_ #0', 'for orow_i in _:_ #1')
    gemmini = gemmini.fuse_loop('for orow_i in _:_ #0', 'for orow_i in _:_ #1')

    gemmini = gemmini.add_loop('for och_o in _:_ #3', 'orow_o', 2)
    gemmini = gemmini.fuse_loop('for orow_o in _:_ #1', 'for orow_o in _:_ #2')
    gemmini = gemmini.fuse_loop('for orow_o in _:_ #1', 'for orow_o in _:_ #2')
    gemmini = gemmini.add_loop('for och_o in _:_ #3', 'orow_i', 28)
    gemmini = gemmini.fuse_loop('for orow_i in _:_ #1', 'for orow_i in _:_ #2')
    gemmini = gemmini.fuse_loop('for orow_i in _:_ #1', 'for orow_i in _:_ #2')

    gemmini = gemmini.fuse_loop('for krow in _:_ #0', 'for krow in _:_ #1')
    gemmini = gemmini.fuse_loop('for kcol in _:_ #0', 'for kcol in _:_ #1')
    gemmini = gemmini.fuse_loop('for krow in _:_ #1', 'for krow in _:_ #2')
    gemmini = gemmini.fuse_loop('for kcol in _:_ #1', 'for kcol in _:_ #2')


    gemmini = gemmini.add_unsafe_guard('ld_i8_block_id2(_) #0', 'orow_i == 0 or krow == 2')
    gemmini = gemmini.add_unsafe_guard('ld_i8_block_id2(_) #1', 'orow_i == 0 or krow == 2')

    gemmini = gemmini.split('orow_i', 7, ['orow_io', 'orow_ii'], perfect=True)
    #gemmini = gemmini.lift_alloc('res : _', n_lifts=1)
    gemmini = gemmini.par_to_seq('for orow_io in _:_')
    gemmini = gemmini.unroll('och_o')
    gemmini = gemmini.unroll('kch_o')
    gemmini = gemmini.unroll('kcol')
    gemmini = gemmini.unroll('krow')
    gemmini = gemmini.simplify()

    # Schedule ends here, 44 lines excluding comments and newlines


    cpu = conv_algorithm_precise().rename("conv_on_cpu")
    cpu = cpu.partial_eval(batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim)

    # These lines are relevant if you want to run the generated C code with GEMMINI simulator
    T.add_proc(cpu)
    T.add_proc(gemmini)

    T.start_timer('cpu')

    T.add_body([f'conv_on_cpu(ctxt, output_cpu, bias, inp, weights, false, scale);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'conv_on_gemmini(ctxt, output_gemmini, bias, inp, weights, false, scale);',
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


    print("")
    print("============= THIS IS THE SCHEDULED CONV ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED CONV ===============")
    print("")

