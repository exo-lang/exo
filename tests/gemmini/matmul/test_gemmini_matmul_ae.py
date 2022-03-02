from __future__ import annotations
import pytest
from ..gemmini import *
from ..harness_gemmini import GemmTestBuilder

def matmul_algorithm():
    @proc
    def matmul(
      N : size,
      M : size,
      K : size,
      scale : f32,
      act   : bool,
      A : i8[N,K] @ DRAM,
      B : i8[K,M] @ DRAM,
      C : i8[N,M] @ DRAM,
    ):

        # Artifact evaluation
        # Algorithm is here, 23? LOC
        for i in par(0,N):
            for j in par(0,M):
                res : i32 @ DRAM
                res = 0.0
                for k in par(0,K):
                    a : i8 @ DRAM
                    a = A[i,k]

                    b : i8 @ DRAM
                    b = B[k,j]

                    a2 : i32
                    b2 : i32
                    a2 = a
                    b2 = b
                    res += a2*b2

                src_tmp : i32
                src_tmp = res
                tmp_res1 : f32
                acc_scale(src_tmp, tmp_res1, scale)
                tmp_res2 : i8
                clamp(tmp_res1, tmp_res2)
                if act == True:
                    tmp_res2 = relu(tmp_res2)
                C[i,j] = tmp_res2

    return matmul

def inline_lift_config(gemmini):
    # part of scheduling count, 25
    gemmini = gemmini.call_eqv(zero_acc_i32_v2, "zero_acc_i32(_, _, _)")
    gemmini = gemmini.inline("zero_acc_i32_v2(_, _, _)")
    gemmini = gemmini.inline_window("dst = res[_]")
    gemmini = lift_config(gemmini, 'config_zero()')

    gemmini = gemmini.call_eqv(ld_i8_block_id1_v2, "ld_i8_block_id1(_)")
    gemmini = gemmini.inline("ld_i8_block_id1_v2(_, _, _, _, _)")
    gemmini = gemmini.inline_window("src = A[_]")
    gemmini = gemmini.inline_window("dst = a[_]")
    gemmini = lift_config(gemmini, 'config_ld_i8_id1()')

    gemmini = gemmini.call_eqv(ld_i8_block_id2_v2, "ld_i8_block_id2(_)")
    gemmini = gemmini.inline("ld_i8_block_id2_v2(_, _, _, _, _)")
    gemmini = gemmini.inline_window("src = B[_]")
    gemmini = gemmini.inline_window("dst = b[_]")
    gemmini = lift_config(gemmini, 'config_ld_i8_id2()')

    gemmini = gemmini.call_eqv(matmul_acc_i8_v2, "matmul_acc_i8(_, _, _, _, _)")
    gemmini = gemmini.inline("matmul_acc_i8_v2(_, _, _, _, _)")
    gemmini = gemmini.inline_window("A = a[_]")
    gemmini = gemmini.inline_window("B = b[_]")
    gemmini = gemmini.inline_window("C = res[_]")
    gemmini = lift_config(gemmini, 'config_matmul()')

    gemmini = gemmini.call_eqv(st_acc_i8_v2, "st_acc_i8(_, _, _, _, _, _)")
    gemmini = gemmini.inline("st_acc_i8_v2(_, _, _, _, _, _)")
    gemmini = gemmini.inline_window("src = res[_]")
    gemmini = gemmini.inline_window("dst = C[_]")
    gemmini = lift_config(gemmini, 'config_st_acc_i8(_)')
    return gemmini


# Best for 512x512x512
def test_matmul_ae():
    NN = 512
    MM = 512
    KK = 512

    cpu = matmul_algorithm().rename("matmul_on_cpu")
    cpu = cpu.partial_eval(NN, MM, KK)
    
    T = GemmTestBuilder('matmul_ae')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_ae_lib_Context *ctxt;"])

    T.alloc_dram_2i8('x', NN, KK, 'i+j')
    T.alloc_dram_2i8('y', KK, MM, 'j*3')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    # Rename the procedure and set gemmini memory
    gemmini = cpu.rename("matmul_on_gemmini")
    # Schedule starts here!
    gemmini = (gemmini.set_memory('res', GEMM_ACCUM).set_memory('a', GEMM_SCRATCH).set_memory('b', GEMM_SCRATCH))

    # Tile outer loops
    gemmini = tile_outer_loops(gemmini)

    # Lift res, so that we can fission the inner loop to use gemmini instructions
    gemmini = gemmini.lift_alloc('res : _ #0', n_lifts=2)
    gemmini = gemmini.lift_alloc('res : _ #0', n_lifts=1, mode='col', size=16)

    # fission loops to zero accum code block, main block, and store block and reorder k up
    gemmini = fission_outer_blocks(gemmini)

    # fission the main block to 4x16x16 blocks, so that we can use gemmini instr
    gemmini = fission_inner_blocks(gemmini)

    # replace to gemmini calls
    gemmini = replace_gemmini_calls(gemmini)

    # inline and lift config
    gemmini = inline_lift_config(gemmini)

    # Real optimization
    # tile
    gemmini = tile(gemmini)

    gemmini = gemmini.lift_alloc('res : _', n_lifts=1)
    gemmini = gemmini.lift_alloc('a : _', n_lifts=4)
    gemmini = gemmini.lift_alloc('b : _', n_lifts=3)

    [ (gemmini := gemmini.par_to_seq(s)) for s in ['for ioo in _:_', 'for io in _:_', 'for jo in _:_', 'for i in _:_'] ]

    [ (gemmini := gemmini.lift_alloc(s, n_lifts=n)) for (s,n) in [('a : i8', 1), ('b : i8', 2), ('res : _', 4)] ]

    gemmini = gemmini.par_to_seq('for ji in _:_')

    [ (gemmini := gemmini.add_guard(s, i, 0)) for (s,i) in [('do_ld_i8_block_id1(_)', 'ji'), ('do_ld_i8_block_id1(_)', 'jo'), ('do_ld_i8_block_id2(_)', 'i'), ('do_ld_i8_block_id2(_)', 'io')] ]

    gemmini = gemmini.fuse_loop('for k in _:_ #0', 'for k in _:_ #1')
    gemmini = gemmini.unroll('j_in_o')
    gemmini = gemmini.simplify()

    # Schedule ends here!

    T.add_proc(gemmini)
    T.add_proc(cpu)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, c_scale, false, x, y, z_cpu);',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_on_gemmini(ctxt, c_scale, false, x, y, z_gemmini);',
                f'gemmini_fence();'])
    T.stop_timer('gemmini', 'Cycles for GEMMINI version')


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

    print(gemmini)
