from __future__ import annotations
import pytest
from exo.platforms.gemmini import *
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

        # Algorithm starts here
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
        # Algorithm ends here. 23 lines excluding newlines

    return matmul


# Matmul test for artifact evaluation. The same algorithm and schedule
# was used for Table 2 (512x521x512) and Table 3 (code size)
def test_matmul_ae():
    NN = 512
    MM = 512
    KK = 512

    cpu = rename(matmul_algorithm(), "matmul_on_cpu") # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(NN, MM, KK)
    
    # These lines are relevant if you have GEMMINI environment set up
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

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_on_gemmini")

    print("")
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print(gemmini)
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print("")

    # Schedule starts here. Below sets buffer to use GEMMINI memories.
    gemmini = set_memory(gemmini, 'res', GEMM_ACCUM)
    gemmini = set_memory(gemmini, 'a', GEMM_SCRATCH)
    gemmini = set_memory(gemmini, 'b', GEMM_SCRATCH)

    # Tile outer loops
    gemmini = tile_outer_loops(gemmini)

    # Lift res, so that we can fission the inner loop to use gemmini instructions
    gemmini = gemmini.lift_alloc('res : _ #0', n_lifts=2)
    gemmini = gemmini.lift_alloc('res : _ #0', n_lifts=1, mode='col', size=16)

    # Fission loops to zero accum code block, main block, and store block and reorder k up
    gemmini = fission_outer_blocks(gemmini)

    # Fission the main block to 4x16x16 blocks, so that we can use gemmini instr
    gemmini = fission_inner_blocks(gemmini)

    # Replace to gemmini calls
    gemmini = replace_gemmini_calls(gemmini)

    # Inline and lift the configuration as high as possible
    # Lift config_zero
    gemmini = gemmini.call_eqv(zero_acc_i32_v2, "zero_acc_i32(_, _, _)")
    gemmini = gemmini.inline("zero_acc_i32_v2(_, _, _)")
    gemmini = inline_window(gemmini, "dst = res[_]")
    gemmini = lift_config(gemmini, 'config_zero()')
    # Lift config_ld_i8_id1
    gemmini = gemmini.call_eqv(ld_i8_block_id1_v2, "ld_i8_block_id1(_)")
    gemmini = gemmini.inline("ld_i8_block_id1_v2(_, _, _, _, _)")
    gemmini = inline_window(gemmini, "src = A[_]")
    gemmini = inline_window(gemmini, "dst = a[_]")
    gemmini = lift_config(gemmini, 'config_ld_i8_id1()')
    # Lift config_ld_i8_id2
    gemmini = gemmini.call_eqv(ld_i8_block_id2_v2, "ld_i8_block_id2(_)")
    gemmini = gemmini.inline("ld_i8_block_id2_v2(_, _, _, _, _)")
    gemmini = inline_window(gemmini, "src = B[_]")
    gemmini = inline_window(gemmini, "dst = b[_]")
    gemmini = lift_config(gemmini, 'config_ld_i8_id2()')
    # Lift config_matmul
    gemmini = gemmini.call_eqv(matmul_acc_i8_v2, "matmul_acc_i8(_, _, _, _, _)")
    gemmini = gemmini.inline("matmul_acc_i8_v2(_, _, _, _, _)")
    gemmini = inline_window(gemmini, "A = a[_]")
    gemmini = inline_window(gemmini, "B = b[_]")
    gemmini = inline_window(gemmini, "C = res[_]")
    gemmini = lift_config(gemmini, 'config_matmul()')
    # Lift config_st_acc_i8
    gemmini = gemmini.call_eqv(st_acc_i8_v2, "st_acc_i8(_, _, _, _, _, _)")
    gemmini = gemmini.inline("st_acc_i8_v2(_, _, _, _, _, _)")
    gemmini = inline_window(gemmini, "src = res[_]")
    gemmini = inline_window(gemmini, "dst = C[_]")
    gemmini = lift_config(gemmini, 'config_st_acc_i8(_)')

    # Futher tile the innner loops
    gemmini = tile(gemmini)

    # Lift the allocations
    gemmini = gemmini.lift_alloc('res : _', n_lifts=1)
    gemmini = gemmini.lift_alloc('a : _', n_lifts=4)
    gemmini = gemmini.lift_alloc('b : _', n_lifts=3)

    [ (gemmini := gemmini.par_to_seq(s)) for s in ['for ioo in _:_', 'for io in _:_', 'for jo in _:_', 'for i in _:_'] ]

    [ (gemmini := gemmini.lift_alloc(s, n_lifts=n)) for (s,n) in [('a : i8', 1), ('b : i8', 2), ('res : _', 4)] ]

    gemmini = gemmini.par_to_seq('for ji in _:_')

    # These schedules correspond to previous add_guard
    gemmini = simplify(gemmini)
    gemmini = autofission(gemmini,
                    gemmini.find('for j_in_o in _:_').after(), n_lifts=5)
    gemmini = autofission(gemmini,
                    gemmini.find('for k in _:_').after(), n_lifts=6)
    gemmini = autofission(gemmini,
                    gemmini.find('do_ld_i8_block_id1(_)').after(), n_lifts=6)
    gemmini = gemmini.add_loop('do_ld_i8_block_id1(_)', 'ji', 4, guard=True)
    gemmini = gemmini.add_loop('if ji == 0: _', 'jo', 2, guard=True)
    gemmini = gemmini.add_loop('do_ld_i8_block_id2(_)', 'i', 8, guard=True)
    gemmini = gemmini.add_loop('if i == 0: _', 'io', 2, guard=True)
    # Fuse_loop cleanup
    gemmini = gemmini.add_loop('for jo in _:_ #1', 'ioo', 2)
    gemmini = gemmini.add_loop('for ji in _:_ #0', 'ioo', 2)
    gemmini = gemmini.fuse_loop('for ioo in _:_ #0', 'for ioo in _:_ #1')
    gemmini = gemmini.fuse_loop('for ioo in _:_ #0', 'for ioo in _:_ #1')
    gemmini = gemmini.fuse_loop('for ioo in _:_ #0', 'for ioo in _:_ #1')
    gemmini = gemmini.add_loop('for ji in _:_ #0', 'jo', 2)
    gemmini = old_reorder(gemmini, 'ji jo')
    gemmini = old_reorder(gemmini, 'ko jo')
    gemmini = old_reorder(gemmini, 'i jo')
    gemmini = old_reorder(gemmini, 'io jo')
    gemmini = gemmini.fuse_loop('for jo in _:_ #0', 'for jo in _:_ #1')
    gemmini = gemmini.fuse_loop('for jo in _:_ #0', 'for jo in _:_ #1')
    gemmini = gemmini.fuse_loop('for jo in _:_ #0', 'for jo in _:_ #1')
    gemmini = old_reorder(gemmini, 'i io')
    gemmini = old_reorder(gemmini, 'k io')
    gemmini = old_reorder(gemmini, 'ko io')
    gemmini = old_reorder(gemmini, 'ji io')
    gemmini = gemmini.add_loop('for ji in _:_ #0', 'io', 2)
    gemmini = gemmini.fuse_loop('for io in _:_ #0', 'for io in _:_ #1')
    gemmini = gemmini.fuse_loop('for io in _:_ #0', 'for io in _:_ #1')
    gemmini = gemmini.fuse_loop('for io in _:_ #0', 'for io in _:_ #1')
    gemmini = old_reorder(gemmini, 'k i')
    gemmini = old_reorder(gemmini, 'ko i')
    gemmini = old_reorder(gemmini, 'ji i')
    gemmini = gemmini.add_loop('for ji in _:_ #0', 'i', 8)
    gemmini = gemmini.fuse_loop('for i in _:_ #0', 'for i in _:_ #1')
    gemmini = gemmini.fuse_loop('for i in _:_ #0', 'for i in _:_ #1')
    gemmini = gemmini.fuse_loop('for i in _:_ #0', 'for i in _:_ #1')
    gemmini = old_reorder(gemmini, 'ko ji')
    gemmini = gemmini.par_to_seq('for ji in _:_ #1')
    gemmini = gemmini.fuse_loop('for ji in _:_ #0', 'for ji in _:_ #1')
    gemmini = gemmini.fuse_loop('for ji in _:_ #0', 'for ji in _:_ #1')
    gemmini = gemmini.fuse_loop('for ji in _:_ #0', 'for ji in _:_ #1')
    gemmini = gemmini.fuse_loop('for ko in _:_ #0', 'for ko in _:_ #1')
    gemmini = gemmini.fuse_loop('for ko in _:_ #0', 'for ko in _:_ #1')

    gemmini = gemmini.fuse_loop('for k in _:_ #0', 'for k in _:_ #1')
    gemmini = gemmini.unroll('j_in_o')
    gemmini = simplify(gemmini)

    # Schedule ends here. 43 lines excluding comments and newlines

    # These lines are relevant if you want to run the generated C code with GEMMINI simulator
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

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")

