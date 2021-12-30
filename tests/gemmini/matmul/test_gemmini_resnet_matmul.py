from __future__ import annotations

import pytest

from ..gemmini import *
from ..harness_gemmini import GemmTestBuilder

def matmul_cpu():
    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      scale : f32,
      act   : bool,
      A : i8[N,K] @ DRAM,
      B : i8[K,M] @ DRAM,
      C : i8[N,M] @ DRAM,
    ):

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

    return matmul_on_cpu

def split_and_reorder(gemmini):
    gemmini = gemmini.split('j', 4, ['jo', 'ji'], perfect=True)
    gemmini = gemmini.split('i', 8, ['io', 'i'], perfect=True)
    gemmini = gemmini.split('io', 2, ['ioo', 'io'], perfect=True)
    gemmini = gemmini.reorder('i','jo')
    gemmini = gemmini.reorder('io','jo')
    return gemmini

# Best for 512x512x512
def test_matmul_512x512x512():
    NN = 512
    MM = 512
    KK = 512

    cpu = matmul_cpu().rename("matmul_on_cpu")
    cpu = cpu.partial_eval(NN, MM, KK)
    
    T = GemmTestBuilder('matmul_512x512x512')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_512x512x512_lib_Context *ctxt;"])

    T.alloc_dram_2i8('x', NN, KK, 'i+j')
    T.alloc_dram_2i8('y', KK, MM, 'j*3')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    gemmini = cpu.rename("matmul_on_gemmini")
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
    gemmini = split_and_reorder(gemmini)

    gemmini = gemmini.lift_alloc('res : _', n_lifts=1)
    gemmini = gemmini.lift_alloc('a : _', n_lifts=4)
    gemmini = gemmini.lift_alloc('b : _', n_lifts=3)

    #gemmini = gemmini.par_to_seq('for ioo in _:_')
    #gemmini = gemmini.par_to_seq('for io in _:_')
    #gemmini = gemmini.par_to_seq('for jo in _:_')
    #gemmini = gemmini.par_to_seq('for i in _:_')

    [ (gemmini := gemmini.par_to_seq(s)) for s in ['for ioo in _:_', 'for io in _:_', 'for jo in _:_', 'for i in _:_'] ]

    [ (gemmini := gemmini.lift_alloc(s, n_lifts=n)) for (s,n) in [('a : i8', 1), ('b : i8', 2), ('res : _', 4)] ]

    #gemmini = gemmini.lift_alloc('a : i8', n_lifts=1)
    #gemmini = gemmini.lift_alloc('b : i8', n_lifts=2)
    #gemmini = gemmini.lift_alloc('res : _', n_lifts=4)

    gemmini = gemmini.par_to_seq('for ji in _:_')

    [ (gemmini := gemmini.add_guard(s, i, 0)) for (s,i) in [('do_ld_i8_block_id1(_)', 'ji'), ('do_ld_i8_block_id1(_)', 'jo'), ('do_ld_i8_block_id2(_)', 'i'), ('do_ld_i8_block_id2(_)', 'io')] ]
    #gemmini = gemmini.add_guard('do_ld_i8_block_id1(_)', 'ji', 0)
    #gemmini = gemmini.add_guard('do_ld_i8_block_id1(_)', 'jo', 0)
    #gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'i', 0)
    #gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'io', 0)

    gemmini = gemmini.fuse_loop('for k in _:_ #0', 'for k in _:_ #1')
    gemmini = gemmini.unroll('j_in_o')
    #gemmini = gemmini.unroll('k')
    gemmini = gemmini.simplify()

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


def test_matmul_4():
    NN = 12544
    MM = 256
    KK = 64

    cpu = matmul_cpu().rename("matmul_on_cpu")
    cpu = cpu.partial_eval(NN, MM, KK)
    
    T = GemmTestBuilder('matmul_4')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_4_lib_Context *ctxt;"])

    T.alloc_dram_2i8('x', NN, KK, 'i+j*2')
    T.alloc_dram_2i8('y', KK, MM, 'j*3+i')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    gemmini = cpu.rename("matmul_on_gemmini")
    gemmini = (gemmini.set_memory('res', GEMM_ACCUM)
                                     .set_memory('a', GEMM_SCRATCH)
                                     .set_memory('b', GEMM_SCRATCH))

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
    gemmini = gemmini.unroll('ko')
    gemmini = gemmini.lift_alloc('res:_')
    gemmini = gemmini.simplify()

    gemmini = gemmini.split('i', 196, ['io', 'i'], perfect=True)
    gemmini = gemmini.lift_alloc('a : _', n_lifts=2)
    gemmini = gemmini.lift_alloc('b : _', n_lifts=2)
     
    # tile
    gemmini = gemmini.par_to_seq('for io in _:_')
    gemmini = gemmini.par_to_seq('for i in _:_')
    gemmini = gemmini.lift_alloc('a : i8', n_lifts=1)
    gemmini = gemmini.lift_alloc('b : i8', n_lifts=1)
    gemmini = gemmini.lift_alloc('res : _', n_lifts=2)

    gemmini = gemmini.par_to_seq('for j in _:_')
    gemmini = gemmini.add_guard('do_ld_i8_block_id1(_)', 'j', 0)
    gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'i', 0)
    gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'io', 0)

    gemmini = gemmini.unroll('j_in_o')
    gemmini = gemmini.unroll('k')

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



def test_matmul_6():
    NN = 12544
    MM = 64
    KK = 256

    cpu = matmul_cpu().rename("matmul_on_cpu")
    cpu = cpu.partial_eval(NN, MM, KK)
    
    T = GemmTestBuilder('matmul_6')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_6_lib_Context *ctxt;"])

    T.alloc_dram_2i8('x', NN, KK, 'i+j*2')
    T.alloc_dram_2i8('y', KK, MM, 'j*3+i')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    gemmini = cpu.rename("matmul_on_gemmini")
    gemmini = (gemmini.set_memory('res', GEMM_ACCUM)
                                     .set_memory('a', GEMM_SCRATCH)
                                     .set_memory('b', GEMM_SCRATCH))

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
    gemmini = gemmini.unroll('j')
    gemmini = gemmini.split('i', 8, ['io', 'i'], perfect=True)
    gemmini = gemmini.lift_alloc('res:_')
    gemmini = gemmini.lift_alloc('a : _')
    gemmini = gemmini.lift_alloc('b : _')
    gemmini = gemmini.simplify()
     
    # tile
    gemmini = gemmini.par_to_seq('for io in _:_')
    gemmini = gemmini.par_to_seq('for i in _:_')
    gemmini = gemmini.lift_alloc('res:_')
    gemmini = gemmini.lift_alloc('a : _')
    gemmini = gemmini.lift_alloc('b : _')

    gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'i', 0)
    gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'io', 0)

    gemmini = gemmini.unroll('j_in_o')
    gemmini = gemmini.unroll('k')

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




def test_matmul_14():
    NN = 3136
    MM = 512
    KK = 128

    cpu = matmul_cpu().rename("matmul_on_cpu")
    cpu = cpu.partial_eval(NN, MM, KK)
    
    T = GemmTestBuilder('matmul_14')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_14_lib_Context *ctxt;"])

    T.alloc_dram_2i8('x', NN, KK, 'i+j*2')
    T.alloc_dram_2i8('y', KK, MM, 'j*3+i')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    gemmini = cpu.rename("matmul_on_gemmini")
    gemmini = (gemmini.set_memory('res', GEMM_ACCUM)
                                     .set_memory('a', GEMM_SCRATCH)
                                     .set_memory('b', GEMM_SCRATCH))

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
    gemmini = gemmini.split('i', 49, ['io', 'i'], perfect=True)
    gemmini = gemmini.lift_alloc('res:_')
    gemmini = gemmini.lift_alloc('a : _', n_lifts=2)
    gemmini = gemmini.lift_alloc('b : _', n_lifts=2)
    gemmini = gemmini.simplify()
     
    # tile
    gemmini = gemmini.par_to_seq('for io in _:_')
    gemmini = gemmini.par_to_seq('for i in _:_')
    gemmini = gemmini.par_to_seq('for j in _:_')
    gemmini = gemmini.lift_alloc('res:_', n_lifts=2)
    gemmini = gemmini.lift_alloc('a : _')
    gemmini = gemmini.lift_alloc('b : _')

    gemmini = gemmini.add_guard('do_ld_i8_block_id1(_)', 'j', 0)
    gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'i', 0)
    gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'io', 0)

    gemmini = gemmini.unroll('j_in_o')
    gemmini = gemmini.unroll('k')

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



def test_matmul_16():
    NN = 3136
    MM = 128
    KK = 512

    cpu = matmul_cpu().rename("matmul_on_cpu")
    cpu = cpu.partial_eval(NN, MM, KK)
    
    T = GemmTestBuilder('matmul_16')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_16_lib_Context *ctxt;"])

    T.alloc_dram_2i8('x', NN, KK, 'i+j*2')
    T.alloc_dram_2i8('y', KK, MM, 'j*3+i')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    gemmini = cpu.rename("matmul_on_gemmini")
    gemmini = (gemmini.set_memory('res', GEMM_ACCUM)
                                     .set_memory('a', GEMM_SCRATCH)
                                     .set_memory('b', GEMM_SCRATCH))

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
    gemmini = gemmini.split('i', 14, ['io', 'i'], perfect=True)
    gemmini = gemmini.lift_alloc('res:_')
    gemmini = gemmini.lift_alloc('a : _', n_lifts=2)
    gemmini = gemmini.lift_alloc('b : _', n_lifts=2)
    gemmini = gemmini.simplify()
     
    # tile
    gemmini = gemmini.par_to_seq('for io in _:_')
    gemmini = gemmini.par_to_seq('for i in _:_')
    gemmini = gemmini.par_to_seq('for j in _:_')
    gemmini = gemmini.lift_alloc('res:_', n_lifts=2)
    gemmini = gemmini.lift_alloc('a : _')
    gemmini = gemmini.lift_alloc('b : _')

    gemmini = gemmini.add_guard('do_ld_i8_block_id1(_)', 'j', 0)
    gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'i', 0)
    gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'io', 0)

    gemmini = gemmini.unroll('j_in_o')
    gemmini = gemmini.unroll('k')

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


def test_matmul_27():
    NN = 784
    MM = 1024
    KK = 256

    cpu = matmul_cpu().rename("matmul_on_cpu")
    cpu = cpu.partial_eval(NN, MM, KK)
    
    T = GemmTestBuilder('matmul_27')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_27_lib_Context *ctxt;"])

    T.alloc_dram_2i8('x', NN, KK, 'i+j*2')
    T.alloc_dram_2i8('y', KK, MM, 'j*3+i')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    gemmini = cpu.rename("matmul_on_gemmini")
    gemmini = (gemmini.set_memory('res', GEMM_ACCUM)
                                     .set_memory('a', GEMM_SCRATCH)
                                     .set_memory('b', GEMM_SCRATCH))

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
    gemmini = gemmini.split('i', 7, ['io', 'i'], perfect=True)
    gemmini = gemmini.split('j', 8, ['jo', 'j'], perfect=True)
    gemmini = gemmini.reorder('i', 'jo')
    gemmini = gemmini.lift_alloc('res:_')
    gemmini = gemmini.lift_alloc('a : _', n_lifts=2)
    gemmini = gemmini.lift_alloc('b : _', n_lifts=2)
    gemmini = gemmini.simplify()
     
    # tile
    gemmini = gemmini.par_to_seq('for io in _:_')
    gemmini = gemmini.par_to_seq('for jo in _:_')
    gemmini = gemmini.par_to_seq('for i in _:_')
    gemmini = gemmini.par_to_seq('for j in _:_')
    gemmini = gemmini.lift_alloc('res:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('a : _', n_lifts=2)
    gemmini = gemmini.lift_alloc('b : _', n_lifts=2)

    gemmini = gemmini.add_guard('do_ld_i8_block_id1(_)', 'j', 0)
    gemmini = gemmini.add_guard('do_ld_i8_block_id1(_)', 'jo', 0)
    gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'i', 0)
    #gemmini = gemmini.add_guard('do_ld_i8_block_id2(_)', 'io', 0)

    gemmini = gemmini.unroll('j_in_o')
    gemmini = gemmini.unroll('k')

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



