from __future__ import annotations

import pytest

from .gemmini import *
from .harness_gemmini import GemmTestBuilder

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


# Best for 512x512x512
def test_matmul_512x512x512():
    NN = 512
    MM = 512
    KK = 512
    tile_size_I = 128
    tile_size_J = 128
    K_SIZE = KK//16

    cpu = matmul_cpu().rename("matmul_on_cpu")
    cpu = cpu.partial_eval(NN, MM, KK)
    
    T = GemmTestBuilder('matmul_512x512x512')
    T.add_body(['gemm_init_mem();',
                'gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_512x512x512_lib_Context *ctxt;"])

    T.alloc_dram_2i8('x', NN, KK, '7')
    T.alloc_dram_2i8('y', KK, MM, '4')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    gemmini = cpu.rename("matmul_on_gemmini")
    gemmini = (gemmini.set_memory('res', GEMM_ACCUM)
                                     .set_memory('a', GEMM_SCRATCH)
                                     .set_memory('b', GEMM_SCRATCH))

    gemmini = gemmini.split('i',tile_size_I,['io','i'], perfect=True)
    gemmini = gemmini.split('j',tile_size_J,['jo','j'], perfect=True)
    gemmini = gemmini.reorder('i','jo')

    gemmini = gemmini.split('i',16,['i','i_in'], perfect=True)
    gemmini = gemmini.reorder('i_in','j')
    gemmini = gemmini.split('j',16,['j','j_in'], perfect=True)

    gemmini = gemmini.lift_alloc('res : _ #0', n_lifts=1)
    gemmini = gemmini.lift_alloc('res : _ #0', n_lifts=1, mode='col', size=16)
    gemmini = gemmini.par_to_seq('for jo in _:_')
    gemmini = gemmini.par_to_seq('for io in _:_')

    gemmini = gemmini.fission_after('res[_] = 0.0 #0', n_lifts=2)

    gemmini = gemmini.fission_after('for k in _:_ #0', n_lifts=2)

    gemmini = gemmini.reorder('i_in','k')
    gemmini = gemmini.reorder('j_in','k')

    gemmini = gemmini.lift_alloc('a : i8', n_lifts=2)
    gemmini = gemmini.lift_alloc('b : i8', n_lifts=2)

    gemmini = gemmini.split('k',16,['k','k_in'], perfect=True)

    gemmini = gemmini.lift_alloc('a : _ #0', n_lifts=1, mode='col')
    gemmini = gemmini.lift_alloc('b : _', n_lifts=1)

    gemmini = gemmini.fission_after('a[_] = _', n_lifts=3)
    gemmini = gemmini.fission_after('b[_] = _', n_lifts=3)

    gemmini = gemmini.reorder('j_in','i_in')
    gemmini = gemmini.replace(zero_acc_i32, "for i_in in _:_ #0")
    gemmini = gemmini.reorder('k_in','i_in')
    gemmini = gemmini.replace(ld_i8_id1, "for i_in in _:_ #0")
    gemmini = gemmini.replace(ld_i8_id2, "for k_in in _:_ #0")
    gemmini = gemmini.reorder('k_in','j_in')
    gemmini = gemmini.replace(matmul_acc_i8, "for i_in in _:_ #0")
    gemmini = gemmini.replace(st_acc_i8, "for i_in in _:_ #0")

    gemmini = gemmini.call_eqv(zero_acc_i32_v2, "zero_acc_i32(_, _, _)")
    gemmini = gemmini.inline("zero_acc_i32_v2(_, _, _)")
    gemmini = gemmini.inline_window("dst = res[_]")

    gemmini = gemmini.call_eqv(ld_i8_id1_v2, "ld_i8_id1(_, _, _, _, _)")
    gemmini = gemmini.inline("ld_i8_id1_v2(_, _, _, _, _)")
    gemmini = gemmini.inline_window("src = A[_]")
    gemmini = gemmini.inline_window("dst = a[_]")

    gemmini = gemmini.call_eqv(ld_i8_id2_v2, "ld_i8_id2(_, _, _, _, _)")
    gemmini = gemmini.inline("ld_i8_id2_v2(_, _, _, _, _)")
    gemmini = gemmini.inline_window("src = B[_]")
    gemmini = gemmini.inline_window("dst = b[_]")

    gemmini = gemmini.call_eqv(st_acc_i8_v2, "st_acc_i8(_, _, _, _, _, _)")
    gemmini = gemmini.inline("st_acc_i8_v2(_, _, _, _, _, _)")
    gemmini = gemmini.inline_window("src = res[_]")
    gemmini = gemmini.inline_window("dst = C[_]")

    gemmini = gemmini.call_eqv(matmul_acc_i8_v2, "matmul_acc_i8(_, _, _, _, _)")
    gemmini = gemmini.inline("matmul_acc_i8_v2(_, _, _, _, _)")
    gemmini = gemmini.inline_window("A = a[_]")
    gemmini = gemmini.inline_window("B = b[_]")
    gemmini = gemmini.inline_window("C = res[_]")


    gemmini = gemmini.reorder_stmts("for k in _:_", "config_st_acc_i8(_, _)")
    gemmini = gemmini.reorder_stmts("do_zero_acc_i32(_, _, _)", "config_st_acc_i8(_, _)")
    gemmini = gemmini.reorder_stmts("config_zero()", "config_st_acc_i8(_, _)")
    gemmini = gemmini.reorder_stmts("res : _", "config_st_acc_i8(_, _)")
    gemmini = gemmini.fission_after("config_st_acc_i8(_, _)", n_lifts=2)
    gemmini = gemmini.fission_after("config_st_acc_i8(_, _)", n_lifts=2)

    gemmini = gemmini.reorder_stmts("res : _", "config_zero(_)")
    gemmini = gemmini.fission_after("config_zero(_)", n_lifts=4)

    gemmini = gemmini.reorder_stmts("b : _", "config_ld_i8_id1(_)")
    gemmini = gemmini.reorder_stmts("a : _", "config_ld_i8_id1(_)")
    gemmini = gemmini.reorder_stmts("do_ld_i8_id1(_)", "config_ld_i8_id2(_)")
    gemmini = gemmini.reorder_stmts("b : _", "config_ld_i8_id2(_)")
    gemmini = gemmini.reorder_stmts("a : _", "config_ld_i8_id2(_)")
    gemmini = gemmini.fission_after("config_ld_i8_id1(_)", n_lifts=1)
    gemmini = gemmini.fission_after("config_ld_i8_id2(_)", n_lifts=1)
    gemmini = gemmini.reorder_stmts("do_zero_acc_i32(_)", "config_ld_i8_id1(_)")
    gemmini = gemmini.reorder_stmts("do_zero_acc_i32(_)", "config_ld_i8_id2(_)")
    gemmini = gemmini.reorder_stmts("res:_", "config_ld_i8_id1(_)")
    gemmini = gemmini.reorder_stmts("res:_", "config_ld_i8_id2(_)")
    gemmini = gemmini.fission_after("config_ld_i8_id1(_)", n_lifts=4)
    gemmini = gemmini.fission_after("config_ld_i8_id2(_)", n_lifts=4)

    gemmini = gemmini.reorder_stmts("do_ld_i8_id2(_,_,_,_)", "config_matmul()")
    gemmini = gemmini.reorder_stmts("do_ld_i8_id1(_,_,_,_)", "config_matmul()")
    gemmini = gemmini.reorder_stmts("b : _", "config_matmul()")
    gemmini = gemmini.reorder_stmts("a : _", "config_matmul()")
    gemmini = gemmini.fission_after("config_matmul()", n_lifts=1)
    gemmini = gemmini.reorder_stmts("do_zero_acc_i32(_, _, _)", "config_matmul()")
    gemmini = gemmini.reorder_stmts("res:_", "config_matmul()")
    gemmini = gemmini.fission_after("config_matmul()", n_lifts=4)

    # Real optimization

    # Why is this lost?
    gemmini = gemmini.par_to_seq('for jo in _:_')
    gemmini = gemmini.par_to_seq('for io in _:_')
    gemmini = gemmini.lift_alloc('a : i8', n_lifts=5)
    gemmini = gemmini.lift_alloc('b : i8', n_lifts=5)

    gemmini = gemmini.lift_alloc('res : _ #0', n_lifts=4)

    gemmini = gemmini.par_to_seq('for i in _:_ #0')
    gemmini = gemmini.par_to_seq('for j in _:_ #0')

    gemmini = gemmini.add_guard('do_ld_i8_id1(_)', 'j', 0)
    gemmini = gemmini.add_guard('do_ld_i8_id2(_)', 'i', 0)

    gemmini = gemmini.unroll('k')
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
"""

"""
