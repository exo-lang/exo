from __future__ import annotations

from SYS_ATL.libs.memories import MDRAM
from .gemmini import *
from .harness_gemmini import GemmTestBuilder

import pytest

pytest.skip("skipping gemmini tests that are bitrotted",
            allow_module_level=True)
# --------------------------------------------------------------------------- #
#   Basic MatMul Test
# --------------------------------------------------------------------------- #

def test_matmul_c_i8():
    T = GemmTestBuilder('matmul_c_i8')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i8_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i8][N,M] @ MDRAM,
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res : i32
                res = 0.0
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                if acc == True:
                    res = relu(res)

                tmp_res1 : f32
                tmp_res1 = res
                tmp_res1 = tmp_res1 * c_scale

                tmp_res2 : i8
                clamp(tmp_res1, tmp_res2)
                C[i,j] = tmp_res2

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i8(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i8][N,M] @ DRAM,
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                zero_acc_i32(16,16,res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,16, c_scale, acc, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                zero_acc_i32(N%16,16,res)

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(N%16,16, c_scale, acc, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                zero_acc_i32(16,16,res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,M%16, c_scale, acc, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            zero_acc_i32(N%16,16,res)

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i8(N%16, M%16, c_scale, acc, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i8)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_cpu, {NN}, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i8(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_gemmini, {NN}, 1 }});',
                f'gemmini_fence();',
                f''])
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


def test_matmul_c_i32():
    T = GemmTestBuilder('matmul_c_i32')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i32_lib_Context *ctxt;"])

    # TODO: fails when 7 7 8
    NN = 70
    MM = 70
    KK = 80

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_f32('a_scale', '1.0f')
    T.alloc_dram_f32('b_scale', '1.0f')
    T.alloc_dram_2i32('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i32('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i32][N,M] @ MDRAM,
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res : i32
                res = 0.0
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                C[i,j] = res

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i32(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i32][N,M] @ DRAM,
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                zero_acc_i32(16,16,res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                zero_acc_i32(N%16,16,res)

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(N%16,16, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                zero_acc_i32(16,16,res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,M%16, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            zero_acc_i32(N%16,16,res)

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i32(N%16, M%16, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i32)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_cpu, {NN}, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i32(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_gemmini, {NN}, 1 }});',
                f'gemmini_fence();',
                f''])
    T.stop_timer('gemmini', 'Cycles for GEMMINI version')

    T.add_body([f'if(check_eq_2i32({NN},{MM}, z_cpu, z_gemmini)) {{',
                 '    printf("Correct\\n");',
                 '} else {',
                 '    printf("Results Don\'t Match\\n");',
                 '    printf("Correct Result (z_cpu):\\n");',
                f'    print_2i32({NN},{MM}, z_cpu);',
                 '    printf("Computed Roundtrip (z_gemmini):\\n");',
                f'    print_2i32({NN},{MM}, z_gemmini);',
                 '    exit(1);',
                 '}',
                 ''])


    T.compile().run()


def test_matmul_c_i8_d_i8():
    T = GemmTestBuilder('matmul_c_i8_d_i8')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i8_d_i8_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_2i8('d', NN, MM, '2')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_f32('d_scale', '4.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      d_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i8][N,M] @ MDRAM,
      D : [i8][N,M] @ MDRAM
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res   : i32
                tmp_d : f32
                tmp_d = D[i,j]
                tmp_d = tmp_d * d_scale
                res   = tmp_d
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                if acc == True:
                    res = relu(res)

                tmp_res1 : f32
                tmp_res1 = res
                tmp_res1 = tmp_res1 * c_scale

                tmp_res2 : i8
                clamp(tmp_res1, tmp_res2)
                C[i,j] = tmp_res2

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i8_d_i8(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      d_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i8][N,M] @ DRAM,
      D : [i8][N,M] @ DRAM
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1
        assert stride(D, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                ld_acc_i8(16, 16, d_scale, D[ 16*i:16*(i+1), 16*j:16*(j+1) ], res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,16, c_scale, acc, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                ld_acc_i8(N%16, 16, d_scale, D[ N-N%16:, 16*j:16*(j+1) ], res)

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(N%16,16, c_scale, acc, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                ld_acc_i8(16, M%16, d_scale, D[ 16*i:16*(i+1), M-M%16: ], res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,M%16, c_scale, acc, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            ld_acc_i8(N%16, M%16, d_scale, D[ N-N%16:, M-M%16: ], res)

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i8(N%16, M%16, c_scale, acc, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i8_d_i8)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, d_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_cpu, {NN}, 1 }}, (struct systl_win_2i8){{ d, {NN}, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i8_d_i8(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, d_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_gemmini, {NN}, 1 }}, (struct systl_win_2i8){{ d, {NN}, 1 }});',
                f'gemmini_fence();',
                f''])
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



def test_matmul_c_i8_d_i8_rep():
    T = GemmTestBuilder('matmul_c_i8_d_i8_rep')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i8_d_i8_rep_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_2i8('d', 1, MM, '2')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_f32('d_scale', '4.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      d_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i8][N,M] @ MDRAM,
      D : [i8][1,M] @ MDRAM
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res   : i32
                tmp_d : f32
                tmp_d = D[0,j]
                tmp_d = tmp_d * d_scale
                res   = tmp_d
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                if acc == True:
                    res = relu(res)

                tmp_res1 : f32
                tmp_res1 = res
                tmp_res1 = tmp_res1 * c_scale

                tmp_res2 : i8
                clamp(tmp_res1, tmp_res2)
                C[i,j] = tmp_res2

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i8_d_i8_rep(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      d_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i8][N,M] @ DRAM,
      D : [i8][1,M] @ DRAM
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1
        assert stride(D, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                for l in par(0, 16):
                    ld_acc_i8(1, 16, d_scale, D[ 0:1, 16*j:16*(j+1) ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,16, c_scale, acc, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                for l in par(0, N%16):
                    ld_acc_i8(1, 16, d_scale, D[ 0:1, 16*j:16*(j+1) ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(N%16,16, c_scale, acc, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                for l in par(0, 16):
                    ld_acc_i8(1, M%16, d_scale, D[ 0:1, M-M%16: ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,M%16, c_scale, acc, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            for l in par(0, N%16):
                ld_acc_i8(1, M%16, d_scale, D[ 0:1, M-M%16: ], res[l:l+1, :])

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i8(N%16, M%16, c_scale, acc, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i8_d_i8_rep)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, d_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_cpu, {NN}, 1 }}, (struct systl_win_2i8){{ d, 1, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i8_d_i8_rep(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, d_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_gemmini, {NN}, 1 }}, (struct systl_win_2i8){{ d, 1, 1 }});',
                f'gemmini_fence();',
                f''])
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



def test_matmul_c_i8_d_i32():
    T = GemmTestBuilder('matmul_c_i8_d_i32')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i8_d_i32_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_2i32('d', NN, MM, '2')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_f32('d_scale', '4.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      d_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i8][N,M] @ MDRAM,
      D : [i32][N,M] @ MDRAM
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res   : i32
                tmp_d : f32
                tmp_d = D[i,j]
                tmp_d = tmp_d * d_scale
                res   = tmp_d
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                if acc == True:
                    res = relu(res)

                tmp_res1 : f32
                tmp_res1 = res
                tmp_res1 = tmp_res1 * c_scale

                tmp_res2 : i8
                clamp(tmp_res1, tmp_res2)
                C[i,j] = tmp_res2

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i8_d_i32(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      d_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i8][N,M] @ DRAM,
      D : [i32][N,M] @ DRAM
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1
        assert stride(D, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                ld_acc_i32(16, 16, d_scale, D[ 16*i:16*(i+1), 16*j:16*(j+1) ], res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,16, c_scale, acc, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                ld_acc_i32(N%16, 16, d_scale, D[ N-N%16:, 16*j:16*(j+1) ], res)

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(N%16,16, c_scale, acc, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                ld_acc_i32(16, M%16, d_scale, D[ 16*i:16*(i+1), M-M%16: ], res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,M%16, c_scale, acc, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            ld_acc_i32(N%16, M%16, d_scale, D[ N-N%16:, M-M%16: ], res)

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i8(N%16, M%16, c_scale, acc, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i8_d_i32)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, d_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_cpu, {NN}, 1 }}, (struct systl_win_2i32){{ d, {NN}, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i8_d_i32(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, d_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_gemmini, {NN}, 1 }}, (struct systl_win_2i32){{ d, {NN}, 1 }});',
                f'gemmini_fence();',
                f''])
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



def test_matmul_c_i8_d_i32_rep():
    T = GemmTestBuilder('matmul_c_i8_d_i32_rep')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i8_d_i32_rep_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_2i32('d', 1, MM, '2')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('c_scale', '2.0f')
    T.alloc_dram_f32('d_scale', '4.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      d_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i8][N,M] @ MDRAM,
      D : [i32][1,M] @ MDRAM
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res   : i32
                tmp_d : f32
                tmp_d = D[0,j]
                tmp_d = tmp_d * d_scale
                res   = tmp_d
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                if acc == True:
                    res = relu(res)

                tmp_res1 : f32
                tmp_res1 = res
                tmp_res1 = tmp_res1 * c_scale

                tmp_res2 : i8
                clamp(tmp_res1, tmp_res2)
                C[i,j] = tmp_res2

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i8_d_i32_rep(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      d_scale : f32,
      acc     : bool,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i8][N,M] @ DRAM,
      D : [i32][1,M] @ DRAM
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1
        assert stride(D, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                for l in par(0, 16):
                    ld_acc_i32(1, 16, d_scale, D[ 0:1, 16*j:16*(j+1) ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,16, c_scale, acc, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                for l in par(0, N%16):
                    ld_acc_i32(1, 16, d_scale, D[ 0:1, 16*j:16*(j+1) ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i8(N%16,16, c_scale, acc, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                for l in par(0, 16):
                    ld_acc_i32(1, M%16, d_scale, D[ 0:1, M-M%16: ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i8(16,M%16, c_scale, acc, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            for l in par(0, N%16):
                ld_acc_i32(1, M%16, d_scale, D[ 0:1, M-M%16: ], res[l:l+1, :])

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i8(N%16, M%16, c_scale, acc, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i8_d_i32_rep)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, d_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_cpu, {NN}, 1 }}, (struct systl_win_2i32){{ d, 1, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i8_d_i32_rep(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, d_scale, false, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i8){{ z_gemmini, {NN}, 1 }}, (struct systl_win_2i32){{ d, 1, 1 }});',
                f'gemmini_fence();',
                f''])
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



def test_matmul_c_i32_d_i8():
    T = GemmTestBuilder('matmul_c_i32_d_i8')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i32_d_i8_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_2i8('d', NN, MM, '2')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('d_scale', '4.0f')
    T.alloc_dram_2i32('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i32('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      d_scale : f32,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i32][N,M] @ MDRAM,
      D : [i8][N,M] @ MDRAM
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res   : i32
                tmp_d : f32
                tmp_d = D[i,j]
                tmp_d = tmp_d * d_scale
                res   = tmp_d
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                C[i,j] = res

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i32_d_i8(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      d_scale : f32,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i32][N,M] @ DRAM,
      D : [i8][N,M] @ DRAM
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1
        assert stride(D, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                ld_acc_i8(16, 16, d_scale, D[ 16*i:16*(i+1), 16*j:16*(j+1) ], res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                ld_acc_i8(N%16, 16, d_scale, D[ N-N%16:, 16*j:16*(j+1) ], res)

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(N%16,16, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                ld_acc_i8(16, M%16, d_scale, D[ 16*i:16*(i+1), M-M%16: ], res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,M%16, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            ld_acc_i8(N%16, M%16, d_scale, D[ N-N%16:, M-M%16: ], res)

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i32(N%16, M%16, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i32_d_i8)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, d_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_cpu, {NN}, 1 }}, (struct systl_win_2i8){{ d, {NN}, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i32_d_i8(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, d_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_gemmini, {NN}, 1 }}, (struct systl_win_2i8){{ d, {NN}, 1 }});',
                f'gemmini_fence();',
                f''])
    T.stop_timer('gemmini', 'Cycles for GEMMINI version')

    T.add_body([f'if(check_eq_2i32({NN},{MM}, z_cpu, z_gemmini)) {{',
                 '    printf("Correct\\n");',
                 '} else {',
                 '    printf("Results Don\'t Match\\n");',
                 '    printf("Correct Result (z_cpu):\\n");',
                f'    print_2i32({NN},{MM}, z_cpu);',
                 '    printf("Computed Roundtrip (z_gemmini):\\n");',
                f'    print_2i32({NN},{MM}, z_gemmini);',
                 '    exit(1);',
                 '}',
                 ''])


    T.compile().run()


def test_matmul_c_i32_d_i8_rep():
    T = GemmTestBuilder('matmul_c_i32_d_i8_rep')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i32_d_i8_rep_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_2i8('d', 1, MM, '2')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('d_scale', '4.0f')
    T.alloc_dram_2i32('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i32('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      d_scale : f32,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i32][N,M] @ MDRAM,
      D : [i8][1,M] @ MDRAM
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res   : i32
                tmp_d : f32
                tmp_d = D[0,j]
                tmp_d = tmp_d * d_scale
                res   = tmp_d
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                C[i,j] = res

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i32_d_i8_rep(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      d_scale : f32,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i32][N,M] @ DRAM,
      D : [i8][1,M] @ DRAM
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1
        assert stride(D, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                for l in par(0, 16):
                    ld_acc_i8(1, 16, d_scale, D[ 0:1, 16*j:16*(j+1) ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                for l in par(0, N%16):
                    ld_acc_i8(1, 16, d_scale, D[ 0:1, 16*j:16*(j+1) ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(N%16,16, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                for l in par(0, 16):
                    ld_acc_i8(1, M%16, d_scale, D[ 0:1, M-M%16: ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,M%16, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            for l in par(0, N%16):
                ld_acc_i8(1, M%16, d_scale, D[ 0:1, M-M%16: ], res[l:l+1, :])

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i32(N%16, M%16, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i32_d_i8_rep)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, d_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_cpu, {NN}, 1 }}, (struct systl_win_2i8){{ d, 1, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i32_d_i8_rep(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, d_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_gemmini, {NN}, 1 }}, (struct systl_win_2i8){{ d, 1, 1 }});',
                f'gemmini_fence();',
                f''])
    T.stop_timer('gemmini', 'Cycles for GEMMINI version')

    T.add_body([f'if(check_eq_2i32({NN},{MM}, z_cpu, z_gemmini)) {{',
                 '    printf("Correct\\n");',
                 '} else {',
                 '    printf("Results Don\'t Match\\n");',
                 '    printf("Correct Result (z_cpu):\\n");',
                f'    print_2i32({NN},{MM}, z_cpu);',
                 '    printf("Computed Roundtrip (z_gemmini):\\n");',
                f'    print_2i32({NN},{MM}, z_gemmini);',
                 '    exit(1);',
                 '}',
                 ''])


    T.compile().run()



def test_matmul_c_i32_d_i32():
    T = GemmTestBuilder('matmul_c_i32_d_i32')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i32_d_i32_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_2i32('d', NN, MM, '2')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('d_scale', '4.0f')
    T.alloc_dram_2i32('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i32('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      d_scale : f32,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i32][N,M] @ MDRAM,
      D : [i32][N,M] @ MDRAM
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res   : i32
                tmp_d : f32
                tmp_d = D[i,j]
                tmp_d = tmp_d * d_scale
                res   = tmp_d
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                C[i,j] = res

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i32_d_i32(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      d_scale : f32,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i32][N,M] @ DRAM,
      D : [i32][N,M] @ DRAM
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1
        assert stride(D, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                ld_acc_i32(16, 16, d_scale, D[ 16*i:16*(i+1), 16*j:16*(j+1) ], res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                ld_acc_i32(N%16, 16, d_scale, D[ N-N%16:, 16*j:16*(j+1) ], res)

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(N%16,16, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                ld_acc_i32(16, M%16, d_scale, D[ 16*i:16*(i+1), M-M%16: ], res)

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,M%16, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            ld_acc_i32(N%16, M%16, d_scale, D[ N-N%16:, M-M%16: ], res)

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i32(N%16, M%16, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i32_d_i32)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, d_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_cpu, {NN}, 1 }}, (struct systl_win_2i32){{ d, {NN}, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i32_d_i32(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, d_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_gemmini, {NN}, 1 }}, (struct systl_win_2i32){{ d, {NN}, 1 }});',
                f'gemmini_fence();',
                f''])
    T.stop_timer('gemmini', 'Cycles for GEMMINI version')

    T.add_body([f'if(check_eq_2i32({NN},{MM}, z_cpu, z_gemmini)) {{',
                 '    printf("Correct\\n");',
                 '} else {',
                 '    printf("Results Don\'t Match\\n");',
                 '    printf("Correct Result (z_cpu):\\n");',
                f'    print_2i32({NN},{MM}, z_cpu);',
                 '    printf("Computed Roundtrip (z_gemmini):\\n");',
                f'    print_2i32({NN},{MM}, z_gemmini);',
                 '    exit(1);',
                 '}',
                 ''])


    T.compile().run()


def test_matmul_c_i32_d_i32_rep():
    T = GemmTestBuilder('matmul_c_i32_d_i32_rep')
    T.add_body(['gemm_init_mem();',
                'init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_c_i32_d_i32_rep_lib_Context *ctxt;"])

    NN = 60
    MM = 70
    KK = 120

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_2i32('d', 1, MM, '2')
    T.alloc_dram_f32('a_scale', '3.0f')
    T.alloc_dram_f32('b_scale', '2.0f')
    T.alloc_dram_f32('d_scale', '4.0f')
    T.alloc_dram_2i32('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i32('z_gemmini', NN, MM, '0')

    @proc
    def matmul_on_cpu(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      d_scale : f32,
      A : [i8][N,K] @ MDRAM,
      B : [i8][K,M] @ MDRAM,
      C : [i32][N,M] @ MDRAM,
      D : [i32][1,M] @ MDRAM
    ):
        assert N <= 16
        assert M <= 16
        assert K <= 16

        for i in par(0,N):
            for j in par(0,M):
                res   : i32
                tmp_d : f32
                tmp_d = D[0,j]
                tmp_d = tmp_d * d_scale
                res   = tmp_d
                for k in par(0,K):
                    tmp_a : f32
                    tmp_b : f32
                    tmp_a = A[i,k]
                    tmp_b = B[k,j]
                    tmp_a = tmp_a * a_scale
                    tmp_b = tmp_b * b_scale
                    a : i32
                    b : i32
                    a = tmp_a
                    b = tmp_b
                    res += a*b

                C[i,j] = res

    T.add_proc(matmul_on_cpu)

    @proc
    def matmul_c_i32_d_i32_rep(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      d_scale : f32,
      A : [i8][N,K] @ DRAM,
      B : [i8][K,M] @ DRAM,
      C : [i32][N,M] @ DRAM,
      D : [i32][1,M] @ DRAM
    ):
        assert stride(A, 1) == 1
        assert stride(B, 1) == 1
        assert stride(C, 1) == 1
        assert stride(D, 1) == 1

        for i in par(0,N/16):
            for j in par(0,M/16):
                res : i32[16,16] @ GEMM_ACCUM
                for l in par(0, 16):
                    ld_acc_i32(1, 16, d_scale, D[ 0:1, 16*j:16*(j+1) ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,16, res, C[ 16*i:16*(i+1), 16*j:16*(j+1) ])

        if N%16 > 0:
            for j in par(0,M/16):
                res : i32[N%16,16] @ GEMM_ACCUM
                for l in par(0, N%16):
                    ld_acc_i32(1, 16, d_scale, D[ 0:1, 16*j:16*(j+1) ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,16, b_scale, B[ 16*k:16*(k+1), 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[N%16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                    ld_i8(K%16,16, b_scale, B[ K-K%16:, 16*j:16*(j+1) ], Bblock)

                    matmul_acc_i8(N%16,16,K%16, Ablock, Bblock, res)

                st_acc_i32(N%16,16, res, C[ N-N%16:, 16*j:16*(j+1) ])

        if M%16 > 0:
            for i in par(0,N/16):
                res : i32[16,16] @ GEMM_ACCUM
                for l in par(0, 16):
                    ld_acc_i32(1, M%16, d_scale, D[ 0:1, M-M%16: ], res[l:l+1, :])

                for k in par(0,K/16):
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[16,16] @ GEMM_SCRATCH
                    ld_i8(16,16, a_scale, A[ 16*i:16*(i+1), 16*k:16*(k+1) ], Ablock)
                    ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,16, Ablock, Bblock, res)

                if K%16 > 0:
                    Ablock : i8[16,16] @ GEMM_SCRATCH
                    Bblock : i8[K%16,16] @ GEMM_SCRATCH
                    ld_i8(16,K%16, a_scale, A[ 16*i:16*(i+1), K-K%16: ], Ablock)
                    ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                    matmul_acc_i8(16,M%16,K%16, Ablock, Bblock, res)

                st_acc_i32(16,M%16, res, C[ 16*i:16*(i+1), M-M%16: ])

        if N%16 > 0 and M%16 > 0:
            res : i32[N%16,16] @ GEMM_ACCUM
            for l in par(0, N%16):
                ld_acc_i32(1, M%16, d_scale, D[ 0:1, M-M%16: ], res[l:l+1, :])

            for k in par(0,K/16):
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[16,16] @ GEMM_SCRATCH
                ld_i8(N%16,16, a_scale, A[ N-N%16:, 16*k:16*(k+1) ], Ablock)
                ld_i8(16,M%16, b_scale, B[ 16*k:16*(k+1), M-M%16: ], Bblock)

                matmul_acc_i8(N%16,M%16,16, Ablock, Bblock, res)

            if K%16 > 0:
                Ablock : i8[N%16,16] @ GEMM_SCRATCH
                Bblock : i8[K%16,16] @ GEMM_SCRATCH
                ld_i8(N%16,K%16, a_scale, A[ N-N%16:, K-K%16: ], Ablock)
                ld_i8(K%16,M%16, b_scale, B[ K-K%16:, M-M%16: ], Bblock)

                matmul_acc_i8(N%16, M%16, K%16, Ablock, Bblock, res)

            st_acc_i32(N%16, M%16, res, C[ N-N%16:, M-M%16: ])


    T.add_proc(matmul_c_i32_d_i32_rep)

    T.start_timer('cpu')
    T.add_body([f'matmul_on_cpu(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, d_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_cpu, {NN}, 1 }}, (struct systl_win_2i32){{ d, 1, 1 }});',
                f'gemmini_fence();'])
    T.stop_timer('cpu', 'Cycles for CPU version')

    T.start_timer('gemmini')
    T.add_body([f'matmul_c_i32_d_i32_rep(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, d_scale, (struct systl_win_2i8){{ x, {NN}, 1 }}, (struct systl_win_2i8){{ y, {KK}, 1 }}, (struct systl_win_2i32){{ z_gemmini, {NN}, 1 }}, (struct systl_win_2i32){{ d, 1, 1 }});',
                f'gemmini_fence();',
                f''])
    T.stop_timer('gemmini', 'Cycles for GEMMINI version')

    T.add_body([f'if(check_eq_2i32({NN},{MM}, z_cpu, z_gemmini)) {{',
                 '    printf("Correct\\n");',
                 '} else {',
                 '    printf("Results Don\'t Match\\n");',
                 '    printf("Correct Result (z_cpu):\\n");',
                f'    print_2i32({NN},{MM}, z_cpu);',
                 '    printf("Computed Roundtrip (z_gemmini):\\n");',
                f'    print_2i32({NN},{MM}, z_gemmini);',
                 '    exit(1);',
                 '}',
                 ''])


    T.compile().run()
