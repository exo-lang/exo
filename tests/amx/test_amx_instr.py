from __future__ import annotations

import os
import pytest

from exo import proc
from exo.stdlib.scheduling import *
from .amx import *
from .harness_amx import AMXTestBuilder

# --------------------------------------------------------------------------- #
#   Individual Load / Store / Zero Tests
# --------------------------------------------------------------------------- #

def test_ldst_i8_16x64(compiler, sde64):
    @proc
    def ldst_i8_16x64(x: i8[16, 64] @ DRAM, y: i8[16, 64] @ DRAM,
                      z: i8[16, 64] @ DRAM):
        config()
        tile0: i8[16, 64] @ AMX_TILE
        tile1: i8[16, 64] @ AMX_TILE
        ld_i8(16, 64, x, tile0)
        st_i8(16, 64, tile0, y)
        ld_i8(16, 64, y, tile1)
        st_i8(16, 64, tile1, z)

    t = AMXTestBuilder(compiler.basename)
    t.add_body([f"{compiler.basename}_Context *ctxt;"])

    t.alloc_dram_2i8('x', 16, 64, 'i+j')
    t.alloc_dram_2i8('y', 16, 64, '0')
    t.alloc_dram_2i8('z', 16, 64, '0')

    t.add_body(['ldst_i8_16x64(ctxt, x, y, z);',
                '',
                'if(check_eq_2i8(16, 64, x, z)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (x):\\n");',
                '    print_2i8(16, 64, x);',
                '    printf("Computed Roundtrip (z):\\n");',
                '    print_2i8(16, 64, z);',
                '    exit(1);',
                '}',
                ''])

    _run_amx(compiler, sde64, [ldst_i8_16x64], t)


def test_dpbssd_i8_16x64(compiler, sde64):
    @proc
    def dpbssd_i8_16x64(x: i8[16, 64] @ DRAM, y: i8[16, 64] @ DRAM,
                        z: i32[16, 16] @ DRAM):
        config()
        tile0: i8[16, 64] @ AMX_TILE
        tile1: i8[16, 64] @ AMX_TILE
        tile2: i32[16, 16] @ AMX_TILE
        ld_i8(16, 64, x, tile0)
        ld_i8(16, 64, y, tile1)
        dpbssd(16, 16, 16, tile0, tile1, tile2)
        st_i32(16, 16, tile2, z)

    t = AMXTestBuilder(compiler.basename)
    t.add_body([f"{compiler.basename}_Context *ctxt;"])

    t.alloc_dram_2i8('x', 16, 64, '1')
    t.alloc_dram_2i8('y', 16, 64, '1')
    t.alloc_dram_2i32('z', 16, 16, '64')  # expected result
    t.alloc_dram_2i32('res', 16, 16, '0')

    t.add_body(['dpbssd_i8_16x64(ctxt, x, y, res);',
                '',
                'if(check_eq_2i32(16, 16, z, res)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (z):\\n");',
                '    print_2i32(16, 16, z);',
                '    printf("Computed Roundtrip (res):\\n");',
                '    print_2i32(16, 16, res);',
                '    exit(1);',
                '}',
                ''])

    _run_amx(compiler, sde64, [dpbssd_i8_16x64], t)


def test_transform_memory(compiler, sde64):
    @proc
    def call_transform_memory(
            m: size,
            n: size,
            src: i8[4 * m, n] @ DRAM,
            dest: i8[m, 4 * n] @ DRAM,
    ):
        for i in seq(0, m):
            for j in seq(0, n):
                for k in seq(0, 4):
                    dest[i, 4 * j + k] = src[4 * i + k, j]

    t = AMXTestBuilder(compiler.basename)
    t.add_body([f"{compiler.basename}_Context *ctxt;"])

    t.alloc_dram_2i8('src', 64, 16, '5')
    t.alloc_dram_2i8('ans', 16, 64, '5')
    t.alloc_dram_2i8('res', 16, 64, '0')
    t.add_body(['call_transform_memory(ctxt, 16, 16, src, res);',
                '',
                'if(check_eq_2i8(16, 64, ans, res)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (ans):\\n");',
                '    print_2i8(16, 64, ans);',
                '    printf("Computed Roundtrip (res):\\n");',
                '    print_2i8(16, 64, res);',
                '    exit(1);',
                '}',
                ''])

    _run_amx(compiler, sde64, [call_transform_memory], t)


def matmul_algorithm_i8():
    @proc
    def matmul_on_cpu(
            M: size,
            K: size,
            N: size,
            A: i8[M, K] @ DRAM,
            B: i8[K, N] @ DRAM,
            C: i32[M, N] @ DRAM,
    ):
        for i in seq(0, M):
            for j in seq(0, N):
                C[i, j] = 0.0
                for k in seq(0, K):
                    # Casts the i8s to i32s before multiplying
                    # (note that i8 is signed).

                    C[i, j] += A[i, k] * B[k, j]

    return matmul_on_cpu


def modified_matmul_algorithm_i8():
    @proc
    def matmul_on_amx(
            M: size,
            K: size,
            N: size,
            A: i8[M, 4 * K] @ DRAM,
            B: i8[K, 4 * N] @ DRAM,
            C: i32[M, N] @ DRAM,
    ):
        assert K % 4 == 0
        config()  # TODO: how to insert this via Exo
        for i in seq(0, M):
            for j in seq(0, N):
                for k in seq(0, K):
                    for byte in seq(0, 4):
                        a: i32
                        b: i32

                        a = A[i, 4 * k + byte]
                        b = B[k, 4 * j + byte]

                        C[i, j] += a * b

    return matmul_on_amx


def get_transform_memory_i8():
    @proc
    def transform_memory(
            m: size,
            n: size,
            src: i8[m, n] @ DRAM,
            dest: i8[m / 4, 4 * n] @ DRAM,
    ):
        assert m % 4 == 0
        for i in seq(0, m / 4):
            for j in seq(0, n):
                for k in seq(0, 4):
                    dest[i, 4 * j + k] = src[4 * i + k, j]

    return transform_memory

def test_matmul_on_amx_by_hand_i8(compiler, sde64):
    @proc
    def matmul_on_amx(
            M: size,
            K: size,
            N: size,
            A: i8[M, 4 * K] @ DRAM,
            B: i8[K, 4 * N] @ DRAM,
            C: i32[M, N] @ DRAM,
    ):
        assert M % 32 == 0
        assert N % 32 == 0
        assert K % 16 == 0
        config()
        for i in seq(0, M / 32):
            for j in seq(0, N / 32):
                tileC1: i32[16, 16] @ AMX_TILE
                tileC2: i32[16, 16] @ AMX_TILE
                tileC3: i32[16, 16] @ AMX_TILE
                tileC4: i32[16, 16] @ AMX_TILE

                ld_i32(16, 16, C[i*32:i*32+16, j*32:j*32+16], tileC1)
                ld_i32(16, 16, C[i*32:i*32+16, j*32+16:j*32+32], tileC2)
                ld_i32(16, 16, C[i*32+16:i*32+32, j*32:j*32+16], tileC3)
                ld_i32(16, 16, C[i*32+16:i*32+32, j*32+16:j*32+32], tileC4)

                for k in seq(0, K / 16):
                    tileA1: i8[16, 64] @ AMX_TILE
                    tileA2: i8[16, 64] @ AMX_TILE
                    tileB1: i8[16, 64] @ AMX_TILE
                    tileB2: i8[16, 64] @ AMX_TILE

                    ld_i8(16, 64, A[16 * 2*i:16 * (2*i+1), 64 * k:64 * (k + 1)],
                          tileA1)
                    ld_i8(16, 64, A[16 * (2*i+1):16 * (2*i+2), 64 * k:64 * (k + 1)],
                          tileA2)
                    ld_i8(16, 64, B[16 * k:16 * (k + 1), 64 * 2*j:64 * (2*j+1)],
                          tileB1)
                    ld_i8(16, 64, B[16 * k:16 * (k + 1), 64 * (2*j+1):64 * (2*j+2)],
                          tileB2)
                    dpbssd(16, 16, 16, tileA1, tileB1, tileC1)
                    dpbssd(16, 16, 16, tileA1, tileB2, tileC2)
                    dpbssd(16, 16, 16, tileA2, tileB1, tileC3)
                    dpbssd(16, 16, 16, tileA2, tileB2, tileC4)

                st_i32(16, 16, tileC1, C[16 * 2*i:16 *
                       (2*i+1), 16 * 2*j:16 * (2*j+1)])
                st_i32(16, 16, tileC2, C[16 * 2*i:16 *
                       (2*i+1), 16 * (2*j+1):16 * (2*j+2)])
                st_i32(16, 16, tileC3, C[16 * (2*i+1):16 * (2*i+2), 16 * 2*j:16 * (2*j+1)])
                st_i32(16, 16, tileC4,
                       C[16 * (2*i+1):16 * (2*i+2), 16 * (2*j+1):16 * (2*j+2)])

    size1 = 256
    size2 = 256

    t = AMXTestBuilder(compiler.basename)
    t.add_body([f"{compiler.basename}_Context *ctxt;"])

    t.alloc_dram_2i8('x', size1, size2, 'i+j')
    t.alloc_dram_2i8('y_orig', size2, size1, 'j+1')  # before transform_memory
    t.alloc_dram_2i8('y', size2 // 4, 4 * size1, '0')  # after transform_memory
    t.alloc_dram_2i32('z', size1, size1, '0')  # expected result
    t.alloc_dram_2i32('res', size1, size1, '0')

    t.add_body([f'transform_memory(ctxt, {size2}, {size1}, y_orig, y);'])
    t.add_body(
        [f'matmul_on_cpu(ctxt, {size1}, {size2}, {size1}, x, y_orig, z);'])
    t.add_body(
        [f'matmul_on_amx(ctxt, {size1}, {size2 // 4}, {size1}, x, y, res);'])

    t.add_body([f'if(check_eq_2i32({size1}, {size1}, z, res)) {{',
                r'    printf("Correct\n");',
                r'} else {',
                r'    printf("Results Don\'t Match\n");',
                r'    printf("Correct Result (z):\n");',
                f'    print_2i32({size1}, {size1}, z);',
                r'    printf("Computed Roundtrip (res):\n");',
                f'    print_2i32({size1}, {size1}, res);',
                r'    exit(1);',
                r'}',
                r''])

    _run_amx(compiler, sde64, [
        matmul_algorithm_i8(), get_transform_memory_i8(), matmul_on_amx
    ], t)

@pytest.fixture
def matmul_i8():
    size1 = 256
    size2 = 256

    amx = modified_matmul_algorithm_i8().partial_eval(size1, size2 // 4, size1)

    print("Base Implementation: ")
    print(amx)

    print("Loop splitting and reordering:")
    amx = divide_loop(amx, 'i', 32, ['io', 'ii'], perfect=True)
    amx = divide_loop(amx, 'j', 32, ['jo', 'ji'], perfect=True)
    amx = reorder_loops(amx, 'ii jo')
    amx = divide_loop(amx, 'k', 16, ['ko', 'ki'], perfect=True)
    amx = reorder_loops(amx, 'ji ko')
    amx = reorder_loops(amx, 'ii ko')
    amx = divide_loop(amx, 'ii', 16, ['ii_unroll', 'ii'], perfect=True)
    amx = divide_loop(amx, 'ji', 16, ['ji_unroll', 'ji'], perfect=True)
    amx = reorder_loops(amx, 'ii ji_unroll')
    print(amx)

    print("Staging A and B memory and replacing their loads")
    amx = amx.partition_loop('ji_unroll', 1)
    B_mem_template = "B[16*ko:16*(ko+1), 128*jo+64*(ji_unroll+{j_lo}):128*jo+64*(ji_unroll+{j_lo}+1)]"
    for i in range(2):
        amx = amx.stage_mem(B_mem_template.format(j_lo=i), f'Btile{i}', f'for ii in _: _ #{i}')
        amx = set_memory(amx, f'Btile{i}', AMX_TILE)
        amx = lift_alloc(amx, f'Btile{i}:_', n_lifts=3)
    amx = fission(amx, amx.find('for i0 in _:_ #0').after(), n_lifts=1)
    amx = fission(amx, amx.find('for i0 in _:_ #1').after(), n_lifts=1)
    amx = amx.reorder_before('for ji_unroll in _:_ #2')
    amx = fission(amx, amx.find('for ji_unroll in _:_ #1').after(), n_lifts=1)
    amx = remove_loop(amx, 'ii_unroll #0')
    amx = amx.partition_loop('ii_unroll', 1)
    A_mem_template = "A[32 * io + 16*(ii_unroll+{i_lo}):32 * io + 16*(ii_unroll+{i_lo}+1), 64*ko:64*(ko+1)]"
    for i in range(2):
        amx = fusion(amx, f'for ji_unroll in _:_ #{i+2}',
                          f'for ji_unroll in _:_ #{i+3}')
        amx = amx.stage_mem(A_mem_template.format(i_lo=i), f'Atile{i}', f'for ji_unroll in _:_ #{i+2}')
        amx = set_memory(amx, f'Atile{i}', AMX_TILE)
        amx = lift_alloc(amx, f'Atile{i}:_', n_lifts=2)
    amx = fission(amx, amx.find('for i0 in _:_ #2').after(), n_lifts=1)
    amx = fission(amx, amx.find('for i0 in _:_ #3').after(), n_lifts=1)
    amx = amx.reorder_before('for ii_unroll in _:_ #2')
    amx = amx.unroll('ji_unroll')
    amx = amx.unroll('ii_unroll')
    amx = simplify(amx)
    amx = amx.replace(ld_i8, 'for i0 in _:_')
    print(amx)

    print("Staging C memory")
    C_mem_template = "C[32*io + 16*({i_lo}):32*io + 16*({i_lo}+1), 32*jo + 16*({j_lo}):32*jo + 16*({j_lo}+1)]"
    for i in range(4):
        amx = amx.stage_mem(C_mem_template.format(i_lo=i//2, j_lo=i%2), f'Ctile{i}', f'for ii in _: _ #{i}')
        amx = set_memory(amx, f'Ctile{i}', AMX_TILE)
        amx = lift_alloc(amx, f'Ctile{i}:_', n_lifts=1)
        for j in range(4+i):
            amx = amx.reorder_before(f'for i0 in _:_ #{i}')
        amx = fission(amx, amx.find(f'for i0 in _:_ #{i}').after(), n_lifts=1)
        for j in range(i+1, 4):
            amx = amx.reorder_before(f'for ii in _:_ #{j}')
        amx = fission(amx, amx.find('for ii in _:_ #3').after(), n_lifts=1)
    amx = simplify(amx)
    for i in range(4):
        amx = remove_loop(amx, 'ko #0')
        amx = amx.replace(ld_i32, 'for i0 in _:_ #0')
    for i in range(4):
        amx = remove_loop(amx, 'ko #1')
        amx = amx.replace(st_i32, 'for i0 in _:_ #0')
    amx = simplify(amx)
    amx = amx.replace(dpbssd, 'for ii in _:_')
    print(amx)

    print("Final scheduled algorithm:")
    print(amx)

    return amx

def test_gen_matmul_i8_amx(matmul_i8):
    pass

def test_matmul_on_amx_scheduled_i8(compiler, sde64, matmul_i8):
    t = AMXTestBuilder(compiler.basename)
    t.add_body([f"{compiler.basename}_Context *ctxt;"])

    # duplicate from the fixture above
    size1 = 256
    size2 = 256

    t.alloc_dram_2i8('x', size1, size2, 'i+j')
    t.alloc_dram_2i8('y_orig', size2, size1, 'j')  # before transform_memory
    t.alloc_dram_2i8('y', size2 // 4, 4 * size1, '0')  # after transform_memory
    t.alloc_dram_2i32('z', size1, size1, '0')  # expected result
    t.alloc_dram_2i32('res', size1, size1, '0')

    t.add_body([f'transform_memory(ctxt, {size2}, {size1}, y_orig, y);'])
    t.add_body(
        [f'matmul_on_cpu(ctxt, {size1}, {size2}, {size1}, x, y_orig, z);'])
    t.add_body(
        [f'matmul_on_amx(ctxt, x, y, res);'])

    t.add_body([f'if(check_eq_2i32({size1}, {size1}, z, res)) {{',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (z):\\n");',
                f'    print_2i32({size1}, {size1}, z);',
                '    printf("Computed Roundtrip (res):\\n");',
                f'    print_2i32({size1}, {size1}, res);',
                '    exit(1);',
                '}',
                ''])

    _run_amx(compiler, sde64, [
        matmul_algorithm_i8(), get_transform_memory_i8(), matmul_i8,
    ], t)


def _run_amx(compiler, sde64, procs, test_source):
    test_exe = compiler.compile(
        procs,
        test_files={'main.c': str(test_source)},
        CMAKE_C_COMPILER=os.getenv('CLANG', os.getenv('CC', 'clang-13')),
        CMAKE_C_FLAGS='-mamx-int8 -mamx-tile',
    )

    sde64(test_exe)
