from __future__ import annotations

import pytest
import platform
# if platform.system() == 'Darwin':
#     pytest.skip("skipping x86 tests on Apple machines for now",
#                 allow_module_level=True)

import sys
from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import AMX_TILE
from .amx import *
from .harness_amx import ENV, AMXTestBuilder


# --------------------------------------------------------------------------- #
#   Individual Load / Store / Zero Tests
# --------------------------------------------------------------------------- #

def test_ldst_i8_16x64():
    T = AMXTestBuilder('ldst_i8_16x64')
    T.add_body(["ldst_i8_16x64_lib_Context *ctxt;"])

    @proc
    def ldst_i8_16x64(x: i8[16, 64] @ DRAM, y: i8[16, 64] @ DRAM, z: i8[16, 64] @ DRAM):
        config()
        tile0: i8[16, 64] @ AMX_TILE
        tile1: i8[16, 64] @ AMX_TILE
        ld_i8(16, 64, x, tile0)
        st_i8(16, 64, tile0, y)
        ld_i8(16, 64, y, tile1)
        st_i8(16, 64, tile1, z)

    T.add_proc(ldst_i8_16x64)

    T.alloc_dram_2i8('x', 16, 64, 'i+j')
    T.alloc_dram_2i8('y', 16, 64, '0')
    T.alloc_dram_2i8('z', 16, 64, '0')

    T.add_body(['ldst_i8_16x64(ctxt, x, y, z);',
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

    T.compile().run()


def test_dpbuud_i8_16x64():
    T = AMXTestBuilder('dpbuud_i8_16x64')
    T.add_body(["dpbuud_i8_16x64_lib_Context *ctxt;"])

    @proc
    def dpbuud_i8_16x64(x: i8[16, 64] @ DRAM, y: i8[16, 64] @ DRAM, z: i32[16, 16] @ DRAM):
        config()
        tile0: i8[16, 64] @ AMX_TILE
        tile1: i8[16, 64] @ AMX_TILE
        tile2: i32[16, 16] @ AMX_TILE
        ld_i8(16, 64, x, tile0)
        ld_i8(16, 64, y, tile1)
        dpbuud(16, 16, 16, tile0, tile1, tile2)
        st_i32(16, 16, tile2, z)

    T.add_proc(dpbuud_i8_16x64)

    T.alloc_dram_2i8('x', 16, 64, '1')
    T.alloc_dram_2i8('y', 16, 64, '1')
    T.alloc_dram_2i32('z', 16, 16, '64')  # expected result
    T.alloc_dram_2i32('res', 16, 16, '0')

    T.add_body(['dpbuud_i8_16x64(ctxt, x, y, res);',
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

    T.compile().run()


def test_transform_memory():
    T = AMXTestBuilder('transform_memory')
    T.add_body(["transform_memory_lib_Context *ctxt;"])

    @proc
    def call_transform_memory(
        m: size,
        n: size,
        src: i8[4*m, n] @ DRAM,
        dest: i8[m, 4*n] @ DRAM,
    ):
        for i in par(0, m):
            for j in par(0, n):
                for k in par(0, 4):
                    dest[i, 4*j+k] = src[4*i + k, j]

    T.add_proc(call_transform_memory)
    T.alloc_dram_2i8('src', 64, 16, '5')
    T.alloc_dram_2i8('ans', 16, 64, '5')
    T.alloc_dram_2i8('res', 16, 64, '0')
    T.add_body(['call_transform_memory(ctxt, 16, 16, src, res);',
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
    T.compile().run()


def test_matmul_c_i8():
    T = AMXTestBuilder('matmul_c_i8')
    T.add_body(["matmul_c_i8_lib_Context *ctxt;"])

    @proc
    def matmul_on_cpu(
        N: size,
        M: size,
        K: size,
        A: [i8][N, K] @ DRAM,
        B: [i8][K, M] @ DRAM,
        C: [i8][N, M] @ DRAM,
    ):
        for i in par(0, N):
            for j in par(0, M):
                C[i, j] = 0.0
                for k in par(0, K):
                    a: i32
                    b: i32

                    a = A[i, k]
                    b = B[k, j]

                    C[i, j] += a * b

    @proc
    def matmul_c_i8(
        M: size,
        K: size,
        N: size,
        # TODO: can't make this windowed. Might be because of the cast I added in memory.py?
        A: i8[M, K] @ DRAM,
        B: i8[K, N] @ DRAM,
        C: i32[M, N] @ DRAM,
    ):
        config()
        for i in par(0, M/16):
            for j in par(0, N/16):
                tileC: i32[16, 16] @ AMX_TILE
                zero_i32(16, 16, tileC)

                for k in par(0, K/64):
                    Bcopy: i8[16, 64] @ DRAM

                    # TODO: any way to separate this into its own function?
                    for x in par(0, 16):
                        for y in par(0, 16):
                            for z in par(0, 4):
                                Bcopy[x, 4*y+z] = B[64*k + 4*x + z, 16*j + y]

                    tileA: i8[16, 64] @ AMX_TILE
                    tileB: i8[16, 64] @ AMX_TILE
                    ld_i8(16, 64, A[16*i:16*(i+1), 64*k:64*(k+1)], tileA)
                    ld_i8(16, 64, Bcopy, tileB)
                    dpbuud(16, 16, 16, tileA, tileB, tileC)

                temp: i32[16, 16] @ DRAM
                st_i32(16, 16, tileC, temp)
                for x in par(0, 16):
                    for y in par(0, 16):
                        C[16*i+x, 16*j+y] += temp[x, y]

    T.add_proc(matmul_on_cpu)
    T.add_proc(matmul_c_i8)

    size1 = 512
    size2 = 512
    T.alloc_dram_2i8('x', size1, size2, '1')
    T.alloc_dram_2i8('y', size2, size1, '1')
    T.alloc_dram_2i32('z', size1, size1, f'{size2}')  # expected result
    T.alloc_dram_2i32('res', size1, size1, '0')

    T.add_body([f'matmul_c_i8(ctxt, {size1}, {size2}, {size1}, x, y, res);',
                '',
                'if(check_eq_2i32(16, 16, z, res)) {',
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

    T.compile().run()
