from __future__ import annotations

import os

from SYS_ATL import proc
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
        for i in par(0, m):
            for j in par(0, n):
                for k in par(0, 4):
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
        for i in par(0, M):
            for j in par(0, N):
                C[i, j] = 0.0
                for k in par(0, K):
                    # Casts the i8s to i32s before multiplying
                    # (note that i8 is signed).
                    a: i32
                    b: i32

                    a = A[i, k]
                    b = B[k, j]

                    C[i, j] += a * b

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
        config()  # TODO: how to insert this via SYS_ATL
        for i in par(0, M):
            for j in par(0, N):
                # C[i, j] = 0.0
                # for k in par(0, K):
                #     for byte in par(0, 4):
                #         a: i32
                #         b: i32

                #         a = A[i, 4*k+byte]
                #         b = B[k, 4*j+byte]

                #         C[i, j] += a * b

                C_tile: i32
                C_tile = 0.0
                for k in par(0, K):
                    for byte in par(0, 4):
                        a: i32
                        b: i32

                        a = A[i, 4 * k + byte]
                        b = B[k, 4 * j + byte]

                        C_tile += a * b
                C[i, j] = C_tile

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
        for i in par(0, m / 4):
            for j in par(0, n):
                for k in par(0, 4):
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
        assert M % 16 == 0
        assert N % 16 == 0
        assert K % 16 == 0
        config()
        for i in par(0, M / 16):
            for j in par(0, N / 16):
                tileC: i32[16, 16] @ AMX_TILE
                zero_i32(16, 16, tileC)

                for k in par(0, K / 16):
                    tileA: i8[16, 64] @ AMX_TILE
                    tileB: i8[16, 64] @ AMX_TILE
                    ld_i8(16, 64, A[16 * i:16 * (i + 1), 64 * k:64 * (k + 1)],
                          tileA)
                    ld_i8(16, 64, B[16 * k:16 * (k + 1), 64 * j:64 * (j + 1)],
                          tileB)
                    dpbssd(16, 16, 16, tileA, tileB, tileC)

                st_i32(16, 16, tileC,
                       C[16 * i:16 * (i + 1), 16 * j:16 * (j + 1)])

    size1 = 256
    size2 = 256

    t = AMXTestBuilder(compiler.basename)
    t.add_body([f"{compiler.basename}_Context *ctxt;"])

    t.alloc_dram_2i8('x', size1, size2, 'i+j')
    t.alloc_dram_2i8('y_orig', size2, size1, 'j')  # before transform_memory
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


def test_matmul_on_amx_scheduled_i8(compiler, sde64):
    size1 = 512
    size2 = 512

    amx = modified_matmul_algorithm_i8().partial_eval(size1, size2 // 4, size1)

    print("Base Implementation: ")
    print(amx)
    amx = amx.set_memory('a', AMX_TILE)
    amx = amx.set_precision('a', 'i8')
    amx = amx.set_memory('b', AMX_TILE)
    amx = amx.set_precision('b', 'i8')
    amx = amx.set_memory('C_tile', AMX_TILE)

    print("Loop splitting and reordering:")
    amx = amx.split('i', 16, ['io', 'ii'], perfect=True)
    amx = amx.split('j', 16, ['jo', 'ji'], perfect=True)
    amx = amx.reorder('ii', 'jo')
    print(amx)

    print("Introducing tiles:")
    amx = amx.lift_alloc('C_tile:_', n_lifts=2)
    amx = amx.fission_after('C_tile[_] = 0.0', n_lifts=2)
    amx = amx.fission_after('for k in _:_', n_lifts=2)
    amx = amx.split('k', 16, ['ko', 'ki'], perfect=True)
    amx = amx.reorder('ji', 'ko')
    amx = amx.reorder('ii', 'ko')
    amx = amx.lift_alloc('a:_', n_lifts=4)
    amx = amx.reorder('ji', 'ki')
    amx = amx.lift_alloc('b:_', n_lifts=4)
    amx = amx.fission_after('a[_] = A[_]', n_lifts=4)
    amx = amx.fission_after('b[_] = B[_]', n_lifts=4)
    print(amx)

    print("Matching the implementation of DPBSSD:")
    amx = amx.reorder('ki #1', 'ji')
    amx = amx.bind_expr('A_temp', 'a[_] #0')
    amx = amx.bind_expr('B_temp', 'b[_] #0')
    amx = amx.set_memory('A_temp', DRAM)
    amx = amx.set_memory('B_temp', DRAM)
    amx = amx.set_precision('A_temp', 'i32')
    amx = amx.set_precision('B_temp', 'i32')
    amx = amx.reorder_before('B_temp:_')
    print(amx)

    print("Replacing with AMX memory commands:")
    amx = amx.replace(zero_i32, "for ii in _: _ #0")
    amx = amx.replace(ld_i8_3d, "for ii in _: _ #0")
    amx = amx.replace(ld_i8_3d, "for ki in _: _ #0")
    amx = amx.replace(dpbssd_3d, 'for ii in _: _ #0')
    amx = amx.replace(st_i32, "for ii in _: _ #0")
    print(amx)

    t = AMXTestBuilder(compiler.basename)
    t.add_body([f"{compiler.basename}_Context *ctxt;"])

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
        matmul_algorithm_i8(), get_transform_memory_i8(), amx,
    ], t)


def _run_amx(compiler, sde64, procs, test_source):
    test_exe = compiler.compile(
        procs,
        test_files={'main.c': str(test_source)},
        CMAKE_C_FLAGS='-mamx-int8 -mamx-tile',
        CMAKE_C_COMPILER=os.getenv('CLANG', 'clang-13')
    )

    sde64(test_exe)
