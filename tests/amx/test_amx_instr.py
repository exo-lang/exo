from __future__ import annotations

import pytest
import platform
# if platform.system() == 'Darwin':
#     pytest.skip("skipping x86 tests on Apple machines for now",
#                 allow_module_level=True)

import sys
from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import AMX_TILE
from tests.gemmini.matmul.test_gemmini_matmul_paper import matmul
from .amx import *
from .harness_amx import ENV, AMXTestBuilder


# --------------------------------------------------------------------------- #
#   Individual Load / Store / Zero Tests
# --------------------------------------------------------------------------- #

@pytest.mark.skip()
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


@pytest.mark.skip()
def test_dpbssd_i8_16x64():
    T = AMXTestBuilder('dpbssd_i8_16x64')
    T.add_body(["dpbssd_i8_16x64_lib_Context *ctxt;"])

    @proc
    def dpbssd_i8_16x64(x: i8[16, 64] @ DRAM, y: i8[16, 64] @ DRAM, z: i32[16, 16] @ DRAM):
        config()
        tile0: i8[16, 64] @ AMX_TILE
        tile1: i8[16, 64] @ AMX_TILE
        tile2: i32[16, 16] @ AMX_TILE
        ld_i8(16, 64, x, tile0)
        ld_i8(16, 64, y, tile1)
        dpbssd(16, 16, 16, tile0, tile1, tile2)
        st_i32(16, 16, tile2, z)

    T.add_proc(dpbssd_i8_16x64)

    T.alloc_dram_2i8('x', 16, 64, '1')
    T.alloc_dram_2i8('y', 16, 64, '1')
    T.alloc_dram_2i32('z', 16, 16, '64')  # expected result
    T.alloc_dram_2i32('res', 16, 16, '0')

    T.add_body(['dpbssd_i8_16x64(ctxt, x, y, res);',
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


@pytest.mark.skip()
def test_transform_memory():
    T = AMXTestBuilder('transform_memory')
    T.add_body(["transform_memory_lib_Context *ctxt;"])

    # TODO: make this transform_memory same as the one below
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


def matmul_algorithm_i8():
    @proc
    def matmul_on_cpu_i8(
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
                    # Casts the i8s to i32s before multiplying (note that i8 is signed).
                    a: i32
                    b: i32

                    a = A[i, k]
                    b = B[k, j]

                    C[i, j] += a * b

    return matmul_on_cpu_i8

def modified_matmul_algorithm_i8():
    @proc
    def modified_matmul_on_cpu_i8(
        M: size,
        K: size,
        N: size,
        A: i8[M, 4*K] @ DRAM,
        B: i8[K, 4*N] @ DRAM,
        C: i32[M, N] @ DRAM,
    ):
        assert K%4 == 0
        for i in par(0, M):
            for j in par(0, N):
                # C_tile: i32
                # C_tile = 0.0
                C[i, j] = 0.0
                for k in par(0, K):
                    for byte in par(0, 4):
                        a: i32
                        b: i32

                        a = A[i, 4*k+byte]
                        b = B[k, 4*j+byte]

                        C[i, j] += a * b
                        # C_tile += a * b
                # C[i, j] = C_tile

    return modified_matmul_on_cpu_i8

def get_transform_memory_i8():
    @proc
    def transform_memory_i8(
        m: size,
        n: size,
        src: i8[m, n] @ DRAM,
        dest: i8[m/4, 4*n] @ DRAM,
    ):
        assert m % 4 == 0
        for i in par(0, m/4):
            for j in par(0, n):
                for k in par(0, 4):
                    dest[i, 4*j+k] = src[4*i + k, j]
    return transform_memory_i8

@pytest.mark.skip()
def test_matmul_on_amx_by_hand_i8():
    size1 = 256
    size2 = 256

    T = AMXTestBuilder('matmul_on_amx_by_hand_i8')
    T.add_body(["matmul_on_amx_by_hand_i8_lib_Context *ctxt;"])

    @proc
    def matmul_on_amx_i8(
        M: size,
        K: size,
        N: size,
        A: i8[M, 4*K] @ DRAM,
        B: i8[K, 4*N] @ DRAM,
        C: i32[M, N] @ DRAM,
    ):
        assert M % 16 == 0
        assert N % 16 == 0
        assert K % 16 == 0
        config()
        for i in par(0, M/16):
            for j in par(0, N/16):
                tileC: i32[16, 16] @ AMX_TILE
                zero_i32(16, 16, tileC)  # bind_expr, expand_dim
                # use partial_eval to work with constants

                for k in par(0, K/16):
                    tileA: i8[16, 64] @ AMX_TILE
                    tileB: i8[16, 64] @ AMX_TILE
                    ld_i8(16, 64, A[16*i:16*(i+1), 64*k:64*(k+1)], tileA)
                    ld_i8(16, 64, B[16*k:16*(k+1), 64*j:64*(j+1)], tileB)
                    dpbssd(16, 16, 16, tileA, tileB, tileC)

                st_i32(16, 16, tileC, C[16*i:16*(i+1), 16*j:16*(j+1)])

    cpu = matmul_algorithm_i8()
    transform_memory = get_transform_memory_i8()
    T.add_proc(cpu.rename("matmul_on_cpu"))
    T.add_proc(transform_memory.rename("transform_memory"))
    T.add_proc(matmul_on_amx_i8.rename("matmul_on_amx"))

    T.alloc_dram_2i8('x', size1, size2, 'i+j')
    T.alloc_dram_2i8('y_orig', size2, size1, 'j')  # before transform_memory
    T.alloc_dram_2i8('y', size2//4, 4*size1, '0')  # after transform_memory
    T.alloc_dram_2i32('z', size1, size1, '0')  # expected result
    T.alloc_dram_2i32('res', size1, size1, '0')

    T.add_body([f'transform_memory(ctxt, {size2}, {size1}, y_orig, y);'])
    T.add_body(
        [f'matmul_on_cpu(ctxt, {size1}, {size2}, {size1}, x, y_orig, z);'])
    T.add_body(
        [f'matmul_on_amx(ctxt, {size1}, {size2//4}, {size1}, x, y, res);'])

    T.add_body([f'if(check_eq_2i32({size1}, {size1}, z, res)) {{',
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


def test_matmul_on_amx_scheduled_i8():
    size1 = 512
    size2 = 512

    T = AMXTestBuilder('matmul_on_amx_scheduled_i8')
    T.add_body(["matmul_on_amx_scheduled_i8_lib_Context *ctxt;"])

    cpu = matmul_algorithm_i8()
    transform_memory = get_transform_memory_i8()

    amx = modified_matmul_algorithm_i8().partial_eval(size1, size2//4, size1)

    print("Base Implementation: ")
    print(amx)

    amx = amx.split('i', 16, ['io', 'ii'], perfect=True)
    amx = amx.split('j', 16, ['jo', 'ji'], perfect=True)
    amx = amx.reorder('ii', 'jo')

    print("After loop splitting and reordering:")
    print(amx)

    # Attempt to use an AMX tile to aggregate results for C
    amx = amx.stage_assn('C_tile', 'C[_] = 0.0')
    amx = amx.stage_assn('C_tile_2', 'C[_] += _')
    amx = amx.data_reuse('C_tile:_', 'C_tile_2:_')

    # amx = amx.fission_after('C[_] = 0.0', n_lifts=2)
    # amx = amx.split('k', 16, ['ko', 'ki'], perfect=True)
    # amx = amx.reorder('ji', 'ko')
    # amx = amx.reorder('ii', 'ko')

    # amx = amx.lift_alloc('a:_', n_lifts=4)
    # amx = amx.lift_alloc('b:_', n_lifts=4)

    # par_to_seq hack deals with allocating too much memory
    # amx = amx.par_to_seq('for ki in _:_')
    # amx = amx.par_to_seq('for byte in _:_')

    # amx = amx.stage_assn('C_tile', 'C[_] += _')
    # amx = amx.lift_alloc('C_tile:_', n_lifts=4)
    # amx = amx.set_memory('C_tile',  AMX_TILE)
    # amx = amx.set_precision('C_tile', 'i32')

    T.add_proc(cpu.rename("matmul_on_cpu"))
    T.add_proc(transform_memory.rename("transform_memory"))
    T.add_proc(amx.rename("matmul_on_amx"))

    print("Post scheduling: ")
    print(amx)

    T.alloc_dram_2i8('x', size1, size2, 'i+j')
    T.alloc_dram_2i8('y_orig', size2, size1, 'j')  # before transform_memory
    T.alloc_dram_2i8('y', size2//4, 4*size1, '0')  # after transform_memory
    T.alloc_dram_2i32('z', size1, size1, '0')  # expected result
    T.alloc_dram_2i32('res', size1, size1, '0')

    T.add_body([f'transform_memory(ctxt, {size2}, {size1}, y_orig, y);'])
    T.add_body(
        [f'matmul_on_cpu(ctxt, {size1}, {size2}, {size1}, x, y_orig, z);'])
    T.add_body(
        [f'matmul_on_amx(ctxt, {size1}, {size2//4}, {size1}, x, y, res);'])

    T.add_body([f'if(check_eq_2i32({size1}, {size1}, z, res)) {{',
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