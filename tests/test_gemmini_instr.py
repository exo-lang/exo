from __future__ import annotations

from .gemmini import *
from .harness_gemmini import GemmTestBuilder


# --------------------------------------------------------------------------- #
#   Individual Load / Store / Zero Tests
# --------------------------------------------------------------------------- #

def test_ldst_i8_16():
    T = GemmTestBuilder('ldst_i8_16')
    T.add_body(['gemm_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["ldst_i8_16_lib_Context *ctxt;"])

    @proc
    def ldst_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
        tmp : i8[16,16] @ GEMM_SCRATCH
        scale : f32
        scale = 1.0
        ld_i8(16,16, scale, x, tmp)
        st_i8(16,16, tmp, y)
    T.add_proc(ldst_i8_16)

    T.alloc_dram_2i8('x', 16, 16, 'i+j')
    T.alloc_dram_2i8('y', 16, 16, '0')

    T.add_body(['ldst_i8_16(ctxt, x, y);',
                '',
                'gemmini_fence();',
                '',
                'if(check_eq_2i8(16,16, x, y)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (x):\\n");',
                '    print_2i8(16,16, x);',
                '    printf("Computed Roundtrip (y):\\n");',
                '    print_2i8(16,16, y);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()


def test_ldst_acc_i8_16():
    T = GemmTestBuilder('ldst_acc_i8_16')
    T.add_body(['gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["ldst_acc_i8_16_lib_Context *ctxt;"])

    @proc
    def ldst_acc_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
        tmp : i32[16,16] @ GEMM_ACCUM
        scale : f32
        scale = 1.0
        ld_acc_i8(16,16, scale, x, tmp)
        st_acc_i8(16,16, scale, False, tmp, y)
    T.add_proc(ldst_acc_i8_16)

    T.alloc_dram_2i8('x', 16, 16, 'i+j')
    T.alloc_dram_2i8('y', 16, 16, '0')

    T.add_body(['ldst_acc_i8_16(ctxt, x, y);',
                '',
                'gemmini_fence();',
                '',
                'if(check_eq_2i8(16,16, x, y)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (x):\\n");',
                '    print_2i8(16,16, x);',
                '    printf("Computed Roundtrip (y):\\n");',
                '    print_2i8(16,16, y);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()


def test_ldst_i8_odd():
    T = GemmTestBuilder('ldst_i8_odd')
    T.add_body(['gemm_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["ldst_i8_odd_lib_Context *ctxt;"])

    @proc
    def ldst_i8_odd( x : i8[15,7] @ DRAM, y : i8[15,7] @ DRAM ):
        tmp : i8[15,16] @ GEMM_SCRATCH
        scale : f32
        scale = 1.0
        ld_i8(15,7, scale, x, tmp)
        st_i8(15,7, tmp, y)
    T.add_proc(ldst_i8_odd)

    T.alloc_dram_2i8('x', 15, 7, 'i+j')
    T.alloc_dram_2i8('y', 15, 7, '0')

    T.add_body(['ldst_i8_odd(ctxt, x, y);',
                '',
                'gemmini_fence();',
                '',
                'if(check_eq_2i8(15,7, x, y)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (x):\\n");',
                '    print_2i8(15,7, x);',
                '    printf("Computed Roundtrip (y):\\n");',
                '    print_2i8(15,7, y);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()


def test_ldst_acc_i8_acc():
    T = GemmTestBuilder('ldst_acc_i8_acc')
    T.add_body(['gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["ldst_acc_i8_acc_lib_Context *ctxt;"])

    @proc
    def ldst_acc_i8_acc( x : i8[7,13] @ DRAM, y : i8[7,13] @ DRAM ):
        tmp : i32[7,16] @ GEMM_ACCUM
        scale : f32
        scale = 1.0
        ld_acc_i8(7,13, scale, x, tmp)
        st_acc_i8(7,13, scale, False, tmp, y)
    T.add_proc(ldst_acc_i8_acc)

    T.alloc_dram_2i8('x', 7, 13, 'i+j')
    T.alloc_dram_2i8('y', 7, 13, '0')

    T.add_body(['ldst_acc_i8_acc(ctxt, x, y);',
                '',
                'gemmini_fence();',
                '',
                'if(check_eq_2i8(7,13, x, y)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (x):\\n");',
                '    print_2i8(7,13, x);',
                '    printf("Computed Roundtrip (y):\\n");',
                '    print_2i8(7,13, y);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()


def test_ldzerost_i8_16():
    T = GemmTestBuilder('ldzerost_i8_16')
    T.add_body(['gemm_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["ldzerost_i8_16_lib_Context *ctxt;"])

    @proc
    def ldzerost_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
        tmp : i8[16,16] @ GEMM_SCRATCH
        scale : f32
        scale = 1.0
        ld_i8(16,16, scale, x, tmp)
        zero_i8(8,8, tmp[4:12,:])
        st_i8(16,16, tmp, y)
    T.add_proc(ldzerost_i8_16)

    T.alloc_dram_2i8('x', 16, 16, 'i+j')
    T.alloc_dram_2i8('y', 16, 16, '0')

    T.add_body(['ldzerost_i8_16(ctxt, x, y);',
                '',
                'gemmini_fence();',
                '',
                '// zero out the same region of x',
                'for(int i=4; i<12; i++) {',
                '    for(int j=0; j<8; j++) {'
                '        x[i*16 + j] = 0;'
                '}}',
                '',
                'if(check_eq_2i8(16,16, x, y)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (x):\\n");',
                '    print_2i8(16,16, x);',
                '    printf("Computed Roundtrip (y):\\n");',
                '    print_2i8(16,16, y);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()


def test_ldzerost_acc_i8_16():
    T = GemmTestBuilder('ldzerost_acc_i8_16')
    T.add_body(['gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["ldzerost_acc_i8_16_lib_Context *ctxt;"])

    @proc
    def ldzerost_acc_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
        tmp : i32[16,16] @ GEMM_ACCUM
        scale : f32
        scale = 1.0
        ld_acc_i8(16,16, scale, x, tmp)
        zero_acc_i32(8,8, tmp[4:12,:])
        st_acc_i8(16,16, scale, False, tmp, y)
    T.add_proc(ldzerost_acc_i8_16)

    T.alloc_dram_2i8('x', 16, 16, 'i+j')
    T.alloc_dram_2i8('y', 16, 16, '0')

    T.add_body(['ldzerost_acc_i8_16(ctxt, x, y);',
                '',
                'gemmini_fence();',
                '',
                '// zero out the same region of x',
                'for(int i=4; i<12; i++) {',
                '    for(int j=0; j<8; j++) {'
                '        x[i*16 + j] = 0;'
                '}}',
                '',
                'if(check_eq_2i8(16,16, x, y)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (x):\\n");',
                '    print_2i8(16,16, x);',
                '    printf("Computed Roundtrip (y):\\n");',
                '    print_2i8(16,16, y);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()


# --------------------------------------------------------------------------- #
#   Individual MatMul Tests
# --------------------------------------------------------------------------- #



def test_matmul_i8_ones_16():
    T = GemmTestBuilder('matmul_i8_ones_16')
    T.add_body(['gemm_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_i8_ones_16_lib_Context *ctxt;"])


    T.alloc_dram_2i8('x', 16, 16, '1')
    T.alloc_dram_2i8('y', 16, 16, '1')
    T.alloc_dram_2i8('z', 16, 16, '16') # expected result
    T.alloc_dram_2i8('res', 16, 16, '0')

    @proc
    def matmul_i8_ones_16(
      x : i8[16,16] @ DRAM,
      y : i8[16,16] @ DRAM,
      res : i8[16,16] @ DRAM,
    ):
        A : i8[16,16] @ GEMM_SCRATCH
        B : i8[16,16] @ GEMM_SCRATCH
        C : i32[16,16] @ GEMM_ACCUM
        scale : f32
        scale = 1.0
        ld_i8(16,16, scale, x, A)
        ld_i8(16,16, scale, y, B)
        zero_acc_i32(16,16, C)

        matmul_i8(16,16,16, A, B, C)

        st_acc_i8(16,16, scale, False, C, res)

    T.add_proc(matmul_i8_ones_16)


    T.add_body(['matmul_i8_ones_16(ctxt, x, y, res);',
                '',
                'gemmini_fence();',
                '',
                'if(check_eq_2i8(16,16, z, res)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (res):\\n");',
                '    print_2i8(16,16, res);',
                '    printf("Computed Roundtrip (z):\\n");',
                '    print_2i8(16,16, z);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()


def test_matmul_i8_ones_odd():
    T = GemmTestBuilder('matmul_i8_ones_odd')
    T.add_body(['gemm_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["malloc_i8_ones_odd_lib_Context *ctxt;"])


    # 15 x 9 x 13
    T.alloc_dram_2i8('x', 15, 9, '1')
    T.alloc_dram_2i8('y', 9, 13, '1')
    T.alloc_dram_2i8('z', 15, 13, '9') # expected result
    T.alloc_dram_2i8('res', 15, 13, '0')

    @proc
    def matmul_i8_ones_odd(
      x : i8[15,9] @ DRAM,
      y : i8[9,13] @ DRAM,
      res : i8[15,13] @ DRAM,
    ):
        A : i8[15,16] @ GEMM_SCRATCH
        B : i8[9,16] @ GEMM_SCRATCH
        C : i32[15,16] @ GEMM_ACCUM
        scale : f32
        scale = 1.0
        ld_i8(15,9, scale, x, A)
        ld_i8(9,13, scale, y, B)
        zero_acc_i32(15,13, C)

        matmul_i8(15,13,9, A, B, C)

        st_acc_i8(15,13, scale, False, C, res)
    T.add_proc(matmul_i8_ones_odd)


    T.add_body(['matmul_i8_ones_odd(ctxt, x, y, res);',
                '',
                'gemmini_fence();',
                '',
                'if(check_eq_2i8(15,13, z, res)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (res):\\n");',
                '    print_2i8(15,13, res);',
                '    printf("Computed Roundtrip (z):\\n");',
                '    print_2i8(15,13, z);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()

def test_ldst_acc_i32_15():
    T = GemmTestBuilder('ldst_acc_i32_15')
    T.add_body(['gemm_acc_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["ldst_acc_i32_15_lib_Context *ctxt;"])

    @proc
    def ldst_acc_i32_15( x : i32[15,15] @ DRAM, y : i32[15,15] @ DRAM ):
        tmp : i32[15,16] @ GEMM_ACCUM
        scale : f32
        scale = 4.0
        ld_acc_i32(15,15, scale, x, tmp)
        st_acc_i32(15,15, tmp, y)
    T.add_proc(ldst_acc_i32_15)

    T.alloc_dram_2i32('x', 15, 15, '1')
    T.alloc_dram_2i32('y', 15, 15, '0')
    T.alloc_dram_2i32('res', 15, 15, '4')

    T.add_body(['ldst_acc_i32_15(ctxt, x, y);',
                '',
                'gemmini_fence();',
                '',
                'if(check_eq_2i32(15,15, y, res)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (res):\\n");',
                '    print_2i32(15,15, res);',
                '    printf("Computed Roundtrip (y):\\n");',
                '    print_2i32(15,15, y);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()



def test_matmul_i8_ones_odd():
    T = GemmTestBuilder('matmul_i8_ones_odd')
    T.add_body(['gemm_init_mem();',
                'gemmini_flush(0);',
                ''])
    T.add_body(["matmul_i8_ones_odd_lib_Context *ctxt;"])


    # 15 x 9 x 13
    T.alloc_dram_2i8('x', 15, 9, '1')
    T.alloc_dram_2i8('y', 9, 13, '1')
    T.alloc_dram_2i8('z', 15, 13, '9') # expected result
    T.alloc_dram_2i8('res', 15, 13, '0')

    @proc
    def matmul_i8_ones_odd(
      x : i8[15,9] @ DRAM,
      y : i8[9,13] @ DRAM,
      res : i8[15,13] @ DRAM,
    ):
        A : i8[15,16] @ GEMM_SCRATCH
        B : i8[9,16] @ GEMM_SCRATCH
        C : i32[15,16] @ GEMM_ACCUM
        scale : f32
        scale = 1.0
        ld_i8(15,9, scale, x, A)
        ld_i8(9,13, scale, y, B)
        zero_acc_i32(15,13, C)

        matmul_i8(15,13,9, A, B, C)

        st_acc_i8(15,13, scale, False, C, res)
    T.add_proc(matmul_i8_ones_odd)


    T.add_body(['matmul_i8_ones_odd(ctxt, x, y, res);',
                '',
                'gemmini_fence();',
                '',
                'if(check_eq_2i8(15,13, z, res)) {',
                '    printf("Correct\\n");',
                '} else {',
                '    printf("Results Don\'t Match\\n");',
                '    printf("Correct Result (res):\\n");',
                '    print_2i8(15,13, res);',
                '    printf("Computed Roundtrip (z):\\n");',
                '    print_2i8(15,13, z);',
                '    exit(1);',
                '}',
                ''])

    T.compile().run()
