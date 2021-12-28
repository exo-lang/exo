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
    T.alloc_dram_2i32('z', 16, 16, '64') # expected result
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
