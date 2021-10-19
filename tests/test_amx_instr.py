from __future__ import annotations

import sys
from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import AMX_TILE
from .amx import *
from .harness_amx import ENV, AMXTestBuilder
import pytest


# --------------------------------------------------------------------------- #
#   Individual Load / Store / Zero Tests
# --------------------------------------------------------------------------- #

def test_ldst_i8_16():
    T = AMXTestBuilder('ldst_i8_16')
    T.add_body(["ldst_i8_16_lib_Context *ctxt;"])

    @proc
    def ldst_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
        config(16, 16)
        tmp : i8[16,16] @ AMX_TILE
        ld_i8(16,16, x, tmp)
        st_i8(16,16, tmp, y)

    T.add_proc(ldst_i8_16)

    T.alloc_dram_2i8('x', 16, 16, 'i+j')
    T.alloc_dram_2i8('y', 16, 16, '0')

    T.add_body(['ldst_i8_16(ctxt, x, y);',
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


