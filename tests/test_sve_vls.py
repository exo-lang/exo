from __future__ import annotations

import pytest

from exo import proc, compile_procs_to_strings
from exo.platforms.sve_vls import *
from exo.stdlib.scheduling import *

SVE_VLS = SVE_VLS(512)


@pytest.fixture
def test_sve_vls_svmla():
    @proc
    def svmla(
        C: f32[16] @ DRAM,
        A: f32[16] @ DRAM,
        B: f32,
    ):
        for i in seq(0, 16):
            C[i] += A[i] * B

    def simple_svmla(p=svmla):
        p = stage_mem(p, "for i in _:_", "C[0:16]", "C_reg")
        p = set_memory(p, "C_reg:_", SVE_VLS.Vector)
        p = stage_mem(p, "for i in _:_", "A[0:16]", "A_reg")
        p = set_memory(p, "A_reg:_", SVE_VLS.Vector)
        p = replace_all(p, SVE_VLS.svld1_f32)
        p = replace_all(p, SVE_VLS.svst1_f32)
        p = replace_all(p, SVE_VLS.svmla_n_f32_x)
        return p

    simple_sve_vls_svmla = simple_svmla()

    return simplify(simple_sve_vls_svmla)


def test_gen_sve_vls_svmla(golden, test_sve_vls_svmla):
    assert str(test_sve_vls_svmla) == golden


def test_compile_sve_vls_svmla(golden, test_sve_vls_svmla):
    c_file, _ = compile_procs_to_strings([test_sve_vls_svmla], "test.h")

    assert c_file == golden
