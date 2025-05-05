from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.sve_vla import *
from exo.stdlib.scheduling import *


@pytest.fixture
def test_sve_vla_svmla():
    @proc
    def svmla(
        N: size,
        C: f32[N] @ DRAM,
        A: f32[N] @ DRAM,
        B: f32,
    ):
        for i in seq(0, N):
            C[i] += A[i] * B

    def simple_svmla(p=svmla):
        p = replace_all(p, svmla_n_f32_x_vla)
        return p

    simple_sve_vla_svmla = simple_svmla()

    return simplify(simple_sve_vla_svmla)


def test_gen_sve_vla_svmla(golden, test_sve_vla_svmla):
    assert str(test_sve_vla_svmla) == golden
