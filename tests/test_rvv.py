from __future__ import annotations

import itertools
import os
import platform

import pytest

from exo import proc
from exo.platforms.rvv import *
from exo.stdlib.scheduling import *
from exo.memory import MemGenError

import numpy as np


@pytest.fixture
def test_rvv_vfmul():
    """
    Compute C[i] = A[i] * B[l]
    """

    @proc
    def vfmul(
        M: size,
        C: f32[M] @ DRAM,
        A: f32[M] @ DRAM,
        B: f32[M] @ DRAM,
    ):
        # pragma: no cover
        assert M == 8
        for i in seq(0, 8):
            C[i] += A[i] * B[i]

    def simple_vfmul(p=vfmul):
        p = divide_loop(p, "i", 4, ["io", "ii"], perfect=True)
        t = "C[4 * io + ii]"
        p = stage_mem(p, "C[_] += _", t, "C_reg")
        p = expand_dim(p, "C_reg", 4, "ii")
        p = expand_dim(p, "C_reg", 2, "io")

        p = lift_alloc(p, "C_reg", n_lifts=2)
        p = autofission(p, p.find("C_reg[_] = _").after(), n_lifts=2)
        p = autofission(p, p.find("C[_] = _").before(), n_lifts=2)

        p = replace(p, "for ii in _: _ #0", rvv_vld_4xf32)
        p = replace(p, "for ii in _: _ #1", rvv_vst_4xf32)

        p = set_memory(p, "C_reg", RVV)
        for buf in ["A", "B"]:
            p = bind_expr(p, f"{buf}[_]", f"{buf}_vec")
            p = expand_dim(p, f"{buf}_vec", 4, "ii")
            p = expand_dim(p, f"{buf}_vec", 2, "io")
            p = lift_alloc(p, f"{buf}_vec", n_lifts=2)
            p = autofission(p, p.find(f"{buf}_vec[_] = _").after(), n_lifts=2)
            p = replace(p, "for ii in _: _ #0", rvv_vld_4xf32)
            p = set_memory(p, f"{buf}_vec", RVV)

        p = replace(p, "for ii in _: _ #0", rvv_vfmacc_4xf32_4xf32)
        return p

    simple_rvv_vfmul = simple_vfmul()

    return simplify(simple_rvv_vfmul)


@pytest.mark.isa("rvv")
def test_gen_rvv_vfmul(golden, test_rvv_vfmul):
    assert str(test_rvv_vfmul) == golden
