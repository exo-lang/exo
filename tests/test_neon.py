from __future__ import annotations

import itertools
import os
import platform

import pytest

from exo import proc
from exo.platforms.neon import *
from exo.stdlib.scheduling import *
from exo.memory import MemGenError

import numpy as np


def test_neon_can_read():
    @proc
    def read_neon(n: size, dst: R[n] @ DRAM, src: R[n] @ Neon):
        for i in seq(0, n):
            dst[i] = src[i]

    with pytest.raises(MemGenError, match="cannot read"):
        read_neon.c_code_str()


@pytest.mark.isa("neon")
def test_neon_memcpy(compiler):
    """
    Compute dst = src
    """

    @proc
    def memcpy_neon(n: size, dst: R[n] @ DRAM, src: R[n] @ DRAM):  # pragma: no cover
        for i in seq(0, (n + 3) / 4):
            if n - 4 * i >= 4:
                tmp: f32[4] @ Neon
                neon_vld_4xf32(tmp, src[4 * i : 4 * i + 4])
                neon_vst_4xf32(dst[4 * i : 4 * i + 4], tmp)
            else:
                for j in seq(0, n - 4 * i):
                    dst[4 * i + j] = src[4 * i + j]

    fn = compiler.compile(
        memcpy_neon, skip_on_fail=True, CMAKE_C_FLAGS="-mcpu=apple-a14"
    )

    for n in (7, 8, 9, 31, 32, 33, 127, 128, 129):
        inp = np.array([float(i) for i in range(n)], dtype=np.float32)
        out = np.array([float(0) for _ in range(n)], dtype=np.float32)
        fn(None, n, out, inp)

        assert np.array_equal(inp, out)


@pytest.mark.isa("neon")
def test_neon_simple_math(compiler):
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_neon(n: size, x: R[n] @ DRAM, y: R[n] @ DRAM):  # pragma: no cover
        assert n % 4 == 0
        for i in seq(0, n / 4):
            xVec1: f32[4] @ Neon
            xVec2: f32[4] @ Neon
            yVec: f32[4] @ Neon
            neon_vld_4xf32(xVec2, x[4 * i : 4 * i + 4])
            neon_vld_4xf32(yVec, y[4 * i : 4 * i + 4])
            neon_vmul_4xf32(xVec1, xVec2, yVec)
            neon_vmul_4xf32(xVec2, xVec1, yVec)
            neon_vst_4xf32(x[4 * i : 4 * i + 4], xVec2)

    fn = compiler.compile(
        simple_math_neon, skip_on_fail=True, CMAKE_C_FLAGS="-mcpu=apple-a14"
    )

    for n in (4, 8, 12, 16, 24, 32, 64, 128):
        x = np.array([float(i) for i in range(n)], dtype=np.float32)
        y = np.array([float(3 * i) for i in range(n)], dtype=np.float32)
        expected = x * y * y

        fn(None, n, x, y)
        assert np.allclose(x, expected)


@pytest.fixture
def test_neon_vfmla():
    """
    Compute C[i] = A[i] * B[l]
    """

    @proc
    def vfmla(
        n: size, C: R[n] @ DRAM, A: R[n] @ DRAM, B: R[n] @ DRAM
    ):  # pragma: no cover
        assert n == 4
        for l in seq(0, 4):
            for i in seq(0, 4):
                C[i] += A[i] * B[l]

    def simple_vfmla(p=vfmla):
        p = stage_mem(p, "C[_] += _", "C[i]", "C_reg")
        p = expand_dim(p, "C_reg", 4, "i")
        p = lift_alloc(p, "C_reg", n_lifts=2)
        p = autofission(p, p.find("C_reg[_] = _").after(), n_lifts=2)
        p = autofission(p, p.find("C[_] = _").before(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_4xf32)
        p = replace(p, "for i in _: _ #1", neon_vst_4xf32)
        p = set_memory(p, "C_reg", Neon)

        p = bind_expr(p, "A[_]", "A_vec")
        p = expand_dim(p, "A_vec", 4, "i")
        p = lift_alloc(p, "A_vec", n_lifts=2)
        p = autofission(p, p.find("A_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_4xf32)
        p = set_memory(p, "A_vec", Neon)

        p = bind_expr(p, "B[_]", "B_vec")
        p = expand_dim(p, "B_vec", 4, "l")
        p = lift_alloc(p, "B_vec", n_lifts=2)
        p = autofission(p, p.find("B_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for l in _: _ #0", neon_vld_4xf32)
        p = set_memory(p, "B_vec", Neon)

        p = replace(p, "for i in _: _ #0", neon_vfmla_4xf32_4xf32)
        p = unroll_loop(p, "l #0")
        return p

    simple_neon_vfmla = simple_vfmla()

    return simplify(simple_neon_vfmla)


@pytest.mark.isa("neon")
def test_gen_neon_vfmla(golden, test_neon_vfmla):
    assert str(test_neon_vfmla) == golden


@pytest.fixture
def test_neon_vfmla_f16():
    """
    Compute C[i] = A[i] * B[l]
    """

    @proc
    def vfmla_f16(
        n: size, C: R[n] @ DRAM, A: R[n] @ DRAM, B: R[n] @ DRAM
    ):  # pragma: no cover
        assert n == 8
        for l in seq(0, 8):
            for i in seq(0, 8):
                C[i] += A[i] * B[l]

    def simple_vfmla_f16(p=vfmla_f16):
        p = stage_mem(p, "C[_] += _", "C[i]", "C_reg")
        p = expand_dim(p, "C_reg", 8, "i")
        p = lift_alloc(p, "C_reg", n_lifts=2)
        p = autofission(p, p.find("C_reg[_] = _").after(), n_lifts=2)
        p = autofission(p, p.find("C[_] = _").before(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_8xf16)
        p = replace(p, "for i in _: _ #1", neon_vst_8xf16)
        p = set_memory(p, "C_reg", Neon)

        p = bind_expr(p, "A[_]", "A_vec")
        p = expand_dim(p, "A_vec", 8, "i")
        p = lift_alloc(p, "A_vec", n_lifts=2)
        p = autofission(p, p.find("A_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_8xf16)
        p = set_memory(p, "A_vec", Neon)

        p = bind_expr(p, "B[_]", "B_vec")
        p = expand_dim(p, "B_vec", 8, "l")
        p = lift_alloc(p, "B_vec", n_lifts=2)
        p = autofission(p, p.find("B_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for l in _: _ #0", neon_vld_8xf16)
        p = set_memory(p, "B_vec", Neon)

        p = replace(p, "for i in _: _ #0", neon_vfmla_8xf16_8xf16)
        p = unroll_loop(p, "l #0")
        return p

    simple_neon_vfmla_f16 = simple_vfmla_f16()

    return simplify(simple_neon_vfmla_f16)


@pytest.mark.isa("neon")
def test_gen_neon_vfmla_f16(golden, test_neon_vfmla_f16):
    assert str(test_neon_vfmla_f16) == golden


@pytest.fixture
def test_neon_vfmla_f16():
    """
    Compute C[i] = A[i] * B[l]
    """

    @proc
    def vfmla_f16(
        n: size, C: R[n] @ DRAM, A: R[n] @ DRAM, B: R[n] @ DRAM
    ):  # pragma: no cover
        assert n == 8
        for l in seq(0, 8):
            for i in seq(0, 8):
                C[i] += A[i] * B[l]

    def simple_vfmla_f16(p=vfmla_f16):
        p = stage_mem(p, "C[_] += _", "C[i]", "C_reg")
        p = expand_dim(p, "C_reg", 8, "i")
        p = lift_alloc(p, "C_reg", n_lifts=2)
        p = autofission(p, p.find("C_reg[_] = _").after(), n_lifts=2)
        p = autofission(p, p.find("C[_] = _").before(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_8xf16)
        p = replace(p, "for i in _: _ #1", neon_vst_8xf16)
        p = set_memory(p, "C_reg", Neon)

        p = bind_expr(p, "A[_]", "A_vec")
        p = expand_dim(p, "A_vec", 8, "i")
        p = lift_alloc(p, "A_vec", n_lifts=2)
        p = autofission(p, p.find("A_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_8xf16)
        p = set_memory(p, "A_vec", Neon)

        p = bind_expr(p, "B[_]", "B_vec")
        p = expand_dim(p, "B_vec", 8, "l")
        p = lift_alloc(p, "B_vec", n_lifts=2)
        p = autofission(p, p.find("B_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for l in _: _ #0", neon_vld_8xf16)
        p = set_memory(p, "B_vec", Neon)

        p = replace(p, "for i in _: _ #0", neon_vfmla_8xf16_8xf16)
        p = unroll_loop(p, "l #0")
        return p

    simple_neon_vfmla_f16 = simple_vfmla_f16()

    return simplify(simple_neon_vfmla_f16)


@pytest.mark.isa("neon")
def test_gen_neon_vfmla_f16(golden, test_neon_vfmla_f16):
    assert str(test_neon_vfmla_f16) == golden


@pytest.fixture
def test_neon_vfmla_f16():
    """
    Compute C[i] = A[i] * B[l]
    """

    @proc
    def vfmla_f16(
        n: size, C: R[n] @ DRAM, A: R[n] @ DRAM, B: R[n] @ DRAM
    ):  # pragma: no cover
        assert n == 8
        for l in seq(0, 8):
            for i in seq(0, 8):
                C[i] += A[i] * B[l]

    def simple_vfmla_f16(p=vfmla_f16):
        p = stage_mem(p, "C[_] += _", "C[i]", "C_reg")
        p = expand_dim(p, "C_reg", 8, "i")
        p = lift_alloc(p, "C_reg", n_lifts=2)
        p = autofission(p, p.find("C_reg[_] = _").after(), n_lifts=2)
        p = autofission(p, p.find("C[_] = _").before(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_8xf16)
        p = replace(p, "for i in _: _ #1", neon_vst_8xf16)
        p = set_memory(p, "C_reg", Neon)

        p = bind_expr(p, "A[_]", "A_vec")
        p = expand_dim(p, "A_vec", 8, "i")
        p = lift_alloc(p, "A_vec", n_lifts=2)
        p = autofission(p, p.find("A_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_8xf16)
        p = set_memory(p, "A_vec", Neon)

        p = bind_expr(p, "B[_]", "B_vec")
        p = expand_dim(p, "B_vec", 8, "l")
        p = lift_alloc(p, "B_vec", n_lifts=2)
        p = autofission(p, p.find("B_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for l in _: _ #0", neon_vld_8xf16)
        p = set_memory(p, "B_vec", Neon)

        p = replace(p, "for i in _: _ #0", neon_vfmla_8xf16_8xf16)
        p = unroll_loop(p, "l #0")
        return p

    simple_neon_vfmla_f16 = simple_vfmla_f16()

    return simplify(simple_neon_vfmla_f16)


@pytest.mark.isa("neon")
def test_gen_neon_vfmla_f16(golden, test_neon_vfmla_f16):
    assert str(test_neon_vfmla_f16) == golden


@pytest.fixture
def simple_math_neon_sched():
    @proc
    def simple_math_neon_sched(
        n: size, x: R[n] @ DRAM, y: R[n] @ DRAM
    ):  # pragma: no cover
        for i in seq(0, n):
            x[i] = x[i] * y[i] * y[i]

    def sched_neon(p=simple_math_neon_sched):
        p = divide_loop(p, "i", 4, ["io", "ii"], tail="cut_and_guard")

        p = stage_mem(p, "for ii in _:_ #0", "x[4 * io : 4 * io + 4]", "xVec")

        p = bind_expr(p, p.find("y[_]", many=True), "yVec")
        p = autolift_alloc(p, "yVec: _", keep_dims=True)
        p = fission(p, p.find("yVec[_] = _").after())

        p = bind_expr(p, "xVec[_] * yVec[_]", "xy")
        p = autolift_alloc(p, "xy: _", keep_dims=True)
        p = fission(p, p.find("xy[_] = _").after())

        p = set_memory(p, "xVec", Neon)
        p = set_memory(p, "yVec", Neon)
        p = set_memory(p, "xy", Neon)
        p = replace(p, "for i0 in _: _ #1", neon_vst_4xf32)
        p = replace_all(p, neon_vld_4xf32)
        p = replace_all(p, neon_vmul_4xf32)

        p = simplify(p)
        return p

    simple_math_neon_sched = sched_neon()

    return simple_math_neon_sched


def test_gen_neon_simple_math_scheduling(golden, simple_math_neon_sched):
    assert str(simple_math_neon_sched) == golden


@pytest.mark.isa("neon")
def test_neon_simple_math_scheduling(compiler, simple_math_neon_sched):
    """
    Compute x = x * y^2
    """

    fn = compiler.compile(
        simple_math_neon_sched, skip_on_fail=True, CMAKE_C_FLAGS="-mcpu=apple-a14"
    )

    for n in (4, 8, 12, 16, 24, 32, 64, 128):
        x = np.array([float(i) for i in range(n)], dtype=np.float32)
        y = np.array([float(3 * i) for i in range(n)], dtype=np.float32)
        expected = x * y * y

        fn(None, n, x, y)
        assert np.allclose(x, expected)
