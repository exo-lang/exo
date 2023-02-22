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
    def read_neon(n: size, dst: R[n] @ DRAM, src: R[n] @ Neon4f):
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
                tmp: f32[4] @ Neon4f
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
            xVec: f32[4] @ Neon4f
            yVec: f32[4] @ Neon4f
            neon_vld_4xf32(xVec, x[4 * i : 4 * i + 4])
            neon_vld_4xf32(yVec, y[4 * i : 4 * i + 4])
            neon_vmul_4xf32(xVec, xVec, yVec)
            neon_vmul_4xf32(xVec, xVec, yVec)
            neon_vst_4xf32(x[4 * i : 4 * i + 4], xVec)

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
        p = expand_dim(p, "C_reg", 4, "i", unsafe_disable_checks=True)
        p = lift_alloc(p, "C_reg", n_lifts=2)
        p = autofission(p, p.find("C_reg[_] = _").after(), n_lifts=2)
        p = autofission(p, p.find("C[_] = _").before(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_4xf32)
        p = replace(p, "for i in _: _ #1", neon_vst_4xf32)
        p = set_memory(p, "C_reg", Neon4f)

        p = bind_expr(p, "A[_]", "A_vec")
        p = expand_dim(p, "A_vec", 4, "i", unsafe_disable_checks=True)
        p = lift_alloc(p, "A_vec", n_lifts=2)
        p = autofission(p, p.find("A_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for i in _: _ #0", neon_vld_4xf32)
        p = set_memory(p, "A_vec", Neon4f)

        p = bind_expr(p, "B[_]", "B_vec")
        p = expand_dim(p, "B_vec", 4, "l", unsafe_disable_checks=True)
        p = lift_alloc(p, "B_vec", n_lifts=2)
        p = autofission(p, p.find("B_vec[_] = _").after(), n_lifts=2)
        p = replace(p, "for l in _: _ #0", neon_vld_4xf32)
        p = set_memory(p, "B_vec", Neon4f)

        p = replace(p, "for i in _: _ #0", neon_vfmla_4xf32_4xf32)
        p = unroll_loop(p, "l #0")
        return p

    simple_neon_vfmla = simple_vfmla()

    return simple_neon_vfmla


@pytest.mark.isa("neon")
def test_gen_neon_vfmla(golden, test_neon_vfmla):
    assert str(test_neon_vfmla) == golden


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

        p = bind_expr(p, "y[_]", "yVec", cse=True)
        p = autolift_alloc(p, "yVec: _", keep_dims=True)
        p = fission(p, p.find("yVec[_] = _").after())

        p = bind_expr(p, "xVec[_] * yVec[_]", "xy")
        p = autolift_alloc(p, "xy: _", keep_dims=True)
        p = fission(p, p.find("xy[_] = _").after())

        p = set_memory(p, "xVec", Neon4f)
        p = set_memory(p, "yVec", Neon4f)
        p = set_memory(p, "xy", Neon4f)
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


@pytest.mark.isa("neon")
def test_neon_mul_alias_hack(compiler):
    @proc
    def neon_vmul_4xf32_alias_hack_wrapper(dst: f32[4] @ DRAM, rhs: f32[4] @ DRAM):
        tmp_buffer_0: f32[4] @ Neon4f
        neon_vld_4xf32(tmp_buffer_0, dst)
        tmp_buffer_1: f32[4] @ Neon4f
        neon_vld_4xf32(tmp_buffer_1, rhs)
        neon_vmul_4xf32_alias_hack(tmp_buffer_0, tmp_buffer_1)
        neon_vst_4xf32(dst, tmp_buffer_0)
        neon_vst_4xf32(rhs, tmp_buffer_1)

    @proc
    def neon_vmul_4xf32_alias_hack_ref(dst: f32[4] @ DRAM, rhs: f32[4] @ DRAM):
        # @instr {dst_data} = vmulq_f32({dst_data}, {rhs_data});
        assert stride(dst, 0) == 1
        assert stride(rhs, 0) == 1
        for i in seq(0, 4):
            dst[i] = dst[i] * rhs[i]

    fn = compiler.compile(
        [neon_vmul_4xf32_alias_hack_wrapper, neon_vmul_4xf32_alias_hack_ref],
        skip_on_fail=True,
    )
    dst = np.array([0.03267257, 0.70744205, 0.0064026015, 0.9334069], dtype=np.float32)
    rhs = np.array([0.8566227, 0.8331061, 0.6870745, 0.078659], dtype=np.float32)
    dst_copy = dst.copy()
    rhs_copy = rhs.copy()
    getattr(fn, "neon_vmul_4xf32_alias_hack_wrapper")(None, dst, rhs)
    getattr(fn, "neon_vmul_4xf32_alias_hack_ref")(None, dst_copy, rhs_copy)
    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(rhs, rhs_copy)


@pytest.mark.isa("neon")
def test_neon_reg_copy(compiler):
    @proc
    def neon_reg_copy_4xf32_wrapper(dst: f32[4] @ DRAM, src: f32[4] @ DRAM):
        tmp_buffer_0: f32[4] @ Neon4f
        neon_vld_4xf32(tmp_buffer_0, dst)
        tmp_buffer_1: f32[4] @ Neon4f
        neon_vld_4xf32(tmp_buffer_1, src)
        neon_reg_copy_4xf32(tmp_buffer_0, tmp_buffer_1)
        neon_vst_4xf32(dst, tmp_buffer_0)
        neon_vst_4xf32(src, tmp_buffer_1)

    @proc
    def neon_reg_copy_4xf32_ref(dst: f32[4] @ DRAM, src: f32[4] @ DRAM):
        # @instr {dst_data} = {src_data};
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 4):
            dst[i] = src[i]

    fn = compiler.compile(
        [neon_reg_copy_4xf32_wrapper, neon_reg_copy_4xf32_ref], skip_on_fail=True
    )
    dst = np.array([0.673626, 0.17301883, 0.3578481, 0.25818807], dtype=np.float32)
    src = np.array([0.077489585, 0.57495946, 0.4729017, 0.93222266], dtype=np.float32)
    dst_copy = dst.copy()
    src_copy = src.copy()
    getattr(fn, "neon_reg_copy_4xf32_wrapper")(None, dst, src)
    getattr(fn, "neon_reg_copy_4xf32_ref")(None, dst_copy, src_copy)
    np.testing.assert_almost_equal(dst, dst_copy)
    np.testing.assert_almost_equal(src, src_copy)
