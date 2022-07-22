from __future__ import annotations

import itertools
import os
import platform

import pytest

from exo import proc
from exo.platforms.neon import *

import numpy as np

@pytest.mark.isa('neon')
def test_neon_memcpy(compiler):
    """
    Compute dst = src
    """

    @proc
    def memcpy_neon(n: size, dst: R[n] @ DRAM,
                    src: R[n] @ DRAM):  # pragma: no cover
        for i in seq(0, (n + 3) / 4):
            if n - 4 * i >= 4:
                tmp: f32[4] @ Neon4f
                neon_vld_4xf32(tmp, src[4 * i:4 * i + 4])
                neon_vst_4xf32(dst[4 * i:4 * i + 4], tmp)
            else:
                for j in seq(0, n - 4 * i):
                    dst[4 * i + j] = src[4 * i + j]

    fn = compiler.compile(memcpy_neon,
                          skip_on_fail=True,
                          CMAKE_C_FLAGS="-mcpu=apple-a14")

    for n in (7, 8, 9, 31, 32, 33, 127, 128, 129):
        inp = np.array([float(i) for i in range(n)], dtype=np.float32)
        out = np.array([float(0) for _ in range(n)], dtype=np.float32)
        fn(None, n, out, inp)

        assert np.array_equal(inp, out)

@pytest.mark.isa('neon')
def test_neon_simple_math(compiler):
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_neon(n: size,
                         x: R[n] @ DRAM,
                         y: R[n] @ DRAM):  # pragma: no cover
        assert n % 4 == 0
        for i in seq(0, n / 4):
            xVec: f32[4] @ Neon4f
            yVec: f32[4] @ Neon4f
            neon_vld_4xf32(xVec, x[4 * i:4 * i + 4])
            neon_vld_4xf32(yVec, y[4 * i:4 * i + 4])
            neon_vmul_4xf32(xVec, xVec, yVec)
            neon_vmul_4xf32(xVec, xVec, yVec)
            neon_vst_4xf32(x[4 * i:4 * i + 4], xVec)

    fn = compiler.compile(simple_math_neon,
                          skip_on_fail=True,
                          CMAKE_C_FLAGS="-mcpu=apple-a14")

    for n in (4, 8, 12, 16, 24, 32, 64, 128):
        x = np.array([float(i) for i in range(n)], dtype=np.float32)
        y = np.array([float(3 * i) for i in range(n)], dtype=np.float32)
        expected = x * y * y

        fn(None, n, x, y)
        assert np.allclose(x, expected)

@pytest.mark.isa('neon')
def test_neon_simple_math_scheduling(compiler):
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_neon_sched(n: size,
                               x: R[n] @ DRAM,
                               y: R[n] @ DRAM):  # pragma: no cover
        for i in seq(0, n):
            x[i] = x[i] * y[i] * y[i]

    print()
    print()

    simple_math_neon_sched = (
        simple_math_neon_sched
            .split('i', 4, ['io', 'ii'], tail='cut_and_guard')
            .stage_assn('xyy', 'x[_] = _ #0')
            .lift_alloc('xyy: _', keep_dims=True)
            .fission_after('xyy[_] = _')

            .bind_expr('xVec', 'x[_]')
            .lift_alloc('xVec: _', keep_dims=True)
            .fission_after('xVec[_] = _')

            .bind_expr('yVec', 'y[_]', cse=True)
            .lift_alloc('yVec: _', keep_dims=True)
            .fission_after('yVec[_] = _')

            .bind_expr('xy', 'xVec[_] * yVec[_]')
            .lift_alloc('xy: _', keep_dims=True)
            .fission_after('xy[_] = _')

            .set_memory('xVec', Neon4f)
            .set_memory('yVec', Neon4f)
            .set_memory('xy', Neon4f)
            .set_memory('xyy', Neon4f)
            .replace(neon_vst_4xf32, 'for ii in _: _ #4')
            .replace_all(neon_vld_4xf32)
            .replace_all(neon_vmul_4xf32)
    )

    print(simple_math_neon_sched)


    fn = compiler.compile(simple_math_neon_sched,
                          skip_on_fail=True,
                          CMAKE_C_FLAGS="-mcpu=apple-a14")

    for n in (4, 8, 12, 16, 24, 32, 64, 128):
        x = np.array([float(i) for i in range(n)], dtype=np.float32)
        y = np.array([float(3 * i) for i in range(n)], dtype=np.float32)
        expected = x * y * y

        fn(None, n, x, y)
        assert np.allclose(x, expected)


