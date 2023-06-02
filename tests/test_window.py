from __future__ import annotations

import pytest

from exo import proc, DRAM


# ------- Window related tests ---------


def test_input1():
    @proc
    def foo(dst1: f32[8] @ DRAM, src1: [f32][8] @ DRAM):
        assert stride(src1, 0) == 1
        assert stride(dst1, 0) == 1

        for i in seq(0, 8):
            dst1[i] = src1[i]


def test_input2():
    @proc
    def foo(dst2: [f32][8] @ DRAM, src2: f32[8] @ DRAM):
        assert stride(src2, 0) == 1
        assert stride(dst2, 0) == 1

        for i in seq(0, 8):
            dst2[i] = src2[i]


def test_input3():
    with pytest.raises(TypeError, match="Could not verify assertion"):

        @proc
        def foo(dst2: [f32][8] @ DRAM, src2: f32[8] @ DRAM):
            assert stride(src2, 0) == 1
            assert stride(dst2, 0) == 1

            for i in seq(0, 8):
                dst2[i] = src2[i]

        @proc
        def bar(x: [f32][8]):
            foo(x, x)


def test_input4():
    @proc
    def foo(dst2: [f32][8] @ DRAM, src2: f32[8] @ DRAM):
        assert stride(src2, 0) == 1
        assert stride(dst2, 0) == 1

        for i in seq(0, 8):
            dst2[i] = src2[i]

    @proc
    def bar(x: [f32][8], y: [f32][8]):
        assert stride(x, 0) == 1
        assert stride(y, 0) == 1
        foo(x, y)


def test_window(golden):
    @proc
    def window(
        n: size,
        m: size,
        src: [i8][n, m] @ DRAM,
        dst: [i8][n, 16] @ DRAM,
    ):
        assert n <= 16
        assert m <= 16

        for i in seq(0, n):
            for j in seq(0, m):
                dst[i, j] = src[i, j]

    assert window.c_code_str() == golden


def test_stride_assert(golden):
    @proc
    def stride_assert(
        n: size,
        m: size,
        src: [i8][n, m] @ DRAM,
        dst: [i8][n, 16] @ DRAM,
    ):
        assert n <= 16
        assert m <= 16
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1

        for i in seq(0, n):
            for j in seq(0, m):
                dst[i, j] = src[i, j]

    assert stride_assert.c_code_str() == golden


def test_window_stmt(golden):
    @proc
    def window_stmt(n: size, m: size, x: f32[n, m]):
        y = x[:, 0]
        z: f32[n]
        for i in seq(0, n):
            z[i] = y[i]

    assert window_stmt.c_code_str() == golden


def test_normalize(golden):
    @proc
    def dot(m: size, x: [f32][m], y: [f32][m], r: f32):
        r = 0.0
        for i in seq(0, m):
            r += x[i] * y[i]

    @proc
    def proj(n: size, m: size, x: f32[n, m], y: f32[m, n]):
        assert n > 4
        assert m > 4
        xy: f32
        y2: f32
        dot(m, x[1, :], y[:, 2], xy)
        dot(m, y[:, 3], x[2, :], y2)

    assert proj.c_code_str() == golden
