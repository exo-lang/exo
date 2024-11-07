from __future__ import annotations

import pytest

from exo import DRAM
from exo.frontend.pyparser import (
    Parser,
    get_src_locals,
    get_ast_from_python,
    ParseError,
)


def to_uast(f):
    body, getsrcinfo = get_ast_from_python(f)
    parser = Parser(
        body,
        getsrcinfo,
        func_globals=f.__globals__,
        srclocals=get_src_locals(depth=2),
        instr=("TEST", ""),
        as_func=True,
    )
    return parser.result()


def test_conv1d(golden):
    def conv1d(
        n: size, m: size, r: size, x: R[n], w: R[m], res: R[r]
    ):  # pragma: no cover
        for i in seq(0, r):
            res[i] = 0.0
        for i in seq(0, r):
            for j in seq(0, n):
                if i <= j < i + m:
                    res[i] += x[j] * w[i - j + m - 1]

    assert str(to_uast(conv1d)) == golden


def test_unary_neg(golden):
    def negate_array(n: size, x: R[n], res: R[n] @ DRAM):  # pragma: no cover
        for i in seq(0, n):
            res[i] = -x[i] + -(x[i]) - -(x[i] + 0.0)

    assert str(to_uast(negate_array)) == golden


def test_alloc_nest(golden):
    def alloc_nest(
        n: size, m: size, x: R[n, m], y: R[n, m] @ DRAM, res: R[n, m] @ DRAM
    ):  # pragma: no cover
        for i in seq(0, n):
            rloc: R[m] @ DRAM
            xloc: R[m] @ DRAM
            yloc: R[m] @ DRAM
            for j in seq(0, m):
                xloc[j] = x[i, j]
            for j in seq(0, m):
                yloc[j] = y[i, j]
            for j in seq(0, m):
                rloc[j] = xloc[j] + yloc[j]
            for j in seq(0, m):
                res[i, j] = rloc[j]

    assert str(to_uast(alloc_nest)) == golden


global_str = "What is 6 times 9?"
global_num = 42


def test_variable_lookup_positive():
    def func(f: f32):
        for i in seq(0, 42):
            f += 1

    reference = to_uast(func)

    def func(f: f32):
        for i in seq(0, global_num):
            f += 1

    test_global = to_uast(func)
    assert str(test_global) == str(reference)

    local_num = 42

    def func(f: f32):
        for i in seq(0, local_num):
            f += 1

    test_local = to_uast(func)
    assert str(test_local) == str(reference)


def test_variable_lookup_type_error():
    def func(f: f32):
        for i in seq(0, global_str):
            f += 1

    with pytest.raises(ParseError, match="type <class 'str'>"):
        to_uast(func)

    local_str = "xyzzy"

    def func(f: f32):
        for i in seq(0, local_str):
            f += 1

    with pytest.raises(ParseError, match="type <class 'str'>"):
        to_uast(func)


def test_variable_lookup_name_error():
    def func(f: f32):
        for i in seq(0, xyzzy):
            f += 1

    with pytest.raises(ParseError, match="'xyzzy' undefined"):
        to_uast(func)
