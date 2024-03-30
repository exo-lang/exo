from __future__ import annotations

from exo import DRAM
from exo.pyparser import Parser, get_src_locals, get_ast_from_python


def to_uast(f):
    body, getsrcinfo = get_ast_from_python(f)
    parser = Parser(
        body,
        getsrcinfo,
        func_globals=f.__globals__,
        srclocals=get_src_locals(depth=3),
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
