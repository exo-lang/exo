from __future__ import annotations

from exo import instr, DRAM


# Merge this file to frontend?

def gen_conv1d():
    @instr("TEST", _testing="UAST")
    def conv1d(n: size, m: size, r: size,
               x: R[n], w: R[m], res: R[r]):
        for i in par(0, r):
            res[i] = 0.0
        for i in par(0, r):
            for j in par(0, n):
                if i <= j < i + m:
                    res[i] += x[j] * w[i - j + m - 1]

    return conv1d


def test_conv1d(golden):
    conv1d = gen_conv1d()
    assert str(conv1d) == golden


def test_unary_neg(golden):
    @instr("TEST", _testing="UAST")
    def negate_array(n: size, x: R[n], res: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, n):
            res[i] = -x[i] + -(x[i]) - -(x[i] + 0.0)

    assert str(negate_array) == golden


def gen_alloc_nest():
    @instr("TEST", _testing="UAST")
    def alloc_nest(n: size, m: size,
                   x: R[n, m], y: R[n, m] @ DRAM, res: R[n, m] @ DRAM):
        for i in par(0, n):
            rloc: R[m] @ DRAM
            xloc: R[m] @ DRAM
            yloc: R[m] @ DRAM
            for j in par(0, m):
                xloc[j] = x[i, j]
            for j in par(0, m):
                yloc[j] = y[i, j]
            for j in par(0, m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0, m):
                res[i, j] = rloc[j]

    return alloc_nest


def test_alloc_nest(golden):
    alloc = gen_alloc_nest()
    assert str(alloc) == golden
