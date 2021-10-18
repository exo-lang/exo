from __future__ import annotations

from SYS_ATL import instr, Procedure, DRAM


# Merge this file to frontend?

def gen_conv1d():
    @instr("TEST", _testing="UAST")
    def conv1d(n : size, m : size, r: size,
               x : R[n], w : R[m], res : R[r] ):
        for i in par(0,r):
            res[i] = 0.0
        for i in par(0,r):
            for j in par(0,n):
                if i <= j < i + m:
                    res[i] += x[j]*w[i-j+m-1]

    return conv1d

def test_conv1d():
    conv1d = gen_conv1d()
    assert type(conv1d) is Procedure
    print(conv1d)


def test_unary_neg():
    @instr("TEST", _testing="UAST")
    def negate_array(n: size, x: R[n], res: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, n):
            res[i] = -x[i] + -(x[i]) - -(x[i] + 0.0)

    assert type(negate_array) is Procedure
    code = str(negate_array)
    print(code)
    assert 'res[i] = -x[i] + -x[i] - -(x[i] + 0.0)' in code


def gen_alloc_nest():
    @instr("TEST", _testing="UAST")
    def alloc_nest(n : size, m : size,
                   x : R[n,m], y: R[n,m] @ DRAM, res : R[n,m] @ DRAM):
        for i in par(0,n):
            rloc : R[m] @DRAM
            xloc : R[m] @DRAM
            yloc : R[m] @DRAM
            for j in par(0,m):
                xloc[j] = x[i,j]
            for j in par(0,m):
                yloc[j] = y[i,j]
            for j in par(0,m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0,m):
                res[i,j] = rloc[j]

    return alloc_nest

def test_alloc_nest():
    alloc = gen_alloc_nest()
    assert type(alloc) is Procedure
    print(alloc)
