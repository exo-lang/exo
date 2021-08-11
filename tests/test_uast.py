from __future__ import annotations
import sys
from SYS_ATL import proc, instr, Procedure, DRAM
sys.path.append(sys.path[0]+"/..")

# Merge this file to frontend?

def gen_conv1d():
    @instr("TEST", _testing="UAST")
    def conv1d(n : index, m : index, r: index,
               x : R[n], w : R[m], res : R[r] ):
        assert n > 0 and m > 0 and r > 0
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
    def negate_array(n: index, x: R[n], res: R[n] @ DRAM):  # pragma: no cover
        assert n > 0
        for i in par(0, n):
            res[i] = -x[i] + -(x[i]) - -(x[i] + 0.0)

    assert type(negate_array) is Procedure
    code = str(negate_array)
    print(code)
    assert 'res[i] = -x[i] + -x[i] - -(x[i] + 0.0)' in code


def gen_alloc_nest():
    @instr("TEST", _testing="UAST")
    def alloc_nest(n : index, m : index,
                   x : R[n,m], y: R[n,m] @ DRAM, res : R[n,m] @ DRAM):
        assert n > 0 and m > 0
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
