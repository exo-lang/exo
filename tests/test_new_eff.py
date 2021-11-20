from __future__ import annotations

import pytest

from SYS_ATL.new_eff import *

from SYS_ATL import proc, DRAM, SchedulingError


print()
print("Dev Tests for new_eff.py")


def test_debug_let_and_mod():
    N = AInt(Sym('N'))
    j = AInt(Sym('j'))
    i = AInt(Sym('i'))
    x = AInt(Sym('x'))

    F =  A.ForAll(i.name,
            A.Let([x.name],
                  [A.Let([j.name],[AInt(64) * i],
                            N + j, T.index, j.srcinfo)],
                  AEq(x % AInt(64), AInt(0)), T.bool, x.srcinfo),
            T.bool, i.srcinfo)

    print(F)

    slv = SMTSolver(verbose=True)

    slv.verify(F)
    print(slv.debug_str(smt=True))


def test_reorder_stmts_fail():

    @proc
    def foo( N : size, x : R[N] ):
        x[0] = 3.0
        x[0] = 4.0

    with pytest.raises(SchedulingError,
                       match='do not commute'):
      foo = foo.reorder_stmts('x[0] = 3.0', 'x[0] = 4.0')
      print(foo)

def test_reorder_loops_success():

    @proc
    def foo( N : size, x : R[N,N] ):
        for i in seq(0,N):
          for j in seq(0,N):
            x[i,j] = x[i,j] * 2.0

    foo = foo.reorder('i','j')
    print(foo)


def test_reorder_loops_fail():

    @proc
    def foo( N : size, x : R[N,N] ):
        for i in seq(0,N):
          for j in seq(0,N):
            x[i,j] = x[j,i] * 2.0

    with pytest.raises(SchedulingError,
                       match='cannot be reordered'):
      foo = foo.reorder('i','j')
      print(foo)



def test_alloc_success():

    @proc
    def foo( N : size, x : R[N,N] ):
        for i in seq(0,N):
          for j in seq(0,N):
            tmp : R
            tmp = x[i,j] * 2.0
            x[i,j] = tmp

    foo = foo.reorder('i','j')
    print(foo)
