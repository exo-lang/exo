from __future__ import annotations

import pytest

from SYS_ATL.new_eff import *

from SYS_ATL import proc, config, DRAM, SchedulingError


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

def test_reorder_alloc_fail():
    @proc
    def foo( N : size, x : R[N] ):
        y : R
        y    = 4.0
        x[0] = y

    with pytest.raises(SchedulingError,
                       match='do not commute'):
        foo = foo.reorder_stmts('y : R', 'y = 4.0')
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


def test_reorder_loops_requiring_seq():
    # the stencil pattern here looks like
    #     o     o
    #       \   |
    #         \ V
    #     o --> x
    #
    # so it isn't safe to _reverse_
    # the iteration order, but it is safe
    # to reorder the loops

    @proc
    def foo( N : size, x : R[N,N] ):
        for i in seq(0,N):
            for j in seq(0,N):
                if i > 0 and j > 0:
                    x[i,j] += -1.0/3.0 * (x[i-1,j] + x[i-1,j-1] + x[i,j-1])

    foo = foo.reorder('i','j')
    print(foo)

def test_reorder_loops_4pt_stencil_succeed():
    # Also, a 4-point stencil being
    # used in e.g. a Gauss-Seidel scheme can be reordered

    @proc
    def foo( N : size, x : R[N,N] ):
        for i in seq(0,N):
            for j in seq(0,N):
                if 0 < i < N-1 and 0 < j < N-1:
                    x[i,j] += -1.0/4.0 * (x[i-1,j] + x[i+1,j] +
                                          x[i,j-1] + x[i,j+1])

    foo = foo.reorder('i','j')
    print(foo)



def test_reorder_loops_failing_seq():
    # But if we do the stencil over the 4 diagonals, then it's not safe

    @proc
    def foo( N : size, x : R[N,N] ):
        for i in seq(0,N):
            for j in seq(0,N):
                if 0 < i < N-1 and 0 < j < N-1:
                    x[i,j] += -1.0/4.0 * (x[i-1,j-1] + x[i-1,j+1] +
                                          x[i+1,j-1] + x[i+1,j+1])

    with pytest.raises(SchedulingError,
                       match='cannot be reordered'):
        foo = foo.reorder('i','j')
        print(foo)



# Should add a test that shows that READing something in an assertion
# does in fact count as a READ effect for the procedure, but not
# for its body.  This can probably distinguish whether certain
# rewrites are allowed or not.




def test_delete_config_basic():
    @config
    class CFG:
        a : index
        b : size

    @proc
    def foo( N : size, x : R[N] ):
        CFG.a = 3
        for i in seq(0,N):
            x[i] = x[i] + 1.0

    foo = foo.delete_config('CFG.a = _')
    print(foo)


def test_delete_config_subproc_basic():
    @config
    class CFG:
        a : index
        b : size

    @proc
    def do_config():
        CFG.a = 3
        CFG.b = 5

    @proc
    def foo( N : size, x : R[N] ):
        do_config()
        for i in seq(0,N):
            x[i] = x[i] + 1.0

    foo = foo.delete_config('do_config()')
    print(foo)

def test_delete_config_fail():
    @config
    class CFG:
        a : index
        b : size

    @proc
    def foo( N : size, x : R[N] ):
        CFG.a = 3
        for i in seq(0,N):
            if i < CFG.a:
                x[i] = x[i] + 1.0

    with pytest.raises(SchedulingError,
                       match='Cannot change configuration values'):
        foo = foo.delete_config('CFG.a = _')
        print(foo)
    
def test_delete_config_subproc_fail():
    @config
    class CFG:
        a : index
        b : size

    @proc
    def do_config():
        CFG.a = 3
        CFG.b = 5

    @proc
    def foo( N : size, x : R[N] ):
        do_config()
        for i in seq(0,N):
            if i < CFG.a:
                x[i] = x[i] + 1.0

    with pytest.raises(SchedulingError,
                       match='Cannot change configuration values'):
        foo = foo.delete_config('do_config()')
        print(foo)












