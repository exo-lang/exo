from __future__ import annotations

import pytest

from exo.stdlib.scheduling import *
from exo import proc
from exo.range_analysis import (
    index_range_analysis,
    arg_range_analyize,
    IndexRangeEnvironment,
)
from exo.LoopIR import LoopIR, T


def test_affine_index_range():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, i + 2):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = index_range_analysis(e, {i_sym: (0, 5)})
    assert e_range == (2, 7)


def test_affine_index_range1():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (i + 2) * 5):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = index_range_analysis(e, {i_sym: (0, 5)})
    assert e_range == (10, 35)


def test_affine_index_range2():
    @proc
    def bar(N: size):
        assert N >= 1
        assert N <= 5
        for i in seq(0, 6):
            for j in seq(0, N + (i + 2) * 5):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    N_sym = bar._loopir_proc.args[0].name
    e_range = index_range_analysis(e, {i_sym: (0, 5), N_sym: (1, 5)})
    assert e_range == (11, 40)


def test_affine_index_range3():
    @proc
    def bar(N: size):
        assert N >= 1
        assert N <= 5
        for i in seq(0, 6):
            for j in seq(0, (N + (i + 2) * 5) / 2):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    N_sym = bar._loopir_proc.args[0].name
    e_range = index_range_analysis(e, {i_sym: (0, 5), N_sym: (1, 5)})
    assert e_range == (5, 20)


def test_affine_index_range4():
    @proc
    def bar(N: size):
        assert N >= 1
        assert N <= 5
        for i in seq(0, 6):
            for j in seq(0, (N + (i + 2) * 5) - 4):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    N_sym = bar._loopir_proc.args[0].name
    e_range = index_range_analysis(e, {i_sym: (0, 5), N_sym: (1, 5)})
    assert e_range == (7, 36)


def test_affine_index_range5():
    @proc
    def bar(N: size):
        assert N >= 1
        assert N <= 5
        for i in seq(0, 6):
            for j in seq(0, (i + 2) * 5 - N):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    N_sym = bar._loopir_proc.args[0].name
    e_range = index_range_analysis(e, {i_sym: (0, 5), N_sym: (1, 5)})
    assert e_range == (5, 34)


def test_affine_index_range6():
    @proc
    def bar(N: size):
        assert N >= 1
        assert N <= 5
        for i in seq(0, 6):
            for j in seq(0, (i + 2) * 5 - N):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    N_sym = bar._loopir_proc.args[0].name
    e_range = index_range_analysis(e, {i_sym: (0, 5), N_sym: (1, 5)})
    assert e_range == (5, 34)


def test_affine_index_range7():
    @proc
    def bar(N: size):
        assert N >= 1
        assert N <= 5
        for i in seq(0, 6):
            for j in seq(0, (-3) + (i + 2) * 5 - N):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    N_sym = bar._loopir_proc.args[0].name
    e_range = index_range_analysis(e, {i_sym: (0, 5), N_sym: (1, 5)})
    assert e_range == (2, 31)


def test_affine_index_range8():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (i + 10) % 4):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = index_range_analysis(e, {i_sym: (0, 5)})
    assert e_range == (0, 4)


def test_affine_index_range9():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (i + 10) % 9):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = index_range_analysis(e, {i_sym: (0, 5)})
    assert e_range == (1, 6)


def test_affine_index_range10():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, 8):
                for k in seq(0, 2 * i + 3 * j):
                    pass

    e = bar.find("for k in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    j_sym = bar.find("for j in _:_")._impl._node.iter
    e_range = index_range_analysis(e, {i_sym: (0, 5), j_sym: (0, 7)})
    assert e_range == (0, 31)


def test_affine_index_range_fail():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (-i) * 3 + i * 4):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = index_range_analysis(e, {i_sym: (0, 5)})
    assert e_range == (None, None)


def test_affine_index_range_fail1():
    @proc
    def bar(N: size):
        for i in seq(0, 6):
            for j in seq(0, (i - 2) / 2 + 10):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = index_range_analysis(e, {i_sym: (0, 5)})
    assert e_range == (None, None)


def test_affine_index_range_fail2():
    @proc
    def bar():
        for i in seq(0, 3):
            for j in seq(0, i * 16 + 16 - i * 16):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = index_range_analysis(e, {i_sym: (0, 2)})
    assert e_range == (None, None)


def test_arg_range():
    @proc
    def foo(N: size):
        pass

    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[0]) == (1, None)


def test_arg_range2():
    @proc
    def foo(N: size, K: size):
        assert N >= 50
        assert K > 20
        pass

    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[0]) == (50, None)
    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[1]) == (21, None)


def test_arg_range3():
    @proc
    def foo(N: size, K: size):
        assert N == 50
        assert K > 20
        assert K >= 100
        pass

    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[0]) == (50, 50)
    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[1]) == (100, None)


def test_arg_range4():
    @proc
    def foo(N: size, K: size):
        assert N < 500
        assert K > 20
        assert K >= 100000
        pass

    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[0]) == (1, 499)
    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[1]) == (
        100000,
        None,
    )


def test_arg_range4():
    @proc
    def foo(N: size, K: size):
        assert N < 500
        assert K > 20
        assert K >= 100000
        pass

    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[0]) == (1, 499)
    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[1]) == (
        100000,
        None,
    )


def test_arg_range5():
    val = 2**32

    @proc
    def foo(N: size, K: size):
        assert N > val
        assert K < val
        pass

    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[0]) == (
        None,
        None,
    )
    assert arg_range_analyize(foo._loopir_proc, foo._loopir_proc.args[1]) == (1, None)


def test_inedex_range_env():
    N_upper_bound = 20
    K_lower_bound = 30
    M_value = 100

    @proc
    def foo(N: size, K: size, M: size):
        assert N <= N_upper_bound
        assert K >= K_lower_bound
        assert M == M_value
        for i in seq(K, N * 2 + K):
            pass
        for j in seq(M, M):
            pass

    env = IndexRangeEnvironment(foo._loopir_proc)
    loop = foo.find_loop("i")
    node = loop._impl._node

    N_read = LoopIR.Read(
        foo._loopir_proc.args[0].name, [], T.size, foo._loopir_proc.srcinfo
    )

    def run_N_asserts():
        assert env.check_expr_bounds(
            0,
            IndexRangeEnvironment.leq,
            N_read,
            IndexRangeEnvironment.leq,
            N_upper_bound,
        )
        assert env.check_expr_bounds(
            1,
            IndexRangeEnvironment.leq,
            N_read,
            IndexRangeEnvironment.lt,
            N_upper_bound + 1,
        )
        assert not env.check_expr_bounds(
            1,
            IndexRangeEnvironment.lt,
            N_read,
            IndexRangeEnvironment.lt,
            N_upper_bound + 1,
        )
        assert not env.check_expr_bounds(
            1, IndexRangeEnvironment.lt, N_read, IndexRangeEnvironment.lt, N_upper_bound
        )
        assert not env.check_expr_bounds(
            N_upper_bound,
            IndexRangeEnvironment.eq,
            N_read,
            IndexRangeEnvironment.eq,
            N_upper_bound,
        )

    run_N_asserts()

    env.enter_scope()

    # I should still be able to see `N`
    run_N_asserts()

    env.add_sym(node.iter, node.lo, node.hi)
    i_read = LoopIR.Read(node.iter, [], T.index, foo._loopir_proc.srcinfo)

    run_N_asserts()

    assert not env.check_expr_bounds(
        0, IndexRangeEnvironment.leq, i_read, IndexRangeEnvironment.leq, 100
    )
    assert not env.check_expr_bounds(
        K_lower_bound,
        IndexRangeEnvironment.leq,
        i_read,
        IndexRangeEnvironment.leq,
        100000,
    )
    assert env.check_expr_bound(0, IndexRangeEnvironment.leq, i_read)
    assert env.check_expr_bound(K_lower_bound, IndexRangeEnvironment.leq, i_read)
    assert not env.check_expr_bound(K_lower_bound, IndexRangeEnvironment.lt, i_read)

    env.exit_scope()

    # I shouldn't be able to see `i` now
    with pytest.raises(AssertionError, match=""):
        env.check_expr_bounds(
            0, IndexRangeEnvironment.leq, i_read, IndexRangeEnvironment.leq, 100
        )

    # I should still be able to see `N` though
    run_N_asserts()

    loop = foo.find_loop("j")
    node = loop._impl._node
    env.add_sym(node.iter, node.lo, node.hi)
    N_read = LoopIR.Read(
        foo._loopir_proc.args[0].name, [], T.size, foo._loopir_proc.srcinfo
    )
    j_read = LoopIR.Read(node.iter, [], T.index, foo._loopir_proc.srcinfo)
    assert env.check_expr_bounds(
        M_value, IndexRangeEnvironment.eq, j_read, IndexRangeEnvironment.eq, M_value
    )
    assert env.check_expr_bounds(
        M_value, IndexRangeEnvironment.eq, j_read, IndexRangeEnvironment.leq, M_value
    )
    assert not env.check_expr_bounds(
        M_value, IndexRangeEnvironment.eq, j_read, IndexRangeEnvironment.lt, M_value
    )
