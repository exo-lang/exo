from __future__ import annotations

from exo.stdlib.scheduling import *
from exo import proc
from exo.range_analysis import (
    constant_bound,
    arg_range_analysis,
    IndexRangeEnvironment,
)
from exo.stdlib.range_analysis import (
    infer_range,
    bounds_inference,
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
    e_range = constant_bound(e, {i_sym: (0, 5)})
    assert e_range == (2, 7)


def test_affine_index_range1():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (i + 2) * 5):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = constant_bound(e, {i_sym: (0, 5)})
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
    e_range = constant_bound(e, {i_sym: (0, 5), N_sym: (1, 5)})
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
    e_range = constant_bound(e, {i_sym: (0, 5), N_sym: (1, 5)})
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
    e_range = constant_bound(e, {i_sym: (0, 5), N_sym: (1, 5)})
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
    e_range = constant_bound(e, {i_sym: (0, 5), N_sym: (1, 5)})
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
    e_range = constant_bound(e, {i_sym: (0, 5), N_sym: (1, 5)})
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
    e_range = constant_bound(e, {i_sym: (0, 5), N_sym: (1, 5)})
    assert e_range == (2, 31)


def test_affine_index_range8():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (i + 10) % 4):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = constant_bound(e, {i_sym: (0, 5)})
    assert e_range == (0, 3)


def test_affine_index_range9():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (i + 10) % 9):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = constant_bound(e, {i_sym: (0, 5)})
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
    e_range = constant_bound(e, {i_sym: (0, 5), j_sym: (0, 7)})
    assert e_range == (0, 31)


def test_affine_index_range_fail():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (-i) * 3 + i * 4):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = constant_bound(e, {i_sym: (0, 5)})
    assert e_range == (-15, 20)


def test_affine_index_range_fail2():
    @proc
    def bar(N: size):
        for i in seq(0, 6):
            for j in seq(0, (i - 2) / 2 + 10):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = constant_bound(e, {i_sym: (0, 5)})
    assert e_range == (9, 11)


def test_affine_index_range_fail3():
    @proc
    def bar():
        for i in seq(0, 3):
            for j in seq(0, i * 16 + 16 - i * 16):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = constant_bound(e, {i_sym: (0, 2)})
    assert e_range == (-16, 48)


def test_arg_range():
    @proc
    def foo(N: size):
        pass

    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[0], fast=False
    ) == (1, None)
    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[0], fast=True
    ) == (1, None)


def test_arg_range2():
    @proc
    def foo(N: size, K: size):
        assert N >= 50
        assert K > 20
        pass

    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[0], fast=False
    ) == (50, None)
    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[1], fast=False
    ) == (21, None)
    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[0], fast=True
    ) == (1, None)


def test_arg_range3():
    @proc
    def foo(N: size, K: size):
        assert N == 50
        assert K > 20
        assert K >= 100
        pass

    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[0], fast=False
    ) == (50, 50)
    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[1], fast=False
    ) == (100, None)


def test_arg_range4():
    @proc
    def foo(N: size, K: size):
        assert N < 500
        assert K > 20
        assert K >= 100000
        pass

    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[0], fast=False
    ) == (1, 499)
    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[1], fast=False
    ) == (
        2**15,
        None,
    )


def test_arg_range5():
    @proc
    def foo(N: size, K: size):
        assert N < 500
        assert K > 20
        assert K >= 100000
        pass

    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[0], fast=False
    ) == (1, 499)
    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[1], fast=False
    ) == (
        2**15,
        None,
    )


def test_arg_range6():
    val = 2**32

    @proc
    def foo(N: size, K: size):
        assert N > val
        assert K < val
        pass

    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[0], fast=False
    ) == (
        2**15,
        None,
    )
    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[1], fast=False
    ) == (1, None)


def test_arg_range7():
    @proc
    def foo(N: index, K: index):
        assert -10 < K
        assert K < 4
        pass

    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[0], fast=False
    ) == (
        None,
        None,
    )
    assert arg_range_analysis(
        foo._loopir_proc, foo._loopir_proc.args[1], fast=False
    ) == (-9, 3)


def test_index_range_env():
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
        for j in seq(M, M + 1):
            pass

    env = IndexRangeEnvironment(foo._loopir_proc, fast=False)
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

    env.add_loop_iter(node.iter, node.lo, node.hi)
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

    run_N_asserts()

    loop = foo.find_loop("j")
    node = loop._impl._node
    env.add_loop_iter(node.iter, node.lo, node.hi)
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


# TODO: write test for IndexRange


def test_infer_range_of_expr():
    # TODO: test loops with non-zero lower bounds
    @proc
    def foo(n: size, m: size, x: i8[n + 2, m + 5]):
        assert n % 4 == 0
        assert m % 3 == 0
        for io in seq(0, n / 4):
            for jo in seq(0, m / 3):
                for ii in seq(0, 6):
                    for ji in seq(0, 8):
                        x[io * 4 + ii, jo * 3 + ji] = 1.0

    idx_c = foo.find("x[_] = _").idx()[0]
    idx = idx_c._impl._node

    bounds = infer_range(idx_c, foo.find_loop("jo"))
    assert str(bounds) == "(io * 4, 0, 5)"

    bounds = infer_range(idx_c, foo.find_loop("io"))
    assert str(bounds) == "(io * 4, 0, 5)"


def test_bounds_inference_tiled_blur2d():
    @proc
    def blur2d_tiled(n: size, consumer: i8[n, n], sin: i8[n + 1, n + 1]):
        assert n % 4 == 0
        producer: i8[n + 1, n + 1]
        for i in seq(0, n + 1):
            for j in seq(0, n + 1):
                producer[i, j] = sin[i, j]
        for io in seq(0, n / 4):
            for jo in seq(0, n / 4):
                for ii in seq(0, 4):
                    for ji in seq(0, 4):
                        consumer[4 * io + ii, 4 * jo + ji] = (
                            producer[4 * io + ii, 4 * jo + ji]
                            + producer[4 * io + ii, 4 * jo + ji + 1]
                            + producer[4 * io + ii + 1, 4 * jo + ji]
                            + producer[4 * io + ii + 1, 4 * jo + ji + 1]
                        ) / 4.0

    loop = blur2d_tiled.find_loop("io")
    io_sym = loop._impl._node.iter
    bound = bounds_inference(loop, "producer", 0, include=["R"])
    assert str(bound) == "(io * 4, 0, 4)"

    loop = blur2d_tiled.find_loop("jo")
    jo_sym = loop._impl._node.iter
    bound = bounds_inference(loop, "producer", 1, include=["R"])
    assert str(bound) == "(jo * 4, 0, 4)"

    bound = bounds_inference(loop, "producer", 0, include=["R"])
    assert str(bound) == "(io * 4, 0, 4)"
    assert bound.get_stride_of(io_sym) == 4
    bound = bounds_inference(loop, "producer", 1, include=["R"])
    assert str(bound) == "(jo * 4, 0, 4)"
    assert bound.get_stride_of(jo_sym) == 4


def test_bounds_inference_fail():
    @proc
    def foo(n: size, x: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                x[i] = 1.0
                x[j] = 1.0

    loop = foo.find_loop("j")
    bound = bounds_inference(loop, "x", 0, include=["W"])
    assert str(bound) == "(0, -inf, inf)"
