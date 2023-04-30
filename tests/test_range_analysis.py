from __future__ import annotations

import pytest

from exo.stdlib.scheduling import *
from exo import proc, DRAM, Procedure
from exo.range_analysis import IndexRangeAnalysis


def test_affine_index_range():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, i + 2):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5)}).result()
    assert e_range == (2, 7)


def test_affine_index_range1():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (i + 2) * 5):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5)}).result()
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
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5), N_sym: (1, 5)}).result()
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
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5), N_sym: (1, 5)}).result()
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
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5), N_sym: (1, 5)}).result()
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
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5), N_sym: (1, 5)}).result()
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
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5), N_sym: (1, 5)}).result()
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
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5), N_sym: (1, 5)}).result()
    assert e_range == (2, 31)


def test_affine_index_range8():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (i + 10) % 4):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5)}).result()
    assert e_range == (0, 4)


def test_affine_index_range9():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (i + 10) % 9):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5)}).result()
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
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5), j_sym: (0, 7)}).result()
    assert e_range == (0, 31)


def test_affine_index_range_fail():
    @proc
    def bar():
        for i in seq(0, 6):
            for j in seq(0, (-i) * 3 + i * 4):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5)}).result()
    assert e_range == None


def test_affine_index_range_fail1():
    @proc
    def bar(N: size):
        for i in seq(0, 6):
            for j in seq(0, (i - 2) / 2 + 10):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 5)}).result()
    assert e_range == None


def test_affine_index_range_fail2():
    @proc
    def bar():
        for i in seq(0, 3):
            for j in seq(0, i * 16 + 16 - i * 16):
                pass

    e = bar.find("for j in _:_").hi()._impl._node
    i_sym = bar.find("for i in _:_")._impl._node.iter
    e_range = IndexRangeAnalysis(e, {i_sym: (0, 2)}).result()
    assert e_range == None
