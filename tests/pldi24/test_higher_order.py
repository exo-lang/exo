from __future__ import annotations

import pytest

from exo import proc, DRAM, Procedure, config
from exo.stdlib.scheduling import *
from exo.API_cursors import *


# The implementation of higher order functions


def try_do(op):
    def func(proc, *args, **kwargs):
        try:
            proc = op(proc, *args, **kwargs)
        except:
            pass
        return proc

    return func


def liftop(op):
    def func(p, c, *args, **kwargs):
        p = op(p, c, *args, **kwargs)
        return (p, c) if isinstance(p, Procedure) else p

    return func


def try_else(op, opelse):
    def func(p, c, *args):
        try:
            p, c = liftop(op)(p, c, *args)
        except:
            p, c = liftop(opelse)(p, c, *args)
        return p, c

    return func


def seq(*ops):
    def func(p, c, *args):
        for op in ops:
            p, c = liftop(op)(p, c, *args)
        return p, c

    return func


def reduce(op, top):
    def func(p, cursor, *args):
        for c in top(cursor):
            p, c = liftop(op)(p, c, *args)
        return p, c

    return func


def lift_stmt(p, s):
    def helper(p, s):
        p = repeat(reorder_stmts)(p, p.forward(s).expand(1, 0))
        p = fission(p, p.forward(s).after())
        return remove_loop(p, p.forward(s).parent())

    return repeat(helper)(p, s)


def traversal_func(c):
    yield c
    yield c.prev()


def lrn(c):
    for c in c.body():
        if isinstance(c, (ForCursor, IfCursor)):
            yield from lrn(c)
        yield c


def repeat(op):
    def func(p, c, *args):
        try:
            while True:
                p, c = liftop(op)(p, c, *args)
        except:
            return p, c

    return func


# Test for the higher order scheduling functions


def test_repeat(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8[n]
                tmp_a[i] = A[i]

    repeated_lift_alloc = repeat(lift_alloc)
    c = bar.find("tmp_a: _")

    assert str(repeated_lift_alloc(bar, c)[0]) == golden


def test_reduce(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8[n]
                tmp_b: i8[n]
                tmp_a[i] = A[i]
                tmp_b[i] = A[i]

    y_c = bar.find("tmp_b:_")

    assert str(reduce(lift_alloc, traversal_func)(bar, y_c)[0]) == golden


def test_lrn(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8[n]
                tmp_b: i8[n]
                tmp_a[i] = A[i]
                tmp_b[i] = A[i]

    str_p = ""
    for p in lrn(bar):
        str_p += str(p) + "\n"
    assert str_p == golden


# Scheduling combinators in 7.1


def reorder_before(p, s):
    return reorder_stmts(p, p.forward(s).expand(1, 0))


def fission_after(p, s):
    return fission(p, p.forward(s).after())


def remove_parent_loop(p, s):
    return remove_loop(p, p.forward(s).parent())


def reframe(nav):
    return lambda p, c: (p, nav(p.forward(c)))


def savec(op):
    return lambda p, c: (op(p, c)[0], c)


def test_reframe(golden):
    @proc
    def bar(n: size, A: i8[n]):
        x: R
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8[n]
                tmp_a[i] = A[i]
                x = 0.0

    a_c = bar.find("tmp_a: _")
    x_c = bar.find("x = _")

    frame = lambda nav_f, op: savec(seq(reframe(nav_f), op))
    reorder_before = frame(lambda c: c.expand(1, 0), reorder_stmts)
    remove_parent_loop = frame(lambda c: c.parent(), remove_loop)
    fission_after = frame(lambda c: c.after(), fission)
    assert (
        str(
            repeat(try_else(seq(fission_after, remove_parent_loop), reorder_before))(
                bar, x_c
            )[0]
        )
        == golden
    )
