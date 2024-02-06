from __future__ import annotations

import pytest

from exo import proc
from exo.stdlib.scheduling import *
from exo.platforms.x86 import *


def test_mult_loop():
    @proc
    def foo():
        for i in seq(0, 10):
            for j in seq(0, 30):
                pass

    i_loop = foo.find_loop("i")
    j_loop = foo.find_loop("j")
    stmt = j_loop.body()[0]
    foo = mult_loops(foo, "i j", "ij")
    assert foo.forward(i_loop) == foo.forward(j_loop) == foo.forward(stmt).parent()


def test_join_loops():
    @proc
    def foo(x: i32[20]):
        for i in seq(0, 10):
            x[i] = 1.0
        for i in seq(10, 20):
            x[i] = 1.0

    loop1 = foo.find_loop("i")
    stmt1 = loop1.body()[0]
    loop2 = foo.find_loop("i #1")
    stmt2 = loop2.body()[0]

    foo = join_loops(foo, loop1, loop2)

    assert foo.forward(loop1) == foo.forward(loop2)
    assert foo.forward(stmt1) == foo.forward(stmt2)


def test_fuse():
    @proc
    def foo(n: size, x: R[n]):
        assert n > 3
        y: R[n]
        for i in seq(3, n):
            y[i] = x[i]
        for j in seq(3, n):
            x[j] = y[j] + 1.0

    loop1 = foo.find_loop("i")
    stmt1 = loop1.body()[0]
    loop2 = foo.find_loop("j")
    stmt2 = loop2.body()[0]

    foo = fuse(foo, loop1, loop2)

    print(foo)
    print(foo.forward(loop1))

    assert foo.forward(loop1) == foo.forward(loop2)
    assert foo.forward(stmt1).next() == foo.forward(stmt2)
