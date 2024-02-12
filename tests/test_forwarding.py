from __future__ import annotations

import pytest

from exo import proc
from exo.stdlib.scheduling import *
from exo.platforms.x86 import *


def test_divide_loop():
    @proc
    def foo():
        for i in seq(0, 10):
            pass
            pass

    i_loop = foo.find_loop("i")
    body = i_loop.body()

    # Loop always forwards to the outermost loop of the main loop nest
    # Loop body forwards to the main loop nest's body
    foo_guard = divide_loop(foo, "i", 5, ["io", "ii"], tail="guard")
    assert foo_guard.forward(i_loop) == foo_guard.find_loop("io")
    assert foo_guard.forward(body).parent().parent() == foo_guard.find_loop("ii")

    foo_cut = divide_loop(foo, "i", 5, ["io", "ii"], tail="cut")
    assert foo_cut.forward(i_loop) == foo_cut.find_loop("io")
    assert foo_cut.forward(body).parent() == foo_cut.find_loop("ii")

    foo_cg = divide_loop(foo, "i", 5, ["io", "ii"], tail="cut_and_guard")
    assert foo_cg.forward(i_loop) == foo_cg.find_loop("io")
    assert foo_cg.forward(body).parent() == foo_cg.find_loop("ii")


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


def test_fuse_loops():
    @proc
    def foo(n: size, x: R[n]):
        assert n > 3
        y: R[n]
        for i in seq(3, n):
            y[i] = x[i]
        for j in seq(3, n):
            x[j] = y[j] + 1.0

    loop1 = foo.find_loop("i")
    body1 = loop1.body()
    loop2 = foo.find_loop("j")
    body2 = loop2.body()

    both_loops = loop1.expand(0, 1)

    foo = fuse(foo, loop1, loop2)

    # The two loops forward to the newly fused loop
    assert foo.forward(loop1) == foo.forward(loop2)

    # Loop bodies are consecutive
    assert foo.forward(body1)[-1].next() == foo.forward(body2)[0]

    # Block containing both loops forwards to block containing fused loop.
    assert foo.forward(both_loops) == foo.find_loop("i").as_block()
