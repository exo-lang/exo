from __future__ import annotations

import pytest

from exo import proc


@pytest.fixture(scope="session")
def proc_foo():
    @proc
    def foo(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

    yield foo


@pytest.fixture(scope="session")
def proc_bar():
    @proc
    def bar(n: size, m: size):
        x: f32
        for i in seq(0, n):
            for j in seq(0, m):
                x = 0.0
                x = 1.0
                x = 2.0
                x = 3.0
                x = 4.0
                x = 5.0

    yield bar


def test_parent_cursor(proc_foo):
    jloop = proc_foo.find("for j in _:_")
    iloop = proc_foo.find("for i in _:_")
    assert jloop.parent() == iloop


def test_child_cursor(proc_foo):
    jloop = proc_foo.find("for j in _:_")
    iloop = proc_foo.find("for i in _:_")
    assert jloop == iloop.body()[0]


def test_asblock_cursor(proc_foo):
    jloop = proc_foo.find("for j in _:_")
    iloop = proc_foo.find("for i in _:_")
    assert jloop.as_block() == iloop.body()


def test_builtin_cursor():
    @proc
    def bar():
        x: f32
        x = select(0.0, 1.0, 2.0, 3.0)

    select_builtin = bar.find("select(_)")
    assert select_builtin == bar.body()[1].rhs()


# Need some more tests here...
