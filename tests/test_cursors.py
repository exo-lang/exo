from __future__ import annotations

import weakref

import pytest

import exo
from exo import proc
from exo.LoopIR import LoopIR
from exo.cursors import Cursor
from exo.syntax import size, par, f32


@proc
def foo(n: size, m: size):
    for i in par(0, n):
        for j in par(0, m):
            x: f32
            x = 0.0
            y: f32
            y = 1.1


def test_get_root():
    cursor = Cursor.root(foo)
    assert isinstance(cursor._proc, weakref.ReferenceType)
    assert isinstance(cursor._proc(), exo.Procedure)
    assert isinstance(cursor._node, weakref.ReferenceType)
    assert isinstance(cursor._node(), LoopIR.proc)
    assert cursor.node() is foo._loopir_proc


def test_get_child():
    cursor = Cursor.root(foo).child(0)
    assert cursor.node() is foo._loopir_proc.body[0]


def test_find_cursor():
    c = foo.find_cursor("for j in _:_")
    assert len(c) == 1
    c = c[0]

    assert c.node() is foo._loopir_proc.body[0].body[0]


# TODO: needs selection cursors
def test_cursor_move():
    c = foo.find_cursor("for j in _:_")
    assert len(c) == 1
    c = c[0]

    c_list = c.body()  # list of j's body
    assert isinstance(c_list, list)
    # TODO: subscriptable class probably shouldn't have a parent() method?
    # c_list_par = c_list.parent()  # for j in _:_
    # assert c is c_list_par

    c1 = c_list[0]  # x : f32
    assert c1.node() is foo._loopir_proc.body[0].body[0].body[0]
    c2 = c1.next()  # x = 0.0
    assert c2.node() is foo._loopir_proc.body[0].body[0].body[1]
    c3 = c1.next(2)  # y : f32
    assert c3.node() is foo._loopir_proc.body[0].body[0].body[2]

    _c2_ = c3.prev()
    assert c2 == _c2_
    _c1_ = c3.prev(2)
    assert c1 == _c1_


def test_cursor_gap():
    # for i in par(0, n):
    #    for j in par(0, m):
    #        x: f32
    #               <- g1
    #        x = 0.0
    #        y: f32
    #               <- g2
    #        y = 1.1
    c = foo.find_cursor("for j in _:_")[0].body()[0]  # x : f32
    assert str(c.node()) == 'x: f32 @ DRAM\n'
    g1 = c.after()
    c1 = foo.find_cursor("x = 0.0")[0]
    _g1_ = c1.before()
    assert g1 == _g1_

    g2 = g1.next(2)
    _g2_ = c1.after(1)
    assert g2 == _g2_

    # Testing gap -> stmt
    c3 = foo.find_cursor("y = 1.1")[0]
    _c3_ = g2.after()
    assert c3 == _c3_


@pytest.mark.skip()
def test_explicit_fwd():
    # for i in par(0, n):
    #    for j in par(0, m):
    #        x: f32 <- c1
    #        x = 0.0
    #               <- g1
    #        y: f32
    #        y = 1.1
    c1 = foo.find_cursor("x : f32")
    g1 = c1.after(2)
    new_foo = foo.fission_at(g1)
    _c1_ = new_foo.fwd(c1)
    assert c1.node() is _c1_.node()


@pytest.mark.skip()
def test_implicit_fwd():
    c1 = foo.find_cursor("x : f32")
    g1 = c1.after(2)
    bar = foo.fission_at(g1)
    # for i in par(0, n):
    #    for j in par(0, m):
    #        x: f32 <- c1
    #        x = 0.0
    #               <- g1
    # for i in par(0, n):
    #    for j in par(0, m):
    #        y: f32
    #        y = 1.1

    bar = foo.lift_alloc(c1, n_lifts=2)
    # This should give the following:
    # x : f32[n,m] <- c1
    # for i in par(0, n):
    #    for j in par(0, m):
    #        x[i,j] = 0.0
    #               <- g1
    # for i in par(0, n):
    #    for j in par(0, m):
    #        y: f32
    #        y = 1.1


@pytest.mark.skip()
def test_cursor_print():
    # I am not sure how exactly this should look like,
    # but print(foo) should print cursors like the comments above
    pass


def test_cursor_loop_bound():
    c_proc = Cursor.root(foo)
    c_fori = c_proc.child(0)
    c_bound = c_fori.child(0)
    assert isinstance(c_bound.node(), LoopIR.Read)
