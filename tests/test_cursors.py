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
    assert cursor._node() is foo._loopir_proc


def test_get_child():
    cursor = Cursor.root(foo).child(0)
    assert cursor._node() is foo._loopir_proc.body[0]


def test_find_cursor():
    c = foo.find_cursor("for j in _:_")
    assert len(c) == 1
    c = c[0]

    assert c._node() is foo._loopir_proc.body[0].body[0]


# TODO: needs selection cursors
def test_cursor_move():
    c = foo.find_cursor("for j in _:_")
    assert len(c) == 1
    c = c[0]

    c_list = c.body()  # list of j's body
    assert isinstance(c_list, list)
    #TODO: subscriptable class probably shouldn't have a parent() method?
    #c_list_par = c_list.parent()  # for j in _:_
    #assert c is c_list_par

    c1 = c_list[0]  # x : f32
    assert c1._node() is foo._loopir_proc.body[0].body[0].body[0]
    c2 = c1.next()  # x = 0.0
    assert c2._node() is foo._loopir_proc.body[0].body[0].body[1]
    c3 = c1.next(2)  # y : f32
    assert c3._node() is foo._loopir_proc.body[0].body[0].body[2]

    _c2_ = c3.prev()
    assert c2 == _c2_
    _c1_ = c3.prev(2)
    assert c1 == _c1_


@pytest.mark.skip()
def test_cursor_gap():
    # for i in par(0, n):
    #    for j in par(0, m):
    #        x: f32
    #               <- g1
    #        x = 0.0
    #        y: f32
    #               <- g2
    #        y = 1.1
    c = foo.find_cursor("for j in _:_").body()[0]  # x : f32
    g1 = c.after()
    c1 = foo.find_cursor("x = 0.0")
    _g1_ = c1.before()
    assert g1 is _g1_

    g2 = g1.next(2)
    _g2_ = c1.after(2)
    assert g2 is _g2_

    # Testing gap -> stmt
    c3 = foo.find_cursor("y = 1.1")
    _c3_ = g2.after()
    assert c3 is _c3_


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
    assert c1._node is _c1_._node


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


# fwd(proc)
"""
find_cursor()

stmt.next() # stmt
prev() # stmt
after() # gap
before() # gap

parent()
body()
ast_type() OR is_seq() is_if() is_alloc() is_list()... \Yuka{I think is_seq() is nicer}
expressions (basically need to traverse LoopIR stmt to get LoopIR.expr):
    hi
    cond
    basetype
    iter
    name
    idx

@config
class ConfigLoad:
    stride: ...

def foo(src_stride: stride):
    pass
    x : i8
    ...
configwrite_at(c, )
configwrite_after('pass', ConfigLoad.stride, foo.arg()[0]._loopir())
def foo(src_stride: stride):
    pass
    ConfigLoad.stride = src_stride
    x : i8
    ...
configwrite_after('stride', None)
"""


def test_cursor_loop_bound():
    c_proc = Cursor.root(foo)
    c_fori = c_proc.child(0)
    c_bound = c_fori.child(0)
    assert isinstance(c_bound._node(), LoopIR.Read)
