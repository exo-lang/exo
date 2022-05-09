from __future__ import annotations

import gc
import weakref

import pytest

from exo import proc, Procedure
from exo.LoopIR import LoopIR, T
from exo.cursors import Cursor, Selection, InvalidCursorError
from exo.syntax import size, par, f32


@proc
def foo(n: size, m: size):
    for i in par(0, n):
        for j in par(0, m):
            x: f32
            x = 0.0
            y: f32
            y = 1.1


@proc
def bar(n: size, m: size):
    x: f32
    for i in par(0, n):
        for j in par(0, m):
            x = 0.0
            x = 1.0
            x = 2.0
            x = 3.0
            x = 4.0
            x = 5.0


def test_get_root():
    cursor = Cursor.root(foo)
    assert isinstance(cursor._proc, weakref.ReferenceType)
    assert isinstance(cursor._proc(), Procedure)
    assert isinstance(cursor._node, weakref.ReferenceType)
    assert isinstance(cursor._node(), LoopIR.proc)
    assert cursor.node() is foo.INTERNAL_proc()


def test_get_child():
    cursor = Cursor.root(foo).children()
    cursor = next(iter(cursor))
    assert cursor.node() is foo.INTERNAL_proc().body[0]


def test_find_cursor():
    c = foo.find_cursor("for j in _:_")
    assert len(c) == 1
    c = c[0]  # One match
    c = c[0]  # First/only node in the selection

    assert c.node() is foo.INTERNAL_proc().body[0].body[0]


def test_gap_insert_pass(golden):
    c = foo.find_cursor('x = 0.0')[0][0]
    assn = c.node()
    g = c.after()
    foo2 = g.insert([LoopIR.Pass(None, assn.srcinfo)])
    assert str(foo2) == golden


def test_insert_root_front(golden):
    c = Cursor.root(foo)
    foo2 = c.body().before().insert([LoopIR.Pass(None, c.node().srcinfo)])
    assert str(foo2) == golden


def test_insert_root_end(golden):
    c = Cursor.root(foo)
    foo2 = c.body().after().insert([LoopIR.Pass(None, c.node().srcinfo)])
    assert str(foo2) == golden


def test_selection_gaps():
    c = bar.find_cursor('for j in _: _')
    assert len(c) == 1
    c = c[0][0]

    body = c.body()
    assert len(body) == 6
    subset = body[1:4]
    assert len(subset) == 3

    cx1 = bar.find_cursor('x = 1.0')[0][0]
    cx3 = bar.find_cursor('x = 3.0')[0][0]
    assert subset[0] == cx1
    assert subset[2] == cx3

    assert subset.before() == cx1.before()
    assert subset.after() == cx3.after()


def test_selection_delete(golden):
    c = bar.find_cursor('for j in _: _')[0][0]
    stmts = c.body()[1:4]

    bar2 = stmts.delete()
    assert str(bar2) == golden


def test_selection_replace(golden):
    c = bar.find_cursor('for j in _: _')[0][0]
    stmts = c.body()[1:4]

    bar2 = stmts.replace([LoopIR.Pass(None, c.node().srcinfo)])
    assert str(bar2) == golden


def test_selection_delete_whole_block(golden):
    c = bar.find_cursor('for j in _: _')[0][0]
    bar2 = c.body().delete()
    assert str(bar2) == golden


def test_cursor_move():
    c = foo.find_cursor("for j in _:_")[0][0]

    c_list = c.body()  # list of j's body
    assert isinstance(c_list, Selection)
    assert len(c_list) == 4
    assert c_list.parent() == c

    c1 = c_list[0]  # x : f32
    assert c1.node() is foo.INTERNAL_proc().body[0].body[0].body[0]
    c2 = c1.next()  # x = 0.0
    assert c2.node() is foo.INTERNAL_proc().body[0].body[0].body[1]
    c3 = c1.next(2)  # y : f32
    assert c3.node() is foo.INTERNAL_proc().body[0].body[0].body[2]

    _c2_ = c3.prev()
    assert c2 == _c2_
    _c1_ = c3.prev(2)
    assert c1 == _c1_


def test_cursor_gap():
    # for i in par(0, n):
    #    for j in par(0, m):
    #        x: f32
    #                 <- g1
    #        x = 0.0  <- c1
    #        y: f32
    #                 <- g2
    #        y = 1.1
    c = foo.find_cursor("for j in _:_")[0][0].body()[0]  # x : f32
    assert str(c.node()) == 'x: f32 @ DRAM\n'
    g1 = c.after()
    assert g1._path[-1] == ('body', 1)
    c1 = foo.find_cursor("x = 0.0")[0][0]
    assert c1._path[-1] == ('body', 1)
    _g1_ = c1.before()
    assert g1 == _g1_

    g2 = g1.next(2)
    _g2_ = c1.after(2)
    assert g2 == _g2_

    # Testing gap -> stmt
    c3 = foo.find_cursor("y = 1.1")[0][0]
    _c3_ = g2.after()
    assert c3 == _c3_


def test_cursor_replace_expr(golden):
    c = foo.find_cursor('m')[0]
    foo2 = c.replace(LoopIR.Const(42, T.size, c.node().srcinfo))
    print(foo2)
    assert str(foo2) == golden


def test_cursor_loop_bound():
    c_proc = Cursor.root(foo)
    c_fori = c_proc.body()[0]
    c_bound = c_fori.child('hi')
    assert isinstance(c_bound.node(), LoopIR.Read)


def test_cursor_lifetime():
    @proc
    def delete_me():
        x: f32
        x = 0.0

    cur = delete_me.find_cursor('x = _')[0][0]
    assert isinstance(cur.node(), LoopIR.Assign)

    del delete_me
    gc.collect()

    with pytest.raises(InvalidCursorError, match='underlying proc was destroyed'):
        cur.proc()

    # TODO: The WeakKeyDictionary-ies in other modules seem to keep the IR alive as
    #   they keep references to them in the values.
    # with pytest.raises(InvalidCursorError, match='underlying node was destroyed'):
    #     cur.node()
