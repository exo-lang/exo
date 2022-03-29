from __future__ import annotations

import weakref

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


# fwd(proc)
"""
stmt.next() # stmt
prev() # stmt
after() # gap
before() # gap

body()
ast_type() OR is_seq() is_if() is_alloc()...
expressions (basically need to traverse LoopIR stmt to get LoopIR.expr):
    hi
    cond
    basetype
    iter
    name
    idx
"""


def test_cursor_loop_bound():
    c_proc = Cursor.root(foo)
    c_fori = c_proc.child(0)
    c_bound = c_fori.child(0)
    assert isinstance(c_bound._node(), LoopIR.Read)
