from __future__ import annotations

import weakref

import exo
from exo import proc
from exo.LoopIR import LoopIR
from exo.pattern_match import Cursor
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
    assert isinstance(cursor.proc, weakref.ReferenceType)
    assert isinstance(cursor.proc(), exo.Procedure)
    assert isinstance(cursor.node, weakref.ReferenceType)
    assert isinstance(cursor.node(), LoopIR.proc)
    assert cursor.node() is foo._loopir_proc


def test_get_child():
    cursor = Cursor.root(foo).child(0)
    assert cursor.node() is foo._loopir_proc.body[0]


def test_basic_prune():
    cursor = Cursor.root(foo).child(0)
    cursor = cursor.child(1)
    cursor = cursor.child(4)
    assert cursor.node() is foo._loopir_proc.body[0].body[0].body[3]

    # n, m, x: f32, x = 0.0, y: f32
    assert len(cursor.prune) == 5
