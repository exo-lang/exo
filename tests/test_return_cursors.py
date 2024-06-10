from __future__ import annotations

import pytest

from exo import proc, ExoType
from exo.stdlib.scheduling import *
from exo.libs.memories import *
from exo.API_cursors import *


def test_add_loop(golden):
    @proc
    def foo(x: i8):
        x = 0.0

    foo, (block, loop) = add_loop(foo, foo.find("x = _"), "i", 4, rc=True)
    assert "\n\n".join([str(v) for v in [block, loop]]) == golden


def test_add_loop_with_guard(golden):
    @proc
    def foo(x: i8):
        x += 1.0

    foo, cs = add_loop(foo, foo.find("x += _"), "i", 4, guard=True, rc=True)
    assert "\n\n".join([str(v) for v in cs]) == golden


def test_unroll_loop(golden):
    @proc
    def foo(x: i8[4]):
        for i in seq(0, 4):
            x[i] = 0.0

    foo, cs = unroll_loop(foo, foo.find_loop("i"), rc=True)
    print(cs.block)
    assert "\n\n".join([str(v) for v in cs]) == golden
