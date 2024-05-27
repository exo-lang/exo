from __future__ import annotations

import pytest

from exo import proc, ExoType
from exo.libs.memories import *
from exo.API_cursors import *

from exo.stdlib.inspection import *
from exo.stdlib.scheduling import *


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


def test_type_and_shape_introspection():
    @proc
    def foo(n: size, m: index, flag: bool):
        assert m >= 0
        a: R[n + m]
        b: i32[n]
        c: i8[n]
        d: f32[n]
        e: f64[2]

    assert foo.find("a:_").type() == ExoType.R
    assert foo.find("b:_").type() == ExoType.I32
    assert foo.find("c:_").type() == ExoType.I8
    assert foo.find("d:_").type() == ExoType.F32
    assert foo.find("e:_").type() == ExoType.F64
    assert foo.args()[0].type() == ExoType.Size
    assert foo.args()[1].type() == ExoType.Index
    assert foo.args()[2].type() == ExoType.Bool
    assert str(foo.find("a:_").shape()[0]._impl._node) == "n + m"
    assert foo.find("e : _").shape()[0].type() == ExoType.Int


def test_arg_cursor(golden):
    @proc
    def scal(n: size, alpha: R, x: [R][n, n]):
        for i in seq(0, n):
            x[i, i] = alpha * x[i, i]

    args = scal.args()

    output = ""
    for arg in args:
        output += f"{arg.name()}, {arg.is_tensor()}"
        if arg.is_tensor():
            for dim in arg.shape():
                output += f", {dim._impl._node}"
        output += "\n"

    assert output == golden


def test_argcursor_introspection():
    @proc
    def bar(n: size, x: f32[n], y: [f64][8], result: R):
        pass

    n_arg = bar.args()[0]
    assert isinstance(n_arg, ArgCursor)
    assert n_arg.name() == "n"
    with pytest.raises(AssertionError, match=""):
        mem = n_arg.mem()
    assert n_arg.is_tensor() == False
    with pytest.raises(AssertionError, match=""):
        shape = n_arg.shape()
    assert n_arg.type() == ExoType.Size

    x_arg = bar.args()[1]
    assert isinstance(x_arg, ArgCursor)
    assert x_arg.name() == "x"
    assert x_arg.mem() == DRAM
    assert x_arg.is_tensor() == True
    x_arg_shape = x_arg.shape()
    assert isinstance(x_arg_shape, ExprListCursor)
    assert len(x_arg_shape) == 1
    assert isinstance(x_arg_shape[0], ReadCursor)
    assert x_arg_shape[0].name() == "n"
    assert isinstance(x_arg_shape[0].idx(), ExprListCursor)
    assert len(x_arg_shape[0].idx()) == 0
    assert x_arg.type() == ExoType.F32

    y_arg = bar.args()[2]
    assert isinstance(y_arg, ArgCursor)
    assert y_arg.name() == "y"
    assert y_arg.mem() == DRAM
    assert y_arg.is_tensor() == True
    y_arg_shape = y_arg.shape()
    assert isinstance(y_arg_shape, ExprListCursor)
    assert len(y_arg_shape) == 1
    assert isinstance(y_arg_shape[0], LiteralCursor)
    assert y_arg_shape[0].value() == 8
    assert y_arg.type() == ExoType.F64

    result_arg = bar.args()[3]
    assert isinstance(result_arg, ArgCursor)
    assert result_arg.name() == "result"
    assert result_arg.mem() == DRAM
    assert result_arg.is_tensor() == False
    with pytest.raises(AssertionError, match=""):
        shape = result_arg.shape()
    assert result_arg.type() == ExoType.R


def test_match_depth(golden):
    @proc
    def foo(x: i8):
        for i in seq(0, 8):
            if i + 3 < -1:
                x = 0.0
                pass
        for i in seq(0, 2):
            x = 1.0

    c = match_depth(foo.find("x = 1.0"), foo.find_loop("i"))
    assert str(c) == golden


def test_match_depth_fail():
    @proc
    def foo(x: i8):
        for i in seq(0, 8):
            x = 1.0
        for j in seq(0, 2):
            x = 2.0

    @proc
    def bar():
        pass

    with pytest.raises(
        CursorNavigationError,
        match="cursor_to_match's parent is not an ancestor of cursor",
    ):
        match_depth(foo.find("x = _ #0"), foo.find("x = _ #1"))

    with pytest.raises(AssertionError, match="cursors originate from different procs"):
        match_depth(foo.find("x = _"), bar.find("pass"))


def test_get_enclosing_loop_by_name(golden):
    @proc
    def foo(x: i8):
        for i in seq(0, 5):
            for j in seq(0, 5):
                if i == 0:
                    x = 1.0

    c1 = get_enclosing_loop(foo, foo.find("x = _"))
    c2 = get_enclosing_loop_by_name(foo, foo.find("x = _"), "i")

    assert "\n\n".join([str(c) for c in [c1, c2]]) == golden


def test_get_enclosing_loop_by_name_fail():
    @proc
    def foo(x: i8):
        for i in seq(0, 8):
            x = 1.0
        for j in seq(0, 2):
            x = 2.0
        x = 3.0

    with pytest.raises(CursorNavigationError, match="no enclosing loop found"):
        get_enclosing_loop_by_name(foo, foo.find("x = _"), foo.find_loop("j"))


def test_is_ancestor_of_and_lca():
    @proc
    def foo():
        for i in seq(0, 10):
            for j in seq(0, 10):
                pass
            x: i8

    i_loop = foo.find_loop("i")
    j_loop = foo.find_loop("j")
    x_alloc = foo.find("x:_")
    pass_stmt = foo.find("pass")

    assert i_loop.is_ancestor_of(i_loop)

    assert i_loop.is_ancestor_of(j_loop)
    assert i_loop.is_ancestor_of(pass_stmt)
    assert i_loop.is_ancestor_of(x_alloc)

    assert not j_loop.is_ancestor_of(i_loop)
    assert j_loop.is_ancestor_of(pass_stmt)
    assert not j_loop.is_ancestor_of(x_alloc)

    assert not pass_stmt.is_ancestor_of(i_loop)
    assert not pass_stmt.is_ancestor_of(j_loop)

    assert not x_alloc.is_ancestor_of(i_loop)
    assert not x_alloc.is_ancestor_of(j_loop)

    assert get_lca(foo, x_alloc, pass_stmt) == i_loop
    assert get_lca(foo, pass_stmt, x_alloc) == i_loop
    assert get_lca(foo, i_loop, x_alloc) == i_loop
    assert get_lca(foo, x_alloc, i_loop) == i_loop


def test_cursor_find_loop():
    @proc
    def foo(n: size, x: i8[n]):
        for i in seq(0, n):
            pass
        if n > 1:
            for i in seq(0, n):
                x[i] = 0.0

    i_loop2 = foo.find("for i in _:_ #1")
    if_stmt = foo.find("if _: _ ")
    i_loop_alternative = if_stmt.find("for i in _: _")
    assert i_loop2 == i_loop_alternative
