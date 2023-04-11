from __future__ import annotations

import pytest

from exo import proc, ExoType
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


def test_basic_forwarding(golden):
    @proc
    def p():
        x: f32
        if True:
            x = 1.0
        if True:
            x = 2.0

    x1 = p.find("x = _ #0")
    x2 = p.find("x = _ #1")
    if1 = x1.parent()
    if2 = x2.parent()
    p = insert_pass(p, if1.before())
    p = fuse(p, if1, if2)
    assert str(p) == golden


def test_basic_forwarding2(golden):
    @proc
    def filter1D(ow: size, kw: size, x: f32[ow + kw - 1], y: f32[ow], w: f32[kw]):
        for o in seq(0, ow):
            sum: f32
            sum = 0.0
            for k in seq(0, kw):
                sum += x[o + k] * w[k]
            y[o] = sum

    filter1D = divide_loop(filter1D, "o", 4, ["outXo", "outXi"], tail="cut_and_guard")

    sum_c = filter1D.find("sum:_")

    filter1D = expand_dim(filter1D, sum_c, "4", "outXi")
    filter1D = lift_alloc(filter1D, sum_c)

    assert str(filter1D.forward(sum_c)) == golden


def test_basic_forwarding3(golden):
    @proc
    def filter1D(ow: size, kw: size, x: f32[ow + kw - 1], y: f32[ow], w: f32[kw]):
        for o in seq(0, ow):
            sum: f32
            sum = 0.0
            for k in seq(0, kw):
                sum += x[o + k] * w[k]
            y[o] = sum

    filter1D = divide_loop(filter1D, "o", 4, ["outXo", "outXi"], tail="cut_and_guard")

    sum_c = filter1D.find("sum:_")

    filter1D = expand_dim(filter1D, sum_c, "4", "outXi")
    filter1D = lift_alloc(filter1D, filter1D.forward(sum_c))

    assert str(filter1D.forward(sum_c)) == golden


def test_simplify_forwarding(golden):
    @proc
    def foo(n: size, m: size):
        x: R[n, 16 * (n + 1) - n * 16, (10 + 2) * m - m * 12 + 10]
        for i in seq(0, 4 * (n + 2) - n * 4 + n * 5):
            y: R[n + m]
            y[n * 4 - n * 4 + 1] = 0.0

    stmt = foo.find("y[_] = _")
    foo1 = simplify(foo)
    assert str(foo.find("y:_").shape()[0]._impl._node) == "n + m"
    assert str(foo1.forward(stmt)._impl._node) == golden


def test_type_and_shape_introspection():
    @proc
    def foo(n: size, m: index, flag: bool):
        assert m >= 0
        a: R[n + m]
        b: i32[n]
        c: i8[n]
        d: f32[n]
        e: f64[n]

    assert foo.find("a:_").type() == ExoType.R
    assert foo.find("b:_").type() == ExoType.I32
    assert foo.find("c:_").type() == ExoType.I8
    assert foo.find("d:_").type() == ExoType.F32
    assert foo.find("e:_").type() == ExoType.F64
    assert foo.args()[0].type() == ExoType.Size
    assert foo.args()[1].type() == ExoType.Index
    assert foo.args()[2].type() == ExoType.Bool
    assert str(foo.find("a:_").shape()[0]._impl._node) == "n + m"


def test_expand_dim_forwarding(golden):
    @proc
    def scal(n: size, alpha: R, x: [R][n]):
        for i in seq(0, n):
            x[i] = alpha * x[i]

    stmt = scal.find("x[_] = _")
    scal1 = divide_loop(scal, "for i in _:_", 8, ("io", "ii"), tail="cut")
    scal2 = bind_expr(scal1, [stmt.rhs().lhs()], "alphaReg")

    # seems to fail for some other scheduling ops besides expand_dim...
    # scal3 = lift_alloc(scal2, scal2.find("alphaReg: _"))
    scal3 = expand_dim(scal2, "alphaReg", "8", "ii")


def test_lift_alloc_forwarding():
    @proc
    def scal(n: size, alpha: R, x: [R][n]):
        for i in seq(0, n):
            x[i] = alpha * x[i]

    stmt = scal.find("x[_] = _")
    scal1 = divide_loop(scal, "for i in _:_", 8, ("io", "ii"), tail="cut")
    scal2 = bind_expr(scal1, [stmt.rhs().lhs()], "alphaReg")
    scal3 = expand_dim(scal2, "alphaReg", "8", "ii")
    scal4 = lift_alloc(scal3, "alphaReg")

    scal1.forward(stmt)
    scal2.forward(stmt)
    scal3.forward(stmt)
    scal4.forward(stmt)


def test_bind_expr_forwarding(golden):
    @proc
    def scal(n: size, alpha: R, x: [R][n]):
        for i in seq(0, n):
            x[i] = alpha * x[i]

    stmt = scal.find("x[_] = _")
    scal1 = divide_loop(scal, "for i in _:_", 8, ("io", "ii"), tail="cut")
    stmt2 = scal1.find("x[_] = _")
    scal2 = bind_expr(scal1, [stmt.rhs().lhs()], "alphaReg")
    assert str(scal2.forward(stmt)._impl.get_root()) == golden
    assert str(scal2.forward(stmt2)._impl.get_root()) == golden


def test_vectorize_forwarding(golden):
    @proc
    def scal(n: size, alpha: R, x: [R][n]):
        for i in seq(0, n):
            x[i] = alpha * x[i]

    stmt = scal.find("x[_] = _")
    scal1 = divide_loop(scal, "for i in _:_", 8, ("io", "ii"), tail="cut")
    scal2 = bind_expr(scal1, [stmt.rhs().lhs()], "alphaReg")
    scal3 = expand_dim(scal2, "alphaReg", "8", "ii")
    scal4 = lift_alloc(scal3, "alphaReg")
    scal5 = fission(scal4, scal4.find("alphaReg[_] = _").after())

    scal1.forward(stmt)
    scal2.forward(stmt)
    scal3.forward(stmt)
    scal4.forward(stmt)
    scal5.forward(stmt)

    assert str(scal5.forward(stmt)) == golden


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

    print(output)
    assert output == golden
