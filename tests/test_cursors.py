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


def test_gap_forwarding(golden):
    @proc
    def p():
        x: f32
        if True:
            x = 1.0
        if True:
            x = 2.0

    if1 = p.find("x = _ #0").parent()
    if2 = p.find("x = _ #1").parent()
    x_alloc = p.find("x: _")
    p = fuse(p, if1, if2)
    p = insert_pass(p, if1.body()[0].after())
    p = insert_pass(p, if2.body()[0].after())
    p = insert_pass(p, x_alloc.after())
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


def test_simplify_predicates_forwarding():
    @proc
    def foo(n: size):
        assert n >= 1
        for i in seq(0, n):
            pass

    loop = foo.find_loop("i")
    foo = simplify(foo)
    loop = foo.forward(loop)
    assert loop == foo.find_loop("i")


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


def test_reorder_loops_forwarding(golden):
    @proc
    def foo():
        for i in seq(0, 4):
            for j in seq(0, 4):
                for k in seq(0, 4):
                    x: i8

    i_loop = foo.find("for i in _:_")
    j_loop = foo.find("for j in _:_")
    k_loop = foo.find("for k in _:_")
    foo = reorder_loops(foo, i_loop)
    foo = reorder_loops(foo, i_loop)
    foo = reorder_loops(foo, j_loop)
    foo = reorder_loops(foo, k_loop)
    assert str(foo) == golden


def test_split_write_forwarding(golden):
    @proc
    def foo(x: i32):
        x += 1 + 2
        x = 3 + 4

    stmt0 = foo.body()[0]
    stmt1 = foo.body()[1]
    body = foo.body()

    foo = split_write(foo, stmt0)
    with pytest.raises(InvalidCursorError, match=""):
        foo.forward(stmt0)
    assert foo.body()[:2] == foo.forward(stmt0.as_block())
    assert foo.body()[2] == foo.forward(stmt1)
    assert foo.body() == foo.forward(body)

    foo = split_write(foo, stmt1)
    with pytest.raises(InvalidCursorError, match=""):
        foo.forward(stmt0.rhs())

    assert foo.body()[:2] == foo.forward(stmt0.as_block())
    assert foo.body()[2:] == foo.forward(stmt1.as_block())
    assert foo.body() == foo.forward(body)


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

    assert str(scal5.forward(stmt)) == golden


def test_unroll_buffer_forwarding(golden):
    @proc
    def foo():
        src: i32[2]
        src[0] = 1.0
        src[1] = 1.0

    assn1 = foo.find("src[_] = _")
    const1 = assn1.rhs()
    assn2 = foo.find("src[_] = _ #1")
    foo1 = unroll_buffer(foo, "src: _", 0)

    tests = [foo1.forward(assn1), foo1.forward(assn2), foo1.forward(const1).parent()]

    assert "\n".join([str(test) for test in tests]) == golden


def test_forwarding_for_procs_with_identical_code():
    @proc
    def foo():
        x: f32[8] @ AVX2
        for i in seq(0, 8):
            x[i] += 1.0

    alloc_cursor = foo.find("x : _")
    foo = set_memory(foo, alloc_cursor, AVX2)
    loop_cursor = foo.find_loop("i")
    foo = expand_dim(foo, alloc_cursor, "1", "0")
    foo.forward(loop_cursor)


def test_delete_pass_forwarding():
    @proc
    def foo(x: R):
        for i in seq(0, 16):
            x = 1.0
            pass
            for j in seq(0, 2):
                pass
                pass
            pass
        x = 0.0

    i_loop = foo.body()[0]
    assign_1 = i_loop.body()[0]
    assign_0 = foo.body()[1]

    foo = delete_pass(foo)
    assert isinstance(foo.forward(i_loop), ForCursor)
    assert isinstance(foo.forward(assign_1), AssignCursor)
    assert isinstance(foo.forward(assign_0), AssignCursor)


def test_extract_subproc_forwarding():
    @proc
    def foo(N: size, M: size, K: size, x: R[N, K + M]):
        assert N >= 8
        x[0, 0] = 0.0
        for i in seq(0, 8):
            x[i, 0] += 2.0

    block = foo.body()
    foo, new = extract_subproc(foo, block, "fooooo")
    block = foo.forward(block)
    assert len(block) == 1
    assert isinstance(block[0], CallCursor)


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


def test_cut_loop_forwarding():
    @proc
    def foo(n: size):
        for i in seq(0, 10):
            pass

    loop_cursor = foo.find_loop("i")
    pass_cursor = loop_cursor.body()[0]
    foo = cut_loop(foo, loop_cursor, 3)
    loop_cursor = foo.forward(loop_cursor)
    second_loop = loop_cursor.next()
    pass_cursor = foo.forward(pass_cursor)

    assert isinstance(loop_cursor, ForCursor)
    assert isinstance(second_loop, ForCursor)
    assert isinstance(pass_cursor, PassCursor)
    assert isinstance(pass_cursor.parent(), ForCursor)
    assert pass_cursor.parent().hi().value() == 3


def test_shift_loop_forwarding():
    @proc
    def foo(x: f32[10]):
        for i in seq(0, 10):
            x[i] = 1.0

    loop_cursor = foo.find_loop("i")
    assign_cursor = loop_cursor.body()[0]
    foo = shift_loop(foo, loop_cursor, 3)
    loop_cursor = foo.forward(loop_cursor)
    assign_cursor = foo.forward(assign_cursor)

    assert isinstance(loop_cursor, ForCursor)
    assert isinstance(assign_cursor, AssignCursor)
    assert isinstance(assign_cursor.parent(), ForCursor)


def test_eliminate_dead_code_forwarding():
    @proc
    def foo():
        x: f32 @ DRAM
        for i in seq(0, 8):
            if i + 3 > -1:
                x = 0.0
                pass
            else:
                x += 1.0
                pass
                pass

    loop_cursor = foo.find_loop("i")
    if_cursor = loop_cursor.body()[0]
    if_true_stmt = if_cursor.body()[0]
    if_false_stmt = if_cursor.orelse()[0]
    foo = eliminate_dead_code(foo, "if _:_ #0")
    loop_cursor = foo.forward(loop_cursor)
    with pytest.raises(InvalidCursorError, match=""):
        if_cursor = foo.forward(if_cursor)
    if_true_stmt = foo.forward(if_true_stmt)
    with pytest.raises(InvalidCursorError, match=""):
        if_false_stmt = foo.forward(if_false_stmt)

    assert isinstance(loop_cursor, ForCursor)
    assert len(loop_cursor.body()) == 2
    assert isinstance(if_true_stmt, AssignCursor)
    assert isinstance(if_true_stmt.parent(), ForCursor)


def test_eliminate_dead_code_forwarding2():
    @proc
    def foo():
        x: f32 @ DRAM
        for i in seq(0, 8):
            if i + 3 < -1:
                x = 0.0
                pass
            else:
                x += 1.0
                pass
                pass

    loop_cursor = foo.find_loop("i")
    if_cursor = loop_cursor.body()[0]
    if_true_stmt = if_cursor.body()[0]
    if_false_stmt = if_cursor.orelse()[0]
    foo = eliminate_dead_code(foo, "if _:_ #0")
    loop_cursor = foo.forward(loop_cursor)
    with pytest.raises(InvalidCursorError, match=""):
        if_cursor = foo.forward(if_cursor)
    with pytest.raises(InvalidCursorError, match=""):
        if_true_stmt = foo.forward(if_true_stmt)
    if_false_stmt = foo.forward(if_false_stmt)

    assert isinstance(loop_cursor, ForCursor)
    assert len(loop_cursor.body()) == 3
    assert isinstance(if_false_stmt, ReduceCursor)
    assert isinstance(if_false_stmt.parent(), ForCursor)


def test_eliminate_dead_code_forwarding3():
    @proc
    def foo():
        x: f32 @ DRAM
        for i in seq(0, 8):
            if i + 3 > -1:
                x = 0.0
                pass

    loop_cursor = foo.find_loop("i")
    if_cursor = loop_cursor.body()[0]
    if_true_stmt = if_cursor.body()[0]
    foo = eliminate_dead_code(foo, "if _:_ #0")
    loop_cursor = foo.forward(loop_cursor)
    with pytest.raises(InvalidCursorError, match=""):
        if_cursor = foo.forward(if_cursor)
    if_true_stmt = foo.forward(if_true_stmt)

    assert isinstance(loop_cursor, ForCursor)
    assert len(loop_cursor.body()) == 2
    assert isinstance(if_true_stmt, AssignCursor)
    assert isinstance(if_true_stmt.parent(), ForCursor)


def test_eliminate_dead_code_forwarding4():
    @proc
    def foo():
        x: f32 @ DRAM
        for i in seq(0, 8):
            if i + 3 < -1:
                x = 0.0
                pass

    loop_cursor = foo.find_loop("i")
    if_cursor = loop_cursor.body()[0]
    if_true_stmt = if_cursor.body()[0]
    foo = eliminate_dead_code(foo, "if _:_ #0")
    loop_cursor = foo.forward(loop_cursor)
    with pytest.raises(InvalidCursorError, match=""):
        if_cursor = foo.forward(if_cursor)
    with pytest.raises(InvalidCursorError, match=""):
        if_true_stmt = foo.forward(if_true_stmt)

    assert isinstance(loop_cursor, ForCursor)
    assert len(loop_cursor.body()) == 1
    assert isinstance(loop_cursor.body()[0], PassCursor)


def test_eliminate_dead_code_forwarding5():
    @proc
    def foo(n: size):
        for i in seq(0, n):
            x: f32

    foo = specialize(foo, foo.find_loop("i"), "0 < n")
    else_loop = foo.find_loop("i #1")
    else_loop_alloc = else_loop.body()[0]
    foo = eliminate_dead_code(foo, else_loop)

    with pytest.raises(InvalidCursorError, match=""):
        foo.forward(else_loop)
    with pytest.raises(InvalidCursorError, match=""):
        foo.forward(else_loop_alloc)


def test_specialize_forwarding():
    @proc
    def foo(n: size, a: f32):
        a = 1.0
        a = 2.0

    body = foo.body()
    foo = specialize(foo, body, ["n > 0"])
    assert foo.forward(body) == foo.body()


def test_match_parent(golden):
    @proc
    def foo(x: i8):
        for i in seq(0, 8):
            if i + 3 < -1:
                x = 0.0
                pass
        for i in seq(0, 2):
            x = 1.0

    c1, c2 = match_parent(foo.find("x = 1.0"), foo.find_loop("i"))
    assert "\n\n".join([str(c) for c in [c1, c2]]) == golden


def test_match_parent_2(golden):
    @proc
    def foo(x: i8):
        for i in seq(0, 8):
            x = 1.0
        for j in seq(0, 2):
            x = 2.0

    c1, c2 = match_parent(foo.find("x = _ #0"), foo.find("x = _ #1"))
    assert "\n\n".join([str(c) for c in [c1, c2]]) == golden


def test_match_parent_fail():
    @proc
    def foo():
        pass

    @proc
    def bar():
        pass

    with pytest.raises(AssertionError, match="cursors originate from different procs"):
        match_parent(foo.find("pass"), bar.find("pass"))


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
