from __future__ import annotations

import pytest

from exo import proc
from exo.stdlib.scheduling import *
from exo.platforms.x86 import *


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


def test_divide_loop_forwarding():
    @proc
    def foo():
        for i in seq(0, 10):
            pass
            pass

    i_loop = foo.find_loop("i")
    body = i_loop.body()

    # Loop always forwards to the outermost loop of the main loop nest
    # Loop body forwards to the main loop nest's body
    foo_guard = divide_loop(foo, "i", 5, ["io", "ii"], tail="guard")
    assert foo_guard.forward(i_loop) == foo_guard.find_loop("io")
    assert foo_guard.forward(body).parent().parent() == foo_guard.find_loop("ii")

    foo_cut = divide_loop(foo, "i", 5, ["io", "ii"], tail="cut")
    assert foo_cut.forward(i_loop) == foo_cut.find_loop("io")
    assert foo_cut.forward(body).parent() == foo_cut.find_loop("ii")

    foo_cg = divide_loop(foo, "i", 5, ["io", "ii"], tail="cut_and_guard")
    assert foo_cg.forward(i_loop) == foo_cg.find_loop("io")
    assert foo_cg.forward(body).parent() == foo_cg.find_loop("ii")


def test_mult_loop_forwarding():
    @proc
    def foo():
        for i in seq(0, 10):
            for j in seq(0, 30):
                pass
                pass

    i_loop = foo.find_loop("i")
    j_loop = foo.find_loop("j")
    body = j_loop.body()
    foo = mult_loops(foo, "i j", "ij")

    assert foo.forward(body) == foo.find_loop("ij").body()


def test_fuse_loops_forwarding():
    @proc
    def foo(n: size, x: R[n]):
        assert n > 3
        y: R[n]
        for i in seq(3, n):
            y[i] = x[i]
        for j in seq(3, n):
            x[j] = y[j] + 1.0

    loop1 = foo.find_loop("i")
    body1 = loop1.body()
    loop2 = foo.find_loop("j")
    body2 = loop2.body()

    both_loops = loop1.expand(0, 1)

    foo = fuse(foo, loop1, loop2)

    # Loop bodies are consecutive
    assert foo.forward(body1)[-1].next() == foo.forward(body2)[0]

    # Block containing both loops forwards to block containing fused loop.
    assert foo.forward(both_loops) == foo.find_loop("i").as_block()


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
