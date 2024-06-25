from __future__ import annotations

import pytest

from exo import proc, SchedulingError, Procedure
from exo.LoopIR import LoopIR, T
from exo.LoopIR_pprint import _print_cursor
from exo.internal_cursors import (
    Cursor,
    Block,
    InvalidCursorError,
    Node,
)
from exo.pattern_match import match_pattern
from exo.prelude import Sym
from exo.syntax import size, f32


def _find_cursors(ctx, pattern):
    if isinstance(ctx, Procedure):
        ctx = ctx._root()

    if isinstance(ctx, LoopIR.proc):
        ctx = Cursor.create(ctx)

    cursors = match_pattern(ctx, pattern, call_depth=1)
    assert isinstance(cursors, list)
    if not cursors:
        raise SchedulingError("failed to find matches", pattern=pattern)
    return cursors


def _find_stmt(ctx, pattern):
    curs = _find_cursors(ctx, pattern)
    assert len(curs) == 1
    curs = curs[0]
    if len(curs) != 1:
        raise SchedulingError(
            "pattern did not match a single statement", pattern=pattern
        )
    return curs[0]


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


@pytest.fixture(scope="session")
def proc_baz():
    @proc
    def baz(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1
                for k in seq(0, n):
                    pass
                    pass

    yield baz


def test_get_root(proc_foo):
    cursor = proc_foo._root()
    assert cursor._node is proc_foo.INTERNAL_proc()


def test_find_cursor(proc_foo):
    c = _find_cursors(proc_foo, "for j in _:_")
    assert len(c) == 1
    c = c[0]  # One match
    c = c[0]  # First/only node in the block

    assert c._node is proc_foo.INTERNAL_proc().body[0].body[0]


def test_find_hole_in_middle(proc_bar):
    c_body_1_5 = _find_cursors(proc_bar, "x = 1.0 ; _ ; x = 4.0")[0]
    for_j = _find_stmt(proc_bar, "for j in _: _")
    assert c_body_1_5 == for_j.body()[1:5]


def test_gap_insert_pass(proc_foo, golden):
    c = _find_stmt(proc_foo, "x = 0.0")
    assn = c._node
    g = c.after()
    foo2, _ = g._insert([LoopIR.Pass(assn.srcinfo)])
    assert str(foo2) == golden


def test_insert_root_front(proc_foo, golden):
    c = proc_foo._root()
    foo2, _ = c.body().before()._insert([LoopIR.Pass(c._node.srcinfo)])
    assert str(foo2) == golden


def test_insert_root_end(proc_foo, golden):
    c = proc_foo._root()
    foo2, _ = c.body().after()._insert([LoopIR.Pass(c._node.srcinfo)])
    assert str(foo2) == golden


def test_block_gaps(proc_bar):
    c = _find_stmt(proc_bar, "for j in _: _")

    body = c.body()
    assert len(body) == 6
    subset = body[1:4]
    assert len(subset) == 3

    cx1 = _find_stmt(proc_bar, "x = 1.0")
    cx3 = _find_stmt(proc_bar, "x = 3.0")
    assert subset[0] == cx1
    assert subset[2] == cx3

    assert subset.before() == cx1.before()
    assert subset.after() == cx3.after()


def test_block_sequence_interface(proc_bar):
    for_j_body = _find_stmt(proc_bar, "for j in _: _").body()
    assert for_j_body[:] is not for_j_body
    assert for_j_body[:] == for_j_body
    assert len(list(for_j_body)) == 6

    with pytest.raises(IndexError, match="block cursors must be contiguous"):
        # noinspection PyStatementEffect
        # Sequence's __getitem__ should throw here, so this does have a side effect
        for_j_body[::2]


def test_block_delete(proc_bar, golden):
    c = _find_stmt(proc_bar, "for j in _: _")
    stmts = c.body()[1:4]

    bar2, _ = stmts._delete()
    assert str(bar2) == golden


def test_block_replace(proc_bar, golden):
    c = _find_stmt(proc_bar, "for j in _: _")
    stmts = c.body()[1:4]

    bar2, _ = stmts._replace([LoopIR.Pass(c._node.srcinfo)])
    assert str(bar2) == golden


def test_block_delete_whole_block(proc_bar, golden):
    c = _find_stmt(proc_bar, "for j in _: _")
    bar2, _ = c.body()._delete()
    assert str(bar2) == golden


def test_node_replace(proc_bar, golden):
    c = _find_stmt(proc_bar, "x = 3.0")
    assert isinstance(c, Node)

    bar2, _ = c._replace([LoopIR.Pass(c._node.srcinfo)])
    assert str(bar2) == golden


def test_cursor_move(proc_foo):
    c = _find_stmt(proc_foo, "for j in _:_")

    c_list = c.body()  # list of j's body
    assert isinstance(c_list, Block)
    assert len(c_list) == 4
    assert c_list.parent() == c

    c1 = c_list[0]  # x : f32
    assert c1.parent() == c
    assert c1._node is proc_foo.INTERNAL_proc().body[0].body[0].body[0]

    c2 = c1.next()  # x = 0.0
    assert c2.parent() == c
    assert c2._node is proc_foo.INTERNAL_proc().body[0].body[0].body[1]

    c3 = c1.next(2)  # y : f32
    assert c3.parent() == c
    assert c3._node is proc_foo.INTERNAL_proc().body[0].body[0].body[2]

    _c2_ = c3.prev()
    assert c2 == _c2_
    _c1_ = c3.prev(2)
    assert c1 == _c1_


def test_cursor_move_invalid(proc_foo):
    # Edge cases near the root
    c = proc_foo._root()
    with pytest.raises(InvalidCursorError, match="cannot move root cursor"):
        c.next()

    with pytest.raises(InvalidCursorError, match="cursor does not have a parent"):
        c.parent()

    # Edge cases near expressions
    c = _find_cursors(proc_foo, "m")[0]
    with pytest.raises(InvalidCursorError, match="cursor is not inside block"):
        c.next()

    # Edge cases near first statement in block
    c = _find_stmt(proc_foo, "x: f32")
    with pytest.raises(InvalidCursorError, match="cursor is out of range"):
        c.prev()

    c.before()  # ok
    with pytest.raises(InvalidCursorError, match="cursor is out of range"):
        c.prev()

    # Edge cases near last statement in block
    c = _find_stmt(proc_foo, "y = 1.1")

    c.after()  # ok
    with pytest.raises(InvalidCursorError, match="cursor is out of range"):
        c.next()


def test_cursor_gap(proc_foo):
    # for i in seq(0, n):
    #    for j in seq(0, m):
    #        x: f32   <- x_alloc
    #                 <- g1
    #        x = 0.0  <- x_assn
    #        y: f32
    #                 <- g2
    #        y = 1.1  <- y_assn
    for_j = _find_stmt(proc_foo, "for j in _:_")
    x_alloc = _find_stmt(proc_foo, "x: f32")
    x_assn = _find_stmt(proc_foo, "x = 0.0")
    y_assn = _find_stmt(proc_foo, "y = 1.1")

    assert str(x_alloc._node) == "x: f32 @ DRAM"

    g1 = x_alloc.after()
    assert g1.anchor() == x_alloc
    assert g1.parent() == for_j

    g2 = g1.anchor().next(2).after()
    assert g2 == x_assn.next().after()
    assert g2.anchor().prev() == x_assn
    assert g2.anchor().next() == y_assn
    assert g2.parent() == for_j


def test_cursor_replace_expr(proc_foo, golden):
    c = _find_cursors(proc_foo, "m")[0]
    foo2, _ = c._replace(LoopIR.Const(42, T.size, c._node.srcinfo))
    assert str(foo2) == golden


def test_cursor_cannot_convert_expr_to_block(proc_foo):
    c = _find_cursors(proc_foo, "m")[0]
    with pytest.raises(InvalidCursorError, match="node is not inside a block"):
        c.as_block()


def test_cursor_replace_expr_deep(golden):
    @proc
    def example():
        x: f32
        x = 1.0 * (2.0 + 3.0)
        # x = 1.0 * (4.0 + 3.0)

    c: Node = _find_cursors(example, "2.0")[0]
    four = LoopIR.Const(4.0, T.f32, c._node.srcinfo)

    example_new, _ = c._replace(four)
    assert str(example_new) == golden


def test_cursor_forward_expr_deep():
    @proc
    def example():
        x: f32
        x = 1.0 * (2.0 + 3.0)
        # x = 1.0 * (4.0 + 3.0)

    c2: Node = _find_cursors(example, "2.0")[0]
    four = LoopIR.Const(4.0, T.f32, c2._node.srcinfo)

    example_new, fwd = c2._replace(four)
    assert fwd(c2) == _find_cursors(example_new, "4.0")[0]

    assert fwd(_find_stmt(example, "x = _")) == _find_stmt(example_new, "x = _")
    assert (
        fwd(_find_cursors(example, "_ + _")[0])
        == _find_cursors(example_new, "_ + _")[0]
    )
    assert fwd(_find_cursors(example, "1.0")[0]) == _find_cursors(example_new, "1.0")[0]
    assert fwd(_find_cursors(example, "3.0")[0]) == _find_cursors(example_new, "3.0")[0]


def test_cursor_loop_bound(proc_foo):
    c_for_i = proc_foo._root().body()[0]
    c_bound = c_for_i._child_node("hi")
    assert isinstance(c_bound._node, LoopIR.Read)


def test_cursor_invalid_child(proc_foo):
    c = proc_foo._root()

    # Quick sanity check
    assert c._child_node("body", 0) == c.body()[0]

    with pytest.raises(AttributeError, match="has no attribute '_invalid'"):
        c._child_node("_invalid", None)

    with pytest.raises(ValueError, match="must index into block attribute"):
        c._child_node("body", None)

    with pytest.raises(InvalidCursorError, match="cursor is out of range"):
        c._child_node("body", 42)


def test_insert_forward_orelse():
    @proc
    def example_old():
        x: f32
        if 1 < 2:
            x = 1.0
        else:
            x = 2.0

    x1_old = _find_stmt(example_old, "x = 1.0")
    x2_old = _find_stmt(example_old, "x = 2.0")
    gap = x1_old.after()

    stmt = [LoopIR.Pass(x1_old._node.srcinfo)]

    example_new, fwd = gap._insert(stmt)
    x2_new = _find_stmt(example_new, "x = 2.0")

    assert fwd(x2_old) == x2_new


def test_double_insert_forwarding(golden):
    @proc
    def proc_s1():
        x: f32
        if 1 < 2:
            x = 1.0
            # x = 2.0  (added in s2)
        else:
            x = 3.0
            # x = 4.0  (added in s3)

    x1_s1 = _find_stmt(proc_s1, "x = 1.0")
    x3_s1 = _find_stmt(proc_s1, "x = 3.0")

    x2_stmt = x1_s1._node.update(rhs=LoopIR.Const(2.0, T.f32, x1_s1._node.srcinfo))
    x4_stmt = x1_s1._node.update(rhs=LoopIR.Const(4.0, T.f32, x1_s1._node.srcinfo))

    proc_s2, fwd_12 = x1_s1.after()._insert([x2_stmt])
    proc_s3, fwd_23 = fwd_12(x3_s1).after()._insert([x4_stmt])
    assert str(proc_s3) == golden

    def fwd_13(cur):
        return fwd_23(fwd_12(cur))

    x1_s3 = _find_stmt(proc_s3, "x = 1.0")
    assert fwd_13(x1_s1) == x1_s3

    x3_s3 = _find_stmt(proc_s3, "x = 3.0")
    assert fwd_13(x3_s1) == x3_s3

    if_pat = "if _: _\nelse: _"
    assert fwd_13(_find_stmt(proc_s1, if_pat)) == _find_stmt(proc_s3, if_pat)

    with pytest.raises(InvalidCursorError, match="cannot forward from unknown root"):
        fwd_23(x1_s1)


@pytest.mark.parametrize(
    "old, new",
    [
        (0, 0),
        (1, 1),
        (2, None),
        (3, None),
        (4, 2),
        (5, 3),
    ],
)
def test_delete_forward_node(proc_bar, old, new):
    for_j = _find_stmt(proc_bar, "for j in _: _").body()
    bar_new, fwd = for_j[2:4]._delete()
    for_j_new = _find_stmt(bar_new, "for j in _: _").body()

    if new is None:
        with pytest.raises(InvalidCursorError, match="node no longer exists"):
            fwd(for_j[old])
    else:
        assert fwd(for_j[old]) == for_j_new[new]


@pytest.mark.parametrize(
    "old, new",
    [
        ("x = 0.0", "x = 0.0"),
        ("x = 1.0", None),
        ("x = 2.0", None),
        ("x = 3.0", None),
        ("x = 4.0", "x = 4.0"),
        ("x = 5.0", "x = 5.0"),
    ],
)
def test_block_replace_forward_node(proc_bar, old, new):
    for_j = _find_stmt(proc_bar, "for j in _: _").body()
    bar_new, fwd = for_j[1:4]._replace(
        [
            LoopIR.Pass(for_j.parent()._node.srcinfo),
            LoopIR.Pass(for_j.parent()._node.srcinfo),
        ]
    )

    old_c = _find_stmt(proc_bar, old)

    if new is None:
        with pytest.raises(InvalidCursorError, match="node no longer exists"):
            fwd(old_c)
    else:
        bar_new = Cursor.create(bar_new)
        assert fwd(old_c) == match_pattern(bar_new, new)[0][0]


def test_cursor_pretty_print_nodes(proc_bar, golden):
    output = []

    ir = proc_bar._loopir_proc

    root = Cursor.create(ir)
    output.append(_print_cursor(root))

    c = _find_stmt(ir, "for i in _: _")
    output.append(_print_cursor(c))

    c = _find_stmt(ir, "for j in _: _")
    output.append(_print_cursor(c))

    c = _find_stmt(ir, "x = 0.0")
    output.append(_print_cursor(c))

    c = _find_stmt(ir, "x = 2.0")
    output.append(_print_cursor(c))

    assert "\n\n".join(output) == golden


def test_cursor_pretty_print_gaps(proc_bar, golden):
    output = []

    c = _find_stmt(proc_bar, "x: f32").before()
    output.append(_print_cursor(c))

    c = _find_stmt(proc_bar, "for i in _: _").before()
    output.append(_print_cursor(c))

    c = _find_stmt(proc_bar, "for j in _: _").before()
    output.append(_print_cursor(c))

    c = _find_stmt(proc_bar, "x = 0.0").before()
    output.append(_print_cursor(c))

    c = _find_stmt(proc_bar, "x = 2.0").before()
    output.append(_print_cursor(c))

    c = _find_stmt(proc_bar, "x = 5.0").after()
    output.append(_print_cursor(c))

    assert "\n\n".join(output) == golden


def test_cursor_pretty_print_blocks(proc_bar, golden):
    output = []

    c = _find_stmt(proc_bar, "for j in _: _").as_block()
    output.append(_print_cursor(c))

    c = _find_cursors(proc_bar, "x = 1.0; _; x = 3.0")[0]
    output.append(_print_cursor(c))

    c = _find_cursors(proc_bar, "x = 0.0; x = 1.0")[0]
    output.append(_print_cursor(c))

    c = _find_cursors(proc_bar, "x = 4.0; x = 5.0")[0]
    output.append(_print_cursor(c))

    c = _find_cursors(proc_bar, "x = 0.0; _; x = 5.0")[0]
    output.append(_print_cursor(c))

    assert "\n\n".join(output) == golden


def test_move_block(proc_bar, golden):
    c = _find_cursors(proc_bar, "x = 1.0 ; x = 2.0")[0]

    # Movement within block
    p0, _ = c._move(c[0].prev().before())
    p1, _ = c._move(c.before())
    p2, _ = c._move(c.after())
    p3, _ = c._move(c[-1].next().after())
    p4, _ = c._move(c[-1].next(2).after())
    p5, _ = c._move(c[-1].next(3).after())

    assert str(p1) == str(p2), "Both before and after should keep block in place."

    # Movement upward
    pu0, _ = c._move(c.parent().before())
    pu1, _ = c._move(c.parent().after())
    pu2, _ = c._move(c.parent().parent().before())
    pu3, _ = c._move(c.parent().parent().prev().before())
    pu4, _ = c._move(c.parent().parent().after())

    # Movement downward (abbreviated)
    c2 = _find_cursors(proc_bar, "x: _")[0]
    pd0, _ = c2._move(c[0].prev().before())
    pd1, _ = c2._move(c.before())
    pd2, _ = c2._move(c2[-1].next().after())

    # Move out a whole loop (needs to insert pass)
    c3 = _find_cursors(proc_bar, "for j in _: _")[0]
    pl0, _ = c3._move(c3.parent().after())

    all_tests = [p0, p1, p2, p3, p4, p5, pu0, pu1, pu2, pu3, pu4, pd0, pd1, pd2, pl0]
    actual = "\n".join(str(p) for p in all_tests)
    assert actual == golden


def _debug_print_forwarding(x, fwd):
    fx = fwd(x)
    print(x._path)
    print(_print_cursor(x))
    print(x._root)
    print("-----------")
    print(fx._path)
    print(_print_cursor(fx))
    print(fx._root)
    print()
    print()


def test_move_block_forwarding(proc_bar, golden):
    c = _find_cursors(proc_bar, "x = 1.0 ; x = 2.0")[0]

    x_orig = []
    for i in range(6):
        x_orig.append(_find_stmt(proc_bar, f"x = {i}.0"))

    def _test_fwd(fwd):
        for x in x_orig:
            # _debug_print_forwarding(x, fwd)
            assert str(fwd(x)._node) == str(x._node)

    # Movement within block
    _, fwd0 = c._move(c[0].prev().before())
    _test_fwd(fwd0)
    _, fwd1 = c._move(c.before())
    _test_fwd(fwd1)
    _, fwd2 = c._move(c.after())
    _test_fwd(fwd2)
    _, fwd3 = c._move(c[-1].next().after())
    _test_fwd(fwd3)
    _, fwd4 = c._move(c[-1].next(2).after())
    _test_fwd(fwd4)
    _, fwd5 = c._move(c[-1].next(3).after())
    _test_fwd(fwd5)

    # Movement upward
    _, fwd6 = c._move(c.parent().before())
    _test_fwd(fwd6)
    _, fwd7 = c._move(c.parent().after())
    _test_fwd(fwd7)
    _, fwd8 = c._move(c.parent().parent().before())
    _test_fwd(fwd8)
    _, fwd9 = c._move(c.parent().parent().prev().before())
    _test_fwd(fwd9)
    _, fwd10 = c._move(c.parent().parent().after())
    _test_fwd(fwd10)

    # Movement downward (abbreviated)
    c2 = _find_cursors(proc_bar, "x: _")[0]
    _, fwd11 = c2._move(c[0].prev().before())
    _test_fwd(fwd11)
    _, fwd12 = c2._move(c.before())
    _test_fwd(fwd12)
    _, fwd13 = c2._move(c2[-1].next().after())
    _test_fwd(fwd13)

    # Move out a whole loop
    c3 = _find_cursors(proc_bar, "for j in _: _")[0]
    _, fwd14 = c3._move(c3.parent().after())
    _test_fwd(fwd14)


def test_wrap_block(proc_bar, golden):
    k = Sym("k")

    def wrapper(body):
        src = body[0].srcinfo
        zero = LoopIR.Const(0, T.index, src)
        eight = LoopIR.Const(8, T.index, src)
        return LoopIR.For(k, zero, eight, body, LoopIR.Seq(), src)

    procs = []
    for i in range(0, 6):
        for j in range(i + 1, 6):
            c = _find_cursors(proc_bar, f"x = {i}.0 ; _ ; x = {j}.0")[0]
            p, _ = c._wrap(wrapper, "body")
            procs.append(p)

    assert "\n".join(map(str, procs)) == golden


def test_wrap_block_forward(proc_bar):
    x_orig = []
    for i in range(6):
        x_orig.append(_find_stmt(proc_bar, f"x = {i}.0"))

    def _test_fwd(fwd):
        for x in x_orig:
            # _debug_print_forwarding(x, fwd)
            assert str(fwd(x)._node) == str(x._node)

    def wrapper(orelse):
        src = orelse[0].srcinfo
        true = LoopIR.Const(True, T.bool, src)
        return LoopIR.If(true, [LoopIR.Pass(src)], orelse, src)

    for i in range(0, 6):
        for j in range(i + 1, 6):
            c = _find_cursors(proc_bar, f"x = {i}.0 ; _ ; x = {j}.0")[0]
            _, fwd = c._wrap(wrapper, "orelse")
            _test_fwd(fwd)


def test_move_forward_diff_scopes_1(golden):
    @proc
    def foo():
        x: i8
        y: i8
        z: i8
        for i in seq(0, 4):
            pass

    alloc_xy = foo.find("x: _")._impl.as_block().expand(0, 1)
    pass_c = foo.find("pass")._impl
    ir, fwd = alloc_xy._move(pass_c.before())
    assert fwd(alloc_xy[0])._path == [("body", 1), ("body", 0)]
    assert str(ir) == golden

    @proc
    def foo():
        z: i8
        for i in seq(0, 4):
            pass
        x: i8
        y: i8

    alloc_xy = foo.find("x: _")._impl.as_block().expand(0, 1)
    pass_c = foo.find("pass")._impl
    ir, fwd = alloc_xy._move(pass_c.before())
    assert fwd(alloc_xy[0])._path == [("body", 1), ("body", 0)]
    assert str(ir) == golden


def test_move_forward_diff_scopes_2():
    @proc
    def foo():
        pass
        for i in seq(0, 4):
            x: i8
            y: i8
            z: i8

    alloc_xy = foo.find("x: _")._impl.as_block().expand(0, 1)
    pass_c = foo.find("pass")._impl
    ir, fwd = alloc_xy._move(pass_c.after())
    assert fwd(alloc_xy[0])._path == [("body", 1)]
    assert fwd(alloc_xy[1])._path == [("body", 2)]

    @proc
    def foo():
        for i in seq(0, 4):
            x: i8
            y: i8
            z: i8
        pass

    alloc_xy = foo.find("x: _")._impl.as_block().expand(0, 1)
    pass_c = foo.find("pass")._impl
    ir, fwd = alloc_xy._move(pass_c.after())
    assert fwd(alloc_xy[0])._path == [("body", 2)]
    assert fwd(alloc_xy[1])._path == [("body", 3)]


def test_move_forward_if_orelse(golden):
    @proc
    def foo():
        if True:
            x: i8
        else:
            y: i8

    alloc_x = foo.find("x: _")._impl
    alloc_y = foo.find("y: _")._impl
    ir, fwd = alloc_x._move(alloc_y.after())
    assert fwd(alloc_x)._path == [("body", 0), ("orelse", 1)]
    assert str(ir) == golden


def test_insert_forwarding_for_blocks(proc_baz, golden):
    c = _find_stmt(proc_baz, "x = 0.0")

    b_above = c.parent()
    b_edit = b_above.body()
    b_below = b_edit[-1].body()
    b_before = b_edit[:2]
    b_after = b_edit[2:]

    _, fwd = c.after()._insert([LoopIR.Pass(c._node.srcinfo)])

    output = []
    output.append(_print_cursor(fwd(b_above)))  # above edit level
    output.append(_print_cursor(fwd(b_below)))  # below edit level
    output.append(_print_cursor(fwd(b_before)))  # same edit level, disjoint
    output.append(_print_cursor(fwd(b_after)))  # same dit level, disjoint

    # Blocks containing the insertion point also work intuitively.
    output.append(_print_cursor(fwd(b_edit)))

    assert "\n\n".join(output) == golden


def test_delete_forwarding_for_blocks(proc_baz, golden):
    c = _find_cursors(proc_baz, "x = 0.0; y: _")[0]

    b_above = c.parent()
    b_edit = b_above.body()
    b_below = b_edit[-1].body()
    b_before = b_edit[:1]
    b_after = b_edit[3:]

    b_aligned_with_deletion_on_left = b_edit[1:]
    b_aligned_with_deletion_on_right = b_edit[:3]
    b_with_endpoint_in_deletion = b_edit[:2]

    _, fwd = c._delete()

    output = []
    output.append(_print_cursor(fwd(b_above)))  # above edit level
    output.append(_print_cursor(fwd(b_below)))  # below edit level
    output.append(_print_cursor(fwd(b_before)))  # same edit level, disjoint
    output.append(_print_cursor(fwd(b_after)))  # same edit level, disjoint

    # Blocks entirely containing the deletion block also work intuitively.
    output.append(_print_cursor(fwd(b_edit)))
    output.append(_print_cursor(fwd(b_aligned_with_deletion_on_left)))
    output.append(_print_cursor(fwd(b_aligned_with_deletion_on_right)))

    # Blocks partially containing the deletion block don't work.
    with pytest.raises(InvalidCursorError, match=r"block no longer exists"):
        fwd(b_with_endpoint_in_deletion)

    assert "\n\n".join(output) == golden


def test_delete_forwarding_for_blocks_fail():
    @proc
    def foo():
        for i in seq(0, 4):
            x: i8
            y: i8
            z: i8

    body = _find_stmt(foo, "for i in _: _").body()
    _, fwd = body.parent()._delete()

    with pytest.raises(
        InvalidCursorError, match=r"block no longer exists \(parent deleted\)"
    ):
        fwd(body)


def test_block_replace_forwarding_for_blocks(proc_baz, golden):
    c = _find_cursors(proc_baz, "x = 0.0; _; y = 1.1")[0]

    b_above = c.parent()
    b_edit = b_above.body()
    b_below = b_edit[-1].body()
    b_before = b_edit[:1]
    b_after = b_edit[4:]
    b_with_endpoint_in_replace = b_edit[:2]
    b_inside = b_edit[2:4]

    # replace 1:4 with two pass stmts
    pass_ir = LoopIR.Pass(c[0]._node.srcinfo)
    _, fwd = c._replace([pass_ir, pass_ir])

    output = []
    output.append(_print_cursor(fwd(b_above)))  # above edit level
    output.append(_print_cursor(fwd(b_below)))  # below edit level
    output.append(_print_cursor(fwd(b_before)))  # same edit level, disjoint
    output.append(_print_cursor(fwd(b_after)))  # same edit level, disjoint

    # Blocks containing the entire replace block work intuitively.
    output.append(_print_cursor(fwd(b_edit)))
    output.append(_print_cursor(fwd(c)))

    # Blocks partially containing the replace block are invalidated.
    with pytest.raises(InvalidCursorError, match=r"block no longer exists"):
        fwd(b_with_endpoint_in_replace)

    # Blocks entirely within the replace block are invalidated
    with pytest.raises(InvalidCursorError, match=r"block no longer exists"):
        fwd(b_inside)

    assert "\n\n".join(output) == golden


def test_node_replace_forwarding(proc_baz):
    c = _find_cursors(proc_baz, "1.1")[0]
    _, fwd = c._replace(LoopIR.Const(42.0, T.f32, c._node.srcinfo))
    assert fwd(c)._node.val == 42.0


def test_block_replace_forwarding_stmt_to_stmt(proc_baz):
    c = _find_stmt(proc_baz, "x = 0")
    _, fwd = c._replace(LoopIR.Pass(c._node.srcinfo))
    assert isinstance(fwd(c)._node, LoopIR.Pass)


def test_wrap_forwarding_for_blocks(proc_baz, golden):
    c = _find_cursors(proc_baz, "y: _; y = 1.1")[0]

    b_above = c.parent()
    b_edit = b_above.body()
    b_below = b_edit[-1].body()

    b_before = b_edit[:2]
    b_after = b_edit[4:]
    b_with_endpoint_in_replace = b_edit[:3]

    # wrap 2:4 with a for loop
    k = Sym("k")

    def wrapper(body):
        src = body[0].srcinfo
        zero = LoopIR.Const(0, T.index, src)
        eight = LoopIR.Const(8, T.index, src)
        return LoopIR.For(k, zero, eight, body, LoopIR.Seq(), src)

    _, fwd = c._wrap(wrapper, "body")

    output = []
    output.append(_print_cursor(fwd(b_above)))  # above edit level
    output.append(_print_cursor(fwd(b_below)))  # below edit level
    output.append(_print_cursor(fwd(b_before)))  # same edit level, disjoint
    output.append(_print_cursor(fwd(b_after)))  # same edit level, disjoint

    # Blocks containing the entire wrap block work intuitively.
    output.append(_print_cursor(fwd(b_edit)))

    # Blocks within the wrap block work intuitively
    output.append(_print_cursor(fwd(c)))

    # Blocks partially containing the wrap block don't work.
    with pytest.raises(InvalidCursorError, match=r"block no longer exists"):
        output.append(_print_cursor(fwd(b_with_endpoint_in_replace)))

    assert "\n\n".join(output) == golden


def test_move_forwarding_for_blocks(proc_baz, golden):
    c = _find_cursors(proc_baz, "y: _; y = 1.1")[0]
    g = c[0].prev().before()

    b_above = c.parent()
    b_edit = b_above.body()
    b_below = b_edit[-1].body()

    b_with_gap = b_edit[:2]
    b_without_gap = b_edit[4:]
    b_with_endpoint_in_moved_block = b_edit[:3]

    _, fwd = c._move(g)

    output = []
    output.append(_print_cursor(fwd(b_above)))  # above edit level
    output.append(_print_cursor(fwd(b_below)))  # below edit level
    output.append(_print_cursor(fwd(b_with_gap)))  # contains gap we are moving block to
    output.append(_print_cursor(fwd(b_without_gap)))

    # Blocks containing the entire moved block work intuitively.
    output.append(_print_cursor(fwd(b_edit)))

    # Blocks within the moved block work intuitively
    output.append(_print_cursor(fwd(c)))

    # Blocks partially containing the wrap block don't work.
    with pytest.raises(
        InvalidCursorError,
        match=r"move cannot forward block because exactly one endpoint",
    ):
        output.append(_print_cursor(fwd(b_with_endpoint_in_moved_block)))

    assert "\n\n".join(output) == golden


def test_move_forwarding_for_blocks_gap_after(proc_baz, golden):
    c = _find_cursors(proc_baz, "y: _; y = 1.1")[0]
    g = _find_stmt(proc_baz, "pass #0").after()

    b_above = c.parent()
    b_moved_block_body = b_above.body()
    b_gap_body = g.anchor().parent().body()

    b_with_endpoint_in_moved_block = b_moved_block_body[:3]

    _, fwd = c._move(g)

    output = []
    output.append(_print_cursor(fwd(b_above)))  # above edit level

    # Blocks containing the entire moved block work intuitively.
    output.append(_print_cursor(fwd(b_moved_block_body)))

    # Blocks containing the gap we are moving block to work intuitively
    output.append(_print_cursor(fwd(b_gap_body)))

    # Blocks within the moved block work intuitively
    output.append(_print_cursor(fwd(c)))

    # Blocks partially containing the wrap block don't work.
    with pytest.raises(
        InvalidCursorError,
        match=r"move cannot forward block because exactly one endpoint",
    ):
        output.append(_print_cursor(fwd(b_with_endpoint_in_moved_block)))

    assert "\n\n".join(output) == golden
