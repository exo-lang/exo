from __future__ import annotations

import gc

import pytest

from exo import proc
from exo.LoopIR import LoopIR, T
from exo.cursors import Cursor, Selection, InvalidCursorError, ForwardingPolicy, Node
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


def test_find_hole_in_middle():
    c_body_1_5 = bar.find_cursor('x = 1.0 ; _ ; x = 4.0')[0]
    for_j = bar.find_stmt('for j in _: _')
    assert c_body_1_5 == for_j.body()[1:5]


def test_gap_insert_pass(golden):
    c = foo.find_stmt('x = 0.0')
    assn = c.node()
    g = c.after()
    foo2, _ = g.insert([LoopIR.Pass(None, assn.srcinfo)])
    assert str(foo2) == golden


def test_insert_root_front(golden):
    c = Cursor.root(foo)
    foo2, _ = c.body().before().insert([LoopIR.Pass(None, c.node().srcinfo)])
    assert str(foo2) == golden


def test_insert_root_end(golden):
    c = Cursor.root(foo)
    foo2, _ = c.body().after().insert([LoopIR.Pass(None, c.node().srcinfo)])
    assert str(foo2) == golden


def test_selection_gaps():
    c = bar.find_stmt('for j in _: _')

    body = c.body()
    assert len(body) == 6
    subset = body[1:4]
    assert len(subset) == 3

    cx1 = bar.find_stmt('x = 1.0')
    cx3 = bar.find_stmt('x = 3.0')
    assert subset[0] == cx1
    assert subset[2] == cx3

    assert subset.before() == cx1.before()
    assert subset.after() == cx3.after()


def test_selection_sequence_interface():
    for_j_body = bar.find_stmt('for j in _: _').body()
    assert for_j_body[:] is not for_j_body
    assert for_j_body[:] == for_j_body
    assert len(list(for_j_body)) == 6

    with pytest.raises(IndexError, match='cursor selections must be contiguous'):
        # noinspection PyStatementEffect
        # Sequence's __getitem__ should throw here, so this does have a side effect
        for_j_body[::2]


def test_selection_delete(golden):
    c = bar.find_stmt('for j in _: _')
    stmts = c.body()[1:4]

    bar2 = stmts.delete()
    assert str(bar2) == golden


def test_selection_replace(golden):
    c = bar.find_stmt('for j in _: _')
    stmts = c.body()[1:4]

    bar2 = stmts.replace([LoopIR.Pass(None, c.node().srcinfo)])
    assert str(bar2) == golden


def test_selection_delete_whole_block(golden):
    c = bar.find_stmt('for j in _: _')
    bar2 = c.body().delete()
    assert str(bar2) == golden


def test_node_replace(golden):
    c = bar.find_stmt('x = 3.0')
    assert isinstance(c, Node)

    bar2 = c.replace([LoopIR.Pass(None, c.node().srcinfo)])
    assert str(bar2) == golden


def test_cursor_move():
    c = foo.find_stmt("for j in _:_")

    c_list = c.body()  # list of j's body
    assert isinstance(c_list, Selection)
    assert len(c_list) == 4
    assert c_list.parent() == c

    c1 = c_list[0]  # x : f32
    assert c1.parent() == c
    assert c1.node() is foo.INTERNAL_proc().body[0].body[0].body[0]

    c2 = c1.next()  # x = 0.0
    assert c2.parent() == c
    assert c2.node() is foo.INTERNAL_proc().body[0].body[0].body[1]

    c3 = c1.next(2)  # y : f32
    assert c3.parent() == c
    assert c3.node() is foo.INTERNAL_proc().body[0].body[0].body[2]

    _c2_ = c3.prev()
    assert c2 == _c2_
    _c1_ = c3.prev(2)
    assert c1 == _c1_


def test_cursor_move_invalid():
    c = Cursor.root(foo)
    with pytest.raises(InvalidCursorError, match='cannot move root cursor'):
        c.next()

    with pytest.raises(InvalidCursorError, match='cursor does not have a parent'):
        c.parent()

    c = foo.find_cursor('m')[0]
    with pytest.raises(InvalidCursorError, match='cursor is not inside block'):
        c.next()


def test_cursor_gap():
    # for i in par(0, n):
    #    for j in par(0, m):
    #        x: f32   <- x_alloc
    #                 <- g1
    #        x = 0.0  <- x_assn
    #        y: f32
    #                 <- g2
    #        y = 1.1  <- y_assn
    for_j = foo.find_stmt("for j in _:_")
    x_alloc = foo.find_stmt("x: f32")
    x_assn = foo.find_stmt("x = 0.0")
    y_assn = foo.find_stmt("y = 1.1")

    assert str(x_alloc.node()) == 'x: f32 @ DRAM\n'

    g1 = x_alloc.after()
    assert g1 == x_assn.before()
    assert g1.before() == x_alloc
    assert g1.after() == x_assn
    assert g1.parent() == for_j

    g2 = g1.next(2)
    assert g2 == x_assn.after(2)
    assert g2.before(2) == x_assn
    assert g2.after() == y_assn
    assert g2.parent() == for_j

    assert g2.prev(2) == g1


def test_cursor_replace_expr(golden):
    c = foo.find_cursor('m')[0]
    foo2 = c.replace(LoopIR.Const(42, T.size, c.node().srcinfo))
    assert str(foo2) == golden


def test_cursor_replace_expr_deep(golden):
    @proc
    def example():
        x: f32
        x = 1.0 * (2.0 + 3.0)
        # x = 1.0 * (4.0 + 3.0)

    c: Node = example.find_cursor('2.0')[0]
    four = LoopIR.Const(4.0, T.f32, c.node().srcinfo)

    example_new = c.replace(four)
    assert str(example_new) == golden


def test_cursor_loop_bound():
    c_for_i = Cursor.root(foo).body()[0]
    c_bound = c_for_i.child('hi')
    assert isinstance(c_bound.node(), LoopIR.Read)


def test_cursor_invalid_child():
    c = Cursor.root(foo)

    # Quick sanity check
    assert c.child('body', 0) == c.body()[0]

    with pytest.raises(AttributeError, match="has no attribute '_invalid'"):
        c.child('_invalid', None)

    with pytest.raises(ValueError, match='must index into block attribute'):
        c.child('body', None)

    with pytest.raises(IndexError, match='list index out of range'):
        c.child('body', 42)


def test_cursor_lifetime():
    @proc
    def delete_me():
        x: f32
        x = 0.0

    cur = delete_me.find_stmt('x = _')
    assert isinstance(cur.node(), LoopIR.Assign)

    del delete_me
    gc.collect()

    with pytest.raises(InvalidCursorError, match='underlying proc was destroyed'):
        cur.proc()

    # TODO: The WeakKeyDictionary-ies in other modules seem to keep the IR alive as
    #   they keep references to them in the values.
    # with pytest.raises(InvalidCursorError, match='underlying node was destroyed'):
    #     cur.node()


@pytest.mark.parametrize("policy", [
    ForwardingPolicy.EagerInvalidation,
    ForwardingPolicy.AnchorPost,
    ForwardingPolicy.AnchorPre,
])
def test_insert_forwarding(policy):
    @proc
    def example_old():
        x: f32
        x = 0.0
        for i in par(0, 10):
            x = 1.0
            for j in par(0, 20):
                x = 2.0
                x = 3.0
                # pass (ins_gap)
                x = 4.0
                x = 5.0
            x = 6.0
        x = 7.0

    x_old = [example_old.find_stmt(f'x = {n}.0') for n in range(8)]
    stmt = [LoopIR.Pass(None, x_old[0].node().srcinfo)]

    ins_gap = x_old[3].after()

    example_new, fwd = ins_gap.insert(stmt, policy)
    x_new = [example_new.find_stmt(f'x = {n}.0') for n in range(8)]

    # Check that the root is forwarded:
    assert fwd(Cursor.root(example_old)) == Cursor.root(example_new)

    # Check that the assignment nodes are forwarded:
    for cur_old, cur_new in zip(x_old, x_new):
        assert fwd(cur_old) == cur_new

    # Check that non-insertion before-gaps (i.e. exclude 4) are forwarded:
    for i in (0, 1, 2, 3, 5, 6, 7):
        assert fwd(x_old[i].before()) == x_new[i].before()

    # Check that non-insertion after-gaps (i.e. exclude 3) are forwarded:
    for i in (0, 1, 2, 4, 5, 6, 7):
        assert fwd(x_old[i].after()) == x_new[i].after()

    # Check that for loops are forwarded:
    for_i_old = example_old.find_stmt('for i in _: _')
    for_i_new = example_new.find_stmt('for i in _: _')
    assert fwd(for_i_old) == for_i_new

    for_j_old = example_old.find_stmt('for j in _: _')
    for_j_new = example_new.find_stmt('for j in _: _')
    assert fwd(for_j_old) == for_j_new

    # Check that for loop bodies are forwarded:
    assert fwd(for_i_old.body()) == for_i_new.body()
    assert fwd(for_j_old.body()) == for_j_new.body()

    # Check that all inserted-body selections (n > 1) are forwarded:
    test_cases = [
        (slice(0, 2), slice(0, 2)),
        (slice(1, 3), slice(1, 4)),
        (slice(2, 4), slice(3, 5)),
        (slice(0, 3), slice(0, 4)),
        (slice(1, 4), slice(1, 5)),
        # full body already tested above.
    ]
    for old_range, new_range in test_cases:
        assert fwd(for_j_old.body()[old_range]) == for_j_new.body()[new_range]


def test_insert_forwarding_policy():
    x2 = bar.find_stmt('x = 2.0')
    gap = x2.after()

    stmt = [LoopIR.Pass(None, x2.node().srcinfo)]

    bar_pre, fwd_pre = gap.insert(stmt, ForwardingPolicy.AnchorPre)
    assert fwd_pre(gap) == bar_pre.find_stmt('x = 2.0').after()

    bar_post, fwd_post = gap.insert(stmt, ForwardingPolicy.AnchorPost)
    assert fwd_post(gap) == bar_post.find_stmt('x = 3.0').before()

    bar_invalid, fwd_invalid = gap.insert(stmt, ForwardingPolicy.EagerInvalidation)
    with pytest.raises(InvalidCursorError, match='insertion gap was invalidated'):
        fwd_invalid(gap)


def test_insert_forward_orelse():
    @proc
    def example_old():
        x: f32
        if 1 < 2:
            x = 1.0
        else:
            x = 2.0

    x1_old = example_old.find_stmt('x = 1.0')
    x2_old = example_old.find_stmt('x = 2.0')
    gap = x1_old.after()

    stmt = [LoopIR.Pass(None, x1_old.node().srcinfo)]

    example_new, fwd = gap.insert(stmt)
    x2_new = example_new.find_stmt('x = 2.0')

    assert fwd(x2_old) == x2_new


def test_double_insert_forwarding():
    @proc
    def proc_s1():
        x: f32
        if 1 < 2:
            x = 1.0
            # x = 2.0
        else:
            x = 3.0
            # x = 4.0

    x1_s1 = proc_s1.find_stmt('x = 1.0')
    x3_s1 = proc_s1.find_stmt('x = 3.0')

    x2_stmt = x1_s1.node().update(rhs=LoopIR.Const(2.0, T.f32, x1_s1.node().srcinfo))
    x4_stmt = x1_s1.node().update(rhs=LoopIR.Const(4.0, T.f32, x1_s1.node().srcinfo))

    proc_s2, fwd_12 = x1_s1.after().insert([x2_stmt])
    x3_s2 = fwd_12(x3_s1)

    proc_s3, fwd_23 = x3_s2.after().insert([x4_stmt])
    print(proc_s3)

    x1_s3 = proc_s3.find_stmt('x = 1.0')
    assert fwd_23(fwd_12(x1_s1)) == x1_s3

    x3_s3 = proc_s3.find_stmt('x = 3.0')
    assert fwd_23(fwd_12(x3_s1)) == x3_s3

    if_pat = 'if _: _\nelse: _'
    assert fwd_23(fwd_12(proc_s1.find_stmt(if_pat))) == proc_s3.find_stmt(if_pat)
