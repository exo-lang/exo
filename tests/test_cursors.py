from __future__ import annotations

import gc
import weakref

import pytest

from exo import proc
from exo.LoopIR import LoopIR, T
from exo.cursors import Cursor, Block, InvalidCursorError, ForwardingPolicy, Node
from exo.syntax import size, par, f32


@pytest.fixture(scope='session')
def proc_foo():
    @proc
    def foo(n: size, m: size):
        for i in par(0, n):
            for j in par(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

    yield foo


@pytest.fixture(scope='session')
def proc_bar():
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

    yield bar


def test_get_root(proc_foo):
    cursor = Cursor.root(proc_foo)
    assert cursor._node() is proc_foo.INTERNAL_proc()


def test_get_child(proc_foo):
    cursor = Cursor.root(proc_foo).children()
    cursor = next(iter(cursor))
    assert cursor._node() is proc_foo.INTERNAL_proc().body[0]


def test_find_cursor(proc_foo):
    c = proc_foo.find_cursors("for j in _:_")
    assert len(c) == 1
    c = c[0]  # One match
    c = c[0]  # First/only node in the block

    assert c._node() is proc_foo.INTERNAL_proc().body[0].body[0]


def test_find_hole_in_middle(proc_bar):
    c_body_1_5 = proc_bar.find_cursors('x = 1.0 ; _ ; x = 4.0')[0]
    for_j = proc_bar.find_stmt('for j in _: _')
    assert c_body_1_5 == for_j.body()[1:5]


def test_gap_insert_pass(proc_foo, golden):
    c = proc_foo.find_stmt('x = 0.0')
    assn = c._node()
    g = c.after()
    foo2, _ = g._insert([LoopIR.Pass(None, assn.srcinfo)])
    assert str(foo2) == golden


def test_insert_root_front(proc_foo, golden):
    c = Cursor.root(proc_foo)
    foo2, _ = c.body().before()._insert([LoopIR.Pass(None, c._node().srcinfo)])
    assert str(foo2) == golden


def test_insert_root_end(proc_foo, golden):
    c = Cursor.root(proc_foo)
    foo2, _ = c.body().after()._insert([LoopIR.Pass(None, c._node().srcinfo)])
    assert str(foo2) == golden


def test_block_gaps(proc_bar):
    c = proc_bar.find_stmt('for j in _: _')

    body = c.body()
    assert len(body) == 6
    subset = body[1:4]
    assert len(subset) == 3

    cx1 = proc_bar.find_stmt('x = 1.0')
    cx3 = proc_bar.find_stmt('x = 3.0')
    assert subset[0] == cx1
    assert subset[2] == cx3

    assert subset.before() == cx1.before()
    assert subset.after() == cx3.after()


def test_block_sequence_interface(proc_bar):
    for_j_body = proc_bar.find_stmt('for j in _: _').body()
    assert for_j_body[:] is not for_j_body
    assert for_j_body[:] == for_j_body
    assert len(list(for_j_body)) == 6

    with pytest.raises(IndexError, match='block cursors must be contiguous'):
        # noinspection PyStatementEffect
        # Sequence's __getitem__ should throw here, so this does have a side effect
        for_j_body[::2]


def test_block_delete(proc_bar, golden):
    c = proc_bar.find_stmt('for j in _: _')
    stmts = c.body()[1:4]

    bar2, _ = stmts._delete()
    assert str(bar2) == golden


def test_block_replace(proc_bar, golden):
    c = proc_bar.find_stmt('for j in _: _')
    stmts = c.body()[1:4]

    bar2, _ = stmts._replace([LoopIR.Pass(None, c._node().srcinfo)])
    assert str(bar2) == golden


def test_block_delete_whole_block(proc_bar, golden):
    c = proc_bar.find_stmt('for j in _: _')
    bar2, _ = c.body()._delete()
    assert str(bar2) == golden


def test_node_replace(proc_bar, golden):
    c = proc_bar.find_stmt('x = 3.0')
    assert isinstance(c, Node)

    bar2, _ = c._replace([LoopIR.Pass(None, c._node().srcinfo)])
    assert str(bar2) == golden


def test_cursor_move(proc_foo):
    c = proc_foo.find_stmt("for j in _:_")

    c_list = c.body()  # list of j's body
    assert isinstance(c_list, Block)
    assert len(c_list) == 4
    assert c_list.parent() == c

    c1 = c_list[0]  # x : f32
    assert c1.parent() == c
    assert c1._node() is proc_foo.INTERNAL_proc().body[0].body[0].body[0]

    c2 = c1.next()  # x = 0.0
    assert c2.parent() == c
    assert c2._node() is proc_foo.INTERNAL_proc().body[0].body[0].body[1]

    c3 = c1.next(2)  # y : f32
    assert c3.parent() == c
    assert c3._node() is proc_foo.INTERNAL_proc().body[0].body[0].body[2]

    _c2_ = c3.prev()
    assert c2 == _c2_
    _c1_ = c3.prev(2)
    assert c1 == _c1_


def test_cursor_move_invalid(proc_foo):
    # Edge cases near the root
    c = Cursor.root(proc_foo)
    with pytest.raises(InvalidCursorError, match="cannot move root cursor"):
        c.next()

    with pytest.raises(InvalidCursorError, match='cursor does not have a parent'):
        c.parent()

    # Edge cases near expressions
    c = proc_foo.find_cursors('m')[0]
    with pytest.raises(InvalidCursorError, match='cursor is not inside block'):
        c.next()

    # Edge cases near first statement in block
    c = proc_foo.find_stmt('x: f32')
    with pytest.raises(InvalidCursorError, match="cursor is out of range"):
        c.prev()

    c.before()  # ok
    with pytest.raises(InvalidCursorError, match="cursor is out of range"):
        c.before(2)

    # Edge cases near last statement in block
    c = proc_foo.find_stmt('y = 1.1')
    with pytest.raises(InvalidCursorError, match="cursor is out of range"):
        c.next()

    c.after()  # ok
    with pytest.raises(InvalidCursorError, match="cursor is out of range"):
        c.after(2)


def test_cursor_gap(proc_foo):
    # for i in par(0, n):
    #    for j in par(0, m):
    #        x: f32   <- x_alloc
    #                 <- g1
    #        x = 0.0  <- x_assn
    #        y: f32
    #                 <- g2
    #        y = 1.1  <- y_assn
    for_j = proc_foo.find_stmt("for j in _:_")
    x_alloc = proc_foo.find_stmt("x: f32")
    x_assn = proc_foo.find_stmt("x = 0.0")
    y_assn = proc_foo.find_stmt("y = 1.1")

    assert str(x_alloc._node()) == 'x: f32 @ DRAM\n'

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


def test_cursor_replace_expr(proc_foo, golden):
    c = proc_foo.find_cursors('m')[0]
    foo2, _ = c._replace(LoopIR.Const(42, T.size, c._node().srcinfo))
    assert str(foo2) == golden


def test_cursor_cannot_convert_expr_to_block(proc_foo):
    c = proc_foo.find_cursors('m')[0]
    with pytest.raises(InvalidCursorError, match='node is not inside a block'):
        c.as_block()


def test_cursor_replace_expr_deep(golden):
    @proc
    def example():
        x: f32
        x = 1.0 * (2.0 + 3.0)
        # x = 1.0 * (4.0 + 3.0)

    c: Node = example.find_cursors('2.0')[0]
    four = LoopIR.Const(4.0, T.f32, c._node().srcinfo)

    example_new, _ = c._replace(four)
    assert str(example_new) == golden


def test_cursor_forward_expr_deep():
    @proc
    def example():
        x: f32
        x = 1.0 * (2.0 + 3.0)
        # x = 1.0 * (4.0 + 3.0)

    c2: Node = example.find_cursors('2.0')[0]
    four = LoopIR.Const(4.0, T.f32, c2._node().srcinfo)

    example_new, fwd = c2._replace(four)
    with pytest.raises(InvalidCursorError, match='cannot forward replaced nodes'):
        fwd(c2)

    assert fwd(example.find_stmt('x = _')) == example_new.find_stmt('x = _')
    assert fwd(example.find_cursors('_ + _')[0]) == example_new.find_cursors('_ + _')[0]
    assert fwd(example.find_cursors('1.0')[0]) == example_new.find_cursors('1.0')[0]
    assert fwd(example.find_cursors('3.0')[0]) == example_new.find_cursors('3.0')[0]


def test_cursor_loop_bound(proc_foo):
    c_for_i = Cursor.root(proc_foo).body()[0]
    c_bound = c_for_i._child_node('hi')
    assert isinstance(c_bound._node(), LoopIR.Read)


def test_cursor_invalid_child(proc_foo):
    c = Cursor.root(proc_foo)

    # Quick sanity check
    assert c._child_node('body', 0) == c.body()[0]

    with pytest.raises(AttributeError, match="has no attribute '_invalid'"):
        c._child_node('_invalid', None)

    with pytest.raises(ValueError, match='must index into block attribute'):
        c._child_node('body', None)

    with pytest.raises(InvalidCursorError, match='cursor is out of range'):
        c._child_node('body', 42)


def test_cursor_lifetime():
    @proc
    def delete_me():
        x: f32
        x = 0.0

    cur = delete_me.find_stmt('x = _')
    assert isinstance(cur._node(), LoopIR.Assign)

    del delete_me
    gc.collect()

    with pytest.raises(InvalidCursorError, match='underlying proc was destroyed'):
        cur.proc()

    # TODO: The WeakKeyDictionary-ies in other modules seem to keep the IR alive as
    #   they keep references to them in the values.
    # with pytest.raises(InvalidCursorError, match='underlying node was destroyed'):
    #     cur._node()


@pytest.mark.parametrize("policy", [
    ForwardingPolicy.PreferInvalidation,
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
    stmt = [LoopIR.Pass(None, x_old[0]._node().srcinfo)]

    ins_gap = x_old[3].after()

    example_new, fwd = ins_gap._insert(stmt, policy)
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

    # Check that all inserted-body blocks (n > 1) are forwarded:
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


def test_insert_forwarding_policy(proc_bar):
    x2 = proc_bar.find_stmt('x = 2.0')
    gap = x2.after()

    stmt = [LoopIR.Pass(None, x2._node().srcinfo)]

    bar_pre, fwd_pre = gap._insert(stmt, ForwardingPolicy.AnchorPre)
    assert fwd_pre(gap) == bar_pre.find_stmt('x = 2.0').after()

    bar_post, fwd_post = gap._insert(stmt, ForwardingPolicy.AnchorPost)
    assert fwd_post(gap) == bar_post.find_stmt('x = 3.0').before()

    bar_invalid, fwd_invalid = gap._insert(stmt, ForwardingPolicy.PreferInvalidation)
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

    stmt = [LoopIR.Pass(None, x1_old._node().srcinfo)]

    example_new, fwd = gap._insert(stmt)
    x2_new = example_new.find_stmt('x = 2.0')

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

    x1_s1 = proc_s1.find_stmt('x = 1.0')
    x3_s1 = proc_s1.find_stmt('x = 3.0')

    x2_stmt = x1_s1._node().update(rhs=LoopIR.Const(2.0, T.f32, x1_s1._node().srcinfo))
    x4_stmt = x1_s1._node().update(rhs=LoopIR.Const(4.0, T.f32, x1_s1._node().srcinfo))

    proc_s2, fwd_12 = x1_s1.after()._insert([x2_stmt])
    proc_s3, fwd_23 = fwd_12(x3_s1).after()._insert([x4_stmt])
    assert str(proc_s3) == golden

    def fwd_13(cur):
        return fwd_23(fwd_12(cur))

    x1_s3 = proc_s3.find_stmt('x = 1.0')
    assert fwd_13(x1_s1) == x1_s3

    x3_s3 = proc_s3.find_stmt('x = 3.0')
    assert fwd_13(x3_s1) == x3_s3

    if_pat = 'if _: _\nelse: _'
    assert fwd_13(proc_s1.find_stmt(if_pat)) == proc_s3.find_stmt(if_pat)

    with pytest.raises(InvalidCursorError, match='cannot forward unknown procs'):
        fwd_23(x1_s1)


@pytest.mark.parametrize('old, new', [
    (0, 0),
    (1, 1),
    (2, None),
    (3, None),
    (4, 2),
    (5, 3),
])
def test_delete_forward_node(proc_bar, old, new):
    for_j = proc_bar.find_stmt('for j in _: _').body()
    bar_new, fwd = for_j[2:4]._delete()
    for_j_new = bar_new.find_stmt('for j in _: _').body()

    if new is None:
        with pytest.raises(InvalidCursorError, match='node no longer exists'):
            fwd(for_j[old])
    else:
        assert fwd(for_j[old]) == for_j_new[new]


@pytest.mark.parametrize('old, new', [
    ((0, Node.before), (0, Node.before)),
    ((0, Node.after), (0, Node.after)),
    ((1, Node.after), (1, Node.after)),
    ((2, Node.after), None),
    ((3, Node.after), (1, Node.after)),
    ((4, Node.before), (1, Node.after)),
    ((4, Node.after), (2, Node.after)),
    ((5, Node.after), (3, Node.after)),
])
def test_delete_forward_gap(proc_bar, old, new):
    for_j = proc_bar.find_stmt('for j in _: _').body()
    bar_new, fwd = for_j[2:4]._delete()
    for_j_new = bar_new.find_stmt('for j in _: _').body()

    old_i, old_gap = old

    if new is None:
        with pytest.raises(InvalidCursorError, match='gap no longer exists'):
            fwd(old_gap(for_j[old_i]))
    else:
        new_i, new_gap = new
        assert fwd(old_gap(for_j[old_i])) == new_gap(for_j_new[new_i])


@pytest.mark.parametrize('old, new', [
    # length 2
    (slice(0, 2), slice(0, 2)),
    (slice(1, 3), None),
    (slice(2, 4), None),  # policy? map deleted block to gap
    (slice(3, 5), None),
    (slice(4, 6), slice(2, 4)),
    # length 3
    (slice(0, 3), None),
    (slice(1, 4), slice(1, 2)),
    (slice(2, 5), slice(2, 3)),
    (slice(3, 6), None),
    # length 4
    (slice(0, 4), slice(0, 2)),
    (slice(1, 5), slice(1, 3)),
    (slice(2, 6), slice(2, 4)),
    # length 5
    (slice(0, 5), slice(0, 3)),
    (slice(1, 6), slice(1, 4)),
    # length 6
    (slice(0, 6), slice(0, 4)),
])
def test_delete_forward_block(proc_bar, old, new):
    for_j = proc_bar.find_stmt('for j in _: _').body()
    bar_new, fwd = for_j[2:4]._delete()
    for_j_new = bar_new.find_stmt('for j in _: _').body()

    if new is None:
        with pytest.raises(InvalidCursorError,
                           match='block (was partially destroyed|no longer exists)'):
            fwd(for_j[old])
    else:
        assert fwd(for_j[old]) == for_j_new[new]


@pytest.mark.parametrize('old, new', [
    # length 2
    ('x = 0.0 ; x = 1.0', 'x = 0.0 ; x = 1.0'),
    ('x = 1.0 ; x = 2.0', None),
    ('x = 2.0 ; x = 3.0', None),  # policy? map deleted block to gap
    ('x = 3.0 ; x = 4.0', None),
    ('x = 4.0 ; x = 5.0', 'x = 4.0 ; x = 5.0'),
    # length 3
    ('x = 0.0 ; _ ; x = 2.0', None),
    ('x = 1.0 ; _ ; x = 3.0', 'x = 1.0'),
    ('x = 2.0 ; _ ; x = 4.0', 'x = 4.0'),
    ('x = 3.0 ; _ ; x = 5.0', None),
    # # length 4
    ('x = 0.0 ; _ ; x = 3.0', 'x = 0.0 ; x = 1.0'),
    ('x = 1.0 ; _ ; x = 4.0', 'x = 1.0 ; x = 4.0'),
    ('x = 2.0 ; _ ; x = 5.0', 'x = 4.0 ; x = 5.0'),
    # length 5
    ('x = 0.0 ; _ ; x = 4.0', 'x = 0.0 ; x = 1.0 ; x = 4.0'),
    ('x = 1.0 ; _ ; x = 5.0', 'x = 1.0 ; x = 4.0 ; x = 5.0'),
    # length 6
    ('x = 0.0 ; _ ; x = 5.0', 'x = 0.0 ; x = 1.0 ; x = 4.0 ; x = 5.0'),
])
def test_delete_forward_block_with_patterns(proc_bar, old, new):
    for_j = proc_bar.find_stmt('for j in _: _').body()
    bar_new, fwd = for_j[2:4]._delete()

    old_c = proc_bar.find_cursors(old)[0]
    if new is None:
        with pytest.raises(InvalidCursorError,
                           match='block (was partially destroyed|no longer exists)'):
            fwd(old_c)
    else:
        assert fwd(old_c) == bar_new.find_cursors(new)[0]


def test_forward_lifetime(proc_bar):
    """
    Make sure that forwarding functions do not keep the cursors that created them
    alive.
    """

    for_j = proc_bar.find_stmt('for j in _: _').body()[2:4]
    for_j_weak = weakref.ref(for_j)

    bar_new, fwd = for_j._delete()
    bar_new_weak = weakref.ref(bar_new)

    c = Cursor.root(bar_new).body()[0]
    assert c._node() is not None

    gc.collect()

    assert for_j_weak() is not None
    assert bar_new_weak() is not None

    del for_j, bar_new
    gc.collect()

    assert for_j_weak() is None
    assert bar_new_weak() is None


@pytest.mark.parametrize('old, new', [
    ('x = 0.0', 'x = 0.0'),
    ('x = 1.0', None),
    ('x = 2.0', None),
    ('x = 3.0', None),
    ('x = 4.0', 'x = 4.0'),
    ('x = 5.0', 'x = 5.0'),
])
def test_block_replace_forward_node(proc_bar, old, new):
    for_j = proc_bar.find_stmt('for j in _: _').body()
    bar_new, fwd = for_j[1:4]._replace(
        [LoopIR.Pass(None, for_j.parent()._node().srcinfo),
         LoopIR.Pass(None, for_j.parent()._node().srcinfo)])

    old_c = proc_bar.find_stmt(old)

    if new is None:
        with pytest.raises(InvalidCursorError, match='node no longer exists'):
            fwd(old_c)
    else:
        assert fwd(old_c) == bar_new.find_stmt(new)


@pytest.mark.parametrize('old, new', [
    # x = 0
    (('x = 0.0', Node.before), ('x = 0.0', Node.before)),
    (('x = 0.0', Node.after), ('x = 0.0', Node.after)),
    (('x = 0.0', Node.after), ('pass #0', Node.before)),
    # x = 1
    (('x = 1.0', Node.before), ('x = 0.0', Node.after)),
    (('x = 1.0', Node.before), ('pass #0', Node.before)),
    (('x = 1.0', Node.after), None),
    # x = 2
    (('x = 2.0', Node.before), None),
    (('x = 2.0', Node.after), None),
    # x = 3
    (('x = 3.0', Node.before), None),
    (('x = 3.0', Node.after), ('pass #1', Node.after)),
    (('x = 3.0', Node.after), ('x = 4.0', Node.before)),
    # x = 4
    (('x = 4.0', Node.before), ('pass #1', Node.after)),
    (('x = 4.0', Node.before), ('x = 4.0', Node.before)),
    (('x = 4.0', Node.after), ('x = 4.0', Node.after)),
    (('x = 4.0', Node.after), ('x = 5.0', Node.before)),
    # x = 5
    (('x = 5.0', Node.before), ('x = 4.0', Node.after)),
    (('x = 5.0', Node.before), ('x = 5.0', Node.before)),
    (('x = 5.0', Node.after), ('x = 5.0', Node.after)),
])
def test_block_replace_forward_gap(proc_bar, old, new):
    for_j = proc_bar.find_stmt('for j in _: _').body()
    bar_new, fwd = for_j[1:4]._replace(
        [LoopIR.Pass(None, for_j.parent()._node().srcinfo),
         LoopIR.Pass(None, for_j.parent()._node().srcinfo)])

    old_pat, old_gap = old
    old_c = proc_bar.find_stmt(old_pat)

    if new is None:
        with pytest.raises(InvalidCursorError, match='gap no longer exists'):
            fwd(old_gap(old_c))
    else:
        new_pat, new_gap = new
        new_c = bar_new.find_stmt(new_pat)
        assert fwd(old_gap(old_c)) == new_gap(new_c)


@pytest.mark.parametrize('old, new', [
    # length 2
    ('x = 0.0 ; x = 1.0', None),
    ('x = 1.0 ; x = 2.0', None),
    ('x = 2.0 ; x = 3.0', None),
    ('x = 3.0 ; x = 4.0', None),
    ('x = 4.0 ; x = 5.0', 'x = 4.0 ; x = 5.0'),
    # length 3
    ('x = 0.0 ; _ ; x = 2.0', None),
    ('x = 1.0 ; _ ; x = 3.0', 'pass ; pass'),
    ('x = 2.0 ; _ ; x = 4.0', None),
    ('x = 3.0 ; _ ; x = 5.0', None),
    # length 4
    ('x = 0.0 ; _ ; x = 3.0', 'x = 0.0 ; pass ; pass'),
    ('x = 1.0 ; _ ; x = 4.0', 'pass ; pass ; x = 4.0'),
    ('x = 2.0 ; _ ; x = 5.0', None),
    # length 5
    ('x = 0.0 ; _ ; x = 4.0', 'x = 0.0 ; pass ; pass ; x = 4.0'),
    ('x = 1.0 ; _ ; x = 5.0', 'pass ; pass ; x = 4.0 ; x = 5.0'),
    # length 6
    ('x = 0.0 ; _ ; x = 5.0', 'x = 0.0 ; pass ; pass ; x = 4.0 ; x = 5.0'),
])
def test_block_replace_forward_block(proc_bar, old, new):
    for_j = proc_bar.find_stmt('for j in _: _').body()
    bar_new, fwd = for_j[1:4]._replace(
        [LoopIR.Pass(None, for_j.parent()._node().srcinfo),
         LoopIR.Pass(None, for_j.parent()._node().srcinfo)])

    print(bar_new)

    old_c = proc_bar.find_cursors(old)[0]
    if new is None:
        with pytest.raises(InvalidCursorError,
                           match=r'block (was partially destroyed|no longer exists)'):
            fwd(old_c)
    else:
        assert fwd(old_c) == bar_new.find_cursors(new)[0]
