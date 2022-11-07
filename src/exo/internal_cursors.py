from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Optional, Iterable, Union
from weakref import ReferenceType

from . import API
from . import LoopIR


class InvalidCursorError(Exception):
    pass


class ForwardingPolicy(Enum):
    # Invalidate any cursor that could be forwarded reasonably in more than one
    # way.
    PreferInvalidation = auto()
    # When forwarding insertions, prefer to forward the insertion gap to the gap
    # BEFORE the inserted block, rather than invalidating it.
    AnchorPre = auto()
    # When forwarding insertions, prefer to forward the insertion gap to the gap
    # AFTER the inserted block, rather than invalidating it.
    AnchorPost = auto()


def _starts_with(a: list, b: list):
    """
    Returns true if the first elements of `a` equal `b` exactly
    >>> _starts_with([1, 2, 3], [1, 2])
    True
    >>> _starts_with(['x'], ['x'])
    True
    >>> _starts_with([1, 2, 3], [])
    True
    >>> _starts_with(['a', 'b', 'c'], ['a', 'b', 'c', 'd'])
    False
    """
    return len(a) >= len(b) and all(a[i] == b[i] for i in range(len(b)))


def _is_sub_range(a: range, b: range):
    """
    Returns true if `a` is a STRICT sub-range of `b`.
    Only applies to step-1 ranges.
    >>> _is_sub_range(range(1, 4), range(1, 4))
    False
    >>> _is_sub_range(range(0, 3), range(3, 6))
    False
    >>> _is_sub_range(range(2, 4), range(2, 5))
    True
    >>> _is_sub_range(range(2, 4), range(1, 4))
    True
    >>> _is_sub_range(range(2, 4), range(1, 5))
    True
    >>> _is_sub_range(range(0, 4), range(1, 4))
    False
    """
    assert a.step == b.step and a.step in (1, None)
    return (a.start >= b.start) and (a.stop <= b.stop) and a != b


def _overlaps_one_side(a: range, b: range):
    """
    Returns True if `a` overlaps `b` on exactly one side, without containing `b`.
    Only applies to step-1 ranges.
    >>> _overlaps_one_side(range(0, 4), range(4, 8))  # fully to left
    False
    >>> _overlaps_one_side(range(0, 5), range(4, 8))  # rightmost overlaps leftmost
    True
    >>> _overlaps_one_side(range(0, 7), range(4, 8))  # almost contains on right side
    True
    >>> _overlaps_one_side(range(0, 8), range(4, 8))  # contains on right side
    False
    >>> _overlaps_one_side(range(4, 7), range(4, 8))  # contained, left-aligned
    True
    >>> _overlaps_one_side(range(4, 8), range(4, 8))  # equal
    False
    >>> _overlaps_one_side(range(5, 8), range(4, 8))  # contained, right-aligned
    False
    >>> _overlaps_one_side(range(4, 12), range(4, 8))  # contains on left side
    False
    >>> _overlaps_one_side(range(5, 12), range(4, 8))  # almost contains on left side
    True
    >>> _overlaps_one_side(range(7, 12), range(4, 8))  # leftmost overlaps rightmost
    True
    >>> _overlaps_one_side(range(8, 12), range(4, 8))  # fully to right
    False
    """
    assert a.step == b.step and a.step in (1, None)
    return (
        a.start < b.start < a.stop < b.stop
        or a.start == b.start < a.stop < b.stop
        or b.start < a.start < b.stop < a.stop
    )


@dataclass
class Cursor(ABC):
    _proc: ReferenceType[API.Procedure]

    # ------------------------------------------------------------------------ #
    # Static constructors
    # ------------------------------------------------------------------------ #

    @staticmethod
    def root(proc: API.Procedure):
        return Node(weakref.ref(proc), [])

    # ------------------------------------------------------------------------ #
    # Validating accessors
    # ------------------------------------------------------------------------ #

    def proc(self):
        if (p := self._proc()) is None:
            raise InvalidCursorError("underlying proc was destroyed")
        return p

    # ------------------------------------------------------------------------ #
    # Navigation (abstract)
    # ------------------------------------------------------------------------ #

    @abstractmethod
    def parent(self) -> Node:
        """Get the node containing the current cursor"""

    @abstractmethod
    def before(self, dist=1) -> Cursor:
        """For gaps, get the node before the gap. Otherwise, get the gap before
        the node or block"""

    @abstractmethod
    def after(self, dist=1) -> Cursor:
        """For gaps, get the node after the gap. Otherwise, get the gap after
        the node or block"""

    @abstractmethod
    def prev(self, dist=1) -> Cursor:
        """Get the previous node/gap in the block. Undefined for blocks."""

    @abstractmethod
    def next(self, dist=1) -> Cursor:
        """Get the next node/gap in the block. Undefined for blocks."""

    # ------------------------------------------------------------------------ #
    # Block Navigation
    # ------------------------------------------------------------------------ #

    def _whole_block(self) -> Block:
        attr, _range = self._path[-1]
        parent = self.parent()
        return parent._child_block(attr)

    # ------------------------------------------------------------------------ #
    # Protected path / mutation helpers
    # ------------------------------------------------------------------------ #

    def _translate(self, ty, dist):
        assert isinstance(self, (Node, Gap))
        if not self._path:
            raise InvalidCursorError("cannot move root cursor")
        attr, i = self._path[-1]
        if i is None:
            raise InvalidCursorError("cursor is not inside block")
        return ty(self.parent(), attr, i + dist)

    @staticmethod
    def _walk_path(n, path):
        for attr, idx in path:
            n = getattr(n, attr)
            if idx is not None:
                n = n[idx]
        return n

    def _rewrite_node(self, fn):
        """
        Applies `fn` to the AST node containing the cursor. The callback is
        passed the raw parent of the pointed-to node/block/gap. The callback is
        expected to return a single, updated node to be placed in the new tree.
        """
        assert isinstance(self, (Node, Block, Gap))

        def impl(node, path):
            if len(path) == 1:
                return fn(node)

            (attr, i), path = path[0], path[1:]
            children = getattr(node, attr)

            if i is None:
                return node.update(**{attr: impl(children, path)})

            return node.update(
                **{attr: children[:i] + [impl(children[i], path)] + children[i + 1 :]}
            )

        return impl(self.proc().INTERNAL_proc(), self._path)

    def _make_forward(self, new_proc, fwd_node, fwd_gap, fwd_sel):
        orig_proc = self._proc
        depth = len(self._path)
        attr = self._path[depth - 1][0]

        def forward(cursor: Cursor) -> Cursor:
            if cursor._proc != orig_proc:
                raise InvalidCursorError("cannot forward unknown procs")

            # TODO: use attrs + attrs.evolve
            def evolve(p):
                return type(cursor)(new_proc, p)

            old_path = cursor._path

            if len(old_path) < depth:
                # Too shallow
                return evolve(old_path)

            old_attr, old_idx = old_path[depth - 1]

            if old_attr != attr:
                # At least as deep, but wrong branch
                return evolve(old_path)

            if len(old_path) > depth or isinstance(cursor, Node):
                idx = fwd_node(old_idx)
            elif isinstance(cursor, Gap):
                idx = fwd_gap(old_idx)
            else:
                assert isinstance(cursor, Block)
                idx = fwd_sel(old_idx)

            return evolve(old_path[: depth - 1] + [(attr, idx)] + old_path[depth:])

        return forward


@dataclass
class Block(Cursor):
    _path: list[tuple[str, Union[int, range]]]  # is range iff last entry

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        return Node(self._proc, self._path[:-1])

    def before(self, dist=1) -> Gap:
        attr, _range = self._path[-1]
        assert len(_range) > 0
        return self.parent()._child_node(attr, _range.start).before(dist)

    def after(self, dist=1) -> Gap:
        attr, _range = self._path[-1]
        assert len(_range) > 0
        return self.parent()._child_node(attr, _range.stop - 1).after(dist)

    def prev(self, dist=1) -> Cursor:
        # TODO: what should this mean?
        #  1. The node after the block?
        #  2. The block shifted over?
        #  3. The block of nodes leading up to the start?
        raise NotImplementedError("Block.prev")

    def next(self, dist=1) -> Cursor:
        # TODO: what should this mean?
        #  1. The node after the block?
        #  2. The block shifted over?
        #  3. The block of nodes past the end?
        raise NotImplementedError("Block.next")

    # ------------------------------------------------------------------------ #
    # Container interface implementation
    # ------------------------------------------------------------------------ #

    def __contains__(self, cur):
        n = len(self._path)
        blk_path = self._path
        cur_path = cur._path

        if n != len(cur_path):
            return False

        if (
            any(cur_path[i] != blk_path[i] for i in range(n - 1))
            or cur_path[-1][0] != blk_path[-1][0]
        ):
            return False

        if isinstance(cur, Node):
            return cur_path[-1][1] in blk_path[-1][1]
        elif isinstance(cur, Gap):
            return blk_path[-1][1].start <= cur_path[-1][1] <= blk_path[-1][1].stop
        else:
            assert isinstance(cur, Block)
            return (
                _is_sub_range(cur_path[-1][1], blk_path[-1][1])
                or cur_path[-1][1] == blk_path[-1][1]
            )

    # ------------------------------------------------------------------------ #
    # Sequence interface implementation
    # ------------------------------------------------------------------------ #

    def __iter__(self):
        attr, _range = self._path[-1]
        block = self.parent()
        for i in _range:
            yield block._child_node(attr, i)

    def __getitem__(self, i):
        attr, r = self._path[-1]
        r = r[i]
        if isinstance(r, range):
            if r.step != 1:
                raise IndexError("block cursors must be contiguous")
            return Block(self._proc, self._path[:-1] + [(attr, r)])
        else:
            return self.parent()._child_node(attr, r)

    def __len__(self):
        _, _range = self._path[-1]
        return len(_range)

    # ------------------------------------------------------------------------ #
    # Block-specific operations
    # ------------------------------------------------------------------------ #

    def expand(self, lo=None, hi=None):
        attr, _range = self._path[-1]
        full_block = self.parent()._child_block(attr)
        _, full_range = full_block._path[-1]
        if lo is None:
            return full_block

        lo = _range.start - lo
        lo = lo if lo >= 0 else 0
        hi = _range.stop + hi
        new_range = full_range[lo:hi]

        return Block(self._proc, self._path[:-1] + [(attr, new_range)])

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def _replace(self, nodes: list, *, empty_default=None):
        """
        This is an UNSAFE internal function for replacing a block in an AST with
        a list of statements and providing a forwarding function as collateral.
        It is meant to be package-private, not class-private, so it may be
        called from other internal classes and modules, but not from end-user
        code.
        """
        assert self._path
        assert len(self) > 0
        assert isinstance(nodes, list)

        def update(parent):
            attr, i = self._path[-1]
            children = getattr(parent, attr)
            new_children = children[: i.start] + nodes + children[i.stop :]
            new_children = new_children or empty_default or []
            return parent.update(**{attr: new_children})

        p = API.Procedure(self._rewrite_node(update))

        return p, self._forward_replace(weakref.ref(p), len(nodes))

    def _forward_replace(self, new_proc, n_ins):
        _, del_range = self._path[-1]
        n_diff = n_ins - len(del_range)

        def fwd_node(i):
            if i in del_range:
                raise InvalidCursorError("node no longer exists")
            return i + n_diff * (i >= del_range.stop)

        def fwd_gap(i):
            if i in del_range[1:]:
                raise InvalidCursorError("gap no longer exists")
            return i + n_diff * (i >= del_range.stop)

        def fwd_sel(rng: range):
            if _is_sub_range(rng, del_range):
                raise InvalidCursorError("block no longer exists")
            if _overlaps_one_side(rng, del_range):
                raise InvalidCursorError("block was partially destroyed")

            start = rng.start + n_diff * (rng.start >= del_range.stop)
            stop = rng.stop + n_diff * (rng.stop >= del_range.stop)

            if start >= stop:
                raise InvalidCursorError("block no longer exists")

            return range(start, stop)

        return self._make_forward(new_proc, fwd_node, fwd_gap, fwd_sel)

    def _delete(self):
        """
        This is an UNSAFE internal function for deleting a block in an AST and
        providing a forwarding function as collateral. It is meant to be
        package-private, not class-private, so it may be called from other
        internal classes and modules, but not from end-user code.
        """
        pass_stmt = [LoopIR.LoopIR.Pass(None, self.parent()._node().srcinfo)]
        return self._replace([], empty_default=pass_stmt)


@dataclass
class Node(Cursor):
    _path: list[tuple[str, Optional[int]]]

    # ------------------------------------------------------------------------ #
    # Validating accessors
    # ------------------------------------------------------------------------ #

    def _node(self):
        """
        Gets the raw underlying node that's pointed-to. This is meant to be
        compiler-internal, not class-private, so other parts of the compiler
        may call this, while users should not.
        """
        if (n := self._node_ref()) is None:
            raise InvalidCursorError("underlying node was destroyed")
        return n

    @cached_property
    def _node_ref(self):
        n = self.proc().INTERNAL_proc()
        n = self._walk_path(n, self._path)
        return weakref.ref(n)

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        if not self._path:
            raise InvalidCursorError("cursor does not have a parent")
        return Node(self._proc, self._path[:-1])

    def before(self, dist=1) -> Gap:
        return self._translate(Node._child_gap, 1 - dist)

    def after(self, dist=1) -> Gap:
        return self._translate(Node._child_gap, dist)

    def prev(self, dist=1) -> Node:
        return self._translate(Node._child_node, -dist)

    def next(self, dist=1) -> Node:
        return self._translate(Node._child_node, dist)

    # ------------------------------------------------------------------------ #
    # Navigation (children)
    # ------------------------------------------------------------------------ #

    def _child_node(self, attr, i=None) -> Node:
        _node = getattr(self._node(), attr)
        if i is not None:
            if 0 <= i < len(_node):
                _node = _node[i]
            else:
                raise InvalidCursorError("cursor is out of range")
        elif isinstance(_node, list):
            raise ValueError("must index into block attribute")
        cur = Node(self._proc, self._path + [(attr, i)])
        # noinspection PyPropertyAccess
        # cached_property is settable, bug in static analysis
        cur._node_ref = weakref.ref(_node)
        return cur

    def _child_gap(self, attr, i=None) -> Gap:
        _node = getattr(self._node(), attr)
        if i is not None:
            if not 0 <= i <= len(_node):
                raise InvalidCursorError("cursor is out of range")
        elif isinstance(_node, list):
            raise ValueError("must index into block attribute")
        return Gap(self._proc, self._path + [(attr, i)])

    def _child_block(self, attr: str):
        stmts = getattr(self._node(), attr)
        assert isinstance(stmts, list)
        return Block(self._proc, self._path + [(attr, range(len(stmts)))])

    def children(self) -> Iterable[Node]:
        n = self._node()
        # Top-level proc
        if isinstance(n, LoopIR.LoopIR.proc):
            yield from self._children_from_attrs(n, "body")
        # Statements
        elif isinstance(n, (LoopIR.LoopIR.Assign, LoopIR.LoopIR.Reduce)):
            yield from self._children_from_attrs(n, "idx", "rhs")
        elif isinstance(n, (LoopIR.LoopIR.WriteConfig, LoopIR.LoopIR.WindowStmt)):
            yield from self._children_from_attrs(n, "rhs")
        elif isinstance(
            n, (LoopIR.LoopIR.Pass, LoopIR.LoopIR.Alloc, LoopIR.LoopIR.Free)
        ):
            yield from []
        elif isinstance(n, LoopIR.LoopIR.If):
            yield from self._children_from_attrs(n, "cond", "body", "orelse")
        elif isinstance(n, LoopIR.LoopIR.Seq):
            yield from self._children_from_attrs(n, "hi", "body")
        elif isinstance(n, LoopIR.LoopIR.Call):
            yield from self._children_from_attrs(n, "args")
        # Expressions
        elif isinstance(n, LoopIR.LoopIR.Read):
            yield from self._children_from_attrs(n, "idx")
        elif isinstance(
            n,
            (
                LoopIR.LoopIR.Const,
                LoopIR.LoopIR.WindowExpr,
                LoopIR.LoopIR.StrideExpr,
                LoopIR.LoopIR.ReadConfig,
            ),
        ):
            yield from []
        elif isinstance(n, LoopIR.LoopIR.USub):
            yield from self._children_from_attrs(n, "arg")
        elif isinstance(n, LoopIR.LoopIR.BinOp):
            yield from self._children_from_attrs(n, "lhs", "rhs")
        elif isinstance(n, LoopIR.LoopIR.BuiltIn):
            yield from self._children_from_attrs(n, "args")
        else:
            assert False, f"case {type(n)} unsupported"

    def _children_from_attrs(self, n, *args) -> Iterable[Node]:
        for attr in args:
            children = getattr(n, attr)
            if isinstance(children, list):
                for i in range(len(children)):
                    yield self._child_node(attr, i)
            else:
                yield self._child_node(attr, None)

    # ------------------------------------------------------------------------ #
    # Navigation (block selectors)
    # ------------------------------------------------------------------------ #

    def body(self) -> Block:
        return self._child_block("body")

    def orelse(self) -> Block:
        return self._child_block("orelse")

    # ------------------------------------------------------------------------ #
    # Conversions
    # ------------------------------------------------------------------------ #

    def as_block(self) -> Block:
        attr, i = self._path[-1]
        if i is None:
            raise InvalidCursorError("node is not inside a block")
        return Block(self._proc, self._path[:-1] + [(attr, range(i, i + 1))])

    # ------------------------------------------------------------------------ #
    # Location queries
    # ------------------------------------------------------------------------ #

    def is_ancestor_of(self, other: Cursor) -> bool:
        """Return true if this node is an ancestor of another"""
        return _starts_with(other._path, self._path)

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def _replace(self, ast):
        """
        This is an UNSAFE internal function for replacing a node in an AST with
        either a list of statements or another single node and providing a
        forwarding function as collateral. It is meant to be package-private,
        not class-private, so it may be called from other internal classes and
        modules, but not from end-user code.
        """
        attr, idx = self._path[-1]
        if idx is not None:
            # noinspection PyProtectedMember
            # delegate block replacement to the Block class
            return self.as_block()._replace(ast)

        # replacing a single expression, or something not in a block
        assert not isinstance(ast, list), "replaced node is not in a block"

        def update(parent):
            return parent.update(**{attr: ast})

        p = API.Procedure(self._rewrite_node(update))

        return p, self._forward_replace(weakref.ref(p))

    def _forward_replace(self, new_proc):
        _, idx = self._path[-1]
        assert idx is None

        def fwd_node(_):
            raise InvalidCursorError("cannot forward replaced nodes")

        def fwd_gap(_):
            assert False, "should never get here"

        def fwd_sel(_):
            assert False, "should never get here"

        return self._make_forward(new_proc, fwd_node, fwd_gap, fwd_sel)


@dataclass
class Gap(Cursor):
    _path: list[tuple[str, int]]

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        assert self._path
        return Node(self._proc, self._path[:-1])

    def before(self, dist=1) -> Node:
        return self._translate(Node._child_node, -dist)

    def after(self, dist=1) -> Node:
        return self._translate(Node._child_node, dist - 1)

    def prev(self, dist=1) -> Gap:
        return self._translate(Node._child_gap, -dist)

    def next(self, dist=1) -> Gap:
        return self._translate(Node._child_gap, dist)

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def _insert(self, stmts: list, policy=ForwardingPolicy.AnchorPre):
        """
        This is an UNSAFE internal function for inserting a list of nodes at a
        particular gap in an AST block and providing a forwarding function as
        collateral. It is meant to be package-private, not class-private, so it
        may be called from other internal classes and modules, but not from
        end-user code.
        """
        assert self._path

        def update(parent):
            attr, i = self._path[-1]
            children = getattr(parent, attr)
            return parent.update(**{attr: children[:i] + stmts + children[i:]})

        p = API.Procedure(self._rewrite_node(update))

        forward = self._forward_insert(weakref.ref(p), len(stmts), policy)
        return p, forward

    def _forward_insert(self, new_proc, ins_len, policy):
        _, ins_idx = self._path[-1]

        def fwd_node(i):
            return i + ins_len * (i >= ins_idx)

        if policy == ForwardingPolicy.AnchorPre:

            def fwd_gap(i):
                return i + ins_len * (i > ins_idx)

        elif policy == ForwardingPolicy.AnchorPost:

            def fwd_gap(i):
                return i + ins_len * (i >= ins_idx)

        else:

            def fwd_gap(i):
                if i == ins_idx:
                    raise InvalidCursorError("insertion gap was invalidated")
                return i + ins_len * (i > ins_idx)

        def fwd_sel(rng):
            return range(
                rng.start + ins_len * (rng.start >= ins_idx),
                rng.stop + ins_len * (rng.stop > ins_idx),
            )

        return self._make_forward(new_proc, fwd_node, fwd_gap, fwd_sel)
