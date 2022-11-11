from __future__ import annotations

import dataclasses
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Optional, Iterable, Union, TypeVar, Generic
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


T = TypeVar("T")


@dataclass(frozen=True)
class Context:
    path: list[(str, Optional[int])] = dataclasses.field(default_factory=list)
    attr: Optional[str] = None

    def follow_path(self, obj):
        for attr, idx in self.path:
            obj = getattr(obj, attr)
            if idx is not None:
                obj = obj[idx]
        if self.attr is not None:
            obj = getattr(obj, self.attr)
        return obj

    def apply(self, obj, fn):
        if self.attr is None:
            return fn(obj)

        def traverse(cur_obj, i):
            assert i <= len(self.path), "bug!"
            if i == len(self.path):
                return cur_obj.update(**{self.attr: fn(getattr(cur_obj, self.attr))})
            attr, j = self.path[i]
            sub = getattr(cur_obj, attr)
            if j is not None:
                return cur_obj.update(
                    **{attr: sub[:j] + [traverse(sub[j], i + 1)] + sub[j + 1 :]}
                )
            return cur_obj.update(**{attr: traverse(sub, i + 1)})

        return traverse(obj, 0)

    def push(self, i, attr) -> Context:
        if self.attr is None:
            assert i is None and not self.path
            return Context([], attr)
        return Context(self.path + [(self.attr, i)], attr)

    def pop(self) -> (Context, (Optional[int], Optional[str])):
        if not self.path:
            return Context(), (None, self.attr)
        attr, i = self.path[-1]
        return Context(self.path[:-1], attr), (i, self.attr)

    def __bool__(self):
        return self.attr is not None


@dataclass(frozen=True)
class Cursor(ABC, Generic[T]):
    _proc: ReferenceType[API.Procedure]
    ctx: Context
    sel: T

    # ------------------------------------------------------------------------ #
    # Static constructors
    # ------------------------------------------------------------------------ #

    @staticmethod
    def root(proc: API.Procedure):
        return Node(weakref.ref(proc), Context(), None)

    def update(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    # ------------------------------------------------------------------------ #
    # Validating accessors
    # ------------------------------------------------------------------------ #

    @property
    def proc(self):
        if (p := self._proc()) is None:
            raise InvalidCursorError("underlying proc was destroyed")
        return p

    # ------------------------------------------------------------------------ #
    # Navigation (abstract)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        # TODO: this is revealing a conceptual ugliness... the popped attr is
        #   irrelevant because we don't have a way of pointing at an attribute
        #   of an ast node (like body or orelse). The parent of an if body is
        #   the same as the parent of its orelse branch.
        #   ...on the other hand, this doesn't need to be abstract anymore!
        if not self.ctx:
            raise InvalidCursorError("cursor does not have a parent")

        ctx, (i, _) = self.ctx.pop()
        return Node(self._proc, ctx, i)

    def _translate(self, ctor, dist):
        if self.sel is None:
            if not self.ctx:
                raise InvalidCursorError("cannot move root cursor")
            raise InvalidCursorError("cursor is not inside block")
        new_i = self.sel + dist
        rng = self.parent().child_block(self.ctx.attr).sel
        if ctor is Gap:
            rng = range(rng.start, rng.stop + 1)
        if new_i not in rng:
            raise InvalidCursorError("cursor is out of range")
        return ctor(self._proc, self.ctx, self.sel + dist)

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
    # Protected path / mutation helpers
    # ------------------------------------------------------------------------ #

    def _rewrite_node(self, fn):
        """
        Applies `fn` to the AST node containing the cursor. The callback is
        passed the raw parent of the pointed-to node/block/gap. The callback is
        expected to return a single, updated node to be placed in the new tree.
        """
        return self.ctx.apply(self.proc.INTERNAL_proc(), fn)

    def _make_local_forward(self, new_proc, fwd_node, fwd_gap, fwd_blk):
        orig_proc = self._proc
        this_ctx = self.ctx
        depth = len(self.ctx.path) + 1
        attr = self.ctx.attr

        def forward(cursor: Cursor) -> Cursor:
            if cursor._proc != orig_proc:
                raise InvalidCursorError("cannot forward unknown procs")

            old_path = cursor.ctx.path + [(cursor.ctx.attr, cursor.sel)]

            # Too shallow?
            if len(old_path) < depth:
                return cursor.update(_proc=new_proc)

            old_attr, old_idx = old_path[depth - 1]

            # Check contexts are the same
            if this_ctx != Context(old_path[: depth - 1], old_attr):
                return cursor.update(_proc=new_proc)

            if len(old_path) > depth or isinstance(cursor, Node):
                idx = fwd_node(old_idx)
            elif isinstance(cursor, Gap):
                idx = fwd_gap(old_idx)
            else:
                assert isinstance(cursor, Block)
                idx = fwd_blk(old_idx)

            new_path = old_path[: depth - 1] + [(attr, idx)] + old_path[depth:]

            return cursor.update(
                _proc=new_proc,
                ctx=Context(new_path[:-1], new_path[-1][0]),
                sel=new_path[-1][1],
            )

        return forward


@dataclass(frozen=True)
class Block(Cursor[Union[int, range]]):  # is range iff last entry
    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def before(self, dist=1) -> Gap:
        return self[0].before(dist)

    def after(self, dist=1) -> Gap:
        return self[-1].after(dist)

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

    def __contains__(self, cur: Cursor):
        if self.ctx != cur.ctx:
            return False

        if isinstance(cur, Node):
            return cur.sel in self.sel

        if isinstance(cur, Gap):
            return self.sel.start <= cur.sel <= self.sel.stop

        assert isinstance(cur, Block)
        return _is_sub_range(cur.sel, self.sel) or cur.sel == self.sel

    # ------------------------------------------------------------------------ #
    # Sequence interface implementation
    # ------------------------------------------------------------------------ #

    def __iter__(self):
        for i in self.sel:
            yield Node(self._proc, self.ctx, i)

    def __getitem__(self, i):
        r = self.sel[i]
        if isinstance(r, range):
            if r.step != 1:
                raise IndexError("block cursors must be contiguous")
            return Block(self._proc, self.ctx, r)
        else:
            return Node(self._proc, self.ctx, r)

    def __len__(self):
        return len(self.sel)

    # ------------------------------------------------------------------------ #
    # Block-specific operations
    # ------------------------------------------------------------------------ #

    def expand(self, lo=None, hi=None):
        full_block = self.parent().child_block(self.ctx.attr).sel

        lo = float("-inf") if lo is None else lo
        lo = max(0, self.sel.start - lo)

        hi = float("inf") if hi is None else hi
        hi = min(len(full_block), self.sel.stop + hi)

        return self.update(sel=range(lo, hi))

    # ------------------------------------------------------------------------ #
    # Location queries
    # ------------------------------------------------------------------------ #

    def is_ancestor_of(self, other: Cursor) -> bool:
        return any(node.is_ancestor_of(other) for node in self)

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def move_to(self, dst: Gap):
        """
        This is an UNSAFE internal function for moving a block in an AST to some
        other location, defined by the gap "dest". It is meant to be
        package-private, not class-private, so it may be called from other
        internal classes and modules, but not from end-user code.
        """
        assert len(self) > 0

        nodes = [x._node() for x in self]

        _, fwd_del = self.delete()
        dst = fwd_del(dst)
        assert isinstance(dst, Gap)
        p, fwd_ins = dst.insert(nodes, policy=ForwardingPolicy.PreferInvalidation)

        def _forward_move_to(c):
            return c.update(_proc=p)

        return p, _forward_move_to

    def replace(self, nodes: list, *, empty_default=None):
        """
        This is an UNSAFE internal function for replacing a block in an AST with
        a list of statements and providing a forwarding function as collateral.
        It is meant to be package-private, not class-private, so it may be
        called from other internal classes and modules, but not from end-user
        code.
        """
        assert len(self) > 0

        i = self.sel

        def update(obj):
            new_children = obj[: i.start] + nodes + obj[i.stop :]
            new_children = new_children or empty_default or []
            return new_children

        p = API.Procedure(self._rewrite_node(update))

        return p, self._forward_replace(weakref.ref(p), len(nodes))

    def _forward_replace(self, new_proc, n_ins):
        del_range = self.sel
        n_diff = n_ins - len(del_range)

        def fwd_node(i):
            if i in del_range:
                raise InvalidCursorError("node no longer exists")
            return i + n_diff * (i >= del_range.stop)

        def fwd_gap(i):
            if i in del_range[1:]:
                raise InvalidCursorError("gap no longer exists")
            return i + n_diff * (i >= del_range.stop)

        def fwd_blk(rng: range):
            if _is_sub_range(rng, del_range):
                raise InvalidCursorError("block no longer exists")
            if _overlaps_one_side(rng, del_range):
                raise InvalidCursorError("block was partially destroyed")

            start = rng.start + n_diff * (rng.start >= del_range.stop)
            stop = rng.stop + n_diff * (rng.stop >= del_range.stop)

            if start >= stop:
                raise InvalidCursorError("block no longer exists")

            return range(start, stop)

        return self._make_local_forward(new_proc, fwd_node, fwd_gap, fwd_blk)

    def delete(self):
        """
        This is an UNSAFE internal function for deleting a block in an AST and
        providing a forwarding function as collateral. It is meant to be
        package-private, not class-private, so it may be called from other
        internal classes and modules, but not from end-user code.
        """
        pass_stmt = [LoopIR.LoopIR.Pass(None, self.parent()._node().srcinfo)]
        return self.replace([], empty_default=pass_stmt)


@dataclass(frozen=True)
class Node(Cursor[Optional[int]]):
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
        n = self.proc.INTERNAL_proc()
        n = self.ctx.follow_path(n)
        if self.sel is not None:
            n = n[self.sel]
        return weakref.ref(n)

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def before(self, dist=1) -> Gap:
        return self._translate(Gap, -(dist - 1))

    def after(self, dist=1) -> Gap:
        return self._translate(Gap, dist)

    def prev(self, dist=1) -> Node:
        return self._translate(Node, -dist)

    def next(self, dist=1) -> Node:
        return self._translate(Node, dist)

    # ------------------------------------------------------------------------ #
    # Navigation (children)
    # ------------------------------------------------------------------------ #

    def child_node(self, attr, i=None) -> Node:
        node = getattr(self._node(), attr)

        if i is not None:
            if 0 <= i < len(node):
                node = node[i]
            else:
                raise InvalidCursorError("cursor is out of range")

        elif isinstance(node, list):
            raise ValueError("must index into block attribute")

        cur = Node(self._proc, self.ctx.push(self.sel, attr), i)
        # This simply overrides the cached property to avoid computing it
        # later since we know the correct value now. Since these classes
        # are frozen, we have to go around Python's back here.
        object.__setattr__(cur, "_node_ref", weakref.ref(node))
        return cur

    def child_gap(self, attr, i=None) -> Gap:
        _node = getattr(self._node(), attr)

        if i is not None:
            if i not in range(len(_node) + 1):
                raise InvalidCursorError("cursor is out of range")

        elif isinstance(_node, list):
            raise ValueError("must index into block attribute")

        return Gap(self._proc, self.ctx.push(self.sel, attr), i)

    def child_block(self, attr: str):
        stmts = getattr(self._node(), attr)
        assert isinstance(stmts, list)
        return Block(self._proc, self.ctx.push(self.sel, attr), range(len(stmts)))

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
                    yield self.child_node(attr, i)
            else:
                yield self.child_node(attr)

    # ------------------------------------------------------------------------ #
    # Navigation (block selectors)
    # ------------------------------------------------------------------------ #

    def body(self) -> Block:
        return self.child_block("body")

    def orelse(self) -> Block:
        return self.child_block("orelse")

    # ------------------------------------------------------------------------ #
    # Conversions
    # ------------------------------------------------------------------------ #

    def as_block(self) -> Block:
        if self.sel is None:
            raise InvalidCursorError("node is not inside a block")
        return Block(self._proc, self.ctx, range(self.sel, self.sel + 1))

    # ------------------------------------------------------------------------ #
    # Location queries
    # ------------------------------------------------------------------------ #

    def is_ancestor_of(self, other: Cursor) -> bool:
        """Return true if this node is an ancestor of another"""
        return _starts_with(
            other.ctx.path + [(other.ctx.attr, other.sel)],
            self.ctx.path + [(self.ctx.attr, self.sel)],
        )

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def replace(self, ast):
        """
        This is an UNSAFE internal function for replacing a node in an AST with
        either a list of statements or another single node and providing a
        forwarding function as collateral. It is meant to be package-private,
        not class-private, so it may be called from other internal classes and
        modules, but not from end-user code.
        """
        if self.sel is not None:
            # noinspection PyProtectedMember
            # delegate block replacement to the Block class
            return self.as_block().replace(ast)

        # replacing a single expression, or something not in a block
        assert not isinstance(ast, list), "replaced node is not in a block"

        def update(_):
            return ast

        p = API.Procedure(self._rewrite_node(update))
        return p, self._forward_replace(weakref.ref(p))

    def _forward_replace(self, new_proc):
        idx = self.sel
        assert idx is None

        def fwd_node(_):
            raise InvalidCursorError("cannot forward replaced nodes")

        def fwd_gap(_):
            assert False, "should never get here"

        def fwd_blk(_):
            assert False, "should never get here"

        return self._make_local_forward(new_proc, fwd_node, fwd_gap, fwd_blk)


@dataclass(frozen=True)
class Gap(Cursor[int]):
    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def before(self, dist=1) -> Node:
        return self._translate(Node, -dist)

    def after(self, dist=1) -> Node:
        return self._translate(Node, dist - 1)

    def prev(self, dist=1) -> Gap:
        return self._translate(Gap, -dist)

    def next(self, dist=1) -> Gap:
        return self._translate(Gap, dist)

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def insert(self, stmts: list, policy=ForwardingPolicy.AnchorPre):
        """
        This is an UNSAFE internal function for inserting a list of nodes at a
        particular gap in an AST block and providing a forwarding function as
        collateral. It is meant to be package-private, not class-private, so it
        may be called from other internal classes and modules, but not from
        end-user code.
        """

        def update(obj):
            return obj[: self.sel] + stmts + obj[self.sel :]

        p = API.Procedure(self._rewrite_node(update))

        forward = self._forward_insert(weakref.ref(p), len(stmts), policy)
        return p, forward

    def _forward_insert(self, new_proc, ins_len, policy):
        ins_idx = self.sel

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

        def fwd_blk(rng):
            return range(
                rng.start + ins_len * (rng.start >= ins_idx),
                rng.stop + ins_len * (rng.stop > ins_idx),
            )

        return self._make_local_forward(new_proc, fwd_node, fwd_gap, fwd_blk)
