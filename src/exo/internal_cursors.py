from __future__ import annotations

import dataclasses
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Iterable, Union

from exo import LoopIR


class InvalidCursorError(Exception):
    pass


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
    return len(a) >= len(b) and all(x == y for x, y in zip(a, b))


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


def forward_identity(p, fwd=None):
    fwd = fwd or (lambda x: x)

    @functools.wraps(fwd)
    def forward(cursor):
        cursor = fwd(cursor)
        return dataclasses.replace(cursor, _root=p)

    return forward


@dataclass
class Cursor(ABC):
    _root: object

    # ------------------------------------------------------------------------ #
    # Static constructors
    # ------------------------------------------------------------------------ #

    @staticmethod
    def create(obj: object):
        return Node(obj, [])

    # ------------------------------------------------------------------------ #
    # Validating accessors
    # ------------------------------------------------------------------------ #

    def get_root(self):
        return self.root()._node

    # ------------------------------------------------------------------------ #
    # Navigation (universal)
    # ------------------------------------------------------------------------ #

    def root(self) -> Node:
        """Get a cursor to the root of the tree this cursor resides in"""
        return Node(self._root, [])

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

        return impl(self.get_root(), self._path)

    def _local_forward(self, new_root, fwd_node):
        """
        Creates a forwarding function for "local" edits to the AST.
        Here, local means that all affected nodes share a common LCA,
        taking into account details about block identity (i.e. body
        vs orelse). It is assumed that the edited locale is the block
        in which the present cursor resides.

        The fwd_node function is applied to one of this cursor's siblings
        and is expected to return a list of new tree edges which are adjusted
        for the edit. See the implementation of insert and delete for some
        simple examples, or wrap for a more complex example (that lengthens
        certain paths).
        """
        orig_root = self._root
        edit_path = self._path
        depth = len(edit_path)
        attr = edit_path[depth - 1][0]

        def forward(cursor: Cursor) -> Cursor:
            if cursor._root != orig_root:
                raise InvalidCursorError("cannot forward from unknown root")

            if not isinstance(cursor, Node):
                raise InvalidCursorError("can only forward nodes")

            def evolve(p):
                return dataclasses.replace(cursor, _root=new_root, _path=p)

            old_path = cursor._path

            if len(old_path) < depth:
                # Too shallow
                return evolve(old_path)

            old_attr, old_idx = old_path[depth - 1]

            if not (_starts_with(old_path, edit_path[:-1]) and old_attr == attr):
                # Same path down tree
                return evolve(old_path)

            return evolve(
                old_path[: depth - 1] + fwd_node(attr, old_idx) + old_path[depth:]
            )

        return forward


@dataclass
class Block(Cursor):
    _path: list[tuple[str, Union[int, range]]]  # is range iff last entry

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        return Node(self._root, self._path[:-1])

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
            return Block(self._root, self._path[:-1] + [(attr, r)])
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
            lo = 0
        else:
            lo = _range.start - lo
            lo = lo if lo >= 0 else 0
        if hi is None:
            hi = len(full_range)
        else:
            hi = _range.stop + hi
        new_range = full_range[lo:hi]

        return Block(self._root, self._path[:-1] + [(attr, new_range)])

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
        # assert len(self) > 0
        assert isinstance(nodes, list)

        def update(parent):
            attr, i = self._path[-1]
            children = getattr(parent, attr)
            new_children = children[: i.start] + nodes + children[i.stop :]
            new_children = new_children or empty_default or []
            return parent.update(**{attr: new_children})

        p = self._rewrite_node(update)
        fwd = self._forward_replace(p, len(nodes))
        return p, fwd

    def _forward_replace(self, new_proc, n_ins):
        _, del_range = self._path[-1]
        n_diff = n_ins - len(del_range)

        def fwd_node(attr, i):
            if i in del_range:
                raise InvalidCursorError("node no longer exists")
            return [(attr, i + n_diff * (i >= del_range.stop))]

        return self._local_forward(new_proc, fwd_node)

    def _delete(self):
        """
        This is an UNSAFE internal function for deleting a block in an AST and
        providing a forwarding function as collateral. It is meant to be
        package-private, not class-private, so it may be called from other
        internal classes and modules, but not from end-user code.
        """
        pass_stmt = [LoopIR.LoopIR.Pass(None, self.parent()._node.srcinfo)]
        return self._replace([], empty_default=pass_stmt)

    def _get_loopir(self):
        # Do this rather than [n._node for n in self] because that would
        # walk down the tree once per node in the block, whereas this walks
        # down once.
        attr, rng = self._path[-1]
        return getattr(self.parent()._node, attr)[rng.start : rng.stop]

    def _wrap(self, ctor, wrap_attr):
        """
        This is an UNSAFE internal function for wrapping a block in an AST
        with another block-containing node and providing a forwarding function
        as collateral. It is meant to be package-private, not class-private,
        so it may be called from other internal classes and modules, but not
        from end-user code.
        """
        nodes = self._get_loopir()
        new_node = ctor(**{wrap_attr: nodes})

        def update(parent):
            orig_attr, i = self._path[-1]
            children = getattr(parent, orig_attr)
            new_children = children[: i.start] + [new_node] + children[i.stop :]
            return parent.update(**{orig_attr: new_children})

        p = self._rewrite_node(update)
        fwd = self._forward_wrap(p, wrap_attr)
        return p, fwd

    def _forward_wrap(self, p, wrap_attr):
        rng = self._path[-1][1]

        def forward(attr, i):
            if i >= rng.stop:
                return [(attr, i - len(rng) + 1)]
            elif i >= rng.start:
                return [(attr, rng.start), (wrap_attr, i - rng.start)]
            else:
                return [(attr, i)]

        return self._local_forward(p, forward)

    def _move(self, target: Gap):
        """
        This is an UNSAFE internal function for relocating a block in an AST
        and providing a forwarding function as collateral. It is meant to be
        package-private, not class-private, so it may be called from other
        internal classes and modules, but not from end-user code.
        """

        if target in self:
            target = self.before()

        nodes = self._get_loopir()

        def _is_before(g: Gap, b: Block):
            b_path = b._path[:-1] + [(b._path[-1][0], b._path[-1][1].start)]

            for (g_attr, g_idx), (b_attr, b_idx) in zip(g._path, b_path):
                if g_attr != b_attr:
                    # arbitrary because they're in disjoint branches
                    return False

                if g_idx != b_idx:
                    return g_idx < b_idx

            return True

        # The following implementation "unsafely" coerces a cursor along the
        # intermediate procs by ordering the edits so that identity-forwarding
        # is actually safe. This is somewhat simpler to reason about than a
        # recursive function that has to walk down two branches simultaneously.
        # Intuition: do the "later" edit first, then the "earlier" one.
        if _is_before(target, self):
            # If the gap comes first in a pre-order traversal, then we want to
            # delete the original block of nodes first, to keep the path to the
            # gap stable, before inserting the nodes in the new position.
            p, _ = self._delete()
            p, _ = dataclasses.replace(target, _root=p)._insert(nodes)
        else:
            # Conversely, if the moved block comes first, then we want to jump
            # ahead and insert the block into the gap position before coming back
            # to delete the original nodes, so that the path to the deletion stays
            # stable.
            p, _ = target._insert(nodes)
            p, _ = dataclasses.replace(self, _root=p)._delete()

        fwd = self._forward_move(p, target)
        return p, fwd

    def _forward_move(self, p, target):
        orig_root = self._root

        block_path = self._path
        block_n = len(block_path)

        edit_n = len(block_path[-1][1])

        gap_path = target._path
        gap_n = len(gap_path)

        def forward(cursor: Node):
            if cursor._root != orig_root:
                raise InvalidCursorError("cannot forward from unknown root")

            if not isinstance(cursor, Node):
                raise InvalidCursorError("can only forward nodes")

            cur_path = list(cursor._path)
            cur_n = len(cur_path)

            # Compute the gap offset when moving within a block
            if (
                block_n == gap_n
                and block_path[: block_n - 1] == gap_path[: block_n - 1]
                and block_path[block_n - 1][0] == gap_path[block_n - 1][0]
                and block_path[block_n - 1][1].stop <= gap_path[block_n - 1][1]
            ):
                gap_off = -edit_n
            else:
                gap_off = 0

            # Handle nodes around the edit points
            offsets = []

            if (
                cur_n >= block_n
                and block_path[: block_n - 1] == cur_path[: block_n - 1]
                and block_path[block_n - 1][0] == cur_path[block_n - 1][0]
            ):
                if block_path[block_n - 1][1].stop <= cur_path[block_n - 1][1]:
                    # if after orig. block end, subtract edit_n
                    offsets.append((block_n - 1, -edit_n))
                elif block_path[block_n - 1][1].start <= cur_path[block_n - 1][1]:
                    # if inside orig block, move to gap location
                    off = cur_path[block_n - 1][1] - block_path[block_n - 1][1].start
                    return dataclasses.replace(
                        cursor,
                        _root=p,
                        _path=(
                            gap_path[:-1]
                            + [(gap_path[-1][0], gap_path[-1][1] + gap_off + off)]
                            + cur_path[block_n:]
                        ),
                    )
                else:
                    # before orig block, do nothing
                    pass

            # if after orig. gap, add edit_n
            if (
                cur_n >= gap_n
                and gap_path[: gap_n - 1] == cur_path[: gap_n - 1]
                and gap_path[gap_n - 1][0] == cur_path[gap_n - 1][0]
                and gap_path[gap_n - 1][1] <= cur_path[gap_n - 1][1]
            ):
                offsets.append((gap_n - 1, edit_n))

            for off_i, off_d in offsets:
                cur_path[off_i] = (cur_path[off_i][0], cur_path[off_i][1] + off_d)

            return dataclasses.replace(cursor, _root=p, _path=cur_path)

        return forward


@dataclass
class Node(Cursor):
    _path: list[tuple[str, Optional[int]]]

    # ------------------------------------------------------------------------ #
    # Validating accessors
    # ------------------------------------------------------------------------ #

    @cached_property
    def _node(self):
        """
        Gets the raw underlying node that's pointed-to. This is meant to be
        compiler-internal, not class-private, so other parts of the compiler
        may call this, while users should not.
        """
        n = self._root

        # TODO: this is what we're trying to remove.
        if isinstance(n, LoopIR.LoopIR.proc):
            pass
        else:
            n = n.INTERNAL_proc()

        for attr, idx in self._path:
            n = getattr(n, attr)
            if idx is not None:
                n = n[idx]

        return n

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        if not self._path:
            raise InvalidCursorError("cursor does not have a parent")
        return Node(self._root, self._path[:-1])

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
        _node = getattr(self._node, attr)
        if i is not None:
            if 0 <= i < len(_node):
                _node = _node[i]
            else:
                raise InvalidCursorError("cursor is out of range")
        elif isinstance(_node, list):
            raise ValueError("must index into block attribute")
        cur = Node(self._root, self._path + [(attr, i)])
        # noinspection PyPropertyAccess
        # cached_property is settable, bug in static analysis
        cur._node = _node
        return cur

    def _child_gap(self, attr, i=None) -> Gap:
        _node = getattr(self._node, attr)
        if i is not None:
            if not 0 <= i <= len(_node):
                raise InvalidCursorError("cursor is out of range")
        elif isinstance(_node, list):
            raise ValueError("must index into block attribute")
        return Gap(self._root, self._path + [(attr, i)])

    def _child_block(self, attr: str):
        stmts = getattr(self._node, attr)
        assert isinstance(stmts, list)
        return Block(self._root, self._path + [(attr, range(len(stmts)))])

    def children(self) -> Iterable[Node]:
        n = self._node
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
        elif isinstance(n, LoopIR.LoopIR.WindowExpr):
            yield from self._children_from_attrs(n, "idx")
        elif isinstance(n, LoopIR.LoopIR.Interval):
            yield from self._children_from_attrs(n, "lo", "hi")
        elif isinstance(n, LoopIR.LoopIR.Point):
            yield from self._children_from_attrs(n, "pt")
        elif isinstance(
            n,
            (
                LoopIR.LoopIR.Const,
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
        return Block(self._root, self._path[:-1] + [(attr, range(i, i + 1))])

    # ------------------------------------------------------------------------ #
    # Location queries
    # ------------------------------------------------------------------------ #

    def get_index(self):
        _, i = self._path[-1]
        return i

    def is_ancestor_of(self, other: Cursor) -> bool:
        """Return true if this node is an ancestor of another"""
        return _starts_with(other._path, self._path)

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def _delete(self):
        return self.as_block()._delete()

    def _move(self, gap: Gap):
        return self.as_block()._move(gap)

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

        p = self._rewrite_node(update)
        fwd = self._forward_replace(p)
        return p, fwd

    def _forward_replace(self, new_root):
        _, idx = self._path[-1]
        assert idx is None

        def fwd_node(*_):
            raise InvalidCursorError("cannot forward replaced nodes")

        return self._local_forward(new_root, fwd_node)


@dataclass
class Gap(Cursor):
    _path: list[tuple[str, int]]

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        assert self._path
        return Node(self._root, self._path[:-1])

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

    def _insert(self, stmts: list):
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

        p = self._rewrite_node(update)
        fwd = self._forward_insert(p, len(stmts))
        return p, fwd

    def _forward_insert(self, new_root, ins_len):
        _, ins_idx = self._path[-1]

        def fwd_node(attr, i):
            return [(attr, i + ins_len * (i >= ins_idx))]

        return self._local_forward(new_root, fwd_node)
