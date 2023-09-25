from __future__ import annotations

import dataclasses
import enum
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional


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
        if isinstance(cursor, Gap):
            return dataclasses.replace(
                cursor, _root=p, _anchor=dataclasses.replace(cursor._anchor, _root=p)
            )
        elif isinstance(cursor, Node):
            return dataclasses.replace(cursor, _root=p)
        else:
            raise InvalidCursorError("cannot forward blocks")

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

    # ------------------------------------------------------------------------ #
    # Protected path / mutation helpers
    # ------------------------------------------------------------------------ #

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

        edit_path = self.parent()._path

        if isinstance(self, Node):
            attr = self._path[-1][0]
        elif isinstance(self, Gap):
            attr = self.anchor()._path[-1][0]
        else:
            assert isinstance(self, Block)
            attr = self._attr

        depth = len(edit_path)

        def forward(cursor: Cursor) -> Cursor:
            if cursor._root is not orig_root:
                raise InvalidCursorError("cannot forward from unknown root")

            if isinstance(cursor, Gap):
                return Gap(new_root, forward(cursor.anchor()), cursor.type())

            if isinstance(cursor, Block):
                raise InvalidCursorError("cannot forward blocks")

            assert isinstance(cursor, Node)

            def evolve(p):
                return dataclasses.replace(cursor, _root=new_root, _path=p)

            old_path = cursor._path

            if len(old_path) < depth + 1:
                # Too shallow
                return evolve(old_path)

            old_attr, old_idx = old_path[depth]

            if not (_starts_with(old_path, edit_path) and old_attr == attr):
                # Same path down tree
                return evolve(old_path)

            return evolve(
                old_path[:depth] + fwd_node(attr, old_idx) + old_path[depth + 1 :]
            )

        return forward


@dataclass
class Block(Cursor):
    _anchor: Node
    _attr: str  # must be 'body' or 'orelse'
    _range: range

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        return self._anchor

    def before(self) -> Gap:
        return self[0].before()

    def after(self) -> Gap:
        return self[-1].after()

    # ------------------------------------------------------------------------ #
    # Container interface implementation
    # ------------------------------------------------------------------------ #

    def __contains__(self, cur):
        if isinstance(cur, Block):
            return (
                cur._anchor == self._anchor
                and cur._attr == self._attr
                and (
                    _is_sub_range(cur._range, self._range) or cur._range == self._range
                )
            )
        elif isinstance(cur, Gap):
            # This shouldn't recurse forever...
            return not cur.is_edge() and cur.anchor() in self
        elif isinstance(cur, Node):
            return (
                cur.parent() == self._anchor
                and cur._path[-1][0] == self._attr
                and cur._path[-1][1] in self._range
            )
        else:
            raise TypeError(f"cannot check containment of {type(cur)}")

    # ------------------------------------------------------------------------ #
    # Sequence interface implementation
    # ------------------------------------------------------------------------ #

    def __iter__(self):
        block = self.parent()
        for i in self._range:
            yield block._child_node(self._attr, i)

    def __getitem__(self, i):
        r = self._range[i]
        if isinstance(r, range):
            if r.step != 1:
                raise IndexError("block cursors must be contiguous")
            return Block(self._root, self._anchor, self._attr, r)
        else:
            return self._anchor._child_node(self._attr, r)

    def __len__(self):
        return len(self._range)

    # ------------------------------------------------------------------------ #
    # Block-specific operations
    # ------------------------------------------------------------------------ #

    def expand(self, delta_lo=None, delta_hi=None):
        """
        [delta_lo] or [delta_hi] being [None] means expand as much as possible to the left
        or right, respectively.
        """
        full_block = self.parent()._child_block(self._attr)
        full_range = full_block._range

        delta_lo = self._range.start if delta_lo is None else delta_lo
        delta_hi = len(full_range) - self._range.stop if delta_hi is None else delta_hi

        lo = max(0, self._range.start - delta_lo)
        hi = min(len(full_range), self._range.stop + delta_hi)

        return Block(self._root, self._anchor, self._attr, range(lo, hi))

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
        assert isinstance(nodes, list)

        def update(n):
            r = self._range
            children = getattr(n, self._attr)
            new_children = children[: r.start] + nodes + children[r.stop :]
            new_children = new_children or empty_default or []
            return n.update(**{self._attr: new_children})

        p = self._anchor._rewrite(update)
        fwd = self._forward_replace(p, len(nodes))
        return p, fwd

    def _forward_replace(self, new_proc, n_ins):
        del_range = self._range
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
        # TODO: refactor this; LoopIR should not be imported here
        from exo.LoopIR import LoopIR

        pass_stmt = [LoopIR.Pass(None, self.parent()._node.srcinfo)]
        return self._replace([], empty_default=pass_stmt)

    def resolve_all(self):
        """
        Do this rather than `[n._node for n in self]` because that would
        walk down the tree once per node in the block, whereas this walks
        down once.
        """
        return getattr(self._anchor._node, self._attr)[
            self._range.start : self._range.stop
        ]

    def _wrap(self, ctor, wrap_attr):
        """
        This is an UNSAFE internal function for wrapping a block in an AST
        with another block-containing node and providing a forwarding function
        as collateral. It is meant to be package-private, not class-private,
        so it may be called from other internal classes and modules, but not
        from end-user code.
        """
        nodes = self.resolve_all()
        new_node = ctor(**{wrap_attr: nodes})

        def update(parent):
            r = self._range
            children = getattr(parent, self._attr)
            new_children = children[: r.start] + [new_node] + children[r.stop :]
            return parent.update(**{self._attr: new_children})

        p = self._anchor._rewrite(update)
        fwd = self._forward_wrap(p, wrap_attr)
        return p, fwd

    def _forward_wrap(self, p, wrap_attr):
        rng = self._range

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

        nodes = self.resolve_all()

        def _is_before(g: Gap, b: Block):
            b_path = b._anchor._path + [(b._attr, b._range.start)]

            g_path = g._anchor._path[:]
            g_path[-1] = (g_path[-1][0], g._insertion_index())

            for (g_attr, g_idx), (b_attr, b_idx) in zip(g_path, b_path):
                if g_attr != b_attr:
                    # arbitrary because they're in disjoint branches
                    return False

                if g_idx != b_idx:
                    return g_idx < b_idx

            return True

        def reroot(x):
            return dataclasses.replace(x, _root=ir)

        # The following implementation "unsafely" coerces a cursor along the
        # intermediate procs by ordering the edits so that identity-forwarding
        # is actually safe. This is somewhat simpler to reason about than a
        # recursive function that has to walk down two branches simultaneously.
        # Intuition: do the "later" edit first, then the "earlier" one.
        if _is_before(target, self):
            # If the gap comes first in a pre-order traversal, then we want to
            # delete the original block of nodes first, to keep the path to the
            # gap stable, before inserting the nodes in the new position.
            ir, _ = self._delete()
            ir, _ = Gap(ir, reroot(target._anchor), target._type)._insert(nodes)
        else:
            # Conversely, if the moved block comes first, then we want to jump
            # ahead and insert the block into the gap position before coming back
            # to delete the original nodes, so that the path to the deletion stays
            # stable.
            ir, _ = target._insert(nodes)
            ir, _ = Block(ir, reroot(self._anchor), self._attr, self._range)._delete()

        fwd = self._forward_move(ir, target)
        return ir, fwd

    def _forward_move(self, p, target: Gap):
        orig_root = self._root

        block_path = self._anchor._path
        block_n = len(block_path)

        block_attr = self._attr

        block_rng = self._range
        edit_n = len(block_rng)

        gap_path = target._anchor._path[:]
        gap_path[-1] = (gap_path[-1][0], target._insertion_index())
        gap_n = len(gap_path) - 1

        def forward(cursor: Node):
            # This is duplicated in _local_forward. If there ever is a third
            # place where this is needed, it should be refactored into a
            # helper function.
            if cursor._root is not orig_root:
                raise InvalidCursorError("cannot forward from unknown root")

            if isinstance(cursor, Gap):
                return Gap(p, forward(cursor.anchor()), cursor.type())

            if isinstance(cursor, Block):
                raise InvalidCursorError("cannot forward blocks")

            assert isinstance(cursor, Node)

            cur_path = list(cursor._path)
            cur_n = len(cur_path)

            # Handle nodes around the edit points
            offsets = []

            if (
                cur_n > block_n
                and block_path == cur_path[:block_n]
                and block_attr == cur_path[block_n][0]
            ):
                if block_rng.stop <= cur_path[block_n][1]:
                    # if after orig. block end, subtract edit_n
                    offsets.append((block_n, -edit_n))
                elif block_rng.start <= cur_path[block_n][1]:
                    new_gap_path = gap_path
                    if block_n <= gap_n:
                        # compute new gap_path
                        block_start_path = block_path + [(block_attr, block_rng.start)]

                        new_gap_path = []
                        for lca_n, (bs, gs) in enumerate(
                            zip(block_start_path, gap_path)
                        ):
                            if bs != gs:
                                if bs[0] == gs[0] and bs[1] < gs[1]:
                                    new_gap_path.append((gs[0], gs[1] - edit_n))
                                else:
                                    new_gap_path.append(gs)
                                break

                            new_gap_path.append(gs)

                        new_gap_path.extend(gap_path[lca_n + 1 :])

                    # if inside orig block, move to gap location
                    off = cur_path[block_n][1] - block_rng.start
                    return dataclasses.replace(
                        cursor,
                        _root=p,
                        _path=(
                            new_gap_path[:-1]
                            + [(new_gap_path[-1][0], new_gap_path[-1][1] + off)]
                            + cur_path[block_n + 1 :]
                        ),
                    )
                else:
                    # before orig block, do nothing
                    pass

            # if after orig. gap, add edit_n
            if (
                cur_n > gap_n
                and gap_path[:gap_n] == cur_path[:gap_n]
                and gap_path[gap_n][0] == cur_path[gap_n][0]
                and gap_path[gap_n][1] <= cur_path[gap_n][1]
            ):
                offsets.append((gap_n, edit_n))

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

    def before(self) -> Gap:
        return Gap(self._root, self, GapType.Before)

    def after(self) -> Gap:
        return Gap(self._root, self, GapType.After)

    def prev(self, dist=1) -> Node:
        return self.next(-dist)

    def next(self, dist=1) -> Node:
        if not self._path:
            raise InvalidCursorError("cannot move root cursor")
        attr, i = self._path[-1]
        if i is None:
            raise InvalidCursorError("cursor is not inside block")
        return self.parent()._child_node(attr, i + dist)

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

    def _child_block(self, attr: str):
        stmts = getattr(self._node, attr)
        assert isinstance(stmts, list)
        return Block(self._root, self, attr, range(len(stmts)))

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
        return Block(self._root, self.parent(), attr, range(i, i + 1))

    # ------------------------------------------------------------------------ #
    # Location queries
    # ------------------------------------------------------------------------ #

    def get_index(self):
        _, i = self._path[-1]
        return i

    def is_ancestor_of(self, other: Cursor) -> bool:
        """Return true if this node is an ancestor of another"""
        if not isinstance(other, Node):
            other = other._anchor
        return _starts_with(other._path, self._path)

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def _rewrite(self, fn):
        """
        Applies `fn` to the current node and rewrites the tree with the result.
        """

        def impl(node, path, j=0):
            if j == len(path):
                return fn(node)

            attr, i = path[j]
            children = getattr(node, attr)

            if i is None:
                return node.update(**{attr: impl(children, path, j + 1)})

            new_nodes = impl(children[i], path, j + 1)
            if not isinstance(new_nodes, list):
                new_nodes = [new_nodes]
            return node.update(**{attr: children[:i] + new_nodes + children[i + 1 :]})

        return impl(self.get_root(), self._path)

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

        p = self._rewrite(lambda _: ast)
        fwd = self._forward_replace(p)
        return p, fwd

    def _forward_replace(self, new_root):
        assert self._path[-1][1] is None

        def fwd_node(*_):
            raise InvalidCursorError("cannot forward replaced nodes")

        return self._local_forward(new_root, fwd_node)


class GapType(enum.Enum):
    Before = 0
    After = 1
    # BodyStart = enum.auto()
    # BodyEnd = enum.auto()
    # OrElseStart = enum.auto()
    # OrElseEnd = enum.auto()


@dataclass
class Gap(Cursor):
    _anchor: Node
    _type: GapType

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        if self.is_edge():
            return self._anchor
        return self._anchor.parent()

    def anchor(self) -> Node:
        return self._anchor

    # ------------------------------------------------------------------------ #
    # Gap type queries
    # ------------------------------------------------------------------------ #

    def is_edge(self):
        return self._type not in (GapType.Before, GapType.After)

    def type(self) -> GapType:
        return self._type

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

        def update(anchor):
            if self._type == GapType.Before:
                return stmts + [anchor]
            elif self._type == GapType.After:
                return [anchor] + stmts
            else:
                assert False, f"case {self._type} not implemented"

        if self.is_edge():
            raise NotImplementedError()

        p = self.anchor()._rewrite(update)
        fwd = self._forward_insert(p, len(stmts))
        return p, fwd

    def _forward_insert(self, new_root, ins_len):
        ins_idx = self._insertion_index()

        def fwd_node(attr, i):
            return [(attr, i + ins_len * (i >= ins_idx))]

        return self._local_forward(new_root, fwd_node)

    def _insertion_index(self):
        _, i = self._anchor._path[-1]
        if self._type == GapType.Before:
            return i
        elif self._type == GapType.After:
            return i + 1
        else:
            assert False, f"case {self.type} not implemented"
