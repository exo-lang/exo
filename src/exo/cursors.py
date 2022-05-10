from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Optional, Iterable, Union
from weakref import ReferenceType

from .API_types import ProcedureBase
from .LoopIR import LoopIR


class InvalidCursorError(Exception):
    pass


class ForwardingPolicy(Enum):
    EagerInvalidation = 0
    AnchorPre = 1
    AnchorPost = 2
    AnchorNearest = 3


@dataclass
class Cursor(ABC):
    _proc: ReferenceType[ProcedureBase]

    # ------------------------------------------------------------------------ #
    # Static constructors
    # ------------------------------------------------------------------------ #

    @staticmethod
    def root(proc: ProcedureBase):
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
        """For gaps, get the node before the gap. Otherwise, get the gap before the node
        or selection"""

    @abstractmethod
    def after(self, dist=1) -> Cursor:
        """For gaps, get the node after the gap. Otherwise, get the gap after the node
        or selection"""

    @abstractmethod
    def prev(self, dist=1) -> Cursor:
        """Get the previous node/gap in the block. Undefined for selections."""

    @abstractmethod
    def next(self, dist=1) -> Cursor:
        """Get the next node/gap in the block. Undefined for selections."""

    # ------------------------------------------------------------------------ #
    # Protected path / mutation helpers
    # ------------------------------------------------------------------------ #

    def _translate(self, ty, path, dist):
        if not path:
            raise InvalidCursorError('cannot move root cursor')
        attr, i = path[-1]
        if i is None:
            raise InvalidCursorError('cursor is not inside block')
        return ty(self._proc, path[:-1] + [(attr, i + dist)])

    @staticmethod
    def _walk_path(n, path):
        for attr, idx in path:
            n = getattr(n, attr)
            if idx is not None:
                n = n[idx]
        return n

    def _rewrite_node(self, fn):
        """
        Applies `fn` to the AST node containing the cursor. The callback gets
        the following arguments:
        1. The parent node
        2. The relevant list of children
        3. The last step in the path (the attribute and index info)
        The callback is expected to return a single, updated node to be placed
        in the new tree.
        """
        assert isinstance(self, (Node, Selection, Gap))

        def impl(node, path):
            if len(path) == 1:
                return fn(node)

            (attr, i), path = path[0], path[1:]
            children = getattr(node, attr)

            if i is None:
                return node.update(**{attr: impl(children, path)})

            return node.update(**{
                attr: children[:i] + [impl(children[i], path)] + children[i + 1:]
            })

        return impl(self.proc().INTERNAL_proc(), self._path)

    def _make_forward(self, new_proc, fwd_node, fwd_gap, fwd_sel):
        orig_proc = self._proc
        depth = len(self._path)
        attr = self._path[depth - 1][0]
        del self

        def forward(cursor: Cursor) -> Cursor:
            if cursor._proc != orig_proc:
                raise InvalidCursorError('cannot forward unknown procs')

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
                assert isinstance(cursor, Selection)
                idx = fwd_sel(old_idx)

            return evolve(old_path[:depth - 1] + [(attr, idx)] + old_path[depth:])

        return forward


@dataclass
class Selection(Cursor):
    _path: list[tuple[str, Union[int, range]]]  # is range iff last entry

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        return Node(self._proc, self._path[:-1])

    def before(self, dist=1) -> Gap:
        attr, _range = self._path[-1]
        assert len(_range) > 0
        return self.parent().child(attr, _range.start).before(dist)

    def after(self, dist=1) -> Gap:
        attr, _range = self._path[-1]
        assert len(_range) > 0
        return self.parent().child(attr, _range.stop - 1).after(dist)

    def prev(self, dist=1) -> Cursor:
        # TODO: what should this mean?
        #  1. The node after the selection?
        #  2. The selection shifted over?
        #  3. The selection of nodes leading up to the start?
        raise NotImplementedError('Selection.prev')

    def next(self, dist=1) -> Cursor:
        # TODO: what should this mean?
        #  1. The node after the selection?
        #  2. The selection shifted over?
        #  3. The selection of nodes past the end?
        raise NotImplementedError('Selection.next')

    # ------------------------------------------------------------------------ #
    # Sequence interface implementation
    # ------------------------------------------------------------------------ #

    def __iter__(self):
        attr, _range = self._path[-1]
        block = self.parent()
        for i in _range:
            yield block.child(attr, i)

    def __getitem__(self, i):
        attr, r = self._path[-1]
        r = r[i]
        if isinstance(r, range):
            if r.step != 1:
                raise IndexError('cursor selections must be contiguous')
            return Selection(self._proc, self._path[:-1] + [(attr, r)])
        else:
            return self.parent().child(attr, r)

    def __len__(self):
        _, _range = self._path[-1]
        return len(_range)

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def replace(self, stmts: list):
        assert self._path
        assert len(self) > 0
        assert stmts

        def update(node):
            attr, i = self._path[-1]
            children = getattr(node, attr)
            return node.update(**{attr: children[:i.start] + stmts + children[i.stop:]})

        from .API import Procedure
        p = Procedure(self._rewrite_node(update))

        return p, self._forward_replace(weakref.ref(p))

    def _forward_replace(self, new_proc):
        def forward(cursor: Cursor):
            raise NotImplementedError('Selection.replace+fwd')

        return forward

    def delete(self):
        assert self._path
        assert len(self) > 0

        def update(node):
            attr, i = self._path[-1]
            children = getattr(node, attr)
            new_children = children[:i.start] + children[i.stop:]
            new_children = new_children or [LoopIR.Pass(None, node.srcinfo)]
            return node.update(**{attr: new_children})

        from .API import Procedure
        p = Procedure(self._rewrite_node(update))

        return p, self._forward_delete(weakref.ref(p))

    def _forward_delete(self, new_proc):
        del_range = self._path[-1][1]

        def fwd_node(i):
            if i in del_range:
                raise InvalidCursorError('cannot forward deleted node')
            return i - len(del_range) * (i >= del_range.stop)

        def fwd_gap(i):
            if del_range.start < i < del_range.stop:
                raise InvalidCursorError('cannot forward deleted node')
            return i - len(del_range) * (i >= del_range.stop)

        def fwd_sel(rng):
            start = rng.start
            if rng.start in del_range:
                start = del_range.start
            stop = rng.stop
            if rng.stop in del_range:
                stop = del_range.stop
            if range(start, stop) == del_range:
                raise InvalidCursorError('cannot forward deleted selection')
            start = start - len(del_range) * (start >= del_range.stop)
            stop = stop - len(del_range) * (stop >= del_range.stop)
            return range(start, stop)

        return self._make_forward(new_proc, fwd_node, fwd_gap, fwd_sel)


@dataclass
class Node(Cursor):
    _path: list[tuple[str, Optional[int]]]

    # ------------------------------------------------------------------------ #
    # Validating accessors
    # ------------------------------------------------------------------------ #

    def node(self):
        if (n := self._node()) is None:
            raise InvalidCursorError('underlying node was destroyed')
        return n

    @cached_property
    def _node(self):
        n = self.proc().INTERNAL_proc()
        n = self._walk_path(n, self._path)
        return weakref.ref(n)

    # ------------------------------------------------------------------------ #
    # Navigation (implementation)
    # ------------------------------------------------------------------------ #

    def parent(self) -> Node:
        if not self._path:
            raise InvalidCursorError('cursor does not have a parent')
        return Node(self._proc, self._path[:-1])

    def before(self, dist=1) -> Gap:
        return self._translate(Gap, self._path, 1 - dist)

    def after(self, dist=1) -> Gap:
        return self._translate(Gap, self._path, dist)

    def prev(self, dist=1) -> Node:
        return self._translate(Node, self._path, -dist)

    def next(self, dist=1) -> Node:
        return self._translate(Node, self._path, dist)

    # ------------------------------------------------------------------------ #
    # Navigation (children)
    # ------------------------------------------------------------------------ #

    def child(self, attr, i=None) -> Node:
        _node = getattr(self.node(), attr)
        if i is not None:
            _node = _node[i]
        elif isinstance(_node, list):
            raise ValueError('must index into block attribute')
        cur = Node(self._proc, self._path + [(attr, i)])
        # noinspection PyPropertyAccess
        # cached_property is settable, bug in static analysis
        cur._node = weakref.ref(_node)
        return cur

    def children(self) -> Iterable[Node]:
        n = self.node()
        # Top-level proc
        if isinstance(n, LoopIR.proc):
            yield from self._children_from_attrs(n, 'body')
        # Statements
        elif isinstance(n, (LoopIR.Assign, LoopIR.Reduce)):
            yield from self._children_from_attrs(n, 'idx', 'rhs')
        elif isinstance(n, (LoopIR.WriteConfig, LoopIR.WindowStmt)):
            yield from self._children_from_attrs(n, 'rhs')
        elif isinstance(n, (LoopIR.Pass, LoopIR.Alloc, LoopIR.Free)):
            yield from []
        elif isinstance(n, LoopIR.If):
            yield from self._children_from_attrs(n, 'cond', 'body', 'orelse')
        elif isinstance(n, (LoopIR.ForAll, LoopIR.Seq)):
            yield from self._children_from_attrs(n, 'hi', 'body')
        elif isinstance(n, LoopIR.Call):
            yield from self._children_from_attrs(n, 'args')
        # Expressions
        elif isinstance(n, LoopIR.Read):
            yield from self._children_from_attrs(n, 'idx')
        elif isinstance(n, (LoopIR.Const, LoopIR.WindowExpr, LoopIR.StrideExpr,
                            LoopIR.ReadConfig)):
            yield from []
        elif isinstance(n, LoopIR.USub):
            yield from self._children_from_attrs(n, 'arg')
        elif isinstance(n, LoopIR.BinOp):
            yield from self._children_from_attrs(n, 'lhs', 'rhs')
        elif isinstance(n, LoopIR.BuiltIn):
            yield from self._children_from_attrs(n, 'args')

    def _children_from_attrs(self, n, *args) -> Iterable[Node]:
        for attr in args:
            children = getattr(n, attr)
            if isinstance(children, list):
                for i in range(len(children)):
                    yield self.child(attr, i)
            else:
                yield self.child(attr, None)

    # ------------------------------------------------------------------------ #
    # Navigation (block selectors)
    # ------------------------------------------------------------------------ #

    def body(self) -> Selection:
        return self._select_attr('body')

    def orelse(self) -> Selection:
        return self._select_attr('orelse')

    def _select_attr(self, attr: str):
        stmts = getattr(self.node(), attr)
        assert isinstance(stmts, list)
        return Selection(self._proc, self._path + [(attr, range(len(stmts)))])

    # ------------------------------------------------------------------------ #
    # Conversions
    # ------------------------------------------------------------------------ #

    def select(self) -> Selection:
        attr, i = self._path[-1]
        if i is None:
            raise InvalidCursorError('cannot select nodes outside of a block')
        return Selection(self._proc, self._path[:-1] + [(attr, range(i, i + 1))])

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def replace(self, ast):
        if self._path[-1][1] is not None:
            return self.select().replace(ast)

        # replacing a single expression, or something not in a block
        def update(node):
            attr, _ = self._path[-1]
            return node.update(**{attr: ast})

        from .API import Procedure
        p = Procedure(self._rewrite_node(update))

        return p, self._forward_replace(weakref.ref(p))

    def _forward_replace(self, new_proc):
        assert self._path[-1][1] is None

        def fwd_node(_):
            raise InvalidCursorError('cannot forward replaced nodes')

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
        return self._translate(Node, self._path, -dist)

    def after(self, dist=1) -> Node:
        return self._translate(Node, self._path, dist - 1)

    def prev(self, dist=1) -> Gap:
        return self._translate(Gap, self._path, -dist)

    def next(self, dist=1) -> Gap:
        return self._translate(Gap, self._path, dist)

    # ------------------------------------------------------------------------ #
    # AST mutation
    # ------------------------------------------------------------------------ #

    def insert(self, stmts: list[LoopIR.stmt], policy=ForwardingPolicy.AnchorPre):
        assert self._path

        def update(node):
            attr, i = self._path[-1]
            children = getattr(node, attr)
            return node.update(**{attr: children[:i] + stmts + children[i:]})

        from .API import Procedure
        p = Procedure(self._rewrite_node(update))

        forward = self._forward_insert(weakref.ref(p), len(stmts), policy)
        return p, forward

    def _forward_insert(self, new_proc, ins_len, policy):
        ins_idx = self._path[-1][1]

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
                    raise InvalidCursorError('insertion gap was invalidated')
                return i + ins_len * (i > ins_idx)

        del policy

        def fwd_sel(rng):
            return range(
                rng.start + ins_len * (rng.start >= ins_idx),
                rng.stop + ins_len * (rng.stop > ins_idx),
            )

        return self._make_forward(new_proc, fwd_node, fwd_gap, fwd_sel)
