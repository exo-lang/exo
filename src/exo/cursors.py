from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Iterable
from weakref import ReferenceType

from .API_types import ProcedureBase
from .LoopIR import LoopIR


class InvalidCursorError(Exception):
    pass


@dataclass
class Cursor(ABC):
    _proc: ReferenceType[ProcedureBase]

    @staticmethod
    def root(proc: ProcedureBase):
        return Node(weakref.ref(proc), [])

    def proc(self):
        if (p := self._proc()) is None:
            raise InvalidCursorError("underlying proc was destroyed")
        return p

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


@dataclass
class Selection(Cursor):
    _block: Node  # leads to block parent (e.g. "if")
    _attr: str  # which block (e.g. "orelse")
    _range: tuple[int, int]  # [lo, hi) of block elements

    def parent(self) -> Node:
        return self._block

    def before(self, dist=1) -> Gap:
        lo, hi = self._range
        assert lo < hi
        return self._block.child(self._attr, lo).before(dist)

    def after(self, dist=1) -> Gap:
        lo, hi = self._range
        assert lo < hi
        return self._block.child(self._attr, hi - 1).after(dist)

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

    def __iter__(self):
        def impl():
            for i in range(*self._range):
                yield self._block.child(self._attr, i)

        return impl()

    def __getitem__(self, i):
        r = range(*self._range)[i]
        if isinstance(r, range):
            if r.step != 1:
                raise IndexError('cursor selections must be contiguous')
            if len(r) == 0:
                raise IndexError('cannot construct an empty selection')
            return Selection(self._proc, self._block, self._attr, (r.start, r.stop))
        else:
            return self._block.child(self._attr, r)

    def __len__(self):
        return len(range(*self._range))


@dataclass
class Node(Cursor):
    _path: list[tuple[str, Optional[int]]]

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

    def node(self):
        if (n := self._node()) is None:
            raise InvalidCursorError('underlying node was destroyed')
        return n

    @cached_property
    def _node(self):
        n = self.proc().INTERNAL_proc()
        n = self._walk_path(n, self._path)
        return weakref.ref(n)

    def child(self, attr, i=None) -> Node:
        if (_node := getattr(self.node(), attr, None)) is None:
            raise ValueError(f'no such attribute {attr}')
        if i is not None:
            _node = _node[i]
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

    def body(self) -> Selection:
        return self._select_attr('body')

    def orelse(self) -> Selection:
        return self._select_attr('orelse')

    def _select_attr(self, attr):
        n = self.node()
        if (stmts := getattr(n, attr, None)) is None:
            raise InvalidCursorError(
                f'node type {type(n).__name__} does not have attribute "{attr}"')
        assert isinstance(stmts, list)
        return Selection(self._proc, self, attr, (0, len(stmts)))


@dataclass
class Gap(Cursor):
    _path: list[tuple[str, int]]

    def parent(self) -> Node:
        assert self._path
        return Node(self._proc, self._path[:-1])

    def before(self, dist=1) -> Node:
        return self._translate(Node, self._path, 1 - dist)

    def after(self, dist=1) -> Node:
        return self._translate(Node, self._path, dist - 1)

    def prev(self, dist=1) -> Gap:
        return self._translate(Gap, self._path, -dist)

    def next(self, dist=1) -> Gap:
        return self._translate(Gap, self._path, dist)
