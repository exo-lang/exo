from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Union, List, Iterable, Optional
from weakref import ReferenceType

from .API_types import ProcedureBase
from .LoopIR import LoopIR


class CursorKind(Enum):
    Node = 0  # Pointing directly at a node
    # BlockFront = 1  # Gap before the first statement of a block
    # BlockEnd = 2  # Gap after the last statement of a block
    GapBefore = 3  # Gap before the node in a block
    GapAfter = 4  # Gap after the node in a block


class ExoIR(Enum):
    Assign = 0
    Reduce = 1
    WriteConfig = 2
    Pass = 3
    If = 4
    For = 5
    Alloc = 6
    Call = 7
    WindowStmt = 8


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
            raise InvalidCursorError()
        return p

    @abstractmethod
    def parent(self) -> Node:
        pass

    @abstractmethod
    def before(self, dist=0) -> Cursor:
        pass

    @abstractmethod
    def after(self, dist=0) -> Cursor:
        pass

    @abstractmethod
    def next(self, dist=0) -> Cursor:
        pass

    @abstractmethod
    def prev(self, dist=0) -> Cursor:
        pass

    @staticmethod
    def _walk_path(n, path):
        for attr, idx in path:
            n = getattr(n, attr)
            if idx is not None:
                n = n[idx]
        return n


@dataclass
class Selection(Cursor):
    _path: list[tuple[str, int]]  # leads to block parent (e.g. "if")
    _attr: str  # which block (e.g. "orelse")
    _range: tuple[int, int]  # [lo, hi) of block elements

    def replace(self):
        pass


@dataclass
class Node(Cursor):
    _path: list[tuple[str, Optional[int]]]

    def parent(self) -> Node:
        if not self._path:
            raise InvalidCursorError('cursor does not have a parent')
        return Node(self._proc, self._path[:-1])

    def before(self, dist=0) -> Gap:
        if not self._path:
            raise InvalidCursorError('cursor has no gap before')
        attr, i = self._path[-1]
        if i is None:
            raise InvalidCursorError('cursor does not point to a block')
        return Gap(self._proc, self._path[:-1] + [(attr, i - dist)])

    def after(self, dist=0) -> Gap:
        return self.before(-1 - dist)

    def next(self, dist=0) -> Node:
        if not self._path:
            raise InvalidCursorError('cursor has no next node')
        attr, i = self._path[-1]
        if i is None:
            raise InvalidCursorError('cursor does not point to a block')
        return Node(self._proc, self._path[:-1] + [(attr, i + dist + 1)])

    def prev(self, dist=0) -> Node:
        return self.next(-2 - dist)

    def node(self):
        if (n := self._node()) is None:
            raise InvalidCursorError()
        return n

    @cached_property
    def _node(self):
        n = self.proc().INTERNAL_proc()
        n = self._walk_path(n, self._path)
        return weakref.ref(n)

    def children(self) -> list[Node]:
        n = self.node()
        # Top-level proc
        if isinstance(n, LoopIR.proc):
            return self._children_from_attrs(n, 'body')
        # Statements
        elif isinstance(n, (LoopIR.Assign, LoopIR.Reduce)):
            return self._children_from_attrs(n, 'idx', 'rhs')
        elif isinstance(n, (LoopIR.WriteConfig, LoopIR.WindowStmt)):
            return self._children_from_attrs(n, 'rhs')
        elif isinstance(n, (LoopIR.Pass, LoopIR.Alloc, LoopIR.Free)):
            return []
        elif isinstance(n, LoopIR.If):
            return self._children_from_attrs(n, 'cond', 'body', 'orelse')
        elif isinstance(n, (LoopIR.ForAll, LoopIR.Seq)):
            return self._children_from_attrs(n, 'hi', 'body')
        elif isinstance(n, LoopIR.Call):
            return self._children_from_attrs(n, 'args')
        # Expressions
        elif isinstance(n, LoopIR.Read):
            return self._children_from_attrs(n, 'idx')
        elif isinstance(n, (LoopIR.Const, LoopIR.WindowExpr, LoopIR.StrideExpr,
                            LoopIR.ReadConfig)):
            return []
        elif isinstance(n, LoopIR.USub):
            return self._children_from_attrs(n, 'arg')
        elif isinstance(n, LoopIR.BinOp):
            return self._children_from_attrs(n, 'lhs', 'rhs')
        elif isinstance(n, LoopIR.BuiltIn):
            return self._children_from_attrs(n, 'args')

    def _children_from_attrs(self, n, *args):
        children = []
        for attr in args:
            children.extend(self._children_from_attr(n, attr))
        return children

    def _children_from_attr(self, n, attr):
        children = getattr(n, attr)
        if not isinstance(children, list):
            return [Node(self._proc, self._path + [(attr, None)])]
        return [Node(self._proc, self._path + [(attr, i)])
                for i in range(len(children))]

    def body(self) -> Selection:
        raise NotImplementedError()

    def orelse(self) -> Selection:
        raise NotImplementedError()


@dataclass
class Gap(Cursor):
    _path: list[tuple[str, int]]
