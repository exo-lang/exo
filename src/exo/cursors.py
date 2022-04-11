from __future__ import annotations

import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Iterable
from weakref import ReferenceType

from .API_types import ProcedureBase
from .LoopIR import LoopIR


class CursorKind(Enum):
    Node = 0  # Pointing directly at a node
    # BlockFront = 1  # Gap before the first statement of a block
    # BlockEnd = 2  # Gap after the last statement of a block
    BeforeNode = 3  # Gap before the node in a block
    AfterNode = 4  # Gap after the node in a block


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
class Cursor:
    _proc: ReferenceType[ProcedureBase]
    _node: ReferenceType[Union[LoopIR.proc, LoopIR.stmt, LoopIR.expr]]
    _path: List[int]
    _kind: CursorKind = CursorKind.Node

    # ------------------------------------------------------------------------ #
    # Static constructors
    # ------------------------------------------------------------------------ #

    @staticmethod
    def root(proc):
        return Cursor.from_node(proc, proc.INTERNAL_proc(), [])

    @staticmethod
    def from_node(proc, node, path):
        return Cursor(weakref.ref(proc), weakref.ref(node), path)

    # ------------------------------------------------------------------------ #
    # Validating getters
    # ------------------------------------------------------------------------ #

    def proc(self):
        if (proc := self._proc()) is None:
            raise InvalidCursorError()
        return proc

    def node(self):
        if (node := self._node()) is None:
            raise InvalidCursorError()
        return node

    # ------------------------------------------------------------------------ #
    # Generic navigation
    # ------------------------------------------------------------------------ #

    def child(self, idx) -> Cursor:
        if self._kind != CursorKind.Node:
            raise TypeError(f"Cursor kind {self._kind} does not have children")

        return self._from_path(self._path + [idx])

    def children(self) -> Iterable[Cursor]:
        for i, node in enumerate(self._get_children(self._node())):
            yield Cursor(self._proc, weakref.ref(node), self._path + [i])

    def parent(self) -> Cursor:
        return self._from_path(self._path[:-1])

    # ------------------------------------------------------------------------ #
    # Type-dependent navigation
    # ------------------------------------------------------------------------ #

    def body(self):
        node = self.node()
        if not isinstance(node, (LoopIR.proc, LoopIR.ForAll, LoopIR.Seq, LoopIR.If)):
            raise TypeError(f"AST {type(node)} does not have a body")
        return list(self.children())[1:len(node.body)+1]

    def orelse(self):
        node = self.node()
        if not isinstance(node, LoopIR.If):
            raise TypeError(f"AST {type(node)} does not have an orelse branch")
        return list(self.children())[1+len(node.body):]

    # ------------------------------------------------------------------------ #
    # Internal implementation
    # ------------------------------------------------------------------------ #

    def _follow_path(self, node, path):
        for i in path:
            node = self._get_children(node)[i]
        return node

    def _is_valid(self):
        if self._kind != CursorKind.Node:
            raise NotImplementedError('Only node cursors are currently supported')

        try:
            node = self.proc().INTERNAL_proc()
            node = self._follow_path(node, self._path)
            return node is self._node()
        except IndexError:
            return False

    def _from_path(self, path):
        try:
            node = self.proc().INTERNAL_proc()
            node = self._follow_path(node, path)
            return Cursor(self._proc, weakref.ref(node), path)
        except IndexError as e:
            raise InvalidCursorError() from e

    # TODO: this should be a feature of ASDL-ADT
    @staticmethod
    def _get_children(node):
        # Procs
        if isinstance(node, LoopIR.proc):
            return node.preds + node.body

        # Statements
        elif isinstance(node, (LoopIR.Assign, LoopIR.Reduce)):
            return node.idx + [node.rhs]
        elif isinstance(node, LoopIR.WriteConfig):
            return [node.rhs]
        elif isinstance(node, LoopIR.If):
            return [node.cond] + node.body + node.orelse
        elif isinstance(node, (LoopIR.ForAll, LoopIR.Seq)):
            return [node.hi] + node.body
        elif isinstance(node, LoopIR.Call):
            return node.args
        elif isinstance(node, LoopIR.WindowStmt):
            return node.rhs

        # Expressions
        elif isinstance(node, LoopIR.Read):
            return node.idx
        elif isinstance(node, LoopIR.USub):
            return [node.arg]
        elif isinstance(node, LoopIR.BinOp):
            return [node.lhs, node.rhs]
        elif isinstance(node, LoopIR.BuiltIn):
            return [node.lhs, node.args]
