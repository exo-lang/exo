from __future__ import annotations

import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Union, List
from weakref import ReferenceType

from .API_types import ProcedureBase
from .LoopIR import LoopIR


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Cursors


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


@dataclass
class _CursorNodeFound(Exception):
    """
    Together with Cursor._run_find, implements an early-return finding monad.
    """

    node: Union[LoopIR.proc, LoopIR.stmt, LoopIR.expr]


class InvalidCursorError(Exception):
    pass


@dataclass
class Cursor:
    _proc: ReferenceType[ProcedureBase]
    _node: ReferenceType[Union[LoopIR.proc, LoopIR.stmt, LoopIR.expr]]
    _path: List[int]
    _kind: CursorKind = CursorKind.Node

    @staticmethod
    def root(proc):
        return Cursor(weakref.ref(proc), weakref.ref(proc.INTERNAL_proc()), [])

    def _from_path(self, path):
        if (node := self._proc()) is None:
            raise InvalidCursorError()

        try:
            node = node.INTERNAL_proc()
            for i in path:
                node = self._get_children(node)[i]
            return Cursor(self._proc, weakref.ref(node), path)
        except IndexError as e:
            raise InvalidCursorError() from e

    def body(self):
        if (node := self._node()) is None:
            raise InvalidCursorError()

        if not isinstance(node, (LoopIR.ForAll, LoopIR.Seq)):
            raise TypeError(f"AST {type(node)} does not have a body")
        return self.child(1)

    def child(self, idx) -> Cursor:
        if (node := self._node()) is None:
            raise InvalidCursorError()

        children = self._get_children(node)
        if idx >= len(children):
            raise InvalidCursorError()

        if self._kind != CursorKind.Node:
            raise TypeError(f"Cursor kind {self._kind} does not have children")

        return Cursor(
            self._proc,
            weakref.ref(children[idx]),
            self._path + [idx],
            self._kind
        )

    def parent(self) -> Cursor:
        return self._from_path(self._path[:-1])

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
