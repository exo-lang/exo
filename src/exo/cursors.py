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
    # Cursor introspection
    # ------------------------------------------------------------------------ #

    def is_gap(self):
        return self._kind != CursorKind.Node

    # ------------------------------------------------------------------------ #
    # Generic navigation
    # ------------------------------------------------------------------------ #

    # TODO: these navigation functions need to take if-statement body/orelse
    #   distinctions into account. "After" the end of the body should not be
    #   the beginning of the orelse.

    def child(self, idx) -> Cursor:
        if self._kind != CursorKind.Node:
            raise TypeError(f"Cursor kind {self._kind} does not have children")

        return self._from_path(self._path + [idx])

    def children(self) -> Iterable[Cursor]:
        if self._kind != CursorKind.Node:
            raise TypeError(f"Cursor kind {self._kind} does not have children")

        for i, node in enumerate(self._get_children(self._node())):
            yield Cursor(self._proc, weakref.ref(node), self._path + [i])

    def parent(self) -> Cursor:
        return self._from_path(self._path[:-1])

    def next(self, idx=1) -> Cursor:
        new_c = self._from_path(self._path[:-1] + [self._path[-1] + idx])
        new_c._kind = self._kind
        return new_c

    def prev(self, idx=1) -> Cursor:
        return self.next(-idx)

    def after(self, idx=0) -> Cursor:
        if self._kind == CursorKind.Node:
            new_c = self._from_path(self._path[:-1] + [self._path[-1] + idx])
            new_c._kind = CursorKind.GapAfter
        elif self._kind == CursorKind.GapBefore:
            new_c = self._from_path(self._path[:-1] + [self._path[-1] + idx])
            new_c._kind = CursorKind.Node
        elif self._kind == CursorKind.GapAfter:
            new_c = self._from_path(self._path[:-1] + [self._path[-1] + idx + 1])
            new_c._kind = CursorKind.Node
        else:
            assert False, "bad case!"
        return new_c

    def before(self, idx=0) -> Cursor:
        if self._kind == CursorKind.Node:
            new_c = self._from_path(self._path[:-1] + [self._path[-1] - idx])
            new_c._kind = CursorKind.GapBefore
        elif self._kind == CursorKind.GapAfter:
            new_c = self._from_path(self._path[:-1] + [self._path[-1] - idx])
            new_c._kind = CursorKind.Node
        elif self._kind == CursorKind.GapBefore:
            new_c = self._from_path(self._path[:-1] + [self._path[-1] - (idx + 1)])
            new_c._kind = CursorKind.Node
        else:
            assert False, "bad case!"
        return new_c

    # ------------------------------------------------------------------------ #
    # Python magic function overloads
    # ------------------------------------------------------------------------ #

    # TODO: must never implement __eq__ without __hash__

    def __eq__(self, other: Cursor):
        if self._kind == other._kind:
            return (self._proc == other._proc
                    and self._path == other._path
                    and self._node == other._node)
        elif self._kind == CursorKind.GapAfter and other._kind == CursorKind.GapBefore:
            return (self._proc == other._proc
                    and self._path[:-1] == other._path[:-1]
                    and self._path[-1] + 1 == other._path[-1])
        elif self._kind == CursorKind.GapBefore and other._kind == CursorKind.GapAfter:
            return (self._proc == other._proc

                    and self._path[:-1] == other._path[:-1]
                    and self._path[-1] - 1 == other._path[-1])
        else:
            return False

    # ------------------------------------------------------------------------ #
    # Type-specific navigation
    # ------------------------------------------------------------------------ #

    # TODO: Should body also return selection cursor?
    def body(self):
        node = self.node()
        if isinstance(node, LoopIR.proc):
            return list(self.children())
        elif isinstance(node, (LoopIR.ForAll, LoopIR.Seq, LoopIR.If)):
            return list(self.children())[1:1 + len(node.body)]
        else:
            raise TypeError(f"AST {type(node)} does not have a body")

    def orelse(self):
        node = self.node()
        if isinstance(node, LoopIR.If):
            return list(self.children())[1 + len(node.body):]
        else:
            raise TypeError(f"AST {type(node)} does not have an orelse branch")

    # ------------------------------------------------------------------------ #
    # Forwarding-aware, persistent, edits
    # ------------------------------------------------------------------------ #

    def insert_ast(self, ast):
        if self._kind == CursorKind.GapBefore:
            pass
        elif self._kind == CursorKind.GapAfter:
            pass
        else:
            raise InvalidCursorError('Must insert at a gap')

    def delete_ast(self):
        if self.is_gap():
            raise InvalidCursorError('Must delete a node')

    def replace_ast(self, replacement):
        """
        Replaces the pointed-to node with "ast", which can be either a single
        node or a list of nodes.
        """
        if self.is_gap():
            raise InvalidCursorError('Must replace a node, not a gap')

        if not isinstance(replacement, list):
            replacement = [replacement]

        def do_edit(ast, path):
            if not path:
                return ast

        new_ast = do_edit(self.proc().INTERNAL_proc(), self._path)
        fwd_fn = _make_replace_fwd(self._path, len(replacement))
        return new_ast, fwd_fn

    def move_to(self, dst: Cursor):
        if not dst.is_gap():
            raise InvalidCursorError('Must move to a gap')
        if self.is_gap():
            raise InvalidCursorError('Cannot move a gap')

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
            return node.body

        # Statements
        elif isinstance(node, (LoopIR.Assign, LoopIR.Reduce)):
            return node.idx + [node.rhs]
        elif isinstance(node, LoopIR.WriteConfig):
            return [node.rhs]
        elif isinstance(node, (LoopIR.Pass, LoopIR.Alloc, LoopIR.Free)):
            return []
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
            return node.args
        elif isinstance(node,
                        (LoopIR.Const, LoopIR.WindowExpr, LoopIR.StrideExpr,
                         LoopIR.ReadConfig)):
            return []

        assert False, f"base case: {type(node)}"


# ---------------------------------------------------------------------------- #
# Cursor forwarding function generators
#   These functions reside at the top-level to prevent them from capturing local
#   state in the cursor editing operations, which could unintentionally extend
#   the lifetimes of nodes. This also reduces the overall memory usage of the
#   closures containing only the relevant information.
# ---------------------------------------------------------------------------- #

def _make_replace_fwd(sub_path, n_sub):
    n_path = len(sub_path)

    def fwd(path):
        # Cursors to the replaced node and below are invalidated.
        if path[:n_path] == sub_path:
            raise InvalidCursorError('Cursor has been invalidated')

        # Cursors to siblings of the replaced node get adjusted based on the
        # substitution size
        sibling_level = path[:n_path - 1]
        if sibling_level == sub_path[:-1]:
            last = path[-1]
            return sibling_level + [last if last < sub_path[-1] else last + n_sub - 1]

        # Otherwise, we're pointing somewhere unrelated and the path is still valid.
        return path

    return fwd
