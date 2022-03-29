from __future__ import annotations

import inspect
import re
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Set
from weakref import ReferenceType

from . import pyparser
from .API_types import ProcedureBase
from .LoopIR import LoopIR, PAST


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Cursors


class CursorKind(Enum):
    Node = 0  # Pointing directly at a node
    BlockFront = 1  # Gap before the first statement of a block
    BlockEnd = 2  # Gap after the last statement of a block
    BeforeNode = 3  # Gap before the node in a block
    AfterNode = 4  # Gap after the node in a block


@dataclass
class _CursorNodeFound(Exception):
    """
    Together with Cursor._run_find, implements an early-return finding monad.
    """

    node: Union[LoopIR.proc, LoopIR.stmt, LoopIR.expr]


@dataclass
class Cursor:
    proc: ReferenceType[ProcedureBase]
    node: Optional[ReferenceType[Union[LoopIR.proc, LoopIR.stmt, LoopIR.expr]]]
    prune: Set[int]  # Stores node ids that cannot be a parent of `node`
    kind: CursorKind = CursorKind.Node

    @staticmethod
    def root(proc):
        return Cursor(weakref.ref(proc), weakref.ref(proc.INTERNAL_proc()), set())

    def child(self, idx) -> Cursor:
        extra_prune = set()

        def _child(children):
            assert isinstance(children, list)
            for child in children[:idx]:
                extra_prune.add(id(child))
            return children[idx]

        if self.node and (node := self.node()):
            return Cursor(
                self.proc,
                self._run_find(self._apply_children, _child, node),
                self.prune.union(extra_prune),
                CursorKind.Node,
            )
        else:
            # invalid
            return self

    def parent(self) -> Cursor:
        def _find_parent(node, parent):
            if node is None or id(node) in self.prune:
                return

            if node is self.node:
                if node.kind in (CursorKind.BlockFront, CursorKind.BlockEnd):
                    # The parent of a gap inside a block is the block, but we're
                    # pointing at the block node, so `parent` is actually the
                    # grandparent here
                    raise _CursorNodeFound(node)
                # Otherwise, we're pointing at something properly inside true parent
                raise _CursorNodeFound(parent)

            def _walk_children(children):
                for child in children:
                    _find_parent(child, node)

            self._apply_children(_walk_children, node)

        return Cursor(
            self.proc,  # Still in the same tree
            self._run_find(_find_parent, self.node(), None),
            self.prune,  # The child's prune-set is never smaller than the parent's
            CursorKind.Node,
        )

    @staticmethod
    def _run_find(fn, *args, **kwargs):
        """
        Together with _CursorNodeFound, implements an early-return finding monad.
        """
        try:
            if node := fn(*args, **kwargs):
                return weakref.ref(node)
        except _CursorNodeFound as e:
            return weakref.ref(e.node)

    @staticmethod
    def _apply_children(fn, node):
        # Procs
        if isinstance(node, LoopIR.proc):
            return fn(node.preds + node.body)

        # Statements
        elif isinstance(node, (LoopIR.Assign, LoopIR.Reduce)):
            return fn(node.idx + [node.rhs])
        elif isinstance(node, LoopIR.WriteConfig):
            return fn([node.rhs])
        elif isinstance(node, LoopIR.If):
            return fn([node.cond] + node.body + node.orelse)
        elif isinstance(node, (LoopIR.ForAll, LoopIR.Seq)):
            return fn([node.hi] + node.body)
        elif isinstance(node, LoopIR.Call):
            return fn(node.args)
        elif isinstance(node, LoopIR.WindowStmt):
            return fn(node.rhs)

        # Expressions
        elif isinstance(node, LoopIR.Read):
            return fn(node.idx)
        elif isinstance(node, LoopIR.USub):
            return fn([node.arg])
        elif isinstance(node, LoopIR.BinOp):
            return fn([node.lhs, node.rhs])
        elif isinstance(node, LoopIR.BuiltIn):
            return fn([node.lhs, node.args])


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Pattern Matching Errors


class PatternMatchError(Exception):
    pass


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# General Pattern-Matching / Pointing Mechanism

"""
We will use <pattern-string>s as a way to point at AST nodes.

A <pattern-string> has the following form:

<pattern-string> ::= <pattern> #<num>
                   | <pattern>
<pattern>        ::= ... -- a UAST statement or expression
                         -- potentially involving one or more holes
                         -- where a hole is written `_`
                         -- specified by LoopIR.PAST
"""


def get_match_no(pattern_str: str) -> Optional[int]:
    """
    Search for a trailing # sign in a pattern string and return the following
    number, or None if it does no # sign exists. Uses `int` to parse the number,
    so is not sensitive to spaces
    >>> get_match_no('foo #34')
    34
    >>> get_match_no('baz # 42 ')
    42
    >>> get_match_no('foo') is None
    True
    >>> get_match_no('foo #bar')
    Traceback (most recent call last):
      ...
    ValueError: invalid literal for int() with base 10: 'bar'
    """
    if (pos := pattern_str.rfind("#")) == -1:
        return None
    return int(pattern_str[pos + 1 :])


def match_pattern(ast, pattern_str, call_depth=0, default_match_no=None):
    # get source location where this is getting called from
    caller = inspect.getframeinfo(inspect.stack()[call_depth + 1][0])

    # break-down pattern_str for possible #<num> post-fix
    if match := re.search(r"^([^#]+)#(\d+)\s*$", pattern_str):
        pattern_str = match[1]
        match_no = int(match[2]) if match[2] is not None else None
    else:
        match_no = default_match_no  # None means match-all

    # parse the pattern we're going to use to match
    p_ast = pyparser.pattern(
        pattern_str, filename=caller.filename, lineno=caller.lineno
    )

    # do the pattern match, to find the nodes in ast
    return PatternMatch(p_ast, ast, match_no=match_no).results()


_PAST_to_LoopIR = {
    # list of stmts
    list: list,
    #
    PAST.Assign: [LoopIR.Assign],
    PAST.Reduce: [LoopIR.Reduce],
    PAST.Pass: [LoopIR.Pass],
    PAST.If: [LoopIR.If],
    PAST.ForAll: [LoopIR.ForAll, LoopIR.Seq],
    PAST.Alloc: [LoopIR.Alloc],
    PAST.Call: [LoopIR.Call],
    PAST.WriteConfig: [LoopIR.WriteConfig],
    PAST.S_Hole: None,
    #
    PAST.Read: [LoopIR.Read],
    PAST.Const: [LoopIR.Const],
    PAST.USub: [LoopIR.USub],
    PAST.BinOp: [LoopIR.BinOp],
    PAST.E_Hole: None,
}


class PatternMatch:
    def __init__(self, pat, src_stmts, match_no=None):
        self._match_i = match_no
        self._results = []

        # prevent the top level of a pattern being just a hole
        if isinstance(pat, PAST.E_Hole):
            raise PatternMatchError("pattern match on 'anything' unsupported")
        elif isinstance(pat, list) and all(isinstance(p, PAST.S_Hole) for p in pat):
            raise PatternMatchError("pattern match on 'anything' unsupported")

        if isinstance(pat, list):
            assert len(pat) > 0
            self.find_stmts_in_stmts(pat, src_stmts)
        else:
            assert isinstance(pat, PAST.expr)
            self.find_e_in_stmts(pat, src_stmts)

    def results(self):
        return self._results

    # -------------------
    #  finding methods

    def find_e_in_stmts(self, pat, stmts):
        for s in stmts:
            self.find_e_in_stmt(pat, s)

    def find_e_in_stmt(self, pat, stmt):
        # short-circuit if we have our one match already...
        if self._match_i is not None and self._match_i < 0:
            return

        if isinstance(stmt, (LoopIR.Assign, LoopIR.Reduce)):
            for e in stmt.idx:
                self.find_e_in_e(pat, e)
            self.find_e_in_e(pat, stmt.rhs)
        elif isinstance(stmt, LoopIR.If):
            self.find_e_in_e(pat, stmt.cond)
            self.find_e_in_stmts(pat, stmt.body)
            self.find_e_in_stmts(pat, stmt.orelse)
        elif isinstance(stmt, (LoopIR.ForAll, LoopIR.Seq)):
            self.find_e_in_e(pat, stmt.hi)
            self.find_e_in_stmts(pat, stmt.body)
        elif isinstance(stmt, LoopIR.Call):
            for e in stmt.args:
                self.find_e_in_e(pat, e)
        else:
            pass  # ignore other statements

    def find_e_in_e(self, pat, e):
        # short-circuit if we have our one match already...
        if self._match_i is not None and self._match_i < 0:
            return

        # try to match
        if self.match_e(pat, e):
            if self._match_i is None:
                self._results.append(e)
            else:
                i = self._match_i
                self._match_i -= 1
                if i == 0:
                    self._results.append(e)
                    return

        # if we need to look for more matches, recurse structurally
        if isinstance(e, LoopIR.BinOp):
            self.find_e_in_e(pat, e.lhs)
            self.find_e_in_e(pat, e.rhs)

        if isinstance(e, LoopIR.USub):
            self.find_e_in_e(pat, e.arg)

    def find_stmts_in_stmts(self, pat, stmts):
        # may encounter empty statement blocks, which we should ignore
        if len(stmts) == 0:
            return
        # short-circuit if we have our one match already...
        if self._match_i is not None and self._match_i < 0:
            return

        # try to match exactly this sequence of statements
        if self.match_stmts(pat, stmts):
            if self._match_i is None:
                self._results.append(stmts)
            else:
                i = self._match_i
                self._match_i -= 1
                if i == 0:
                    self._results.append(stmts)
                    return

        # if we need to look for more matches, recurse structurally ...

        # first, look for any subsequences of statements in the first
        # statement of the sequence `stmts`
        if isinstance(stmts[0], LoopIR.If):
            self.find_stmts_in_stmts(pat, stmts[0].body)
            self.find_stmts_in_stmts(pat, stmts[0].orelse)
        elif isinstance(stmts[0], (LoopIR.ForAll, LoopIR.Seq)):
            self.find_stmts_in_stmts(pat, stmts[0].body)
        else:
            pass  # other forms of statement do not contain stmt blocks

        # second, recurse on the tail of this sequence...
        self.find_stmts_in_stmts(pat, stmts[1:])

    # -------------------
    #  matching methods

    def match_stmts(self, pats, stmts):
        # variable for keeping track of whether we
        # can currently match arbitrary statements
        # into a statement hole
        in_hole = False

        # shallow copy of the stmt list that we can
        stmts = stmts.copy()

        stmt_idx = 0
        for p in pats:
            if isinstance(p, PAST.S_Hole):
                in_hole = True
                continue

            if stmt_idx >= len(stmts):
                # fail match because there are no more statements to match
                # this non-hole pattern
                return False

            # otherwise, if we're in a hole try to find the first match
            if in_hole:
                while not self.match_stmt(p, stmts[stmt_idx]):
                    stmt_idx += 1
                # now stmt_idx is a match

            # if we're not in a hole, failing to match is a failure
            # of matching the entire stmt block
            elif not self.match_stmt(p, stmts[stmt_idx]):
                return False

            # If we successfully matched a statement to a non-hole pattern
            # then we've moved past any statement hole---if we were in one.
            # If we weren't in a hole, this operation has no effect.
            if not isinstance(p, PAST.S_Hole):
                in_hole = False

            # finally, if we're here we found a match and can advance the
            # stmt_idx counter
            stmt_idx += 1

        # if we made it to the end of the function, then
        # the match was successful
        return True

    def match_stmt(self, pat, stmt):
        assert not isinstance(
            pat, PAST.S_Hole
        ), "holes should be handled in match_stmts"

        # first ensure that the pattern and statement
        # are the same constructor

        if not isinstance(
            stmt, (LoopIR.WindowStmt,) + tuple(_PAST_to_LoopIR[type(pat)])
        ):
            return False

        # then handle each constructor as a structural case
        if isinstance(stmt, (LoopIR.Assign, LoopIR.Reduce)):
            return (
                self.match_name(pat.name, stmt.name)
                and all(self.match_e(pi, si) for pi, si in zip(pat.idx, stmt.idx))
                and self.match_e(pat.rhs, stmt.rhs)
            )
        elif isinstance(stmt, LoopIR.WindowStmt):
            if isinstance(pat, PAST.Assign):
                return (
                    self.match_name(pat.name, stmt.lhs)
                    and pat.idx == []
                    and self.match_e(pat.rhs, stmt.rhs)
                )
            else:
                return False
        elif isinstance(stmt, LoopIR.Pass):
            return True
        elif isinstance(stmt, LoopIR.If):
            return (
                self.match_e(pat.cond, stmt.cond)
                and self.match_stmts(pat.body, stmt.body)
                and self.match_stmts(pat.orelse, stmt.orelse)
            )
        elif isinstance(stmt, (LoopIR.ForAll, LoopIR.Seq)):
            return (
                self.match_name(pat.iter, stmt.iter)
                and self.match_e(pat.hi, stmt.hi)
                and self.match_stmts(pat.body, stmt.body)
            )
        elif isinstance(stmt, LoopIR.Alloc):
            if isinstance(stmt.type, LoopIR.Tensor):
                return all(
                    self.match_e(pi, si) for pi, si in zip(pat.sizes, stmt.type.hi)
                ) and self.match_name(pat.name, stmt.name)
            else:  # scalar
                return self.match_name(pat.name, stmt.name)
        elif isinstance(stmt, LoopIR.Call):
            return self.match_name(pat.f, stmt.f.name)
        elif isinstance(stmt, LoopIR.WriteConfig):
            return self.match_name(stmt.config.name(), pat.config) and self.match_name(
                stmt.field, pat.field
            )
        else:
            assert False, f"bad case: {type(stmt)}"

    def match_e(self, pat, e):
        # expression holes can match anything, and we
        # don't have to worry about Kleene-Star behavior
        if isinstance(pat, PAST.E_Hole):
            return True

        # first ensure that the pattern and statement are the same constructor
        if not isinstance(e, (LoopIR.WindowExpr,) + tuple(_PAST_to_LoopIR[type(pat)])):
            return False

        if isinstance(e, LoopIR.Read):
            return self.match_name(pat.name, e.name) and all(
                self.match_e(pi, si) for pi, si in zip(pat.idx, e.idx)
            )
        elif isinstance(e, LoopIR.WindowExpr):
            if isinstance(pat, PAST.Read):
                # TODO: Should we be able to handle window slicing matching? Nah..
                if len(pat.idx) != 1 or not isinstance(pat.idx[0], PAST.E_Hole):
                    return False
                return self.match_name(pat.name, e.name)
            else:
                return False
        elif isinstance(e, LoopIR.Const):
            return pat.val == e.val
        elif isinstance(e, LoopIR.BinOp):
            return (
                pat.op == e.op
                and self.match_e(pat.lhs, e.lhs)
                and self.match_e(pat.rhs, e.rhs)
            )
        elif isinstance(e, LoopIR.USub):
            return self.match_e(pat.arg, e.arg)
        else:
            assert False, "bad case"

    @staticmethod
    def match_name(pat_nm, ir_sym):
        return pat_nm == "_" or pat_nm == str(ir_sym)
