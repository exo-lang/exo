from __future__ import annotations

import inspect
import re
from typing import Optional

from . import pyparser
from .LoopIR import LoopIR, PAST
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Pattern Matching Errors
from .cursors import Cursor, Node, Selection


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
    return int(pattern_str[pos + 1:])


def match_pattern(proc, pattern_str, call_depth=0, default_match_no=None):
    return _match_pattern_impl(proc, pattern_str, call_depth,
                               default_match_no).results()


def match_cursors(proc, pattern_str, call_depth=0, default_match_no=None):
    return _match_pattern_impl(proc, pattern_str, call_depth,
                               default_match_no).cursors()


def _match_pattern_impl(proc, pattern_str, call_depth, default_match_no):
    # break-down pattern_str for possible #<num> post-fix
    if match := re.search(r"^([^#]+)#(\d+)\s*$", pattern_str):
        pattern_str = match[1]
        match_no = int(match[2]) if match[2] is not None else None
    else:
        match_no = default_match_no  # None means match-all

    # get source location where this is getting called from
    caller = inspect.getframeinfo(inspect.stack()[call_depth + 2][0])

    # parse the pattern we're going to use to match
    p_ast = pyparser.pattern(
        pattern_str, filename=caller.filename, lineno=caller.lineno
    )

    # do the pattern match, to find the nodes in ast
    return PatternMatch(proc, p_ast, match_no=match_no)


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


class _MatchComplete(Exception):
    pass


class PatternMatch:
    def __init__(self, proc, pat, match_no=None):
        self._match_i = match_no
        self._results = []

        # prevent the top level of a pattern being just a hole
        if isinstance(pat, PAST.E_Hole):
            raise PatternMatchError("pattern match on 'anything' unsupported")
        elif isinstance(pat, list) and all(isinstance(p, PAST.S_Hole) for p in pat):
            raise PatternMatchError("pattern match on 'anything' unsupported")

        cur = Cursor.root(proc)

        try:
            if isinstance(pat, list):
                assert len(pat) > 0
                self.find_stmts(pat, cur.body())
            else:
                assert isinstance(pat, PAST.expr)
                self.find_expr(pat, cur)
        except _MatchComplete:
            pass

    def results(self):
        return [[n.node() for n in cur] if isinstance(cur, Selection) else cur.node()
                for cur in self._results]

    def cursors(self):
        return self._results

    def _add_result(self, result):
        assert isinstance(result, (Node, Selection))

        if self._match_i is None:
            self._results.append(result)
            return

        i = self._match_i
        self._match_i -= 1
        if i == 0:
            self._results.append(result)
            raise _MatchComplete()

    # -------------------
    #  finding methods

    def find_expr(self, pat, cur):
        # try to match
        if self.match_e(pat, cur.node()):
            self._add_result(cur)

        for child in cur.children():
            self.find_expr(pat, child)

    def find_stmts(self, pats, curs):
        # may encounter empty statement blocks, which we should ignore
        if len(curs) == 0:
            return

        # try to match a prefix of this sequence of statements
        if m := self.match_stmts(pats, curs):
            self._add_result(m)

        # if we need to look for more matches, recurse structurally ...

        # first, look for any subsequences of statements in the first
        # statement of the sequence `stmts`
        if isinstance(curs[0].node(), LoopIR.If):
            self.find_stmts(pats, curs[0].body())
            self.find_stmts(pats, curs[0].orelse())
        elif isinstance(curs[0].node(), (LoopIR.proc, LoopIR.ForAll, LoopIR.Seq)):
            self.find_stmts(pats, curs[0].body())
        else:
            pass  # other forms of statement do not contain stmt blocks

        # second, recurse on the tail of this sequence...
        self.find_stmts(pats, curs[1:])

    # -------------------
    #  matching methods

    def match_stmts(self, pats, cur):
        i, j = 0, 0
        while i < len(pats) and j < len(cur):
            if isinstance(pats[i], PAST.S_Hole):
                if i + 1 == len(pats):
                    return cur  # No lookahead, guaranteed match
                if self.match_stmt(pats[i + 1], cur[j]):
                    i += 2  # Lookahead matches, skip hole and lookahead
            elif self.match_stmt(pats[i], cur[j]):
                i += 1
            else:
                return None
            j += 1

        # Return the matched portion on success
        return cur[:j] if i == len(pats) else None

    def match_stmt(self, pat, cur):
        assert not isinstance(pat, PAST.S_Hole), "holes must be handled in match_stmts"

        # first ensure that the pattern and statement
        # are the same constructor

        stmt = cur.node()

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
                and self.match_stmts(pat.body, cur.body()) is not None
                and self.match_stmts(pat.orelse, cur.orelse()) is not None
            )
        elif isinstance(stmt, (LoopIR.ForAll, LoopIR.Seq)):
            return (
                self.match_name(pat.iter, stmt.iter)
                and self.match_e(pat.hi, stmt.hi)
                and self.match_stmts(pat.body, cur.body()) is not None
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
