from __future__ import annotations

import inspect
import re
from typing import Optional

from . import pyparser
from .LoopIR import LoopIR, PAST


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Pattern Matching Errors
from .cursors import Cursor


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
    # break-down pattern_str for possible #<num> post-fix
    if match := re.search(r"^([^#]+)#(\d+)\s*$", pattern_str):
        pattern_str = match[1]
        match_no = int(match[2]) if match[2] is not None else None
    else:
        match_no = default_match_no  # None means match-all

    # get source location where this is getting called from
    caller = inspect.getframeinfo(inspect.stack()[call_depth + 1][0])

    # parse the pattern we're going to use to match
    p_ast = pyparser.pattern(
        pattern_str, filename=caller.filename, lineno=caller.lineno
    )

    # do the pattern match, to find the nodes in ast
    return PatternMatch(proc, p_ast, match_no=match_no).results()


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
    def __init__(self, proc, pat, match_no=None):
        self._match_i = match_no
        self._results = []

        # prevent the top level of a pattern being just a hole
        if isinstance(pat, PAST.E_Hole):
            raise PatternMatchError("pattern match on 'anything' unsupported")
        elif isinstance(pat, list) and all(isinstance(p, PAST.S_Hole) for p in pat):
            raise PatternMatchError("pattern match on 'anything' unsupported")

        ast = [proc.INTERNAL_proc()]
        cur = Cursor.root(proc)

        if isinstance(pat, list):
            assert len(pat) > 0
            self.find_stmts(pat, ast)
        else:
            assert isinstance(pat, PAST.expr)
            self.find_expr(pat, cur)

    def results(self):
        return self._results

    # -------------------
    #  finding methods

    def find_expr(self, pat, cur):
        # short-circuit if we have our one match already...
        if self._match_i is not None and self._match_i < 0:
            return

        # try to match
        if self.match_e(pat, cur.node()):
            if self._match_i is None:
                self._results.append(cur.node())
            else:
                i = self._match_i
                self._match_i -= 1
                if i == 0:
                    self._results.append(cur.node())
                    return

        for child in cur.children():
            self.find_expr(pat, child)

    def find_stmts(self, pat, stmts):
        # short-circuit if we have our one match already...
        if self._match_i is not None and self._match_i < 0:
            return

        # may encounter empty statement blocks, which we should ignore
        if len(stmts) == 0:
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
            self.find_stmts(pat, stmts[0].body)
            self.find_stmts(pat, stmts[0].orelse)
        elif isinstance(stmts[0], (LoopIR.proc, LoopIR.ForAll, LoopIR.Seq)):
            self.find_stmts(pat, stmts[0].body)
        else:
            pass  # other forms of statement do not contain stmt blocks

        # second, recurse on the tail of this sequence...
        self.find_stmts(pat, stmts[1:])

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
