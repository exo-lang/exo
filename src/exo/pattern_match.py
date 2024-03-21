from __future__ import annotations

import inspect
import re
from typing import Optional, Iterable

import exo.pyparser as pyparser
from exo.LoopIR import LoopIR, PAST

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Pattern Matching Errors
from exo.internal_cursors import Cursor, Node, Block


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


def match_pattern(
    context: Cursor,
    pattern_str: str,
    call_depth=0,
    default_match_no=None,
    use_sym_id=False,
):
    """
    If [default_match_no] is None, then all matches are returned

    If [use_sym_id] is True, all symbols matchesrare additionally checked
    against their unique id, rather than just matching against the name.
    """
    assert isinstance(context, Cursor), f"Expected Cursor, got {type(context)}"

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
    return PatternMatch().find(context, p_ast, match_no=match_no, use_sym_id=use_sym_id)


_PAST_to_LoopIR = {
    # list of stmts
    list: list,
    #
    PAST.Assign: [LoopIR.Assign],
    PAST.Reduce: [LoopIR.Reduce],
    PAST.Pass: [LoopIR.Pass],
    PAST.If: [LoopIR.If],
    PAST.For: [LoopIR.For],
    PAST.Alloc: [LoopIR.Alloc],
    PAST.Call: [LoopIR.Call],
    PAST.WriteConfig: [LoopIR.WriteConfig],
    PAST.S_Hole: None,
    #
    PAST.Read: [LoopIR.Read],
    PAST.StrideExpr: [LoopIR.StrideExpr],
    PAST.Const: [LoopIR.Const],
    PAST.USub: [LoopIR.USub],
    PAST.BinOp: [LoopIR.BinOp],
    PAST.BuiltIn: [LoopIR.BuiltIn],
    PAST.ReadConfig: [LoopIR.ReadConfig],
    PAST.E_Hole: None,
}


class _MatchComplete(Exception):
    pass


class PatternMatch:
    def __init__(self):
        self._match_no = None
        self._results = []
        self._use_sym_id = False

    def find(self, cur, pat, match_no=None, use_sym_id=False):
        self._match_no = match_no
        self._results = []
        self._use_sym_id = use_sym_id

        # prevent the top level of a pattern being just a hole
        if isinstance(pat, PAST.E_Hole):
            raise PatternMatchError("pattern match on 'anything' unsupported")
        elif isinstance(pat, list) and all(isinstance(p, PAST.S_Hole) for p in pat):
            raise PatternMatchError("pattern match on 'anything' unsupported")

        try:
            if isinstance(pat, list):
                assert len(pat) > 0
                self.find_stmts(pat, cur)
            else:
                assert isinstance(pat, PAST.expr)
                self.find_expr(pat, cur)
        except _MatchComplete:
            pass

        return self._results

    def _add_result(self, result):
        assert isinstance(result, (Node, Block))

        if self._match_no is None:
            self._results.append(result)
            return

        i = self._match_no
        self._match_no -= 1
        if i == 0:
            self._results.append(result)
            raise _MatchComplete()

    ## -------------------
    ##  finding methods

    def find_expr(self, pat, cur):
        # try to match
        if self.match_e(pat, cur._node):
            self._add_result(cur)

        for child in _children(cur):
            self.find_expr(pat, child)

    def find_stmts(self, pats, cur: Node):
        if isinstance(cur._node, LoopIR.proc):
            return self.find_stmts_in_block(pats, cur.body())

        return self.find_stmts_in_block(pats, cur.as_block())

    def find_stmts_in_block(self, pats, curs: Block):
        # may encounter empty statement blocks, which we should ignore
        if len(curs) == 0:
            return

        # try to match a prefix of this sequence of statements
        if m := self.match_stmts(pats, curs):
            self._add_result(m)

        # if we need to look for more matches, recurse structurally ...

        # first, look for any subsequences of statements in the first
        # statement of the sequence `stmts`
        if isinstance(curs[0]._node, LoopIR.If):
            self.find_stmts_in_block(pats, curs[0].body())
            self.find_stmts_in_block(pats, curs[0].orelse())
        elif isinstance(curs[0]._node, LoopIR.For):
            self.find_stmts_in_block(pats, curs[0].body())
        else:
            pass  # other forms of statement do not contain stmt blocks

        # second, recurse on the tail of this sequence...
        self.find_stmts_in_block(pats, curs[1:])

    ## -------------------
    ##  matching methods

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

        stmt = cur._node

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
                    self.match_name(pat.name, stmt.name)
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
        elif isinstance(stmt, LoopIR.For):
            return (
                self.match_name(pat.iter, stmt.iter)
                and self.match_e(pat.lo, stmt.lo)
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
        # expression holes can match anything
        # and we don't have to worry about Kleene-Star behavior
        if isinstance(pat, PAST.E_Hole):
            return True

        # Special case: -3 can be parsed as USub(Const(3))... it should match Const(-3)
        if (
            isinstance(pat, PAST.USub)
            and isinstance(pat.arg, PAST.Const)
            and isinstance(e, LoopIR.Const)
        ):
            pat = pat.arg.update(val=-pat.arg.val)

        # first ensure that the pattern and statement
        # are the same constructor
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
            # TODO: do we need to handle associativity? (a + b) + c vs a + (b + c)?
            return (
                pat.op == e.op
                and self.match_e(pat.lhs, e.lhs)
                and self.match_e(pat.rhs, e.rhs)
            )
        elif isinstance(e, LoopIR.USub):
            return self.match_e(pat.arg, e.arg)
        elif isinstance(e, LoopIR.BuiltIn):
            return pat.f is e.f and all(
                self.match_e(pa, sa) for pa, sa in zip(pat.args, e.args)
            )
        elif isinstance(e, LoopIR.ReadConfig):
            return pat.config == e.config.name() and pat.field == e.field
        elif isinstance(e, LoopIR.StrideExpr):
            return self.match_name(pat.name, e.name) and (
                pat.dim == e.dim or not bool(pat.dim)
            )
        else:
            assert False, "bad case"

    def match_name(self, pat_nm, ir_sym):
        # We use repr(sym) as a way of checking both the Sym name and id
        ir_sym = repr(ir_sym) if self._use_sym_id else str(ir_sym)
        return pat_nm == "_" or pat_nm == ir_sym


def _children(cur) -> Iterable[Node]:
    n = cur._node
    # Top-level proc
    if isinstance(n, LoopIR.proc):
        yield from _children_from_attrs(cur, n, "body")
    # Statements
    elif isinstance(n, (LoopIR.Assign, LoopIR.Reduce)):
        yield from _children_from_attrs(cur, n, "idx", "rhs")
    elif isinstance(n, (LoopIR.WriteConfig, LoopIR.WindowStmt)):
        yield from _children_from_attrs(cur, n, "rhs")
    elif isinstance(n, (LoopIR.Pass, LoopIR.Alloc, LoopIR.Free)):
        yield from []
    elif isinstance(n, LoopIR.If):
        yield from _children_from_attrs(cur, n, "cond", "body", "orelse")
    elif isinstance(n, LoopIR.For):
        yield from _children_from_attrs(cur, n, "lo", "hi", "body")
    elif isinstance(n, LoopIR.Call):
        yield from _children_from_attrs(cur, n, "args")
    # Expressions
    elif isinstance(n, LoopIR.Read):
        yield from _children_from_attrs(cur, n, "idx")
    elif isinstance(n, LoopIR.WindowExpr):
        yield from _children_from_attrs(cur, n, "idx")
    elif isinstance(n, LoopIR.Interval):
        yield from _children_from_attrs(cur, n, "lo", "hi")
    elif isinstance(n, LoopIR.Point):
        yield from _children_from_attrs(cur, n, "pt")
    elif isinstance(
        n,
        (
            LoopIR.Const,
            LoopIR.StrideExpr,
            LoopIR.ReadConfig,
        ),
    ):
        yield from []
    elif isinstance(n, LoopIR.USub):
        yield from _children_from_attrs(cur, n, "arg")
    elif isinstance(n, LoopIR.BinOp):
        yield from _children_from_attrs(cur, n, "lhs", "rhs")
    elif isinstance(n, LoopIR.BuiltIn):
        yield from _children_from_attrs(cur, n, "args")
    else:
        assert False, f"case {type(n)} unsupported"


def _children_from_attrs(cur, n, *args) -> Iterable[Node]:
    for attr in args:
        children = getattr(n, attr)
        if isinstance(children, list):
            for i in range(len(children)):
                yield cur._child_node(attr, i)
        else:
            yield cur._child_node(attr, None)
