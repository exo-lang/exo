import inspect
import re
from typing import Optional

from . import pyparser
from .LoopIR import LoopIR, PAST


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
    if (pos := pattern_str.rfind('#')) == -1:
        return None
    return int(pattern_str[pos + 1:])


def match_pattern(ast, pattern_str, call_depth=0, default_match_no=None):
    # get source location where this is getting called from
    caller = inspect.getframeinfo(inspect.stack()[call_depth + 1][0])

    # break-down pattern_str for possible #<num> post-fix
    if match := re.search(r"^([^\#]+)\#(\d+)\s*$", pattern_str):
        pattern_str = match[1]
        match_no = int(match[2]) if match[2] is not None else None
    else:
        match_no = default_match_no  # None means match-all

    # parse the pattern we're going to use to match
    p_ast = pyparser.pattern(pattern_str,
                             filename=caller.filename,
                             lineno=caller.lineno)

    # do the pattern match, to find the nodes in ast
    return PatternMatch(p_ast, ast, match_no=match_no).results()


_PAST_to_LoopIR = {
  # list of stmts
  list:               list,
  #
  PAST.Assign:        [LoopIR.Assign],
  PAST.Reduce:        [LoopIR.Reduce],
  PAST.Pass:          [LoopIR.Pass],
  PAST.If:            [LoopIR.If],
  PAST.ForAll:        [LoopIR.ForAll, LoopIR.Seq],
  PAST.Alloc:         [LoopIR.Alloc],
  PAST.Call:          [LoopIR.Call],
  PAST.WriteConfig:   [LoopIR.WriteConfig],
  PAST.S_Hole:        None,
  #
  PAST.Read:          [LoopIR.Read],
  PAST.Const:         [LoopIR.Const],
  PAST.USub:          [LoopIR.USub],
  PAST.BinOp:         [LoopIR.BinOp],
  PAST.E_Hole:        None,
}


class PatternMatch:
    def __init__(self, pat, src_stmts, match_no=None):
        self._match_i   = match_no
        self._results   = []

        # prevent the top level of a pattern being just a hole
        if isinstance(pat, PAST.E_Hole):
            raise PatternMatchError("pattern match on 'anything' unsupported")
        elif (isinstance(pat, list) and
              all(isinstance(p, PAST.S_Hole) for p in pat)):
            raise PatternMatchError("pattern match on 'anything' unsupported")

        if isinstance(pat, list):
            assert len(pat) > 0
            self.find_stmts_in_stmts(pat, src_stmts)
        else:
            assert isinstance(pat, PAST.expr)
            self.find_e_in_stmts(pat, src_stmts)

    def results(self):
        return self._results

    ## -------------------
    ##  finding methods

    def find_e_in_stmts(self, pat, stmts):
        for s in stmts:
            self.find_e_in_stmt(pat, s)

    def find_e_in_stmt(self, pat, stmt):
        # short-circuit if we have our one match already...
        if self._match_i is not None and self._match_i < 0:
            return

        styp = type(stmt)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            for e in stmt.idx:
                self.find_e_in_e(pat, e)
            self.find_e_in_e(pat, stmt.rhs)
        elif styp is LoopIR.If:
            self.find_e_in_e(pat, stmt.cond)
            self.find_e_in_stmts(pat, stmt.body)
            self.find_e_in_stmts(pat, stmt.orelse)
        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            self.find_e_in_e(pat, stmt.hi)
            self.find_e_in_stmts(pat, stmt.body)
        elif styp is LoopIR.Call:
            for e in stmt.args:
                self.find_e_in_e(pat, e)
        else: pass # ignore other statements

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
        etyp = type(e)
        if etyp is LoopIR.BinOp:
            self.find_e_in_e(pat, e.lhs)
            self.find_e_in_e(pat, e.rhs)

        if etyp is LoopIR.USub:
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
        styp = type(stmts[0])
        if  styp is LoopIR.If:
            self.find_stmts_in_stmts(pat, stmts[0].body)
            self.find_stmts_in_stmts(pat, stmts[0].orelse)
        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            self.find_stmts_in_stmts(pat, stmts[0].body)
        else: pass # other forms of statement do not contain stmt blocks

        # second, recurse on the tail of this sequence...
        self.find_stmts_in_stmts(pat, stmts[1:])


    ## -------------------
    ##  matching methods

    def match_stmts(self, pats, stmts):
        # variable for keeping track of whether we
        # can currently match arbitrary statements
        # into a statement hole
        in_hole = False

        # shallow copy of the stmt list that we can
        stmts   = stmts.copy()

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
        assert not isinstance(pat, PAST.S_Hole), ("holes should be handled "
                                                  "in match_stmts")

        # first ensure that we the pattern and statement
        # are the same constructor
        styp = type(stmt)

        if (styp not in _PAST_to_LoopIR[type(pat)] and
                styp is not LoopIR.WindowStmt):
            return False

        # then handle each constructor as a structural case
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            return ( self.match_name(pat.name, stmt.name) and
                     all( self.match_e(pi,si)
                          for pi,si in zip(pat.idx,stmt.idx) ) and
                     self.match_e(pat.rhs, stmt.rhs) )
        elif styp is LoopIR.WindowStmt:
            if isinstance(pat, PAST.Assign):
                return (self.match_name(pat.name, stmt.lhs) and
                        pat.idx == [] and
                        self.match_e(pat.rhs, stmt.rhs))
            else:
                return False
        elif styp is LoopIR.Pass:
            return True
        elif styp is LoopIR.If:
            return ( self.match_e(pat.cond, stmt.cond) and
                     self.match_stmts(pat.body, stmt.body) and
                     self.match_stmts(pat.orelse, stmt.orelse) )
        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            return ( self.match_name(pat.iter, stmt.iter) and
                     self.match_e(pat.hi, stmt.hi) and
                     self.match_stmts(pat.body, stmt.body) )
        elif styp is LoopIR.Alloc:
            if isinstance(stmt.type, LoopIR.Tensor):
                return ( all( self.match_e(pi,si)
                              for pi,si in zip(pat.sizes, stmt.type.hi) )
                         and self.match_name(pat.name, stmt.name) )
            else: # scalar
                return ( self.match_name(pat.name, stmt.name) )
        elif styp is LoopIR.Call:
            return ( self.match_name(pat.f, stmt.f.name) )
        elif styp is LoopIR.WriteConfig:
            return ( self.match_name( stmt.config.name(), pat.config ) and
                     self.match_name( stmt.field, pat.field ) )
        else: assert False, f"bad case: {styp}"


    def match_e(self, pat, e):
        # expression holes can match anything
        # and we don't have to worry about Kleene-Star behavior
        if isinstance(pat, PAST.E_Hole):
            return True

        # first ensure that we the pattern and statement
        # are the same constructor
        etyp = type(e)
        if (etyp not in _PAST_to_LoopIR[type(pat)] and
                etyp is not LoopIR.WindowExpr):
            return False

        if etyp is LoopIR.Read:
            return ( self.match_name(pat.name, e.name) and
                     all( self.match_e(pi,si)
                          for pi,si in zip(pat.idx,e.idx) ) )
        elif etyp is LoopIR.WindowExpr:
            if isinstance(pat, PAST.Read):
                # TODO: Should we be able to handle window slicing matching? Nah..
                if len(pat.idx) != 1 or not isinstance(pat.idx[0], PAST.E_Hole):
                    return False
                return self.match_name(pat.name, e.name)
            else:
                return False
        elif etyp is LoopIR.Const:
            return pat.val == e.val
        elif etyp is LoopIR.BinOp:
            return ( pat.op == e.op and
                     self.match_e(pat.lhs, e.lhs) and
                     self.match_e(pat.rhs, e.rhs) )
        elif etyp is LoopIR.USub:
            return self.match_e(pat.arg, e.arg)
        else:
            assert False, "bad case"

    def match_name(self, pat_nm, ir_sym):
        return pat_nm == '_' or pat_nm == str(ir_sym)
