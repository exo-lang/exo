from .prelude import *
from .LoopIR import LoopIR, LoopIR_Rewrite, Alpha_Rename, LoopIR_Do, PAST
from . import pyparser
from .LoopIR import T
import re

import inspect

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

def match_pattern(ast, pattern_str, call_depth=0, default_match_no=None):
    # get source location where this is getting called from
    caller = inspect.getframeinfo(inspect.stack()[call_depth+1][0])

    # break-down pattern_str for possible #<num> post-fix
    match = re.search(r"^([^\#]+)\#(\d+)\s*$", pattern_str)
    if match:
        pattern_str = match[1]
        match_no    = int(match[2]) if match[2] is not None else None
    else:
        match_no    = default_match_no # None means match-all

    # parse the pattern we're going to use to match
    p_ast         = pyparser.pattern(pattern_str,
                                     filename=caller.filename,
                                     lineno=caller.lineno)

    # do the pattern match, to find the nodes in ast
    return PatternMatch(p_ast, ast, match_no=match_no).results()


_PAST_to_LoopIR = {
  # list of stmts
  list:               list,
  #
  PAST.Assign:        LoopIR.Assign,
  PAST.Reduce:        LoopIR.Reduce,
  PAST.Pass:          LoopIR.Pass,
  PAST.If:            LoopIR.If,
  PAST.ForAll:        LoopIR.ForAll,
  PAST.Seq   :        LoopIR.Seq,
  PAST.Alloc:         LoopIR.Alloc,
  PAST.Call:          LoopIR.Call,
  PAST.S_Hole:        None,
  #
  PAST.Read:          LoopIR.Read,
  PAST.Const:         LoopIR.Const,
  PAST.USub:          LoopIR.USub,
  PAST.BinOp:         LoopIR.BinOp,
  PAST.E_Hole:        None,
}


class PatternMatch:
    def __init__(self, pat, src_stmts, match_no=None):
        self._match_i   = match_no
        self._results   = []

        # prevent the top level of a pattern being just a hole
        if type(pat) is PAST.E_Hole:
            raise PatternMatchError("pattern match on 'anything' unsupported")
        elif (type(pat) is list and
              all( type(p) is PAST.S_Hole for p in pat )):
            raise PatternMatchError("pattern match on 'anything' unsupported")

        if type(pat) is list:
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
            if type(p) is PAST.S_Hole:
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
            if type(p) is not PAST.S_Hole:
                in_hole = False

            # finally, if we're here we found a match and can advance the
            # stmt_idx counter
            stmt_idx += 1

        # if we made it to the end of the function, then
        # the match was successful
        return True


    def match_stmt(self, pat, stmt):
        assert type(pat) is not PAST.S_Hole, ("holes should be handled "+
                                              "in match_stmts")

        # first ensure that we the pattern and statement
        # are the same constructor
        styp = type(stmt)
        if _PAST_to_LoopIR[type(pat)] is not styp:
            return False

        # then handle each constructor as a structural case
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            return ( self.match_name(pat.name, stmt.name) and
                     all( self.match_e(pi,si)
                          for pi,si in zip(pat.idx,stmt.idx) ) and
                     self.match_e(pat.rhs, stmt.rhs) )
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
            return ( self.match_name(pat.name, stmt.name) )
        elif styp is LoopIR.Call:
            return ( self.match_name(pat.f, stmt.f.name) )
        else: assert False, f"bad case: {styp}"


    def match_e(self, pat, e):
        # expression holes can match anything
        # and we don't have to worry about Kleene-Star behavior
        if type(pat) is PAST.E_Hole:
            return True

        # first ensure that we the pattern and statement
        # are the same constructor
        etyp = type(e)
        if _PAST_to_LoopIR[type(pat)] is not etyp:
            return False

        if etyp is LoopIR.Read:
            return ( self.match_name(pat.name, e.name) and
                     all( self.match_e(pi,si)
                          for pi,si in zip(pat.idx,e.idx) ) )
        elif etyp is LoopIR.Const:
            return ( pat.val == e.val )
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
