import inspect
from collections import ChainMap

from . import pyparser
from .LoopIR import T, LoopIR_Do, LoopIR, PAST


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Parse Fragment Errors

class ParseFragmentError(Exception):
    pass


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# General Fragment Parsing

def parse_fragment(proc, fragment, ctx_stmt, call_depth=0, scope="before"):
    # get source location where this is getting called from
    caller = inspect.getframeinfo(inspect.stack()[call_depth+1][0])

    # parse the pattern we're going to use to match
    p_ast         = pyparser.pattern(fragment,
                                     filename=caller.filename,
                                     lineno=caller.lineno)
    if isinstance(p_ast, PAST.expr):
        return ParseFragment(p_ast, proc, ctx_stmt, scope).results()
    else:
        assert len(p_ast) == 1
        return ParseFragment(p_ast[0], proc, ctx_stmt, scope).results()


_PAST_to_LoopIR = {
  # list of exprs
  list:               list,
  #
  PAST.Read:          LoopIR.Read,
  PAST.Const:         LoopIR.Const,
  PAST.USub:          LoopIR.USub,
  PAST.BinOp:         LoopIR.BinOp,
  PAST.StrideExpr:    LoopIR.StrideExpr,
}


class BuildEnv(LoopIR_Do):
    def __init__(self, proc, stmt):
        self.env       = ChainMap()
        self.result    = None
        self.trg       = stmt
        self.proc      = proc

        for a in self.proc.args:
            self.env[a.name] = a.type
            self.do_t(a.type)
        for p in self.proc.preds:
            self.do_e(p)

        self.do_stmts(self.proc.body)

    def result(self):
        return self.result

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    def do_s(self, s):
        if s == self.trg:
            self.result = self.env.copy()

        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            self.env[s.name] = s.type
            for e in s.idx:
                self.do_e(e)
            self.do_e(s.rhs)
            self.do_t(s.type)
        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            self.push()
            self.env[s.iter] = T.index
            self.do_e(s.hi)
            self.do_stmts(s.body)
            self.pop()
        elif styp is LoopIR.If:
            self.push()
            self.do_e(s.cond)
            self.do_stmts(s.body)
            if len(s.orelse) > 0:
                self.do_stmts(s.orelse)
            self.pop()
        elif styp is LoopIR.Alloc:
            self.env[s.name] = s.type
            self.do_t(s.type)
        else:
            super().do_s(s)


class BuildEnv_after(LoopIR_Do):
    def __init__(self, proc, stmt):
        self.env       = ChainMap()
        self.in_scope  = False
        self.stack     = []
        self.trg       = stmt
        self.proc      = proc
        self.do_stmts(self.proc.body)

    def result(self):
        return self.env

    def do_s(self, s):
        if s == self.trg:
            self.in_scope = True

        styp = type(s)
        if self.in_scope:
            if styp is LoopIR.Assign or styp is LoopIR.Reduce:
                self.env[s.name] = s.type
                for e in s.idx:
                    self.do_e(e)
                self.do_e(s.rhs)
                self.do_t(s.type)
            elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
                self.env[s.iter] = T.index
                self.do_e(s.hi)
                self.do_stmts(s.body)
            elif styp is LoopIR.If:
                self.do_e(s.cond)
                self.do_stmts(s.body)
                if len(s.orelse) > 0:
                    self.do_stmts(s.orelse)
            elif styp is LoopIR.Alloc:
                self.env[s.name] = s.type
                self.do_t(s.type)
            else:
                super().do_s(s)
        else:
            # Can introduce scope
            if styp is LoopIR.ForAll or styp is LoopIR.Seq:
                self.do_e(s.hi)
                self.do_stmts(s.body)
                if self.in_scope:
                    self.in_scope = False
            elif styp is LoopIR.If:
                self.do_e(s.cond)
                self.do_stmts(s.body)
                if len(s.orelse) > 0:
                    self.do_stmts(s.orelse)
                if self.in_scope:
                    self.in_scope = False
            else:
                super().do_s(s)



class ParseFragment:
    def __init__(self, pat, proc, stmt, scope):
        assert isinstance(stmt, LoopIR.stmt) or (stmt is None)
        assert isinstance(pat, PAST.expr)

        self._results   = None # results should be expression
        self.stmt = stmt
        self.env  = ChainMap()

        if stmt is None:
            self.srcinfo = proc.srcinfo

            # If stmt is None, env should be just arguments
            for a in proc.args:
                self.env[a.name] = a.type
        else:
            self.srcinfo = stmt.srcinfo

            if scope == "before":
                self.env = BuildEnv(proc, stmt).result
            elif scope == "after":
                self.env = BuildEnv_after(proc, stmt).result()
            elif scope == "before_after":
                env1 = BuildEnv(proc, stmt).result
                env2 = BuildEnv_after(proc, stmt).result()
                self.env = ChainMap(env1 , env2)
            else:
                assert False, "bad case"

        self._results = self.parse_e(pat)

    def parse_e(self, pat):
        if isinstance(pat, PAST.Read):
            nm = self.find_sym(pat.name)
            if nm is None:
                raise ParseFragmentError(f"{pat.name} not found in the "+
                                          "current environment")
            idx = [self.find_sym(i) for i in pat.idx]
            return LoopIR.Read(nm, idx, self.env[nm], self.srcinfo)
        elif isinstance(pat, PAST.BinOp):
            lhs = self.parse_e(pat.lhs)
            rhs = self.parse_e(pat.rhs)
            return LoopIR.BinOp(pat.op, lhs, rhs, self.type_for_binop(pat.op),
                                self.srcinfo)
        elif isinstance(pat, PAST.StrideExpr):
            nm = self.find_sym(pat.name)
            return LoopIR.StrideExpr(nm, pat.dim, T.stride, self.srcinfo)
        elif isinstance(pat, PAST.Const):
            typ = {float: T.R, bool: T.bool, int: T.int}.get(type(pat.val))
            assert typ is not None, "bad type!"
            return LoopIR.Const(pat.val, typ, self.srcinfo)

    def find_sym(self, expr):
        for k in self.env.keys():
            if expr == str(k):
                return k

    def results(self):
        return self._results

    def type_for_binop(self, op):
        return {
            "+":   T.int,
            "-":   T.int,
            "*":   T.int,
            "/":   T.int,
            "%":   T.int,
            #
            "<":   T.bool,
            ">":   T.bool,
            "<=":  T.bool,
            ">=":  T.bool,
            "==":  T.bool,
            #
            "and": T.bool,
            "or":  T.bool,
        }[op]
