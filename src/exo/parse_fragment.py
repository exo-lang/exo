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


def parse_fragment(
    proc, fragment, ctx_stmt, call_depth=0, configs=[], scope="before", expr_holes=None
):
    # get source location where this is getting called from
    caller = inspect.getframeinfo(inspect.stack()[call_depth + 1][0])

    # parse the pattern we're going to use to match
    p_ast = pyparser.pattern(fragment, filename=caller.filename, lineno=caller.lineno)
    if isinstance(p_ast, PAST.expr):
        return ParseFragment(
            p_ast, proc, ctx_stmt, configs, scope, expr_holes
        ).results()
    else:
        assert len(p_ast) == 1
        return ParseFragment(
            p_ast[0], proc, ctx_stmt, configs, scope, expr_holes
        ).results()


_PAST_to_LoopIR = {
    # list of exprs
    list: list,
    #
    PAST.Read: LoopIR.Read,
    PAST.Const: LoopIR.Const,
    PAST.USub: LoopIR.USub,
    PAST.BinOp: LoopIR.BinOp,
    PAST.StrideExpr: LoopIR.StrideExpr,
    PAST.BuiltIn: LoopIR.BuiltIn,
    PAST.ReadConfig: LoopIR.ReadConfig,
}


class BuildEnv(LoopIR_Do):
    def __init__(self, proc, stmt):
        self.env = ChainMap()
        self.result = None
        self.trg = stmt
        self.proc = proc
        self.halt = False

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
            self.result = self.env
            self.halt = True
        if self.halt:
            return

        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            self.env[s.name] = s.type
            for e in s.idx:
                self.do_e(e)
            self.do_e(s.rhs)
            self.do_t(s.type)
        elif styp is LoopIR.For:
            self.push()
            self.env[s.iter] = T.index
            self.do_e(s.lo)
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
        self.env = ChainMap()
        self.in_scope = False
        self.stack = []
        self.trg = stmt
        self.proc = proc
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
            elif styp is LoopIR.For:
                self.env[s.iter] = T.index
                self.do_e(s.lo)
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
            if styp is LoopIR.For:
                self.do_e(s.lo)
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
    def __init__(self, pat, proc, stmt, configs, scope, expr_holes):
        assert isinstance(stmt, LoopIR.stmt) or (stmt is None)
        assert isinstance(pat, PAST.expr)

        self._results = None  # results should be expression
        self.stmt = stmt
        self.env = ChainMap()
        self.configs = {c.name(): c for c in configs}
        self.expr_holes = expr_holes

        if stmt is None:
            self.srcinfo = proc.srcinfo

            # If stmt is None, env should be just arguments
            for a in proc.args:
                self.env[a.name] = a.type
        else:
            self.srcinfo = stmt.srcinfo

            assert scope == "before", (
                "Non-before scope for parsing "
                "fragments is not well-defined; "
                "it needs to be removed from the code"
            )

            if scope == "before":
                self.env = BuildEnv(proc, stmt).result
            elif scope == "after":
                self.env = BuildEnv_after(proc, stmt).result()
            elif scope == "before_after":
                env1 = BuildEnv(proc, stmt).result
                env2 = BuildEnv_after(proc, stmt).result()
                self.env = ChainMap(env1, env2)
            else:
                assert False, "bad case"

        self._results = self.parse_e(pat)

    def parse_e(self, pat):
        if isinstance(pat, PAST.Read):
            nm = self.find_sym(pat.name)
            if nm is None:
                raise ParseFragmentError(
                    f"{pat.name} not found in the " + "current environment"
                )
            idx = [self.parse_e(i) for i in pat.idx]
            return LoopIR.Read(nm, idx, self.env[nm], self.srcinfo)
        elif isinstance(pat, PAST.BinOp):
            lhs = self.parse_e(pat.lhs)
            rhs = self.parse_e(pat.rhs)
            return LoopIR.BinOp(
                pat.op, lhs, rhs, self.type_for_binop(pat.op, lhs, rhs), self.srcinfo
            )
        elif isinstance(pat, PAST.USub):
            arg = self.parse_e(pat.arg)
            return LoopIR.USub(arg, arg.type, self.srcinfo)
        elif isinstance(pat, PAST.StrideExpr):
            nm = self.find_sym(pat.name)
            return LoopIR.StrideExpr(nm, pat.dim, T.stride, self.srcinfo)
        elif isinstance(pat, PAST.Const):
            typ = {float: T.R, bool: T.bool, int: T.int}.get(type(pat.val))
            assert typ is not None, "bad type!"
            return LoopIR.Const(pat.val, typ, self.srcinfo)
        elif isinstance(pat, PAST.BuiltIn):
            args = [self.parse_e(a) for a in pat.args]
            try:
                typ = pat.f.typecheck(args)
            except BuiltIn_Typecheck_Error as err:
                raise ParseFragmentError(err)

            return LoopIR.BuiltIn(pat.f, args, typ, self.srcinfo)
        elif isinstance(pat, PAST.ReadConfig):
            if pat.config not in self.configs:
                raise ParseFragmentError(
                    f"Could not find Config named '{pat.config}'. "
                    f"Try supplying a list of Config objects via an "
                    f"optional 'configs' argument"
                )

            cfg = self.configs[pat.config]
            if not cfg.has_field(pat.field):
                raise ParseFragmentError(
                    f"Config named '{pat.config}' does not have a field "
                    f"named '{pat.field}'"
                )

            typ = cfg.lookup_type(pat.field)
            return LoopIR.ReadConfig(cfg, pat.field, typ, self.srcinfo)
        elif isinstance(pat, PAST.E_Hole):
            if self.expr_holes == None:
                raise ParseFragmentError("String cannot contain holes")
            if len(self.expr_holes) == 0:
                raise ParseFragmentError(
                    "String contains more holes than expressions provided"
                )
            subtree = self.expr_holes[0]
            self.expr_holes = self.expr_holes[1:]
            assert isinstance(subtree, LoopIR.expr)
            return self.rebuild_ast(subtree)
        else:
            assert False, f"bad case: {type(pat)}"

    def rebuild_ast(self, loopIR_expr):
        def expr_hole_parsing_error(name, msg):
            raise ParseFragmentError(
                f"{name} used in an expression to fill a string hole, but {msg}"
            )

        def check_sym_consistency(sym):
            env_sym = self.find_sym(str(sym))
            if env_sym != sym:
                expr_hole_parsing_error(sym, "not found in current environment")
            if self.env[env_sym] != self.env[sym]:
                expr_hole_parsing_error(
                    sym, "has a different type in current environment"
                )

        if isinstance(loopIR_expr, LoopIR.Read):
            check_sym_consistency(loopIR_expr.name)
            idx = [self.rebuild_ast(i) for i in loopIR_expr.idx]
            return loopIR_expr.update(idx=idx, srcinfo=self.srcinfo)
        elif isinstance(loopIR_expr, LoopIR.Const):
            return loopIR_expr.update(srcinfo=self.srcinfo)
        elif isinstance(loopIR_expr, LoopIR.USub):
            return loopIR_expr.update(
                arg=self.rebuild_ast(loopIR_expr.arg), srcinfo=self.srcinfo
            )
        elif isinstance(loopIR_expr, LoopIR.BinOp):
            return loopIR_expr.update(
                lhs=self.rebuild_ast(loopIR_expr.lhs),
                rhs=self.rebuild_ast(loopIR_expr.rhs),
                srcinfo=self.srcinfo,
            )
        elif isinstance(loopIR_expr, LoopIR.BuiltIn):
            args = [self.rebuild_ast(a) for a in loopIR_expr.args]

            try:
                typ = loopIR_expr.f.typecheck(args)
            except BuiltIn_Typecheck_Error as err:
                raise ParseFragmentError(err)

            if typ != loopIR_expr.typ:
                expr_hole_parsing_error(
                    loopIR_expr.f.name(),
                    "builtin is called with different argument types",
                )

            return loopIR_expr.update(args=args, srcinfo=self.srcinfo)
        elif isinstance(loopIR_expr, LoopIR.WindowExpr):
            raise ParseFragmentError(
                "Using a window expression is not allowed to fill a hole in an expression"
            )
        elif isinstance(loopIR_expr, LoopIR.StrideExpr):
            check_sym_consistency(loopIR_expr.name)
            return loopIR_expr.update(srcinfo=self.srcinfo)
        elif isinstance(loopIR_expr, LoopIR.ReadConfig):
            config_name = loopIR_expr.config.name()
            if config_name not in self.configs:
                raise ParseFragmentError(
                    f"Could not find Config named '{config_name}'. "
                    f"Try supplying a list of Config objects via an "
                    f"optional 'configs' argument"
                )

            cfg = self.configs[config_name]
            if not cfg.has_field(loopIR_expr.field):
                raise ParseFragmentError(
                    f"Config named '{config_name}' does not have a field "
                    f"named '{loopIR_expr.field}'"
                )

            typ = cfg.lookup_type(loopIR_expr.field)
            if typ != loopIR_expr.typ:
                raise ParseFragmentError(
                    f"Type of field '{loopIR_expr.field}' in Config named '{config_name}' is different"
                )
            return loopIR_expr.update(srcinfo=self.srcinfo)
        else:
            assert False, "Bad Case"

    def find_sym(self, expr):
        for k in self.env.keys():
            if expr == str(k):
                return k

    def results(self):
        return self._results

    def type_for_binop(self, op, lhs, rhs):
        if (lhs.type is T.size and rhs.type is T.size) and (op == "+" or op == "*"):
            return T.size

        return {
            "+": T.index,
            "-": T.index,
            "*": T.index,
            "/": T.index,
            "%": T.index,
            #
            "<": T.bool,
            ">": T.bool,
            "<=": T.bool,
            ">=": T.bool,
            "==": T.bool,
            #
            "and": T.bool,
            "or": T.bool,
        }[op]
