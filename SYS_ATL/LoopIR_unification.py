from .prelude import *
from .LoopIR import LoopIR, T, LoopIR_Rewrite

from collections import ChainMap
import sympy as SYMPY


class DoReplace(LoopIR_Rewrite):
    def __init__(self, proc, subproc, stmt_block):
        # ensure that subproc and stmt_block match in # of statements
        n_stmts = len(subproc.body)
        if len(stmt_block) < n_stmts:
            raise SchedulingError("Not enough statements to match")
        stmt_block = stmt_block[:n_stmts]

        self.subproc        = subproc
        self.target_block   = stmt_block

        super().__init__(proc)

    def map_stmts(self, stmts):
        # see if we can find the target block in this block
        n_stmts = len(self.target_block)
        match_i = None
        for i,s in enumerate(stmts):
            if s == self.target_block[0]:
                if stmts[i:i+n_stmts] == self.target_block:
                    match_i = i
                    break

        if match_i is None:
            return super().map_stmts(stmts)
        else: # process the match
            raise NotImplementedError("need to work out splicing")
            #Unify(subproc.body?, stmt_block)

            new_args = [self.sym_to_expr(s, stmt.srcinfo) for s in res]
            new_call = LoopIR.Call(self.subproc, new_args, None, stmt.srcinfo)

            return stmts[ : match_i] + [new_call] + stmts[match_i+n_stmts : ]

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self,e):
        return e
    def map_t(self,t):
        return t
    def map_eff(self,eff):
        return eff

# For Unification we have a problem of the form
#   Stmt block (w/free variables)   ===   subproc(?a, ?b, ?c[?d:?e])
#                                   ===   [args |> ?]asserts ; (need to prove)
#                                           (also assertions about windows
#                                            and sizes)
#                                         [args |> ?]body
#
#   <===>
#   (list of stmts)                 ===   [args |> ?](list of stmts)
#   + check on assertions
#
#   <===>
#       s1 === [args |> ?]s1'
#       s2 === [args |> ?]s2'
#       s3 === [args |> ?]s3' ...
#   + check on assertions
#
#   <===>
#   if s and s' are Assign or Reduce:
#       s.name === [args|>?]s'.name
#       s.type === s'.type
#       s.cast === s'.cast
#       s.idx  === [args|>?]s'.idx
#       s.rhs  === [args|>?]s'.rhs
#   if s and s' are Pass:
#       True
#   if s and s' are If:
#       s.cond === s'.cond
#       s.body === [args|>?]s'.body
#       s.orelse === [args|>?]s'.orelse
#   if s and s' are ForAll:
#       s.iter === [args|>?]s'.iter
#       s.hi   === [args|>?]s'.hi
#       s.body === [args|>?]s'.body
#   if s and s' are Alloc or Free:
#       s.name === [args|>?]s'.name
#       s.type === s'.type
#       s.mem  === s'.mem
#   if s and s' are Call:
#       s.f    === s'.f
#       s.args === [args|>?]s'.args
#   if s and s' are WindowStmt:
#       s.lhs  === [args|>?]s'.lhs
#       s.rhs  === [args|>?]s'.rhs
#
#
#   (list of exprs) === [args |> ?](list of exprs)
#   when
#       e1 === [args|>?]e1'
#       e2 === [args|>?]e2' ...
#
#   if e and e' are Read:
#       e.name === [args|>?]e'.name
#       e.idx  === [args|>?]e'.idx
#   if e and e' are Const:
#       e.val  === e'.val
#   if e and e' are USub:
#       e.arg  === [args|>?]e'.arg
#   if e and e' are BinOp:
#       if e and e' are numeric:
#           e.op === e'.op
#           e.lhs === [args|>?]e'.lhs
#           e.rhs === [args|>?]e'.rhs
#       if e and e' are indexable:
#           symbolic eval(e) === symbolic eval(e')
#   if e and e' are WindowExpr:
#       e.name  === [args|>?]e'.name
#       e.idx   === [args|>?]e'.idx
#
#
#   (list of w_access) === [args |> ?](list of w_access)
#   when
#       w1 === [args|>?]w1'
#       w2 === [args|>?]w2' ...
#
#   if w and w' are Interval:
#       w.lo === [args|>?]w'.lo
#       w.hi === [args|>?]w'.hi
#   if w and w' are Point:
#       w.pt === [args|>?]w'.pt
#
#

#  [s ; pass] === [s]
# (\...  body)(?, ?, ?[?:?]) === [... |-> ?, ?, ?[?:?]] asserts + body
#
# So how do we break that down?
#
# step 1: inline call to subproc
#
#

def subst(proc, stmt, call):
    body = []
    for s in proc.body:
        if s == stmt:
            body.append(call)
        else:
            body.append(s)

    return LoopIR.proc(name    = proc.name,
                       args    = proc.args,
                       preds   = proc.preds,
                       body    = body,
                       instr   = proc.instr,
                       eff     = proc.eff,
                       srcinfo = proc.srcinfo)

class Unification:
    def __init__(self, loopir, subproc, stmt):
        self.orig_proc = loopir
        self.new_proc  = None
        self.equations = []
        self.env       = ChainMap() # sym to sympy's symbol

        unknowns = []
        # initizliaze arg
        for a in subproc.args:
            self.env[a.name] = SYMPY.symbols(f"{a.name}")
            unknowns.append(self.env[a.name])

        self.check_stmts(subproc.body, [stmt])

        print()
        print(unknowns)
        res = SYMPY.solve(self.equations, tuple(unknowns), dict=False)
        res = list(res)
        print(res)

        call = LoopIR.Call(subproc, [self.sym_to_expr(s, stmt.srcinfo) for s in res],
                           None, stmt.srcinfo)

        self.new_proc = subst(self.orig_proc, stmt, call)

    # TODO: We basically need to copy sympy parser logic here?
    # Also we need sympy symbol to sym list?
    def sym_to_expr(self, sym, srcinfo):
        if type(sym) is SYMPY.Add:
            op = "+"
        elif type(sym) is SYMPY.Mul:
            op = "*"
        elif type(sym) is SYMPY.Integer:
            return LoopIR.Const(int(sym), T.int, srcinfo)
        elif type(sym) is SYMPY.Symbol:
            return LoopIR.Const(100, T.int, srcinfo)
        # TODO: No Div nor mod! Handle via Pow
        else:
            assert False, "bad case!"

        args = list(sym.args)
        assert len(args) >= 2

        ir = LoopIR.BinOp(op, sym_to_expr(args[0], srcinfo),
                              sym_to_expr(args[1], srcinfo),
                              None,
                              srcinfo)
        for a in args[2:]:
            ir = LoopIR.BinOp(op, ir, sym_to_expr(a, srcinfo),
                              None, srcinfo)

        return ir

    def result(self):
        return self.new_proc

    def err(self):
        raise TypeError("subproc and pattern don't match")

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    # return sympy for an expr
    # TODO: not implemented
    def expr_to_sym(self, e):
        assert e.type.is_indexable()

        if type(e) is LoopIR.BinOp:
            rhs = self.expr_to_sym(e.rhs)
            lhs = self.expr_to_sym(e.lhs)

            if e.op == "+":
                return SYMPY.Plus(lhs, rhs)
            elif e.op == "-":
                return SYMPY.Minus(lhs, rhs)
            elif e.op == "*":
                return SYMPY.Times(lhs, rhs)
            elif e.op == "/":
                assert type(expr.rhs) is E.Const
                assert expr.rhs.val > 0
                return SYMPY.Div(lhs, rhs)
            elif e.op == "%":
                assert type(expr.rhs) is E.Const
                assert expr.rhs.val > 0
                return SYMPY.Mod(lhs, rhs)
            else:
                self.err()

        elif type(e) is LoopIR.Read:
            if e.name not in self.env:
                sym = symbols(f"{e.name}")
                self.env[e.name] = sym
            else:
                sym = self.env[e.name]
            return sym
        elif type(e) is LoopIR.Const:
            return e.val
        elif type(e) is LoopIR.USub:
            return -self.expr_to_sym(e.arg)
        else:
            assert False, "bad case!"

    # symbol equality
    def eq_sym(self, s1, s2):
        s1_sym, s2_sym = SYMPY.symbols(f"{s1} {s2}")
        self.env[s1] = s1_sym
        self.env[s2] = s2_sym
        self.equations.append(s1_sym - s2_sym)

    # expr equality
    def eq_expr(self, e1, e2):
        e1 = self.expr_to_sym(e1)
        e2 = self.expr_to_sym(e2)
        self.equations.append(e1 - e2)

    def check_stmts(self, subproc_body, stmts):
        # TODO!: how to pattern match multiple stmts??
        # This shouldn't be hardcoded to 1!
        if len(subproc_body) != len(stmts):
            self.err()

        for sub,s in zip(subproc_body, stmts):
            self.check_stmt(sub, s)

    def check_stmt(self, sub_stmt, stmt):
        # If stmt type are different, emit error
        if type(stmt) is not type(sub_stmt):
            self.err()

        if type(stmt) is LoopIR.ForAll:
            self.eq_sym(sub_stmt.iter, stmt.iter)
            self.eq_expr(sub_stmt.hi, stmt.hi)

            self.check_stmts(sub_stmt.body, stmt.body)
        elif type(stmt) is LoopIR.If:
            self.check_expr(sub_stmt.cond, stmt.cond)

            self.check_stmts(sub_stmt.body, stmt.body)
            self.check_stmts(sub_stmt.orelse, stmt.orelse)
        elif type(stmt) is LoopIR.Assign or type(stmt) is LoopIR.Reduce:
            self.eq_sym(sub_stmt.name, stmt.name)
            for se, e in zip(sub_stmt.idx, stmt.idx):
                self.eq_expr(se, e)

        elif type(stmt) is LoopIR.Pass:
            pass

    def check_expr(self, e1, e2):
        if type(e1) is not type(e2):
            self.err()

        # Construct an equation
        if e1.type.is_indexable():
            self.eq_expr(e1, e2)
        else:
        # numeric type should match syntactically
            if type(e1) is LoopIR.Read:
                if e1.name != e2.name:
                    self.err()

                for i1,i2 in zip(e1.idx, e2.idx):
                    self.check_expr(i1, i2)
            elif type(e1) is LoopIR.Const:
                if e1.val != e2.name:
                    self.err()
            elif type(e1) is LoopIR.USub:
                self.check_expr(e1.arg, e2.arg)
            elif type(e1) is LoopIR.BinOp:
                if e1.op != e2.op:
                    self.err()
                self.check_expr(e1.lhs, e2.lhs)
                self.check_expr(e1.rhs, e2.rhs)
            elif type(e1) is LoopIR.WindowExpr:
                if e1.name != e2.name:
                    self.err()

                for i1,i2 in zip(e1.idx, e1.idx):
                    if type(i1) != type(i2):
                        self.err()

                    if type(i1) is LoopIR.Interval:
                        self.check_expr(i1.lo, i2.lo)
                        self.check_expr(i1.hi, i2.hi)
                    elif type(i1) is LoopIR.Point:
                        self.check_expr(i1.pt)
            elif type(e1) is LoopIR.StrideAssert:
                pass
            else:
                assert False, "bad case"
