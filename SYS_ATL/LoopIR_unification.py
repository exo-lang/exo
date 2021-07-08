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
            res = Unification(subproc, stmt_block)

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



class Unification:
    def __init__(self, subproc, stmt_block):
        self.equations = []
        self.unknowns = []

        # Initialize asserts
        for a in subproc.preds:
            self.equations.append(a)

        # Initialize size
        for a in subproc.args:
            if a.type.is_size():
                self.equation.append(a > 0)

        self.check_stmts(subproc.body, stmt_block)


    def err(self):
        raise TypeError("subproc and pattern don't match")

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents


    def check_stmts(self, body, stmt_block):
        assert len(subproc_body) == len(stmts)

        for sub,s in zip(body, stmt_block):
            self.check_stmt(sub, s)

    def check_stmt(self, sub_stmt, stmt):
        # If stmt type are different, emit error
        if type(stmt) is not type(sub_stmt):
            self.err()

        if type(stmt) is LoopIR.ForAll:
            # Substitute all occurance of subproc_stmt.iter in 
            # subproc body to stmt.iter
            sub_body = subst(sub_stmt.body, sub_stmt.iter, stmt.iter)

            # hi == hi
            self.equations.append(sub_stmt.hi, stmt.hi)

            self.check_stmts(sub_body, stmt.body)

        elif type(stmt) is LoopIR.If:
            self.check_expr(sub_stmt.cond, stmt.cond)

            self.check_stmts(sub_stmt.body, stmt.body)
            self.check_stmts(sub_stmt.orelse, stmt.orelse)
        elif type(stmt) is LoopIR.Assign or type(stmt) is LoopIR.Reduce:
        #stmt    = Assign ( sym name, type type, string? cast, expr* idx, expr rhs )
        #TODO: Think!

        elif type(stmt) is LoopIR.Pass:
            pass

        elif type(stmt) is LoopIR.Alloc:
            # Substitute the rest of body by this new symbol?
        elif type(stmt) is LoopIR.WindowStmt:
            # Substitute the rest of body by this new symbol?
        elif type(stmt) is LoopIR.Call:
            if sub_stmt.f != stmt.f:
                self.err()

            for e1, e2 in zip(sub_stmt.args, stmt.args):
                self.check_expr(e1, e2)

    stmt    = Assign ( sym name, type type, string? cast, expr* idx, expr rhs )
            | Reduce ( sym name, type type, string? cast, expr* idx, expr rhs )
            | Alloc  ( sym name, type type, mem? mem )
            | WindowStmt( sym lhs, expr rhs )

    def check_expr(self, e1, e2):
        if type(e1) is not type(e2):
            self.err()

        # numeric type should match syntactically
        if type(e1) is LoopIR.Read:
            # TODO: Think!
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
            # TODO: Think!
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
