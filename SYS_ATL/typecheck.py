from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *
from .LoopIR import UAST, LoopIR, front_ops, bin_ops
from . import shared_types as T

from .instruction_type import is_valid_mem

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# This pass converts from UAST to LoopIR
#   In the process, different kinds of expressions are identified and
#   checked for proper usage.
#
#   Non-Goals:
#     - The UAST already has symbol conversion sorted out, so we
#       don't need to do that or to capture additional Python values
#       due to metaprogramming etc.
#     - We're not going to check bounds correctness here
#     - We're not going to check for parallelism-induced race-conditions here
#


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The typechecker


class TypeChecker:
    def __init__(self, proc):
        self.uast_proc = proc
        self.env = Environment()
        self.errors = []

        args = []
        for sz in proc.sizes:
            self.env[sz] = T.size
            args.append(LoopIR.fnarg(sz, T.size, None, None, proc.srcinfo))

        for a in proc.args:
            self.env[a.name] = a.type
            mem = a.mem
            if mem and not is_valid_mem(mem):
                self.err(a, f"invalid memory name '{mem}'")
                mem = None
            args.append(LoopIR.fnarg(a.name, a.type, a.effect, mem, a.srcinfo))


        body = self.check_stmts(proc.body)

        self.loopir_proc = LoopIR.proc(name=proc.name,
                                       args=args,
                                       body=body,
                                       srcinfo=proc.srcinfo)

        # do error checking here
        if len(self.errors) > 0:
            raise TypeError("Errors occurred during typechecking:\n" +
                            "\n".join(self.errors))

    def get_loopir(self):
        return self.loopir_proc

    def err(self, node, msg):
        self.errors.append(f"{node.srcinfo}: {msg}")

    def check_stmts(self, body):
        assert len(body) > 0
        stmts = []
        for s in body:
            stmts.append( self.check_single_stmt(s) )
        return stmts

    def check_access(self, node, nm, idx, lvalue=False):
        # check indexing
        idx = [ self.check_e(i) for i in idx ]
        for i in idx:
            if i.type != T.err and not i.type.is_indexable():
                self.err(i, f"cannot index with expression of type '{i.type}'")

        # check compatibility with buffer type
        typ     = self.env[nm]
        if typ is T.err:
            pass
        elif typ.is_numeric():
            if len(typ.shape()) != len(idx):
                self.err(node, f"expected access of variable " +
                               f"'{nm}' of type '{typ}' to have " +
                               f"{len(typ.shape())} " +
                               f"indices, but {len(idx)} were found.")
                typ = T.err
            else:
                typ = typ.base()
        elif lvalue:
            self.err(node, f"cannot assign/reduce to '{nm}', " +
                           f"a non-numeric variable of type '{typ}'")
            typ = T.err

        return idx, typ

    def check_single_stmt(self, stmt):
        if type(stmt) is UAST.Assign or type(stmt) is UAST.Reduce:
            idx, typ = self.check_access(stmt, stmt.name, stmt.idx,
                                         lvalue=True)
            rhs     = self.check_e(stmt.rhs)
            if rhs.type != T.err and not rhs.type.is_numeric():
                self.err(rhs, f"cannot assign/reduce a "+
                              f"'{rhs.type}' type value")

            IRnode  = (LoopIR.Assign if type(stmt) is UAST.Assign else
                       LoopIR.Reduce)
            return IRnode(stmt.name, idx, rhs, stmt.srcinfo)

        elif type(stmt) is UAST.Pass:
            return LoopIR.Pass(stmt.srcinfo)

        elif type(stmt) is UAST.If:
            cond    = self.check_e(stmt.cond)
            if cond.type != T.err and cond.type != T.bool:
                self.err(cond, f"expected a bool expression")
            body    = self.check_stmts(stmt.body)
            ebody   = []
            if len(stmt.orelse) > 0:
                ebody   = self.check_stmts(stmt.orelse)
            return LoopIR.If(cond, body, ebody, stmt.srcinfo)

        elif type(stmt) is UAST.ForAll:
            self.env[stmt.iter] = T.index

            # handle standard ParRanges
            parerr = ("currently only supporting for-loops of the form:\n" +
                      "  for _ in par(0, affine_expression):")

            if (type(stmt.cond) is not UAST.ParRange or
                    type(stmt.cond.lo) is not UAST.Const or
                    stmt.cond.lo.val != 0):
                self.err(stmt.cond, parerr)

            hi = self.check_e(stmt.cond.hi)
            if hi.type != T.err and not hi.type.is_sizeable():
                self.err(hi, "expected loop bound of type 'int' or "
                             "type 'size'")

            body = self.check_stmts(stmt.body)
            return LoopIR.ForAll(stmt.iter, hi, body, stmt.srcinfo)

        elif type(stmt) is UAST.Alloc:
            self.env[stmt.name] = stmt.type
            mem = stmt.mem
            if mem and not is_valid_mem(mem):
                self.err(stmt, f"invalid memory name '{mem}'")
                mem = None
            return LoopIR.Alloc(stmt.name, stmt.type, mem, stmt.srcinfo)

        elif type(stmt) is UAST.Instr:
            body = self.check_single_stmt(stmt.body)
            stmt.op.typecheck(body, self)
            return LoopIR.Instr(stmt.op, body, stmt.srcinfo)

        else:
            assert False, "not a loopir in check_stmts"

    def check_e(self, e):
        if type(e) is UAST.Read:
            idx, typ = self.check_access(e, e.name, e.idx, lvalue=False)
            return LoopIR.Read(e.name, idx, typ, e.srcinfo)

        elif type(e) is UAST.Const:
            if type(e.val) is float:
                return LoopIR.Const(e.val, T.R, e.srcinfo)
            elif type(e.val) is int:
                return LoopIR.Const(e.val, T.int, e.srcinfo)
            elif type(e.val) is bool:
                return LoopIR.Const(e.val, T.bool, e.srcinfo)
            else:
                self.err(e, f"literal of unexpected type '{type(e.val)}' "
                            f"and value: {e.val}")
                return LoopIR.Const(0, T.err, e.srcinfo)

        elif type(e) is UAST.USub:
            arg = self.check_e(e.arg)
            if arg.type == T.R or arg.type.is_indexable():
                neg1 = -1.0 if arg.type == T.R else -1
                return LoopIR.BinOp("*",
                                    LoopIR.Const(neg1, arg.type, e.srcinfo),
                                    arg,
                                    arg.type, e.srcinfo)

        elif type(e) is UAST.BinOp:
            lhs = self.check_e(e.lhs)
            rhs = self.check_e(e.rhs)
            typ = T.err
            if lhs.type == T.err or rhs.type == T.err:
                typ = T.err
            elif e.op == "and" or e.op == "or":
                if lhs.type is not T.bool:
                    self.err(lhs, "expected 'bool' argument to logical op")
                if rhs.type is not T.bool:
                    self.err(rhs, "expected 'bool' argument to logical op")
                typ = T.bool
            elif e.op == "==" and lhs.type == T.bool and rhs.type == T.bool:
                typ = T.bool
            elif (e.op == "<" or e.op == "<=" or e.op == "==" or
                  e.op == ">" or e.op == ">="):
                if not lhs.type.is_indexable():
                    self.err(lhs, f"expected 'index' or 'size' argument to "+
                                  f"comparison op: {e.op}")
                if not rhs.type.is_indexable():
                    self.err(rhs, f"expected 'index' or 'size' argument to "+
                                  f"comparison op: {e.op}")
                typ = T.bool
            elif (e.op == "+" or e.op == "-" or e.op == "*" or
                  e.op == "/" or e.op == "%"):
                if lhs.type == T.R:
                    if rhs.type != T.R:
                        self.err(rhs, "expected numeric type")
                        typ = T.err
                    elif e.op == "%":
                        self.err(e, "cannot compute modulus of 'num' values")
                        typ = T.err
                    else:
                        typ = T.R
                elif lhs.type == T.bool:
                    self.err(lhs, "cannot perform arithmetic on 'bool' values")
                    typ = T.err
                elif rhs.type == T.bool:
                    self.err(rhs, "cannot perform arithmetic on 'bool' values")
                    typ = T.err
                else:
                    assert lhs.type.is_indexable()
                    assert rhs.type.is_indexable()
                    if e.op == "/" or e.op == "%":
                        if rhs.type != T.int:
                            self.err(rhs, "cannot divide or modulo by a "+
                                          "non-constant value")
                        typ = lhs.type
                    elif e.op == "*":
                        if lhs.type == T.int:
                            typ = rhs.type
                        elif rhs.type == T.int:
                            typ = lhs.type
                        else:
                            self.err(e, "cannot multiply two non-constant "+
                                        "indexing/sizing expressions, since "+
                                        "the result would be non-affine")
                            typ = T.err
                    else: # + or -
                        if lhs.type == T.index or rhs.type == T.index:
                            typ = T.index
                        elif lhs.type == T.size or rhs.type == T.size:
                            typ = T.size
                        else:
                            typ = T.int

            else: assert False, f"bad op: '{e.op}'"

            return LoopIR.BinOp(e.op, lhs, rhs, typ, e.srcinfo)

        elif type(e) is UAST.ParRange:
            assert False, ("parser should not place ParRange anywhere "+
                           "outside of a for-loop condition")
        else:
            assert False, "not a LoopIR in check_e"
