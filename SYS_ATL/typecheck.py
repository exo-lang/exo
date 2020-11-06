from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *
from .LoopIR import UAST, LoopIR, front_ops, bin_ops, pred_ops
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


TT = ADT("""
module TypeCheckerTypes {
    type = Size()
        | Idx()
        | Bool()
}""", {})
ADTmemo(T._Types, ['Size', 'Idx', 'Bool'], {})
sizeT = TT.Size()
idxT = TT.Idx()
boolT = TT.Bool()


@extclass(TT.Size)
def __str__(self):
    return "size"


@extclass(TT.Idx)
def __str__(self):
    return "index"


@extclass(TT.Bool)
def __str__(self):
    return "bool"


del __str__


class TypeChecker:
    def __init__(self, proc):
        self.uast_proc = proc
        self.env = Environment()
        self.errors = []

        for sz in proc.sizes:
            self.env[sz] = sizeT

        args = []
        for a in proc.args:
            self.env[a.name] = a.type
            mem = a.mem
            if mem and not is_valid_mem(mem):
                self.err(a, f"invalid memory name '{mem}'")
                mem = None
            args.append(LoopIR.fnarg(a.name, a.type, a.effect, mem, a.srcinfo))

        body = self.check_stmts(proc.body)

        self.loopir_proc = LoopIR.proc(name=proc.name,
                                       sizes=proc.sizes,
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
        final = self.check_single_stmt(body[0])
        for stmt in body[1:]:
            loopstmt = self.check_single_stmt(stmt)
            final = LoopIR.Seq(final, loopstmt, loopstmt.srcinfo)
        return final

    def check_access(self, node, nm, idx, lvalue=False):
        # check indexing
        idx = [self.check_a(i) for i in idx]

        # check compatibility with buffer type
        buftyp = self.env[nm]
        if not T.is_type(buftyp):
            if lvalue:
                self.err(node, f"cannot assign/reduce to '{nm}', " +
                               f"a {buftyp} variable")
            else:
                self.err(node, f"cannot read '{nm}', expected " +
                               f"a scalar or tensor number type, but got " +
                               f"a {buftyp} variable")
        elif buftyp is T.err:
            pass
        elif len(buftyp.shape()) != len(idx):
            self.err(node, f"expected access of variable " +
                           f"'{nm}' of type {buftyp} to have " +
                           f"{len(buftyp.shape())} " +
                           f"indices, but {len(idx)} were found.")
        return idx

    def check_single_stmt(self, stmt):
        if type(stmt) is UAST.Assign or type(stmt) is UAST.Reduce:
            idx = self.check_access(stmt, stmt.name, stmt.idx, lvalue=True)
            rhs = self.check_e(stmt.rhs)

            IRnode = (LoopIR.Assign if type(stmt) is UAST.Assign else
                      LoopIR.Reduce)
            return IRnode(stmt.name, idx, rhs, stmt.srcinfo)

        elif type(stmt) is UAST.Pass:
            return LoopIR.Pass(stmt.srcinfo)

        elif type(stmt) is UAST.If:
            if len(stmt.orelse) > 0:
                self.err(stmt, "else is not supported yet, sorry about that")

            cond = self.check_p(stmt.cond)
            body = self.check_stmts(stmt.body)
            return LoopIR.If(cond, body, stmt.srcinfo)

        elif type(stmt) is UAST.ForAll:
            self.env[stmt.iter] = idxT

            # handle standard ParRanges
            parerr = ("currently only supporting for-loops of the form:\n" +
                      "  for _ in par(0, affine_expression):")

            if (type(stmt.cond) is not UAST.ParRange or
                    type(stmt.cond.lo) is not UAST.Const or
                    stmt.cond.lo.val != 0):
                self.err(stmt.cond, parerr)

            hi = self.check_a(stmt.cond.hi)
            # check that no index variables are used
            def index_free_a(a):
                if type(a) is LoopIR.AVar:
                    return False
                elif type(a) is LoopIR.ASize or type(a) is LoopIR.AConst:
                    return True
                elif type(a) is LoopIR.AScale:
                    return index_free_a(a.rhs)
                elif type(a) is LoopIR.AScaleDiv:
                    return index_free_a(a.lhs)
                else:
                    return index_free_a(a.lhs) and index_free_a(a.rhs)
            if not index_free_a(hi):
                self.err(stmt.cond.hi, "expected upper bound of loop to "+
                                       "only use size variables, not index "+
                                       "variables")

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
            idx = self.check_access(e, e.name, e.idx, lvalue=False)
            return LoopIR.Read(e.name, idx, e.srcinfo)

        elif type(e) is UAST.Const:
            if type(e.val) is float or e.val is int:
                return LoopIR.Const(float(e.val), e.srcinfo)
            else:
                self.err(e, f"literal of unexpected type: {type(e.val)}  " +
                            f"Value: {e.val}")
                return LoopIR.Const(0, e.srcinfo)

        elif type(e) is UAST.USub:
            arg = self.check_e(e.arg)
            return LoopIR.BinOp("*", LoopIR.Const(-1.0, e.srcinfo),
                                     arg, e.srcinfo)

        elif type(e) is UAST.BinOp:
            if e.op not in bin_ops:
                self.err(
                    e, f"cannot perform op '{e.op}' on scalar number values")
                return LoopIR.Const(0, e.srcinfo)

            lhs = self.check_e(e.lhs)
            rhs = self.check_e(e.rhs)
            return LoopIR.BinOp(e.op, lhs, rhs, e.srcinfo)

        elif type(e) is UAST.ParRange:
            assert False, ("parser should not place ParRange anywhere "+
                           "outside of a for-loop condition")
        else:
            assert False, "not a LoopIR in check_e"

    def check_a(self, a):
        if type(a) is UAST.Read:
            if len(a.idx) > 0:
                self.err(a, "cannot access buffers inside affine expressions")

            # check compatibility with buffer type
            nmtyp = self.env[a.name]
            if nmtyp is T.err:
                return LoopIR.AConst(0,a.srcinfo)
            elif nmtyp is idxT:
                return LoopIR.AVar(a.name, a.srcinfo)
            elif nmtyp is sizeT:
                return LoopIR.ASize(a.name, a.srcinfo)
            else:
                self.err(a, f"expected variable '{a.name}' to be an index "+
                            f"or size variable")
                return LoopIR.AConst(0,a.srcinfo)

        elif type(a) is UAST.Const:
            # AConst
            if type(a.val) is int:
                return LoopIR.AConst(int(a.val), a.srcinfo)
            else:
                self.err(a, f"Affine literal of unexpected type: "+
                            f"{type(a.val)}  Value: {a.val}")
                return LoopIR.AConst(0, a.srcinfo)

        elif type(a) is UAST.BinOp:
            lhs = self.check_a(a.lhs)
            rhs = self.check_a(a.rhs)
            # AScale, AScaleDiv, AAdd, or ASub
            if a.op == "+" or a.op == "-":
                IRnode = LoopIR.AAdd if a.op == "+" else LoopIR.ASub
                return IRnode(lhs, rhs, a.srcinfo)
            elif a.op == "*":
                if type(lhs) is LoopIR.AConst:
                    return LoopIR.AScale(lhs.val, rhs, a.srcinfo)
                elif type(rhs) is LoopIR.AConst:
                    return LoopIR.AScale(rhs.val, lhs, a.srcinfo)
                else:
                    self.err(a, "The product of two (non-constant) affine "+
                                "expressions is not affine.")

            elif a.op == "/":
                if type(rhs) is LoopIR.AConst:
                    if rhs.val == 0:
                        self.err(a, "divide-by-zero not allowed")
                    elif rhs.val < 0:
                        self.err(a, "affine-division by negative not allowed")
                    return LoopIR.AScaleDiv(lhs, rhs.val, a.srcinfo)
                else:
                    self.err(a, "Cannot divide an affine expression by "+
                                "anything except a constant.")

            else:
                self.err(a, f"Is not an affine index operation: {a.op}")

            # fall-through for error cases
            return LoopIR.AConst(0,a.srcinfo)

    def check_p(self, p):
        if type(p) is UAST.Const:
            if p.val is bool:
                return LoopIR.BConst(bool(p.val), p.srcinfo)
            else:
                self.err(p, f"expected boolean literal, but got a literal of "+
                            f"type: {type(p.val)}  " +
                            f"literal-value: {p.val}")
                return LoopIR.BConst(False, p.srcinfo)
        elif type(p) is UAST.BinOp:
            if p.op == "and" or p.op == "or":
                lhs = self.check_p(p.lhs)
                rhs = self.check_p(p.rhs)
                IRnode = LoopIR.And if p.op == "and" else LoopIR.Or
                return IRnode(lhs, rhs, p.srcinfo)

            elif p.op in pred_ops:
                lhs = self.check_a(p.lhs)
                rhs = self.check_a(p.rhs)
                return LoopIR.Cmp(p.op, lhs, rhs, p.srcinfo)

            else:
                self.err(p, f"Is not a predicate operation: {p.op}")
                return LoopIR.BConst(False, p.srcinfo)
