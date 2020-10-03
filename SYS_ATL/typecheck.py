from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *
from .LoopIR  import UAST, LoopIR, front_ops, bin_ops, pred_ops
from . import shared_types as T

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

def unzip(xys):
    xs = [ x for x,y in xys ]
    ys = [ y for x,y in xys ]
    return xs, ys


TT = ADT("""
module TypeCheckerTypes {
    type = Size()
        | Idx()
        | Bool()
}""",{})
ADTmemo(_Types,['Size','Idx','Bool'],{})
sizeT = TT.Size()
idxT  = TT.Idx()
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
        self.uast_proc  = proc
        self.env        = Environment()
        self.errors     = []

        for sz in proc.sizes:
            self.env[sz] = sizeT

        args = []
        for a in proc.args:
            self.env[a.name] = a.type
            args.append( LoopIR.fnarg(a.name, a.type, a.effect, a.srcinfo) )


        body = self.check_stmts(proc.body)

        self.loopir_proc = LoopIR.proc( name        = proc.name,
                                        sizes       = proc.sizes,
                                        args        = args,
                                        body        = body,
                                        srcinfo     = proc.srcinfo )

        # do error checking here
        if len(self.errors) > 0:
            raise TypeError("Errors occurred during typechecking:\n"+
                            "\n".join(self.errors))

    def get_loopir(self):
        return self.loopir_proc

    def err(self, node, msg):
        self.errors.append(f"{node.srcinfo}: {msg}")

    def check_stmts(self, body):
        assert len(body) > 0
        final   = self.check_single_stmt(body[0])
        for stmt in body[1:]:
            loopstmt = self.check_single_stmt(stmt)
            final = LoopIR.Seq(final, loopstmt, loopstmt.srcinfo)
        return final

    def check_access(self, node, nm, idx, lvalue=False):
        # check indexing
        idx     = [ self.check_a(i) for i in idx ]

        # check compatibility with buffer type
        buftyp      = self.env[nm]
        if not T.is_type(buftyp):
            if lvalue:
                self.err(node, f"cannot assign/reduce to '{nm}', "+
                               f"a {buftyp} variable")
            else:
                self.err(node, f"cannot read '{nm}', expected "+
                               f"a scalar or tensor number type, but got "+
                               f"a {buftyp} variable")
        elif buftyp is T.err:
            pass
        elif len(buftyp.shape()) != len(idx):
            self.err(node, f"expected access of variable "+
                           f"'{nm}' of type {buftyp} to have "+
                           f"{len(buftyp.shape())} "+
                           f"indices, but {len(idx)} were found.")
        return idx

    def check_single_stmt(self,stmt):
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

            cond, ctyp  = self.check_p(stmt.cond)
            body        = self.check_stmts(stmt.body, stmt.srcinfo)
            return LoopIR.If(cond, body, stmt.srcinfo)

        elif type(stmt) is UAST.ForAll:
            self.env[stmt.iter] = idxT

            # handle standard ParRanges
            parerr = ("currently only supporting for-loops of the form:\n"+
                      "  for _ in par(0,size_var):")
            if ( type(stmt.cond) is not UAST.ParRange or
                 type(stmt.cond.lo) is not UAST.Const or
                 stmt.cond.lo.val != 0 or
                 type(stmt.cond.hi) is not UAST.Read or
                 len(stmt.cond.hi.idx) > 0
               ):
                self.err(stmt.cond, parerr)
            size_var = stmt.cond.hi.name
            size_typ = self.env[size_var]
            if size_typ is not T.err and size_typ is not sizeT:
                self.err(stmt.cond.hi, f"expected upper bound of loop "+
                                       f"'{size_var}' to be a size variable")

            body = self.check_stmts(stmt.body)
            return LoopIR.ForAll(stmt.iter, size_var, body, stmt.srcinfo)

        elif type(stmt) is UAST.Alloc:
            self.env[stmt.name] = stmt.type
            return LoopIR.Alloc(stmt.name, stmt.type, stmt.srcinfo)

        else: assert False, "not a loopir in check_stmts"

  def check_e(self, e):
    if type(e) is UAST.Read:
        idx = self.check_access(stmt, stmt.name, stmt.idx, lvalue=False)
        return IR.Read(e.name, idx, e.srcinfo)

    elif type(e) is UAST.Const:
        if type(e.val) is float or e.val is int:
            return LoopIR.Const(float(e.val), e.srcinfo)
        else:
            self.err(e, f"literal of unexpected type: {type(e.val)}  "+
                        f"Value: {e.val}")
            return LoopIR.Const(0, e.srcinfo)

    elif type(e) is UAST.USub:
        arg = self.check_e(e.arg)
        return LoopIR.BinOp( LoopIR.Const(-1.0, e.srcinfo), arg, e.srcinfo )

    elif type(e) is UAST.BinOp:
        if e.op not in bin_ops:
            self.err(e, f"cannot perform op '{e.op}' on scalar number values")
            return LoopIR.Const(0, e.srcinfo)

        lhs = self.check_e(e.lhs)
        rhs = self.check_e(e.rhs)
        return LoopIR.BinOp(op, lhs, rhs, e.srcinfo)

    elif type(e) is UAST.ParRange:
        assert False, ("parser should not place ParRange anywhere outside "+
                       "of a for-loop condition")
    else: assert False, "not a LoopIR in check_e"

  def check_a(self, e):
          elif type(e) is UAST.Const:
              if type(e.val) is float or e.val is int:
                  return LoopIR.Const(e.val, e.srcinfo)
              else:
                  self.err(e, f"literal of unexpected type: {type(e.val)}  "+
                              f"Value: {e.val}")
                  return LoopIR.Const(0, e.srcinfo)

      if expect_idx:
          val = e.val
          if type(e.val) is not int:
              self.err(e, "expected an integer")
              val = 0
          return LoopIR.AConst(val, e.srcinfo), idxT
      else:

      pass

  def check_p(self, e):
                if type(e.val) is bool:
                    return LoopIR.BConst(e.val, e.srcinfo), boolT
      pass
