from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *
from .LoopIR  import UAST, LoopIR, front_ops
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

class TypeChecker:
  def __init__(self, proc):
    self.uast_proc  = proc
    self.env        = Environment()
    self.errors     = []

    for sz in proc.sizes:
        self.env[sz] = sizeT

    args = []
    for a in proc.args:
        self.env[a.name] = type
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
    pass

  def check_e(self, e):
    if type(e) is UAST.Read:
        idx_T   = [ self.check_e(i) for i in e.idx ]
        idx     = [ i[0] for i in idx_T ]
        retT    = T.num
        for i,t in idx_T:
            if t == T.err:
                continue
            elif i.type != idxT:
                self.err(i, "expected an index expression")

        # lookup name ...
            ( sym name, expr* idx )
    pass
