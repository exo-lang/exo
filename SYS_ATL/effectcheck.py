from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *
from .LoopIR import UAST, LoopIR, front_ops, bin_ops
from . import shared_types as T
from .LoopIR_effects import Effects as E
from .LoopIR_effects import eff_union, eff_filter, eff_bind

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Helper Functions

# convert from LoopIR.expr to E.expr
def lift_expr(e):
    if type(e) is LoopIR.Read:
        assert len(e.idx) == 0
        return E.Var( e.name, e.type, e.srcinfo )
    elif type(e) is LoopIR.Const:
        return E.Const( e.val, e.type, e.srcinfo )
    elif type(e) is LoopIR.BinOp:
        return E.BinOp( e.op, lift_expr(e.lhs), lift_expr(e.rhs),
                        e.type, e.srcinfo )
    else: assert False, "bad case"

def negate_expr(e):
    assert e.type == T.bool, "can only negate predicates"
    if type(e) is E.Const:
        return E.Const( not e.val, e.type, e.srcinfo )
    elif type(e) is E.BinOp:
        def change_op(op,lhs=e.lhs,rhs=e.rhs):
            return E.BinOp(op, lhs, rhs, e.type, e.srcinfo)

        if e.op == "and":
            return change_op("or", negate_expr(e.lhs), negate_expr(e.rhs))
        elif e.op == "or":
            return change_op("and", negate_expr(e.lhs), negate_expr(e.rhs))
        elif e.op == ">":
            return change_op("<=")
        elif e.op == "<":
            return change_op(">=")
        elif e.op == ">=":
            return change_op("<")
        elif e.op == "<=":
            return change_op(">")
        elif e.op == "==":
            return E.BinOp("or", change_op("<"), change_op(">"),
                           T.bool, e.srcinfo)
    assert False, "bad case"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Annotation of an AST with Effects

class InferEffects():
    def __init__(self, proc):
        self.env = Environment()
        self.errors = []

        args = []
        for a in proc.args:
            self.env[a.name] = a.type
            args.append(LoopIR.fnarg(a.name, a.type, a.effect, mem, a.srcinfo))

        body, eff = self.infer_stmts(proc.body)

        if not proc.name:
            self.err(proc, "expected all procedures to be named")

        self.loopir_proc = LoopIR.proc(name=proc.name or "anon",
                                       args=args,
                                       body=body,
                                       srcinfo=proc.srcinfo)

    def get_loopir(self):
        return self.loopir_proc

    def err(self, node, msg):
        self.errors.append(f"{node.srcinfo}: {msg}")

    def infer_stmts(self, body):
        assert len(body) > 0
        eff   = eff_null()
        stmts = []
        for s in body:
             new_s = self.infer_single_stmt(s)
             stmts.append(new_s)
             eff = eff_union(eff, new_s.effect)
        return (stmts, eff)


    def infer_single_stmt(self, stmt):
        if type(stmt) is LoopIR.Assign:
            buffer = stmt.name
            loc    = lift_expr(stmt.idx)
            effects = E.effect([], E.effset(buffer, loc,
                                            [], None, self.srcinfo),
                               [], self.srcinfo)

            return LoopIR.Assign(stmt.name, stmt.idx, stmt.rhs,
                                 effects, stmt.srcinfo)

        if type(stmt) is LoopIR.Reduce:
            buffer = stmt.name
            loc    = infer_expr(stmt.idx)

            return Effects.effect([],
                                  [],
                                  Effects.effset(buffer, loc, self.srcinfo),
                                  self.srcinfo)
        if type(stmt) is LoopIR.If:
            body_effects = infer_stmts(stmt.body)
            orelse_effects = infer_stmts(stmt.orelse)
            cond = lift_expr(stmt.cond)
            # Helper for negating all Binoops
            body_effects = eff_filter(cond ,body_effects)
            orelse_effects = eff_filter(negate_expr(cond), orelse_effects)
            effects = eff_union(body_effects, orelse_effects)

            # what should we do about orelse?

        if type(stmt) is LoopIR.ForAll:
            bound = stmt.iter
            pred  = (0 <= bound <= stmt.hi)
            body_effect = infer_stmts(stmt.body)
            effects = eff_bind(bound, body_effect, pred=pred)

            return LoopIR.ForAll()

    stmt    = Assign ( sym name, expr* idx, expr rhs )
            | Reduce ( sym name, expr* idx, expr rhs )
            | Pass()
            | If     ( expr cond, stmt* body, stmt* orelse )
            | ForAll ( sym iter, expr hi, stmt* body )
            | Alloc  ( sym name, type type, mem? mem )
            | Free   ( sym name, type type, mem? mem )
            | Call   ( proc f, expr* args )
            attributes( effect? eff, srcinfo srcinfo )

    expr    = Var( sym name )
            | Const( object val )
            | BinOp( binop op, expr lhs, expr rhs )
            attributes( type type, srcinfo srcinfo )
    effset  = ( sym         buffer,
                expr*       loc,    -- e.g. reading at (i+1,j+1)
                sym*        names,
                expr?       pred,
                srcinfo     srcinfo )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Check Bounds and Parallelism semantics for an effect-annotated AST
