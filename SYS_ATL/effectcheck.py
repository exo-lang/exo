from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo
import pysmt
from pysmt import shortcuts as SMT

from .prelude import *
from .LoopIR import UAST, LoopIR, front_ops, bin_ops, LoopIR_Rewrite
from . import shared_types as T
from .LoopIR_effects import Effects as E
from .LoopIR_effects import (eff_union, eff_filter, eff_bind,
                             eff_null, eff_remove_buf, effect_as_str)

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
    else: assert False, "bad case, e is " + str(type(e))

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

# We don't need effect for Const and BinOp
# !! Should we add effects to exprs as well? Or to propagate up to stmt
def read_effect(e):
    if type(e) is LoopIR.Read:
        if e.type.is_numeric():
            loc = [ lift_expr(idx) for idx in e.idx ]
            return E.effect([E.effset(e.name, loc, [], None, e.srcinfo)]
                            ,[] ,[] , e.srcinfo)
        else:
            return eff_null(e.srcinfo)
    elif type(e) is LoopIR.BinOp:
        return eff_union(read_effect(e.lhs), read_effect(e.rhs),
                         srcinfo=e.srcinfo)
    elif type(e) is LoopIR.Const:
        return eff_null(e.srcinfo)
    else:
        assert False, "bad case"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Annotation of an AST with Effects

class InferEffects:
    def __init__(self, proc):
        self.orig_proc  = proc
        body, eff = self.map_stmts(self.orig_proc.body)

        self.proc = LoopIR.proc(name    = self.orig_proc.name,
                                args    = self.orig_proc.args,
                                body    = body,
                                srcinfo = self.orig_proc.srcinfo)

        self.effect = eff

    def get_effect(self):
        return self.effect

    def result(self):
        return self.proc

    def map_stmts(self, body):
        assert len(body) > 0
        eff   = eff_null(body[0].srcinfo)
        stmts = []
        for s in reversed(body):
            new_s = self.map_s(s)
            stmts.append(new_s)
            if type(new_s) is LoopIR.Alloc:
                eff = eff_remove_buf(new_s.name, eff)
            else:
                eff = eff_union(eff, new_s.eff)
        return ([s for s in reversed(stmts)], eff)

    def map_s(self, stmt):
        if type(stmt) is LoopIR.Assign:
            buf = stmt.name
            loc = [ lift_expr(idx) for idx in stmt.idx ]
            rhs_eff = read_effect(stmt.rhs)
            effects = E.effect([], [E.effset(buf, loc,
                                            [], None, stmt.srcinfo)],
                               [], stmt.srcinfo)
            effects = eff_union(rhs_eff, effects)

            return LoopIR.Assign(stmt.name, stmt.idx, stmt.rhs,
                                 effects, stmt.srcinfo)

        elif type(stmt) is LoopIR.Reduce:
            buf = stmt.name
            loc = [ lift_expr(idx) for idx in stmt.idx ]
            rhs_eff = read_effect(stmt.rhs)
            effects = E.effect([], [],
                               [E.effset(buf, loc, [], None, stmt.srcinfo)]
                               , stmt.srcinfo)
            effects = eff_union(rhs_eff, effects)

            return LoopIR.Reduce(stmt.name, stmt.idx, stmt.rhs,
                                 effects, stmt.srcinfo)

        elif type(stmt) is LoopIR.If:
            cond = lift_expr(stmt.cond)
            body, body_effects = self.map_stmts(stmt.body)
            body_effects = eff_filter(cond ,body_effects)
            orelse_effects = eff_null(stmt.srcinfo)
            orelse = stmt.orelse
            if len(stmt.orelse) > 0:
                orelse, orelse_effects = self.map_stmts(stmt.orelse)
                orelse_effects = eff_filter(negate_expr(cond), orelse_effects)
            effects = eff_union(body_effects, orelse_effects)

            return LoopIR.If(stmt.cond, body, orelse,
                             effects, stmt.srcinfo)

        elif type(stmt) is LoopIR.ForAll:
            # pred is: 0 <= bound <= stmt.hi
            bound = E.Var(stmt.iter, T.index, stmt.srcinfo)
            lhs   = E.BinOp("<=", E.Const(0, T.int, stmt.srcinfo)
                                , bound, T.bool, stmt.srcinfo)
            rhs   = E.BinOp("<=", bound, lift_expr(stmt.hi)
                                       , T.bool, stmt.srcinfo)
            pred  = E.BinOp("and", lhs, rhs, T.bool, stmt.srcinfo)

            body, body_effect = self.map_stmts(stmt.body)
            effects = eff_bind(stmt.iter, body_effect, pred=pred)

            return LoopIR.ForAll(stmt.iter, stmt.hi, body,
                                 effects, stmt.srcinfo)

        elif type(stmt) is LoopIR.Call:
            # Do we need to check arguments types here?
            # ^ Maybe in CheckEffects
            proc_eff = InferEffects(stmt.f).get_effect()

            return LoopIR.Call(stmt.f, stmt.args,
                               proc_eff, stmt.srcinfo)

        elif type(stmt) is LoopIR.Pass:
            return LoopIR.Pass(eff_null(stmt.srcinfo), stmt.srcinfo)
        elif type(stmt) is LoopIR.Alloc:
            return LoopIR.Alloc(stmt.name, stmt.type, stmt.mem,
                                eff_null(stmt.srcinfo), stmt.srcinfo)

        else:
            assert False, "Invalid statement"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Check Bounds and Parallelism semantics for an effect-annotated AST

#
#   What is bounds checking?
#
#       (x : T)  ;  s
#       s has effect e
#       Also, we may assume certain things about the context
#       Let us assume some CTXT_PRED
#
#       Then, x is memory safe iff. all accesses to x in s are "in-bounds"
#       There is a relationship between buffer types (i.e. shapes)
#       and effect-types, which says that the effect is "in-bounds"
#       with respect to the buffer type (and you need to know the buffer name)
#       
#       What we really want to check is that
#           CTXT_PRED ==> IN_BOUNDS( x, T, e )
#
#       IN_BOUNDS( x, T, e ) =
#           AND es in e: IN_BOUNDS( x, T, es )
#       IN_BOUNDS( x, T, (y, ...) ) = TRUE
#       IN_BOUNDS( x, T, (x, (i,j), nms, pred ) ) =
#           forall nms in Z, pred ==> in_bounds(T, (i,j))
#       
#
#   (assert CTXT_PRED_1)
#   (assert CTXT_PRED_2)
#   (valid IN_BOUNDS( x, T, e ) )
#   (valid IN_BOUNDS( y, T2, e2 ) )
#
#
#   for i in par(0,n):
#       ...
#       y : R[n]
#
#       y[i] = 32
#       
#       for j in par(0,n):
#           if i+j < n:
#               y[i+j] = 32
#
#   s has effect WRITE { y : (i+j) for j in int if 0 <= j < n and i+j < n }
#
#   CTXT_PRED is 0 <= i < n


#
#   What is parallelism checking?
#
#       In general the situation is that we have a parallel for loop
#      
#       for i in par(0,n): s
#       
#       s has effect e
#
#       We want to check that 
#           forall i0,i1: 0 <= i0 < i1 < n ==> COMMUTES( [i |-> i0]e,
#                                                        [i |-> i1]e )
#
#       R, W, +
#
#       R commutes with R
#       + commutes with +
#       any two other effects do not commute
#
#       COMMUTES( (r0, w0, p0), (r1, w1, p1) ) =
#           AND ( NOT_CONFLICTS( r0, w1 )
#                 NOT_CONFLICTS( r0, p1 )
#                 NOT_CONFLICTS( w0, r1 )
#                 NOT_CONFLICTS( w0, w1 )
#                 NOT_CONFLICTS( w0, p1 )
#                 NOT_CONFLICTS( p0, r1 )
#                 NOT_CONFLICTS( p0, w1 ) )
#
#       NOT_CONFLICTS( (x,...), (y,...) ) = TRUE
#       NOT_CONFLICTS( (x, loc0, nms0, pred0), (x, loc1, nms1, pred1) ) =
#           forall nms0, nms1: pred0 AND pred1 ==> loc0 != loc1
#
#       Let's try to re-develop these ideas in the setting where
#       we assume that the two effects are identical except for
#       our substitution
#
#       COMMUTES( i, n, e ) =
#           forall i0,i1: 0 <= i0 < i1 < n ==>
#                       COMMUTES( [i |-> i0]e, [i |-> i1]e )
#           
#       COMMUTES( i, n, (r, w, p) ) =
#           AND( NOT_CONFLICTS(i, n, r, w)
#                NOT_CONFLICTS(i, n, r, p)
#                NOT_CONFLICTS(i, n, w, w)
#                NOT_CONFLICTS(i, n, w, p) )
#
#       NOT_CONFLICTS( i, n, (x,...), (y,...) ) = TRUE
#       NOT_CONFLICTS( i, n, (x, loc0, nms0, pred0),
#                            (x, loc1, nms1, pred1) ) =
#           forall i0,i1: 0 <= i0 < i1 < n ==>
#               forall nms0, nms1:
#                   [sub i0,nms0]pred0 AND [sub i1,nms1]pred1 ==>
#                   [sub i0,nms0]loc0 != [sub i1,nms1]loc1
#           
#       cond ==> (x AND y)   ===   (cond ==> x) AND (cond ==> y)
#   
#           AND ( forall _: _ ==> NOT_CONFLICTS( r0, w1 )
#                 forall _: _ ==> NOT_CONFLICTS( r0, p1 )
#                 forall _: _ ==> NOT_CONFLICTS( w0, r1 )
#                 forall _: _ ==> NOT_CONFLICTS( w0, w1 )
#                 forall _: _ ==> NOT_CONFLICTS( w0, p1 )
#                 forall _: _ ==> NOT_CONFLICTS( p0, r1 )
#                 forall _: _ ==> NOT_CONFLICTS( p0, w1 ) )
#
#

# Check if Alloc sizes and function arg sizes are actually larger than bounds
class CheckEffects:
    def __init__(self, proc):
        self.orig_proc  = proc

        # Map sym to z3 variable
        self.env = Environment()

        self.effects = []
        self.map_stmts(self.orig_proc.body)

    def result(self):
        return self.orig_proc

    def sym_to_smt(sym):
        if env[sym] is None:
            env[sym] = SMT.Symbol(str(sym), SMT.INT)
        return env[sym]

    def expr_to_smt(expr):
        if type(expr) is E.Const:
            assert type(expr.val) is int, "Effect must be int"
            return SMT.INT(expr.val)
        elif type(expr) is E.Var:
            return sym_to_smt(expr.name)
        elif type(expr) is E.BinOp:
            lhs = expr_to_smt(expr.lhs)
            rhs = expr_to_smt(expr.rhs)
            if expr.op == "+":
                return lhs + rhs
            elif expr.op == "-":
                return lhs - rhs
            elif expr.op == "*":
                return lhs * rhs
            elif expr.op == "/":
                return lhs / rhs
            elif expr.op == "%":
                return lhs % rhs
            elif expr.op == "<":
                return SMT.LT(lhs, rhs)
            elif expr.op == ">":
                return SMT.GT(lhs, rhs)
            elif expr.op == "<=":
                return SMT.LE(lhs, rhs)
            elif expr.op == ">=":
                return SMT.GE(lhs, rhs)
            elif expr.op == "==":
                return SMT.Equals(lhs, rhs)
            elif expr.op == "and":
                return SMT.And(lhs, rhs)
            elif expr.op == "or":
                return SMT.Or(lhs, rhs)

    def shape_loc_smt(shape, loc):
        assert len(shape) == len(loc), "buffer loc should be same as buffer shape"
        smt = SMT.Bool(True)
        for i in len(shape):
            # 1 <= loc[i] <= shape[i]
            loc_e = expr_to_smt(loc[i])
            lhs = SMT.LE(Int(1), loc_e)
            rhs = SMT.LE(loc_e, expr_to_smt(shape[i]))
            smt = SMT.And(smt, SMT.And(lhs, rhs))

        return smt

    def in_bounds(sym, shape, eff):
        if type(eff) is E.effect:
            f = SMT.And(SMT.And(in_bounds(sym, shape, eff.reads),
                        in_bounds(sym, shape, eff.writes)),
                        in_bounds(sym, shape, eff.reduces))
            # ??
            return self.solver.verify(f)

        elif type(eff) is E.effset:
            if sym != eff.buffer:
                return SMT.Bool(True)
            else:
#       IN_BOUNDS( x, T, (x, (i,j), nms, pred ) ) =
#           forall nms in Z, pred ==> in_bounds(T, (i,j))
                nms_e = [sym_to_z3(e) for e in eff.nms]
                forall_e = SMT.ForAll(nms_e,
                            SMT.Implies(expr_to_smt(eff.pred),
                                shape_loc_smt(shape, eff.loc)))
                return forall_e

    def map_stmts(self, body):
        assert len(body) > 0
        self.context.append(eff_null(body[0].srcinfo))

        for i in len(body):
            stmt = body[i]
            self.context = eff_union(self.context, stmt.eff)

            if type(stmt) is LoopIR.ForAll or type(stmt) is LoopIR.If:
                body_eff = self.map_stmts(stmt.body)
                self.context = eff_union(self.context, body_eff)
            elif type(stmt) is LoopIR.Alloc:
                s_eff = self.map_stmts(body[:i+1])
                in_bounds(stmt.name, stmt.type.shape(), s_eff)

        return self.context.pop()
