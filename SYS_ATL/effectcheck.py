from .prelude import *
from .LoopIR import UAST, LoopIR, front_ops, bin_ops, LoopIR_Rewrite
from .LoopIR import T
from .LoopIR_effects import Effects as E
from .LoopIR_effects import (eff_union, eff_filter, eff_bind,
                             eff_null, eff_remove_buf, effect_as_str)

from collections import ChainMap

import pysmt
from pysmt import shortcuts as SMT

def _get_smt_solver():
    factory = pysmt.factory.Factory(pysmt.shortcuts.get_env())
    slvs    = factory.all_solvers()
    if len(slvs) == 0: raise OSError("Could not find any SMT solvers")
    return pysmt.shortcuts.Solver(name=next(iter(slvs)))

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

def expr_subst(env, e):
    """ perform the substitutions specified by env in expression e """
    if type(e) is E.Const:
        return e
    elif type(e) is E.Var:
        if e.name in env:
            return E.Var(env[e.name], e.type, e.srcinfo)
        else:
            return e
    elif type(e) is E.BinOp:
        return E.BinOp(e.op, expr_subst(env, e.lhs), expr_subst(env, e.rhs),
                       e.type, e.srcinfo)
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
                                preds   = self.orig_proc.preds,
                                body    = body,
                                instr   = self.orig_proc.instr,
                                eff     = eff,
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

            return LoopIR.Assign(stmt.name, stmt.type, stmt.cast,
                                 stmt.idx, stmt.rhs,
                                 effects, stmt.srcinfo)

        elif type(stmt) is LoopIR.Reduce:
            buf = stmt.name
            loc = [ lift_expr(idx) for idx in stmt.idx ]
            rhs_eff = read_effect(stmt.rhs)
            effects = E.effect([], [],
                               [E.effset(buf, loc, [], None, stmt.srcinfo)]
                               , stmt.srcinfo)
            effects = eff_union(rhs_eff, effects)

            return LoopIR.Reduce(stmt.name, stmt.type, stmt.cast,
                                 stmt.idx, stmt.rhs,
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
            rhs   = E.BinOp("<", bound, lift_expr(stmt.hi)
                                       , T.bool, stmt.srcinfo)
            pred  = E.BinOp("and", lhs, rhs, T.bool, stmt.srcinfo)

            body, body_effect = self.map_stmts(stmt.body)
            effects = eff_bind(stmt.iter, body_effect, pred=pred)

            return LoopIR.ForAll(stmt.iter, stmt.hi, body,
                                 effects, stmt.srcinfo)

        elif type(stmt) is LoopIR.Call:
            assert stmt.f.eff is not None
            # build up a substitution dictionary....
            # sig is a LoopIR.fnarg, arg is a LoopIR.expr
            subst = dict()
            for sig,arg in zip(stmt.f.args, stmt.args):
                if sig.type.is_numeric():
                    assert type(arg) is LoopIR.Read
                    subst[sig.name] = arg.name
                elif sig.type.is_indexable():
                    # in this case we have a LoopIR expression...
                    subst[sig.name] = lift_expr(arg)
                else: assert False, "bad case"

            eff = stmt.f.eff
            # TODO: Add read effect to call args
            #for sig in stmt.f.args:
            #    eff = eff_union(read_effect(arg), eff)
            eff = eff.subst(subst)

            return LoopIR.Call(stmt.f, stmt.args,
                               eff, stmt.srcinfo)

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
        self.env        = ChainMap()
        #self.context    = E.Const(True, T.bool, proc.srcinfo)
        self.errors     = []

        self.solver     = _get_smt_solver()

        self.push()
        # Add assersions
        for arg in proc.args:
            if type(arg.type) is T.Size:
                pos_sz = SMT.LT(SMT.Int(0), self.sym_to_smt(arg.name))
                self.solver.add_assertion(pos_sz)
        for p in proc.preds:
            self.solver.add_assertion(self.expr_to_smt(lift_expr(p)))

        body_eff = self.map_stmts(self.orig_proc.body)

        for arg in proc.args:
            if arg.type.is_numeric():
                shape = [ lift_expr(s) for s in arg.type.shape() ]
                # check that all sizes are positive
                for s in shape:
                    self.check_pos_size(s)
                # check the bounds
                self.check_bounds(arg.name, shape, body_eff)
        self.pop()

        # do error checking here
        if len(self.errors) > 0:
            raise TypeError("Errors occurred during effect checking:\n" +
                            "\n".join(self.errors))

    def result(self):
        return self.orig_proc

    def push(self):
        self.solver.push()
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents
        self.solver.pop()

    def err(self, node, msg):
        self.errors.append(f"{node.srcinfo}: {msg}")

    # TODO: Add allow_allocation arg here, to check if we're introducing new
    # symbols from the right place.
    def sym_to_smt(self, sym):
        if sym not in self.env:
            self.env[sym] = SMT.Symbol(repr(sym), SMT.INT)
        return self.env[sym]

    def expr_to_smt(self, expr):
        assert isinstance(expr, E.expr), "expected Effects.expr"
        if type(expr) is E.Const:
            if expr.type == T.bool:
                return SMT.Bool(expr.val)
            elif expr.type.is_indexable():
                return SMT.Int(expr.val)
            else: assert False, "unrecognized const type: {type(expr.val)}"
        elif type(expr) is E.Var:
            return self.sym_to_smt(expr.name)
        elif type(expr) is E.BinOp:
            lhs = self.expr_to_smt(expr.lhs)
            rhs = self.expr_to_smt(expr.rhs)
            if expr.op == "+":
                return SMT.Plus(lhs, rhs)
            elif expr.op == "-":
                return SMT.Minus(lhs, rhs)
            elif expr.op == "*":
                return SMT.Times(lhs, rhs)
            elif expr.op == "/":
                assert type(expr.rhs) is E.Const
                assert expr.rhs.val > 0
                # Introduce new Sym (z in formula below)
                div_tmp = self.sym_to_smt(Sym("div_tmp"))
                # rhs*z <= lhs < rhs*(z+1)
                rhs_eq  = SMT.LE(SMT.Times(rhs, div_tmp), lhs)
                lhs_eq  = SMT.LT(lhs,
                        SMT.Times(rhs, SMT.Plus(div_tmp, SMT.Int(1))))
                self.solver.add_assertion(SMT.And(rhs_eq, lhs_eq))
                return div_tmp
            elif expr.op == "%":
                # x % y is defined as x - floor(x/y)
                # Similar to Div, but return lhs - rhs*z instead
                mod_tmp = self.sym_to_smt(Sym("mod_tmp"))
                # rhs*z <= lhs < rhs*(z+1)
                raise NotImplementedError("TODO: support modulo")
                # lhs - (rhs * mod_tmp)
                rhs_eq  = SMT.LE(SMT.Times(rhs, mod_tmp), lhs)
                lhs_eq  = SMT.LT(lhs,
                        SMT.Times(rhs, SMT.Plus(mod_tmp, SMT.Int(1))))
                self.solver.add_assertion(SMT.And(rhs_eq, lhs_eq))
                return SMT.Minus(lhs, SMT.Times(rhs, mod_tmp))
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
        else: assert False, "bad case"

    def check_in_bounds(self, sym, shape, eff, eff_str):
        assert type(eff) is E.effset, "effset should be passed to in_bounds"

        if sym == eff.buffer:
#       IN_BOUNDS( x, T, (x, (i,j), nms, pred ) ) =
#           forall nms in Z, pred ==> in_bounds(T, (i,j))

            self.solver.push()
            if eff.pred is not None:
                self.solver.add_assertion(self.expr_to_smt(eff.pred))
            in_bds = SMT.Bool(True)

            assert len(eff.loc) == len(shape)
            for e, hi in zip(eff.loc, shape):
                # 1 <= loc[i] < shape[i]
                e   = self.expr_to_smt(e)
                lhs = SMT.LE(SMT.Int(0), e)
                rhs = SMT.LT(e, self.expr_to_smt(hi))
                in_bds = SMT.And(in_bds, SMT.And(lhs, rhs))

            # TODO: Extract counter example from SMT solver
            if not self.solver.is_valid(in_bds):
                self.err(eff, f"{sym} is {eff_str} out-of-bounds")

            self.solver.pop()

    def check_bounds(self, sym, shape, eff):
        effs = [(eff.reads, "read"), (eff.writes, "write"),
                (eff.reduces, "reduce")]

        for (es,y) in effs:
            for e in es:
                self.check_in_bounds(sym, shape, e, y)

#       NOT_CONFLICTS( i, n, (x,...), (y,...) ) = TRUE
#       NOT_CONFLICTS( i, n, (x, loc0, nms0, pred0),
#                            (x, loc1, nms1, pred1) ) =
#           forall i0,i1: 0 <= i0 < i1 < n ==>
#               forall nms0, nms1:
#                   [sub i0,nms0]pred0 AND [sub i1,nms1]pred1 ==>
#                   [sub i0,nms0]loc0 != [sub i1,nms1]loc1
    def not_conflicts(self, iter, hi, e1, e2):
        if e1.buffer == e2.buffer:
            self.solver.push()
            # determine name substitutions
            iter1   = iter.copy()
            iter2   = iter.copy()
            iter1_smt = self.sym_to_smt(iter1)
            iter2_smt = self.sym_to_smt(iter2)
            iter_pred = SMT.And(SMT.And(SMT.LE(SMT.Int(0), iter1_smt),
                                SMT.LT(iter1_smt, iter2_smt)),
                                SMT.LT(iter2_smt, self.expr_to_smt(hi)))
            self.solver.add_assertion(iter_pred)

            sub1    = { nm : nm.copy() for nm in e1.names }
            sub1[iter] = iter1
            sub2    = { nm : nm.copy() for nm in e2.names }
            sub2[iter] = iter2
            if e1.pred is not None:
                pred1   = expr_subst(sub1, e1.pred)
                self.solver.add_assertion(self.expr_to_smt(pred1))
            if e2.pred is not None:
                pred2   = expr_subst(sub2, e2.pred)
                self.solver.add_assertion(self.expr_to_smt(pred2))

            loc1    = [ self.expr_to_smt(expr_subst(sub1, i)) for i in e1.loc ]
            loc2    = [ self.expr_to_smt(expr_subst(sub2, i)) for i in e2.loc ]
            loc_neq = SMT.Bool(False)
            for i1, i2 in zip(loc1,loc2):
                loc_neq = SMT.Or(loc_neq, SMT.NotEquals(i1, i2))

            if not self.solver.is_valid(loc_neq):
                self.err(e1, f"data race conflict with statement on "+
                             f"{e2.srcinfo} while accessing {e1.buffer} "+
                             f"in loop over {iter}.")

            self.solver.pop()


#       COMMUTES( i, n, (r, w, p) ) =
    def check_commutes(self, iter, hi, eff):

#           AND( NOT_CONFLICTS(i, n, r, w)
        for r in eff.reads:
            for w in eff.writes:
                self.not_conflicts(iter, hi, r, w)
#                NOT_CONFLICTS(i, n, r, p)
        for r in eff.reads:
            for p in eff.reduces:
                self.not_conflicts(iter, hi, r, p)
#                NOT_CONFLICTS(i, n, w, w)
        for w1 in eff.writes:
            for w2 in eff.writes:
                self.not_conflicts(iter, hi, w1, w2)
#                NOT_CONFLICTS(i, n, w, p) )
        for w in eff.writes:
            for p in eff.reduces:
                self.not_conflicts(iter, hi, w, p)

        return

    def check_pos_size(self, expr):
        e_pos = SMT.LT( SMT.Int(0), self.expr_to_smt(expr) )
        if not self.solver.is_valid(e_pos):
            self.err(expr, "expected expression to always be positive")

    def check_call_shape_eqv(self, argshp, sigshp, node):
        assert len(argshp) == len(sigshp)
        eqv_dim = SMT.Bool(True)
        for a,s in zip(argshp, sigshp):
            eqv_dim = SMT.And(eqv_dim,
                              SMT.Equals(self.expr_to_smt(a),
                                         self.expr_to_smt(s)))
        if not self.solver.is_valid(eqv_dim):
            self.err(node, "type-shape of calling argument may not equal "+
                           "the required type-shape")

    def map_stmts(self, body):
        """ Returns an effect for the argument `body`
            And also checks bounds/parallelism for any
            allocations/loops within `body`
        """
        assert len(body) > 0
        body_eff = eff_null(body[-1].srcinfo)

        for stmt in reversed(body):
            if type(stmt) is LoopIR.ForAll:
                self.push()
                def bd_pred(x,hi,srcinfo):
                    zero    = E.Const(0, T.int, srcinfo)
                    x       = E.Var(x, T.int, srcinfo)
                    hi      = lift_expr(hi)
                    return E.BinOp("and",
                                E.BinOp("<=", zero, x, T.bool, srcinfo),
                                E.BinOp("<",  x,   hi, T.bool, srcinfo),
                            T.bool, srcinfo)

                self.solver.add_assertion(
                    self.expr_to_smt(bd_pred(stmt.iter, stmt.hi,
                                             stmt.srcinfo)))

                sub_body_eff = self.map_stmts(stmt.body)
                self.pop()

                # Parallelism checking here
                self.check_commutes(stmt.iter, lift_expr(stmt.hi), sub_body_eff)

                body_eff = eff_union(body_eff, stmt.eff)

            if type(stmt) is LoopIR.If:
                # first, do the if-branch
                self.push()
                self.solver.add_assertion(self.expr_to_smt(
                                                lift_expr(stmt.cond)))
                self.map_stmts(stmt.body)
                self.pop()

                # then the else-branch
                if len(stmt.orelse) > 0:
                    self.push()
                    neg_cond = negate_expr( lift_expr(stmt.cond) )
                    self.solver.add_assertion(self.expr_to_smt(neg_cond))
                    self.map_stmts(stmt.orelse)
                    self.pop()

                body_eff = eff_union(body_eff, stmt.eff)

            elif type(stmt) is LoopIR.Alloc:
                shape = [ lift_expr(s) for s in stmt.type.shape() ]
                # check that all sizes are positive
                for s in shape:
                    self.check_pos_size(s)
                # check that all accesses are in bounds
                self.check_bounds(stmt.name, shape, body_eff)
                body_eff = eff_remove_buf(stmt.name, body_eff)

            elif type(stmt) is LoopIR.Call:
                subst = dict()
                for sig,arg in zip(stmt.f.args, stmt.args):
                    if sig.type.is_numeric():
                        # need to check that the argument shape
                        # has all positive dimensions
                        arg_shape = [ lift_expr(s) for s in arg.type.shape() ]
                        for e in arg_shape:
                            self.check_pos_size(e)
                        # also, need to check that the argument shape
                        # is exactly the shape specified in the signature
                        sig_shape = [ lift_expr(s) for s in sig.type.shape() ]
                        sig_shape = [ s.subst(subst) for s in sig_shape ]
                        self.check_call_shape_eqv(arg_shape, sig_shape, arg)
                    elif sig.type.is_indexable():
                        # in this case we have a LoopIR expression...
                        e_arg           = lift_expr(arg)
                        subst[sig.name] = e_arg
                        if sig.type == T.size:
                            self.check_pos_size(e_arg)

                    else: assert False, "bad case"

                for p in stmt.f.preds:
                    pred = lift_expr(p).subst(subst)
                    # Check that asserts are correct
                    if not self.solver.is_valid(self.expr_to_smt(pred)):
                        self.err(stmt, f"Could not verify assertion in "+
                                       f"{stmt.f.name} at {p.srcinfo}")

                body_eff = eff_union(body_eff, stmt.eff)

            else:
                body_eff = eff_union(body_eff, stmt.eff)


        return body_eff # Returns union of all effects
