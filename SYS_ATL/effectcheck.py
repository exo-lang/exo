from collections import ChainMap

import pysmt
from pysmt import shortcuts as SMT

from .LoopIR import LoopIR
from .LoopIR import T
from .LoopIR import lift_to_eff_expr as lift_expr
from .LoopIR_dataflow import LoopIR_Dependencies
from .LoopIR_effects import Effects as E
from .LoopIR_effects import (eff_null, eff_read, eff_write, eff_reduce,
                             eff_config_read, eff_config_write,
                             eff_concat, eff_union, eff_remove_buf,
                             eff_filter, eff_bind)
from .prelude import *


def _get_smt_solver():
    factory = pysmt.factory.Factory(pysmt.shortcuts.get_env())
    slvs    = factory.all_solvers()
    if len(slvs) == 0: raise OSError("Could not find any SMT solvers")
    return pysmt.shortcuts.Solver(name=next(iter(slvs)))

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Helper Functions

def loopir_subst(e, subst):
    if isinstance(e, LoopIR.Read):
        assert not e.type.is_numeric()
        return subst[e.name] if e.name in subst else e
    elif isinstance(e, (LoopIR.Const, LoopIR.ReadConfig)):
        return e
    elif isinstance(e, LoopIR.USub):
        return LoopIR.USub(loopir_subst(e.arg, subst),
                           e.type, e.srcinfo)
    elif isinstance(e, LoopIR.BinOp):
        return LoopIR.BinOp(e.op,
                            loopir_subst(e.lhs, subst),
                            loopir_subst(e.rhs, subst),
                            e.type, e.srcinfo)
    elif isinstance(e, LoopIR.StrideExpr):
        if e.name not in subst:
            return e
        lookup = subst[e.name]
        if isinstance(lookup, LoopIR.Read):
            assert len(lookup.idx) == 0
            return LoopIR.StrideExpr(lookup.name, e.dim,
                                     e.type, e.srcinfo)
        elif isinstance(lookup, LoopIR.WindowExpr):
            windowed_orig_dims = [d for d, w in enumerate(lookup.idx)
                                  if isinstance(w, LoopIR.Interval)]
            dim = windowed_orig_dims[e.dim]
            return LoopIR.StrideExpr(lookup.name, dim,
                                     e.type, e.srcinfo)

        else: assert False, f"bad case: {type(lookup)}"

    else: assert False, f"bad case: {type(e)}"



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Annotation of an AST with Effects

class InferEffects:
    def __init__(self, proc):
        self.orig_proc  = proc

        self._types     = {}
        for a in proc.args:
            self._types[a.name] = a.type
        self.rec_stmts_types(self.orig_proc.body)

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

    def rec_stmts_types(self, body):
        assert len(body) > 0
        for s in body:
            self.rec_s_types(s)

    def rec_s_types(self, stmt):
        if isinstance(stmt, LoopIR.If):
            self.rec_stmts_types(stmt.body)
            if len(stmt.orelse) > 0:
                self.rec_stmts_types(stmt.orelse)
        elif isinstance(stmt, (LoopIR.ForAll, LoopIR.Seq)):
            self.rec_stmts_types(stmt.body)
        elif isinstance(stmt, LoopIR.Alloc):
            self._types[stmt.name] = stmt.type
        elif isinstance(stmt, LoopIR.WindowStmt):
            self._types[stmt.lhs] = stmt.rhs.type
        else:
            pass

    def map_stmts(self, body):
        assert len(body) > 0
        eff   = eff_null(body[0].srcinfo)
        stmts = []
        for s in reversed(body):
            new_s = self.map_s(s)
            stmts.append(new_s)
            if isinstance(new_s, LoopIR.Alloc):
                eff = eff_remove_buf(new_s.name, eff)
            else:
                eff = eff_concat(new_s.eff, eff)
        return (list(reversed(stmts)), eff)

    def map_s(self, stmt):
        if isinstance(stmt, (LoopIR.Assign, LoopIR.Reduce)):
            styp = type(stmt)
            buf = stmt.name
            loc = [lift_expr(idx) for idx in stmt.idx]
            rhs_eff = self.eff_e(stmt.rhs)
            if styp is LoopIR.Assign:
                effects = eff_write(buf, loc, stmt.srcinfo)
            else:  # Reduce
                effects = eff_reduce(buf, loc, stmt.srcinfo)

            effects = eff_concat(rhs_eff, effects)

            return styp(stmt.name, stmt.type, stmt.cast,
                        stmt.idx, stmt.rhs,
                        effects, stmt.srcinfo)

        elif isinstance(stmt, LoopIR.WriteConfig):
            rhs_eff = self.eff_e(stmt.rhs)
            if stmt.rhs.type.is_numeric():
                rhs = E.Var(Sym("opaque_rhs"), stmt.rhs.type, stmt.rhs.srcinfo)
            else:
                rhs = lift_expr(stmt.rhs)
            cw_eff  = eff_config_write(stmt.config, stmt.field,
                                       rhs, stmt.srcinfo)
            eff     = eff_concat(rhs_eff, cw_eff)
            return LoopIR.WriteConfig(stmt.config, stmt.field, stmt.rhs,
                                      eff, stmt.srcinfo)

        elif isinstance(stmt, LoopIR.If):
            cond = lift_expr(stmt.cond)
            body, body_effects = self.map_stmts(stmt.body)
            body_effects = eff_filter(cond, body_effects)
            orelse_effects = eff_null(stmt.srcinfo)
            orelse = stmt.orelse
            if len(stmt.orelse) > 0:
                orelse, orelse_effects = self.map_stmts(stmt.orelse)
                orelse_effects = eff_filter(cond.negate(), orelse_effects)
            # union is appropriate because guards on effects are disjoint
            effects = eff_union(body_effects, orelse_effects)

            return LoopIR.If(stmt.cond, body, orelse,
                             effects, stmt.srcinfo)

        elif isinstance(stmt, (LoopIR.ForAll, LoopIR.Seq)):
            styp = type(stmt)
            # pred is: 0 <= bound < stmt.hi
            bound = E.Var(stmt.iter, T.index, stmt.srcinfo)
            hi    = lift_expr(stmt.hi)
            zero  = E.Const(0, T.int, stmt.srcinfo)
            lhs   = E.BinOp("<=", zero, bound, T.bool, stmt.srcinfo)
            rhs   = E.BinOp("<",  bound, hi, T.bool, stmt.srcinfo)
            pred  = E.BinOp("and", lhs, rhs, T.bool, stmt.srcinfo)
            config_pred = E.BinOp("<", zero, hi, T.bool, stmt.srcinfo)

            body, body_effect = self.map_stmts(stmt.body)
            effects = eff_bind(stmt.iter, body_effect,
                               pred=pred, config_pred=config_pred)

            return styp(stmt.iter, stmt.hi, body,
                                   effects, stmt.srcinfo)

        elif isinstance(stmt, LoopIR.Call):
            assert stmt.f.eff is not None
            # build up a substitution dictionary....
            # sig is a LoopIR.fnarg, arg is a LoopIR.expr
            subst = {}
            for sig, arg in zip(stmt.f.args, stmt.args):
                if sig.type.is_numeric():
                    if isinstance(arg.type, T.Window):
                        pass  # handle below
                    elif isinstance(arg, LoopIR.Read):
                        subst[sig.name] = arg.name
                    elif isinstance(arg, LoopIR.ReadConfig):
                        pass #?
                    else:
                        assert False, "bad case"
                elif sig.type.is_indexable() or sig.type is T.bool:
                    # in this case we have a LoopIR expression...
                    subst[sig.name] = lift_expr(arg)
                elif sig.type.is_stridable():
                    subst[sig.name] = lift_expr(arg)
                    #pass
                else: assert False, "bad case"

            eff = stmt.f.eff
            eff = eff.subst(subst)

            # translate effects occuring on windowed arguments
            for sig,arg in zip(stmt.f.args, stmt.args):
                if sig.type.is_numeric():
                    if isinstance(arg.type, T.Window):
                        eff = self.translate_eff(eff, sig.name, arg.type)

            return LoopIR.Call(stmt.f, stmt.args,
                               eff, stmt.srcinfo)

        elif isinstance(stmt, LoopIR.Pass):
            return LoopIR.Pass(eff_null(stmt.srcinfo), stmt.srcinfo)
        elif isinstance(stmt, LoopIR.Alloc):
            return LoopIR.Alloc(stmt.name, stmt.type, stmt.mem,
                                eff_null(stmt.srcinfo), stmt.srcinfo)
        elif isinstance(stmt, LoopIR.WindowStmt):
            return LoopIR.WindowStmt(stmt.lhs, stmt.rhs,
                                     eff_null(stmt.srcinfo), stmt.srcinfo)

        else:
            assert False, "Invalid statement"

    # extract effects from this expression; return E.effect
    def eff_e(self, e):
        if isinstance(e, LoopIR.Read):
            if e.type.is_numeric():
                # we may assume that we're not in a call-argument position
                assert e.type.is_real_scalar()
                loc = [lift_expr(idx) for idx in e.idx]
                eff = eff_read(e.name, loc, e.srcinfo)

                # x[...], x
                buf_typ = self._types[e.name]
                if isinstance(buf_typ, T.Window):
                    eff = self.translate_eff(eff, e.name, buf_typ)

                return eff
            else:
                return eff_null(e.srcinfo)
        elif isinstance(e, LoopIR.BinOp):
            return eff_concat(self.eff_e(e.lhs), self.eff_e(e.rhs),
                              srcinfo=e.srcinfo)
        elif isinstance(e, LoopIR.USub):
            return self.eff_e(e.arg)
        elif isinstance(e, LoopIR.Const):
            return eff_null(e.srcinfo)
        elif isinstance(e, LoopIR.WindowExpr):
            return eff_null(e.srcinfo)
        elif isinstance(e, LoopIR.BuiltIn):
            return eff_null(e.srcinfo)
        elif isinstance(e, LoopIR.StrideExpr):
            return eff_null(e.srcinfo)
        elif isinstance(e, LoopIR.ReadConfig):
            return eff_config_read(e.config, e.field, e.srcinfo)
        else:
            assert False, "bad case"

    def translate_eff(self, eff, buf_name, win_typ):
        assert isinstance(eff, E.effect)
        assert isinstance(win_typ, T.Window)

        def translate_set(es):
            if es.buffer != buf_name:
                return es
            # otherwise, need to translate through the window
            #   Let `i` = es.loc
            #       `x` = es.buffer
            # For a windowing operation `x = y[:,lo:hi,3]`
            #   Let `j,k` be the indices into `y`.
            # Then,
            #   j == i + lo
            #   k == 3
            # which means we can get the transformed locations
            # by simply adding the `lo` offsets from windowing operations
            loc = es.loc
            buf = buf_name
            typ = win_typ
            while isinstance(typ, T.Window):
                buf     = typ.src_buf
                idx     = typ.idx
                typ     = self._types[buf]
                loc_i   = 0
                new_loc = []
                for w_acc in idx:
                    if isinstance(w_acc, LoopIR.Point):
                        new_loc.append(lift_expr(w_acc.pt))
                    elif isinstance(w_acc, LoopIR.Interval):
                        j = E.BinOp("+", loc[loc_i], lift_expr(w_acc.lo),
                                    T.index, w_acc.lo.srcinfo)
                        new_loc.append(j)
                        loc_i += 1
                assert loc_i == len(loc)
                loc = new_loc

            return E.effset( buf, loc, es.names, es.pred, es.srcinfo)

        return E.effect([ translate_set(es) for es in eff.reads ],
                        [ translate_set(es) for es in eff.writes ],
                        [ translate_set(es) for es in eff.reduces ],
                        eff.config_reads,
                        eff.config_writes,
                        eff.srcinfo)


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



#
#   What is parallelism checking for Configuration Effects?
#
#   Recall that we want to check that
#           forall i0,i1: 0 <= i0 < i1 < n ==> COMMUTES( [i |-> i0]e,
#                                                        [i |-> i1]e )
#
#   Unlike for numeric buffers we need only be concerned with
#       reads and writes: R & W
#   If one loop iteration reads a configuration value set by another
#       loop iteration, then we could have a problem.
#   So as long as the reads and writes to configuration fields are
#       disjoint, we should be fine.
#
#   Additionally, we've chosen to not handle quantification
#       of loop iteration variables, so we need to show a lack of
#       effect dependence on the loop iteration variables.
#   As a consequence of this, two different loop iterations must have
#       the same write effect, and so we may conclude that
#       writes are guaranteed to commute with each other
#
#   First check that
#       i \not\in FreeVars(cr0) U FreeVars(cw0)
#
#   Then, since there is no iteration variable dependence,
#   we can simplify to
#       COMMUTES( (cr, cw) ) = NOT_CONFLICTS( cr, cw )
#
#
#       NOT_CONFLICTS( (c1,...), (c2,...) )         = TRUE
#       NOT_CONFLICTS( (c1,f1,...), (c1,f2,...) )   = TRUE
#       NOT_CONFLICTS( (c1,f1, _, pred1),
#                      (c1,f1, _, pred2) ) = NOT pred1 or NOT pred2
#
#       (that is, there is no conflict when the predicates are disjoint)
#
#



# Check if Alloc sizes and function arg sizes are actually larger than bounds
class CheckEffects:
    def __init__(self, proc):
        self.orig_proc  = proc

        # Map sym to z3 variable
        self.env        = ChainMap()
        self.config_env = ChainMap()
        self.errors     = []

        self.stride_sym = dict()

        self.solver     = _get_smt_solver()

        self.push()

        # Add assertions
        for arg in proc.args:
            if isinstance(arg.type, T.Size):
                pos_sz = SMT.LT(SMT.Int(0), self.sym_to_smt(arg.name))
                self.solver.add_assertion(pos_sz)
            elif arg.type.is_tensor_or_window() and not arg.type.is_win():
                self.assume_tensor_strides(arg, arg.name, arg.type.shape())

        for p in proc.preds:
            # Check whether the assert is even potentially correct
            smt_p = self.expr_to_smt(lift_expr(p))
            if not self.solver.is_sat(smt_p):
                self.err(p, f"The assertion {p} at {p.srcinfo} "
                            f"is always unsatisfiable.")
            # independently, we will assume the assertion is
            # true while checking the rest of this procedure body
            self.solver.add_assertion(smt_p)

        self.preprocess_stmts(self.orig_proc.body)
        body_eff = self.map_stmts(self.orig_proc.body)

        for arg in proc.args:
            if arg.type.is_numeric():
                shape = [ lift_expr(s) for s in arg.type.shape() ]
                # check that all sizes/indices are positive
                for s in shape:
                    self.check_pos_size(s)
                # check the bounds
                self.check_bounds(arg.name, shape, body_eff)
        self.pop()

        # do error checking here
        if len(self.errors) > 0:
            raise TypeError("Errors occurred during effect checking:\n" +
                            "\n".join(self.errors))

    def counter_example(self):
        smt_syms = [ smt for sym,smt in self.env.items()
                         if smt.get_type() == SMT.INT ]
        val_map = self.solver.get_py_values(smt_syms)

        mapping = []
        for sym,smt in self.env.items():
            if smt.get_type() == SMT.INT:
                mapping.append(f" {sym} = {val_map[smt]}")

        return ",".join(mapping)

    def push(self):
        self.solver.push()
        self.env = self.env.new_child()
        self.config_env = self.config_env.new_child()

    def pop(self):
        self.env = self.env.parents
        self.config_env = self.config_env.parents
        self.solver.pop()

    def err(self, node, msg):
        self.errors.append(f"{node.srcinfo}: {msg}")

    # TODO: Add allow_allocation arg here, to check if we're introducing new
    # symbols from the right place.
    def sym_to_smt(self, sym, typ=T.index):
        if sym not in self.env:
            if typ.is_indexable() or typ.is_stridable():
                self.env[sym] = SMT.Symbol(repr(sym), SMT.INT)
            elif typ is T.bool:
                self.env[sym] = SMT.Symbol(repr(sym), SMT.BOOL)
        return self.env[sym]

    def config_to_smt(self, config, field, typ):
        c = (config, field)
        if c not in self.config_env:
            if typ.is_indexable() or typ.is_stridable():
                self.config_env[c] = SMT.Symbol(f"{config.name()}_{field}", SMT.INT)
            elif typ is T.bool:
                self.config_env[c] = SMT.Symbol(f"{config.name()}_{field}", SMT.BOOL)
            elif typ.is_scalar():
                self.config_env[c] = SMT.Symbol(f"{config.name()}_{field}", SMT.REAL)
            else:
                assert False, "bad case!"
        return self.config_env[c]

    def expr_to_smt(self, expr):
        assert isinstance(expr, E.expr), "expected Effects.expr"
        if isinstance(expr, E.Const):
            if expr.type == T.bool:
                return SMT.Bool(expr.val)
            elif expr.type.is_indexable():
                return SMT.Int(expr.val)
            else:
                assert False, f"unrecognized const type: {type(expr.val)}"
        elif isinstance(expr, E.Var):
            return self.sym_to_smt(expr.name, expr.type)
        elif isinstance(expr, E.Not):
            arg = self.expr_to_smt(expr.arg)
            return SMT.Not(arg)
        elif isinstance(expr, E.Stride):
            key = (expr.name, expr.dim)
            if key in self.stride_sym:
                stride_sym = self.stride_sym[key]
            else:
                stride_sym = Sym(f"{expr.name}_stride_{expr.dim}")
                self.stride_sym[key] = stride_sym
            return self.sym_to_smt(stride_sym)
        elif isinstance(expr, E.Select):
            cond = self.expr_to_smt(expr.cond)
            tcase = self.expr_to_smt(expr.tcase)
            fcase = self.expr_to_smt(expr.fcase)
            return SMT.Ite(cond, tcase, fcase)
        elif isinstance(expr, E.ConfigField):
            return self.config_to_smt(expr.config, expr.field, expr.type)
        elif isinstance(expr, E.BinOp):
            lhs = self.expr_to_smt(expr.lhs)
            rhs = self.expr_to_smt(expr.rhs)
            if expr.op == "+":
                return SMT.Plus(lhs, rhs)
            elif expr.op == "-":
                return SMT.Minus(lhs, rhs)
            elif expr.op == "*":
                return SMT.Times(lhs, rhs)
            elif expr.op == "/":
                assert isinstance(expr.rhs, E.Const)
                assert expr.rhs.val > 0
                # x // y is defined as floor(x/y)
                # Let z == floor(x/y)
                # Suppose we have P(x // y).
                # Then,
                #   P(x % y) =~= forall z, z == x // y ==> P(z)
                #   P(x % y) =~= exists z, z == x // y /\ P(z)
                # These two statements are not formally the same, so let's
                # work with both in the following...
                #
                # Consider now that
                #       z == x // y =~=  z == floor(x/y)
                #                   =~=  z <= x/y < z + 1
                #                   =~=  y*z <= x < y*(z+1)
                # which is an affine equation when y is constant.
                #
                # Let's substitute this back into the two quantifier forms
                #   forall z, y*z <= x < y*(z+1) ==> P(z)
                #   exists z, y*z <= x < y*(z+1) /\ P(z)
                #
                # My concern is that we are placing this rewrite into both
                # the position of hypothesis and goal.  So for
                #       forall x, H ==> G  (which =~= forall x, ~H \/ G)
                # If we place the forall form above into the G position,
                # everything works out pretty easily...
                #       forall x, H ==> (forall z, C ==> P(z))
                #   =~= forall x, ~H \/ (forall z, C ==> P(z))
                #   =~= forall x, forall z, ~H \/ (C ==> P(z))
                #   =~= forall x, forall z, H ==> (C ==> P(z))
                #   =~= forall x, forall z, H /\ C ==> P(z)
                # If we place the forall form above into the H position,
                # we get
                #       forall x, (forall z, C ==> P(z)) ==> G
                #   =~= forall x, ~(forall z, C ==> P(z)) \/ G
                #   =~= forall x, (exists z, ~(C ==> P(z)) \/ G
                #   =~= forall x, exists z, ~(C ==> P(z) \/ G
                #   =~= forall x, exists z, ~(~C \/ P(z)) \/ G
                #   =~= forall x, exists z, (C /\ P(z)) \/ G
                # This is a mess!
                #
                # What about if in the hypothesis case, we try using
                # the alternate `exists ...` quantifier form to begin with...
                # Then we get,
                #       forall x, (exists z, C /\ P(z)) ==> G
                #   =~= forall x, ~(exists z, C /\ P(z)) \/ G
                #   =~= forall x, (forall z, ~C \/ ~P(z)) \/ G
                #   =~= forall x, forall z, ~C \/ ~P(z) \/ G
                #   =~= forall x, forall z, C ==> ~P(z) \/ G
                #   =~= forall x, forall z, C ==> (P(z) ==> G)
                #   =~= forall x, forall z, (C /\ P(z)) ==> G
                # This is now the same thing we were expecting to get
                # in the goal position!  So it turns out it's safe too!
                #

                # Introduce new Sym (z in formula below)
                div_tmp = self.sym_to_smt(Sym("div_tmp"))
                # rhs*z <= lhs < rhs*(z+1)
                rhs_eq  = SMT.LE(SMT.Times(rhs, div_tmp), lhs)
                lhs_eq  = SMT.LT(lhs,
                        SMT.Times(rhs, SMT.Plus(div_tmp, SMT.Int(1))))
                self.solver.add_assertion(SMT.And(rhs_eq, lhs_eq))
                return div_tmp
            elif expr.op == "%":
                assert isinstance(expr.rhs, E.Const)
                assert expr.rhs.val > 0
                # In the below, copy the logic above for division
                # to construct `mod_tmp` s.t.
                #   mod_tmp = floor(lhs / rhs)
                # Then,
                #   lhs % rhs = lhs - rhs * mod_tmp
                mod_tmp = self.sym_to_smt(Sym("mod_tmp"))
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
                if expr.lhs.type == T.bool and expr.rhs.type == T.bool:
                    return SMT.Iff(lhs, rhs)
                elif (expr.lhs.type.is_indexable() and
                      expr.rhs.type.is_indexable()):
                    return SMT.Equals(lhs, rhs)
                elif (expr.lhs.type.is_stridable() and
                      expr.rhs.type.is_stridable()):
                    return SMT.Equals(lhs, rhs)
                else:
                    assert False, "bad case"
            elif expr.op == "and":
                return SMT.And(lhs, rhs)
            elif expr.op == "or":
                return SMT.Or(lhs, rhs)
        else: assert False, f"bad case: {type(expr)}"


    def assume_tensor_strides(self, node, name, shape):
        # compute statically knowable strides from the shape
        strides = [None] * len(shape)
        strides[-1] = 1
        for i, sz in reversed(list(enumerate(shape))):
            if i > 0:
                if not isinstance(sz, LoopIR.Const):
                    break
                else:
                    strides[i-1] = sz.val * strides[i]

        # for all statically knowable strides, set the appropriate variable.
        for dim,s in enumerate(strides):
            if s is not None:
                s_expr  = LoopIR.StrideExpr(name, dim, T.stride, node.srcinfo)
                s_const = LoopIR.Const(s, T.int, node.srcinfo)
                eq      = LoopIR.BinOp('==', s_expr, s_const,
                                       T.bool, node.srcinfo)
                self.solver.add_assertion(self.expr_to_smt(lift_expr(eq)))

    def check_in_bounds(self, sym, shape, eff, eff_str):
        assert isinstance(eff, E.effset), "effset should be passed to in_bounds"

        if sym == eff.buffer:
#       IN_BOUNDS( x, T, (x, (i,j), nms, pred ) ) =
#           forall nms in Z, pred ==> in_bounds(T, (i,j))

            self.push()
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

            if not self.solver.is_valid(in_bds):
                eg = self.counter_example()
                self.err(eff, f"{sym} is {eff_str} out-of-bounds "
                              f"when:\n  {eg}.")

            self.pop()

    def check_bounds(self, sym, shape, eff):
        effs = [(eff.reads, "read"), (eff.writes, "written"),
                (eff.reduces, "reduced")]

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
        if e1.buffer != e2.buffer:
            return

        self.push()
        # determine name substitutions
        iter1   = iter.copy()
        iter2   = iter.copy()
        iter1_smt = self.sym_to_smt(iter1)
        iter2_smt = self.sym_to_smt(iter2)
        iter_pred = SMT.And(SMT.And(SMT.LE(SMT.Int(0), iter1_smt),
                            SMT.LT(iter1_smt, iter2_smt)),
                            SMT.LT(iter2_smt, self.expr_to_smt(hi)))
        self.solver.add_assertion(iter_pred)

        sub1    = { nm : E.Var(nm.copy(), T.index, null_srcinfo())
                    for nm in e1.names }
        sub1[iter] = E.Var(iter1, T.index, null_srcinfo())
        sub2    = { nm : E.Var(nm.copy(), T.index, null_srcinfo())
                    for nm in e2.names }
        sub2[iter] = E.Var(iter2, T.index, null_srcinfo())
        
        if e1.pred is not None:
            pred1   = e1.pred.subst(sub1)
            self.solver.add_assertion(self.expr_to_smt(pred1))
        if e2.pred is not None:
            pred2   = e2.pred.subst(sub2)
            self.solver.add_assertion(self.expr_to_smt(pred2))

        loc1    = [ self.expr_to_smt(i.subst(sub1))
                    for i in e1.loc ]
        loc2    = [ self.expr_to_smt(i.subst(sub2))
                    for i in e2.loc ]
        loc_neq = SMT.Bool(False)
        for i1, i2 in zip(loc1,loc2):
            loc_neq = SMT.Or(loc_neq, SMT.NotEquals(i1, i2))

        if not self.solver.is_valid(loc_neq):
            eg = self.counter_example()
            self.err(e1, f"data race conflict with statement on "
                         f"{e2.srcinfo} while accessing {e1.buffer} "
                         f"in loop over {iter}, when:\n  {eg}.")

        self.pop()

    def not_conflicts_config(self, e1, e2):
        if e1.config == e2.config and e1.field == e2.field:
            self.push()
            pred1, pred2 = SMT.Bool(True), SMT.Bool(True)
            if e1.pred:
                pred1   = self.expr_to_smt(e1.pred)
            if e2.pred:
                pred2   = self.expr_to_smt(e2.pred)
            disjoint    = SMT.Not(SMT.And(pred1, pred2))

            if not self.solver.is_valid(disjoint):
                eg = self.counter_example()
                self.err(e1, f"conflict on configuration variable "
                             f"{e1.config.name()}.{e1.field} between "
                             f"access at {e1.srcinfo} and access at "
                             f"{e1.srcinfo}, occurs when:\n  {eg}.")

            self.pop()

#       COMMUTES( i, n, (r, w, p) ) =
    def check_commutes(self, iter, hi, eff, body):

        # for numeric buffers
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
                if w1.buffer == w2.buffer:
                    # If we're writing a loop invariant value to
                    # the same buffer, that is safe
                    if iter not in LoopIR_Dependencies(w1.buffer, body).result():
                        continue

                self.not_conflicts(iter, hi, w1, w2)

#                NOT_CONFLICTS(i, n, w, p) )
        for w in eff.writes:
            for p in eff.reduces:
                self.not_conflicts(iter, hi, w, p)

        # for configuration variables
        for r in eff.config_reads:
            for w in eff.config_writes:
                self.not_conflicts_config(r, w)

        return

    def check_config_no_loop_depend(self, iter, eff, body):
        for ce in eff.config_reads:
            if iter in LoopIR_Dependencies((ce.config, ce.field), body).result():
                self.err(ce, f"The read of config variable "
                             f"{ce.config.name()}.{ce.field} depends on "
                             f"the loop iteration variable {iter}")
        for ce in eff.config_writes:
            if iter in LoopIR_Dependencies((ce.config, ce.field), body).result():
                self.err(ce, f"The value written to config variable "
                             f"{ce.config.name()}.{ce.field} depends on "
                             f"the loop iteration variable {iter}")

    def check_pos_size(self, expr):
        e_pos = SMT.LT( SMT.Int(0), self.expr_to_smt(expr) )
        if not self.solver.is_valid(e_pos):
            eg = self.counter_example()
            self.err(expr, "expected expression to always be positive. "
                           f"It can be non positive when:\n  {eg}.")

    def check_non_negative(self, expr):
        e_nn = SMT.LE( SMT.Int(0), self.expr_to_smt(expr) )
        if not self.solver.is_valid(e_nn):
            eg = self.counter_example()
            self.err(expr, "expected expression to always be non-negative. "
                           f"It can be negative when:\n  {eg}.")

    def check_call_shape_eqv(self, argshp, sigshp, node):
        assert len(argshp) == len(sigshp)
        eqv_dim = SMT.Bool(True)
        for a,s in zip(argshp, sigshp):
            eq_here = SMT.Equals(self.expr_to_smt(a),
                       self.expr_to_smt(s))
            eqv_dim = SMT.And(eqv_dim, eq_here)
        if not self.solver.is_valid(eqv_dim):
            eg = self.counter_example()
            self.err(node, "type-shape of calling argument may not equal "
                           "the required type-shape: "
                           f"[{','.join(map(str,argshp))}] vs. "
                           f"[{','.join(map(str,sigshp))}]."
                           f" It could be non equal when:\n  {eg}")

    def preprocess_stmts(self, body):
        for stmt in body:
            if isinstance(stmt, LoopIR.If):
                self.preprocess_stmts(stmt.body)
                self.preprocess_stmts(stmt.orelse)
            elif isinstance(stmt, (LoopIR.ForAll, LoopIR.Seq)):
                self.preprocess_stmts(stmt.body)
            elif isinstance(stmt, LoopIR.Alloc):
                if stmt.type.is_tensor_or_window():
                    self.assume_tensor_strides(stmt, stmt.name,
                                               stmt.type.shape())
            elif isinstance(stmt, LoopIR.WindowStmt):
                # src_shape   = stmt.rhs.type.src_type.shape()
                w_idx = stmt.rhs.type.idx
                src_buf = stmt.rhs.type.src_buf
                dst_buf = stmt.lhs

                src_dims = [d for d, w in enumerate(w_idx)
                            if isinstance(w, LoopIR.Interval)]
                for dst_dim, src_dim in enumerate(src_dims):
                    src = LoopIR.StrideExpr(src_buf, src_dim,
                                            T.stride, stmt.srcinfo)
                    dst = LoopIR.StrideExpr(dst_buf, dst_dim,
                                            T.stride, stmt.srcinfo)
                    eq  = LoopIR.BinOp('==',src,dst,T.bool,stmt.srcinfo)
                    self.solver.add_assertion(self.expr_to_smt(lift_expr(eq)))
            else:
                pass

    def map_stmts(self, body):
        """ Returns an effect for the argument `body`
            And also checks bounds/parallelism for any
            allocations/loops within `body`
        """
        assert len(body) > 0
        body_eff = eff_null(body[-1].srcinfo)

        for stmt in reversed(body):
            if isinstance(stmt, (LoopIR.ForAll, LoopIR.Seq)):
                self.push()

                def bd_pred(x, hi, srcinfo):
                    zero = E.Const(0, T.int, srcinfo)
                    x = E.Var(x, T.int, srcinfo)
                    hi = lift_expr(hi)
                    return E.BinOp("and",
                                   E.BinOp("<=", zero, x, T.bool, srcinfo),
                                   E.BinOp("<", x, hi, T.bool, srcinfo),
                                   T.bool, srcinfo)

                # Check if for-loop bound is non-negative
                # with the context, before adding assertion
                self.check_non_negative(lift_expr(stmt.hi))

                self.solver.add_assertion(
                    self.expr_to_smt(bd_pred(stmt.iter, stmt.hi,
                                             stmt.srcinfo)))

                sub_body_eff = self.map_stmts(stmt.body)
                self.pop()

                # TODO: This check is outdated
                #self.check_config_no_loop_depend(stmt.iter, sub_body_eff, stmt.body)
                # Parallelism checking here
                # Don't check parallelism is it's a seq loop
                if isinstance(stmt, LoopIR.ForAll):
                    self.check_commutes(stmt.iter, lift_expr(stmt.hi),
                                        sub_body_eff, stmt.body)

                body_eff = eff_concat(stmt.eff, body_eff)

            if isinstance(stmt, LoopIR.If):
                # first, do the if-branch
                self.push()
                self.solver.add_assertion(
                    self.expr_to_smt(lift_expr(stmt.cond)))
                self.map_stmts(stmt.body)
                self.pop()

                # then the else-branch
                if len(stmt.orelse) > 0:
                    self.push()
                    neg_cond = lift_expr(stmt.cond).negate()
                    self.solver.add_assertion(self.expr_to_smt(neg_cond))
                    self.map_stmts(stmt.orelse)
                    self.pop()

                body_eff = eff_concat(stmt.eff, body_eff)

            elif isinstance(stmt, LoopIR.Alloc):
                shape = [lift_expr(s) for s in stmt.type.shape()]
                # check that all sizes are positive
                for s in shape:
                    self.check_pos_size(s)
                # check that all accesses are in bounds
                self.check_bounds(stmt.name, shape, body_eff)
                body_eff = eff_remove_buf(stmt.name, body_eff)

            elif isinstance(stmt, LoopIR.Call):
                subst = dict()
                for sig, arg in zip(stmt.f.args, stmt.args):
                    if sig.type.is_numeric():
                        # need to check that the argument shape
                        # has all positive dimensions
                        arg_shape = [lift_expr(s) for s in arg.type.shape()]
                        for e in arg_shape:
                            self.check_pos_size(e)
                        # also, need to check that the argument shape
                        # is exactly the shape specified in the signature
                        sig_shape = [ lift_expr(loopir_subst(s, subst))
                                      for s in sig.type.shape() ]
                        self.check_call_shape_eqv(arg_shape, sig_shape, arg)

                        # bind potential window-expression
                        subst[sig.name] = arg

                    elif ( sig.type.is_indexable() or
                           sig.type == T.bool or sig.type.is_stridable() ):
                        # in this case we have a LoopIR expression...
                        subst[sig.name] = arg
                        if sig.type == T.size:
                            e_arg       = lift_expr(arg)
                            self.check_pos_size(e_arg)

                    else: assert False, "bad case"

                for p in stmt.f.preds:
                    # Check that asserts are correct
                    p_subst     = loopir_subst(p, subst)
                    smt_pred    = self.expr_to_smt(lift_expr(p_subst))
                    if not self.solver.is_valid(smt_pred):
                        eg = self.counter_example()
                        self.err(stmt, f"Could not verify assertion {p} in "
                                       f"{stmt.f.name} at {p.srcinfo}."
                                       f" Assertion is false when:\n  {eg}")

                body_eff = eff_concat(stmt.eff, body_eff)

            else:
                body_eff = eff_concat(stmt.eff, body_eff)


        return body_eff # Returns union of all effects
