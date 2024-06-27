from collections import ChainMap
from asdl_adt import ADT, validators

import pysmt
from pysmt import shortcuts as SMT

from .LoopIR import LoopIR, T, Operator, Config
from .prelude import *


# --------------------------------------------------------------------------- #
# Effects for bounds checking
# --------------------------------------------------------------------------- #

E = ADT(
    """
module Effects {
    effect      = ( effset*     reads,
                    effset*     writes,
                    effset*     reduces,
                    config_eff* config_reads,
                    config_eff* config_writes,
                    srcinfo     srcinfo )

    effset      = ( sym         buffer,
                    expr*       loc,    -- e.g. reading at (i+1,j+1)
                    sym*        names,
                    expr?       pred,
                    srcinfo     srcinfo )

    config_eff  = ( config      config, -- blah
                    string      field,
                    expr?       value, -- need not be supplied for reads
                    expr?       pred,
                    srcinfo     srcinfo )

    expr        = Var( sym name )
                | Not( expr arg )
                | Const( object val )
                | BinOp( binop op, expr lhs, expr rhs )
                | Stride( sym name, int dim )
                | Select( expr cond, expr tcase, expr fcase )
                | ConfigField( config config, string field )
                attributes( type type, srcinfo srcinfo )

} """,
    {
        "sym": Sym,
        "type": LoopIR.type,
        "binop": validators.instance_of(Operator, convert=True),
        "config": Config,
        "srcinfo": SrcInfo,
    },
)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


# convert from LoopIR.expr to E.expr
def lift_expr(e):
    if isinstance(e, LoopIR.Read):
        assert len(e.idx) == 0
        return E.Var(e.name, e.type, e.srcinfo)

    elif isinstance(e, LoopIR.Const):
        return E.Const(e.val, e.type, e.srcinfo)

    elif isinstance(e, LoopIR.BinOp):
        lhs = lift_expr(e.lhs)
        rhs = lift_expr(e.rhs)
        return E.BinOp(e.op, lhs, rhs, e.type, e.srcinfo)

    elif isinstance(e, LoopIR.USub):
        zero = E.Const(0, e.type, e.srcinfo)
        arg = lift_expr(e.arg)
        return E.BinOp("-", zero, arg, e.type, e.srcinfo)

    elif isinstance(e, LoopIR.StrideExpr):
        return E.Stride(e.name, e.dim, e.type, e.srcinfo)

    elif isinstance(e, LoopIR.ReadConfig):
        cfg_val = e.config.lookup_type(e.field)
        return E.ConfigField(e.config, e.field, cfg_val, e.srcinfo)

    assert False, f"bad case, e is {type(e)}"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Substitution of Effect Variables in an effect


@extclass(E.expr)
def negate(self):
    return negate_expr(self)


del negate


def negate_expr(e):
    Tbool = T.bool
    assert e.type == Tbool, "can only negate predicates"
    if isinstance(e, E.Const):
        return E.Const(not e.val, e.type, e.srcinfo)
    elif isinstance(e, E.Var) or isinstance(e, E.ConfigField):
        return E.Not(e, e.type, e.srcinfo)
    elif isinstance(e, E.Not):
        return e.arg
    elif isinstance(e, E.BinOp):

        def change_op(op, lhs=e.lhs, rhs=e.rhs):
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
            if e.lhs.type is Tbool and e.rhs.type is Tbool:
                l = E.BinOp("and", e.lhs, negate_expr(e.rhs), Tbool, e.srcinfo)
                r = E.BinOp("and", negate_expr(e.lhs), e.rhs, Tbool, e.srcinfo)

                return E.BinOp("or", l, r, Tbool, e.srcinfo)
            elif e.lhs.type.is_indexable() and e.rhs.type.is_indexable():
                return E.BinOp("or", change_op("<"), change_op(">"), Tbool, e.srcinfo)
            else:
                assert False, "TODO: add != support explicitly..."
    elif isinstance(e, E.Select):
        return E.Select(
            e.cond, negate_expr(e.tcase), negate_expr(e.fcase), e.type, e.srcinfo
        )
    assert False, "bad case"


@extclass(E.effect)
@extclass(E.effset)
@extclass(E.expr)
def subst(self, env):
    return eff_subst(env, self)


del subst


def eff_subst(env, eff):
    if isinstance(eff, E.effset):
        assert all(nm not in env for nm in eff.names)
        buf = env[eff.buffer] if eff.buffer in env else eff.buffer
        pred = eff_subst(env, eff.pred) if eff.pred else None
        return E.effset(
            buf, [eff_subst(env, e) for e in eff.loc], eff.names, pred, eff.srcinfo
        )
    elif isinstance(eff, E.config_eff):
        value = eff_subst(env, eff.value) if eff.value else None
        pred = eff_subst(env, eff.pred) if eff.pred else None
        return E.config_eff(eff.config, eff.field, value, pred, eff.srcinfo)
    elif isinstance(eff, E.Var):
        return env[eff.name] if eff.name in env else eff
    elif isinstance(eff, E.Not):
        return E.Not(eff_subst(env, eff.arg), eff.type, eff.srcinfo)
    elif isinstance(eff, E.Const):
        return eff
    elif isinstance(eff, E.BinOp):
        return E.BinOp(
            eff.op,
            eff_subst(env, eff.lhs),
            eff_subst(env, eff.rhs),
            eff.type,
            eff.srcinfo,
        )
    elif isinstance(eff, E.Stride):
        name = env[eff.name] if eff.name in env else eff
        return E.Stride(name, eff.dim, eff.type, eff.srcinfo)
    elif isinstance(eff, E.Select):
        return E.Select(
            eff_subst(env, eff.cond),
            eff_subst(env, eff.tcase),
            eff_subst(env, eff.fcase),
            eff.type,
            eff.srcinfo,
        )
    elif isinstance(eff, E.ConfigField):
        return eff
    elif isinstance(eff, E.effect):
        return E.effect(
            [eff_subst(env, es) for es in eff.reads],
            [eff_subst(env, es) for es in eff.writes],
            [eff_subst(env, es) for es in eff.reduces],
            [eff_subst(env, ce) for ce in eff.config_reads],
            [eff_subst(env, ce) for ce in eff.config_writes],
            eff.srcinfo,
        )
    else:
        assert False, f"bad case: {type(eff)}"


@extclass(E.effect)
@extclass(E.effset)
@extclass(E.expr)
def config_subst(self, env):
    return _subcfg(env, self)


del config_subst


def _subcfg(env, eff):
    if isinstance(eff, E.effset):
        return E.effset(
            eff.buffer,
            [_subcfg(env, e) for e in eff.loc],
            eff.names,
            _subcfg(env, eff.pred) if eff.pred else None,
            eff.srcinfo,
        )
    elif isinstance(eff, E.config_eff):
        value = _subcfg(env, eff.value) if eff.value else None
        pred = _subcfg(env, eff.pred) if eff.pred else None
        return E.config_eff(eff.config, eff.field, value, pred, eff.srcinfo)
    elif isinstance(eff, (E.Var, E.Const)):
        return eff
    elif isinstance(eff, E.Not):
        return E.Not(_subcfg(env, eff.arg), eff.type, eff.srcinfo)
    elif isinstance(eff, E.BinOp):
        return E.BinOp(
            eff.op, _subcfg(env, eff.lhs), _subcfg(env, eff.rhs), eff.type, eff.srcinfo
        )
    elif isinstance(eff, E.Stride):
        return eff
    elif isinstance(eff, E.Select):
        return E.Select(
            _subcfg(env, eff.cond),
            _subcfg(env, eff.tcase),
            _subcfg(env, eff.fcase),
            eff.type,
            eff.srcinfo,
        )
    elif isinstance(eff, E.ConfigField):
        if (eff.config, eff.field) in env:
            return env[(eff.config, eff.field)]
        else:
            return eff
    elif isinstance(eff, E.effect):
        return E.effect(
            [_subcfg(env, es) for es in eff.reads],
            [_subcfg(env, es) for es in eff.writes],
            [_subcfg(env, es) for es in eff.reduces],
            [_subcfg(env, ce) for ce in eff.config_reads],
            [_subcfg(env, ce) for ce in eff.config_writes],
            eff.srcinfo,
        )
    else:
        assert False, f"bad case: {type(eff)}"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Construction/Composition Functions


def eff_null(srcinfo=null_srcinfo()):
    return E.effect([], [], [], [], [], srcinfo)


def eff_read(buf, loc, srcinfo=null_srcinfo()):
    read = E.effset(buf, loc, [], None, srcinfo)
    return E.effect([read], [], [], [], [], srcinfo)


def eff_write(buf, loc, srcinfo=null_srcinfo()):
    write = E.effset(buf, loc, [], None, srcinfo)
    return E.effect([], [write], [], [], [], srcinfo)


def eff_reduce(buf, loc, srcinfo=null_srcinfo()):
    reduction = E.effset(buf, loc, [], None, srcinfo)
    return E.effect([], [], [reduction], [], [], srcinfo)


def eff_config_read(config, field, srcinfo=null_srcinfo()):
    read = E.config_eff(config, field, None, None, srcinfo)
    return E.effect([], [], [], [read], [], srcinfo)


def eff_config_write(config, field, value, srcinfo=null_srcinfo()):
    write = E.config_eff(config, field, value, None, srcinfo)
    return E.effect([], [], [], [], [write], srcinfo)


def _and_preds(a, b):
    return (
        a if b is None else b if a is None else E.BinOp("and", a, b, T.bool, a.srcinfo)
    )


def _or_preds(a, b):
    return None if a is None or b is None else E.BinOp("or", a, b, T.bool, a.srcinfo)


def eff_union(e1, e2, srcinfo=None):
    srcinfo = srcinfo or e1.srcinfo

    return E.effect(
        e1.reads + e2.reads,
        e1.writes + e2.writes,
        e1.reduces + e2.reduces,
        e1.config_reads + e2.config_reads,
        e1.config_writes + e2.config_writes,
        srcinfo,
    )


# handle complex logic for Configuration when computing
#   EffectOf( s1 ; s2 ) in terms of EffectOf( s1 ) and EffectOf( s2 )
def eff_concat(e1, e2, srcinfo=None):
    srcinfo = srcinfo or e1.srcinfo

    # Step 1: substitute references in e2 to fields written by e1
    # value expression of config.field after this effect
    def write_val(ce):
        assert ce.value is not None
        if not ce.pred:
            return ce.value
        else:
            # TODO: Fix! I'm not sure what is the intent here..
            old_val = E.ConfigField(ce.config, ce.field, T.bool, srcinfo)
            return E.Select(ce.pred, ce.value, old_val, T.bool, srcinfo)

    # substitute on the basis of writes in the first effect
    env = {(ce.config, ce.field): write_val(ce) for ce in e1.config_writes}
    e2 = e2.config_subst(env)

    # Step 2: merge writes from the two effects to the same field
    def merge_writes(config_writes_1, config_writes_2):
        cws1 = {(w.config, w.field): w for w in config_writes_1}
        cws2 = {(w.config, w.field): w for w in config_writes_2}
        overlap = set(cws1.keys()).intersection(set(cws2.keys()))

        def merge(w1, w2):
            # in the case of the second write being unconditional
            if w2.pred is None:
                return w2
            else:
                typ = w1.config.lookup_type(w1.field)
                assert typ == w2.config.lookup_type(w2.field)

                pred = _or_preds(w1.pred, w2.pred)
                val = E.Select(w2.pred, w2.value, w1.value, typ, w2.srcinfo)

                return E.config_eff(w1.config, w1.field, val, pred, w2.srcinfo)

        return (
            [cws1[w] for w in cws1 if w not in overlap]
            + [cws2[w] for w in cws2 if w not in overlap]
            + [merge(cws1[key], cws2[key]) for key in overlap]
        )

    # Step 3: filter out config reads in e2 if they are just
    #         reading the config value written in e1
    def shadow_config_reads(config_writes, config_reads):
        results = []
        for read in config_reads:
            assert read.value is None

            # find the corresponding write
            write = None
            for w in config_writes:
                if w.config == read.config and w.field == read.field:
                    write = w
                    break

            # handle the shadowing
            if write is None:
                results.append(read)
            elif write.pred is None:
                # unconditional write, so remove the read
                pass
            else:
                # conditional write, so guard the read
                pred = _and_preds(read.pred, write.pred.negate())
                results.append(
                    E.config_eff(read.config, read.field, None, pred, read.srcinfo)
                )

        return results

    config_reads = e1.config_reads + shadow_config_reads(
        e1.config_writes, e2.config_reads
    )
    config_writes = merge_writes(e1.config_writes, e2.config_writes)

    reads = e1.reads + e2.reads

    return E.effect(
        reads,
        e1.writes + e2.writes,
        e1.reduces + e2.reduces,
        config_reads,
        config_writes,
        srcinfo,
    )


def eff_remove_buf(buf, e):
    return E.effect(
        [es for es in e.reads if es.buffer != buf],
        [es for es in e.writes if es.buffer != buf],
        [es for es in e.reduces if es.buffer != buf],
        e.config_reads,
        e.config_writes,
        e.srcinfo,
    )


# handle conditional
def eff_filter(pred, e):
    def filter_es(es):
        return E.effset(
            es.buffer, es.loc, es.names, _and_preds(pred, es.pred), es.srcinfo
        )

    def filter_ce(ce):
        return E.config_eff(
            ce.config, ce.field, ce.value, _and_preds(pred, ce.pred), ce.srcinfo
        )

    return E.effect(
        [filter_es(es) for es in e.reads],
        [filter_es(es) for es in e.writes],
        [filter_es(es) for es in e.reduces],
        [filter_ce(ce) for ce in e.config_reads],
        [filter_ce(ce) for ce in e.config_writes],
        e.srcinfo,
    )


# handle for loop
def eff_bind(bind_name, e, pred=None, config_pred=None):
    assert isinstance(bind_name, Sym)

    def bind_es(es):
        return E.effset(
            es.buffer,
            es.loc,
            [bind_name] + es.names,
            _and_preds(pred, es.pred),
            es.srcinfo,
        )

    def filter_ce(ce):
        return E.config_eff(
            ce.config, ce.field, ce.value, _and_preds(pred, config_pred), ce.srcinfo
        )

    return E.effect(
        [bind_es(es) for es in e.reads],
        [bind_es(es) for es in e.writes],
        [bind_es(es) for es in e.reduces],
        [filter_ce(ce) for ce in e.config_reads],
        [filter_ce(ce) for ce in e.config_writes],
        e.srcinfo,
    )


def _get_smt_solver():
    factory = pysmt.factory.Factory(pysmt.shortcuts.get_env())
    slvs = factory.all_solvers()
    if len(slvs) == 0:
        raise OSError("Could not find any SMT solvers")
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
        return LoopIR.USub(loopir_subst(e.arg, subst), e.type, e.srcinfo)
    elif isinstance(e, LoopIR.BinOp):
        return LoopIR.BinOp(
            e.op,
            loopir_subst(e.lhs, subst),
            loopir_subst(e.rhs, subst),
            e.type,
            e.srcinfo,
        )
    elif isinstance(e, LoopIR.StrideExpr):
        if e.name not in subst:
            return e
        lookup = subst[e.name]
        if isinstance(lookup, LoopIR.Read):
            assert len(lookup.idx) == 0
            return LoopIR.StrideExpr(lookup.name, e.dim, e.type, e.srcinfo)
        elif isinstance(lookup, LoopIR.WindowExpr):
            windowed_orig_dims = [
                d for d, w in enumerate(lookup.idx) if isinstance(w, LoopIR.Interval)
            ]
            dim = windowed_orig_dims[e.dim]
            return LoopIR.StrideExpr(lookup.name, dim, e.type, e.srcinfo)

        else:
            assert False, f"bad case: {type(lookup)}"

    else:
        assert False, f"bad case: {type(e)}"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Check if Alloc sizes and function arg sizes are actually larger than bounds


class CheckBounds:
    def __init__(self, proc):
        self.orig_proc = proc

        # Map sym to z3 variable
        self.env = ChainMap()
        self.config_env = ChainMap()
        self.errors = []

        self.stride_sym = dict()

        self.solver = _get_smt_solver()

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
                self.err(
                    p, f"The assertion {p} at {p.srcinfo} is always unsatisfiable."
                )
            # independently, we will assume the assertion is
            # true while checking the rest of this procedure body
            self.solver.add_assertion(smt_p)

        self.preprocess_stmts(proc.body)

        body_eff = self.map_stmts(proc.body, self.rec_proc_types(proc))

        for arg in proc.args:
            if arg.type.is_numeric():
                shape = [lift_expr(s) for s in arg.type.shape()]
                # check that all sizes/indices are positive
                for s in shape:
                    self.check_pos_size(s)
                # check the bounds
                self.check_bounds(arg.name, shape, body_eff)

        self.pop()

        # do error checking here
        if len(self.errors) > 0:
            raise TypeError(
                "Errors occurred during effect checking:\n" + "\n".join(self.errors)
            )

    def rec_proc_types(self, proc):
        type_env = {}
        for a in proc.args:
            type_env[a.name] = a.type
        return type_env | self.rec_stmts_types(proc.body)

    def rec_stmts_types(self, body):
        assert len(body) > 0
        type_env = {}
        for s in body:
            self.rec_s_types(s, type_env)
        return type_env

    def rec_s_types(self, stmt, type_env):
        if isinstance(stmt, LoopIR.If):
            type_env |= self.rec_stmts_types(stmt.body)
            if len(stmt.orelse) > 0:
                type_env |= self.rec_stmts_types(stmt.orelse)
        elif isinstance(stmt, LoopIR.For):
            type_env |= self.rec_stmts_types(stmt.body)
        elif isinstance(stmt, LoopIR.Alloc):
            type_env[stmt.name] = stmt.type
        elif isinstance(stmt, LoopIR.WindowStmt):
            type_env[stmt.name] = stmt.rhs.type
        else:
            pass

    def counter_example(self):
        smt_syms = [smt for sym, smt in self.env.items() if smt.get_type() == SMT.INT]
        val_map = self.solver.get_py_values(smt_syms)

        mapping = []
        for sym, smt in self.env.items():
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
                rhs_eq = SMT.LE(SMT.Times(rhs, div_tmp), lhs)
                lhs_eq = SMT.LT(lhs, SMT.Times(rhs, SMT.Plus(div_tmp, SMT.Int(1))))
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
                rhs_eq = SMT.LE(SMT.Times(rhs, mod_tmp), lhs)
                lhs_eq = SMT.LT(lhs, SMT.Times(rhs, SMT.Plus(mod_tmp, SMT.Int(1))))
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
                elif expr.lhs.type.is_indexable() and expr.rhs.type.is_indexable():
                    return SMT.Equals(lhs, rhs)
                elif expr.lhs.type.is_stridable() and expr.rhs.type.is_stridable():
                    return SMT.Equals(lhs, rhs)
                else:
                    assert False, "bad case"
            elif expr.op == "and":
                return SMT.And(lhs, rhs)
            elif expr.op == "or":
                return SMT.Or(lhs, rhs)
        else:
            assert False, f"bad case: {type(expr)}"

    def assume_tensor_strides(self, node, name, shape):
        # compute statically knowable strides from the shape
        strides = [None] * len(shape)
        strides[-1] = 1
        for i, sz in reversed(list(enumerate(shape))):
            if i > 0:
                if not isinstance(sz, LoopIR.Const):
                    break
                else:
                    strides[i - 1] = sz.val * strides[i]

        # for all statically knowable strides, set the appropriate variable.
        for dim, s in enumerate(strides):
            if s is not None:
                s_expr = LoopIR.StrideExpr(name, dim, T.stride, node.srcinfo)
                s_const = LoopIR.Const(s, T.int, node.srcinfo)
                eq = LoopIR.BinOp("==", s_expr, s_const, T.bool, node.srcinfo)
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
                e = self.expr_to_smt(e)
                lhs = SMT.LE(SMT.Int(0), e)
                rhs = SMT.LT(e, self.expr_to_smt(hi))
                in_bds = SMT.And(in_bds, SMT.And(lhs, rhs))

            if not self.solver.is_valid(in_bds):
                eg = self.counter_example()
                self.err(eff, f"{sym} is {eff_str} out-of-bounds when:\n  {eg}.")

            self.pop()

    def check_bounds(self, sym, shape, eff):
        effs = [(eff.reads, "read"), (eff.writes, "written"), (eff.reduces, "reduced")]

        for es, y in effs:
            for e in es:
                self.check_in_bounds(sym, shape, e, y)

    def check_pos_size(self, expr):
        e_pos = SMT.LT(SMT.Int(0), self.expr_to_smt(expr))
        if not self.solver.is_valid(e_pos):
            eg = self.counter_example()
            self.err(
                expr,
                f"expected expression {expr} to always be positive. "
                f"It can be non positive when:\n  {eg}.",
            )

    def check_non_negative(self, expr):
        e_nn = SMT.LE(SMT.Int(0), self.expr_to_smt(expr))
        if not self.solver.is_valid(e_nn):
            eg = self.counter_example()
            self.err(
                expr,
                f"expected expression {expr} to always be non-negative. "
                f"It can be negative when:\n  {eg}.",
            )

    def check_call_shape_eqv(self, argshp, sigshp, node):
        assert len(argshp) == len(sigshp)
        eqv_dim = SMT.Bool(True)
        for a, s in zip(argshp, sigshp):
            eq_here = SMT.Equals(self.expr_to_smt(a), self.expr_to_smt(s))
            eqv_dim = SMT.And(eqv_dim, eq_here)
        if not self.solver.is_valid(eqv_dim):
            eg = self.counter_example()
            self.err(
                node,
                "type-shape of calling argument may not equal "
                "the required type-shape: "
                f"[{','.join(map(str,argshp))}] vs. "
                f"[{','.join(map(str,sigshp))}]."
                f" It could be non equal when:\n  {eg}",
            )

    def preprocess_stmts(self, body):
        for stmt in body:
            if isinstance(stmt, LoopIR.If):
                self.preprocess_stmts(stmt.body)
                self.preprocess_stmts(stmt.orelse)
            elif isinstance(stmt, LoopIR.For):
                self.preprocess_stmts(stmt.body)
            elif isinstance(stmt, LoopIR.Alloc):
                if stmt.type.is_tensor_or_window():
                    self.assume_tensor_strides(stmt, stmt.name, stmt.type.shape())
            elif isinstance(stmt, LoopIR.WindowStmt):
                # src_shape   = stmt.rhs.type.src_type.shape()
                w_idx = stmt.rhs.type.idx
                src_buf = stmt.rhs.type.src_buf
                dst_buf = stmt.name

                src_dims = [
                    d for d, w in enumerate(w_idx) if isinstance(w, LoopIR.Interval)
                ]
                for dst_dim, src_dim in enumerate(src_dims):
                    src = LoopIR.StrideExpr(src_buf, src_dim, T.stride, stmt.srcinfo)
                    dst = LoopIR.StrideExpr(dst_buf, dst_dim, T.stride, stmt.srcinfo)
                    eq = LoopIR.BinOp("==", src, dst, T.bool, stmt.srcinfo)
                    self.solver.add_assertion(self.expr_to_smt(lift_expr(eq)))
            else:
                pass

    def map_stmts(self, body, type_env):
        """
        Returns an effect for the argument `body`
        And also checks bounds/parallelism for any
        allocations/loops within `body`
        """
        assert len(body) > 0
        body_eff = eff_null(body[-1].srcinfo)

        for stmt in reversed(body):
            if isinstance(stmt, (LoopIR.Assign, LoopIR.Reduce)):
                loc = [lift_expr(idx) for idx in stmt.idx]
                rhs_eff = self.eff_e(stmt.rhs, type_env)
                if isinstance(stmt, LoopIR.Assign):
                    effects = eff_write(stmt.name, loc, stmt.srcinfo)
                else:  # Reduce
                    effects = eff_reduce(stmt.name, loc, stmt.srcinfo)

                stmt_eff = eff_concat(rhs_eff, effects)
                body_eff = eff_concat(stmt_eff, body_eff)

            elif isinstance(stmt, LoopIR.WriteConfig):
                rhs_eff = self.eff_e(stmt.rhs, type_env)
                if stmt.rhs.type.is_numeric():
                    rhs = E.Var(Sym("opaque_rhs"), stmt.rhs.type, stmt.rhs.srcinfo)
                else:
                    rhs = lift_expr(stmt.rhs)
                cw_eff = eff_config_write(stmt.config, stmt.field, rhs, stmt.srcinfo)
                stmt_eff = eff_concat(rhs_eff, cw_eff)
                body_eff = eff_concat(stmt_eff, body_eff)

            elif isinstance(stmt, LoopIR.For):
                self.push()

                def bd_pred(x, lo, hi, srcinfo):
                    x = E.Var(x, T.int, srcinfo)
                    lo = lift_expr(lo)
                    hi = lift_expr(hi)
                    return E.BinOp(
                        "and",
                        E.BinOp("<=", lo, x, T.bool, srcinfo),
                        E.BinOp("<", x, hi, T.bool, srcinfo),
                        T.bool,
                        srcinfo,
                    ), E.BinOp("<", lo, hi, T.bool, srcinfo)

                # Check if for-loop bound is non-negative
                # with the context, before adding assertion
                iters = LoopIR.BinOp("-", stmt.hi, stmt.lo, T.index, stmt.srcinfo)
                self.check_non_negative(lift_expr(iters))

                pred, config_pred = bd_pred(stmt.iter, stmt.lo, stmt.hi, stmt.srcinfo)
                self.solver.add_assertion(self.expr_to_smt(pred))

                child_eff = self.map_stmts(stmt.body, type_env)

                self.pop()

                stmt_eff = eff_bind(
                    stmt.iter, child_eff, pred=pred, config_pred=config_pred
                )

                body_eff = eff_concat(stmt_eff, body_eff)

            elif isinstance(stmt, LoopIR.If):
                # first, do the if-branch
                self.push()
                cond = lift_expr(stmt.cond)
                self.solver.add_assertion(self.expr_to_smt(cond))
                body_effects = self.map_stmts(stmt.body, type_env)
                self.pop()

                body_effects = eff_filter(cond, body_effects)
                orelse_effects = eff_null(stmt.srcinfo)
                orelse = stmt.orelse

                # then the else-branch
                if len(stmt.orelse) > 0:
                    self.push()
                    neg_cond = cond.negate()
                    self.solver.add_assertion(self.expr_to_smt(neg_cond))
                    orelse_effects = self.map_stmts(stmt.orelse, type_env)
                    orelse_effects = eff_filter(cond.negate(), orelse_effects)
                    self.pop()

                stmt_eff = eff_union(body_effects, orelse_effects)
                body_eff = eff_concat(stmt_eff, body_eff)

            elif isinstance(stmt, LoopIR.Alloc):
                shape = [lift_expr(s) for s in stmt.type.shape()]
                # check that all sizes are positive
                for s in shape:
                    self.check_pos_size(s)
                # check that all accesses are in bounds
                self.check_bounds(stmt.name, shape, body_eff)
                body_eff = eff_remove_buf(stmt.name, body_eff)

            elif isinstance(stmt, LoopIR.Call):

                self.push()

                bind = dict()
                subst = dict()

                for sig, arg in zip(stmt.f.args, stmt.args):
                    # Add type assertion from the size signature
                    if isinstance(sig.type, T.Size):
                        pos_sz = SMT.LT(SMT.Int(0), self.sym_to_smt(sig.name))
                        self.solver.add_assertion(pos_sz)

                        # check the caller argument always be positive for sizes
                        e_arg = lift_expr(arg)
                        self.check_pos_size(e_arg)

                    # Add type assertion from the caller types
                    if arg.type.is_tensor_or_window() and not arg.type.is_win():
                        self.assume_tensor_strides(sig, sig.name, arg.type.shape())

                    # bind potential window-expression
                    subst[sig.name] = arg

                    if sig.type.is_numeric():
                        if isinstance(arg, LoopIR.Read):
                            bind[sig.name] = arg.name

                        # need to check that the argument shape
                        # has all positive dimensions
                        arg_shape = [lift_expr(s) for s in arg.type.shape()]
                        for e in arg_shape:
                            self.check_pos_size(e)
                        # also, need to check that the argument shape
                        # is exactly the shape specified in the signature
                        sig_shape = [
                            lift_expr(loopir_subst(s, subst)) for s in sig.type.shape()
                        ]
                        self.check_call_shape_eqv(arg_shape, sig_shape, arg)

                    else:
                        bind[sig.name] = lift_expr(arg)

                # map body of the subprocedure
                self.preprocess_stmts(stmt.f.body)
                eff = self.map_stmts(stmt.f.body, self.rec_proc_types(stmt.f))
                eff = eff.subst(bind)

                # translate effects occuring on windowed arguments
                for sig, arg in zip(stmt.f.args, stmt.args):
                    if sig.type.is_numeric():
                        if isinstance(arg.type, T.Window):
                            eff = self.translate_eff(eff, sig.name, arg.type, type_env)

                # Check that asserts are correct
                for p in stmt.f.preds:
                    p_subst = loopir_subst(p, subst)
                    smt_pred = self.expr_to_smt(lift_expr(p_subst))
                    if not self.solver.is_valid(smt_pred):
                        eg = self.counter_example()
                        self.err(
                            stmt,
                            f"Could not verify assertion {p} in "
                            f"{stmt.f.name} at {p.srcinfo}."
                            f" Assertion is false when:\n  {eg}",
                        )

                self.pop()

                body_eff = eff_concat(eff, body_eff)

            elif isinstance(stmt, (LoopIR.Pass, LoopIR.WindowStmt)):
                pass

            else:
                assert False, "bad case!!"

        return body_eff  # Returns union of all effects

    # extract effects from this expression; return E.effect
    def eff_e(self, e, type_env):
        if isinstance(e, LoopIR.Read):
            if e.type.is_numeric():
                # we may assume that we're not in a call-argument position
                assert e.type.is_real_scalar()
                loc = [lift_expr(idx) for idx in e.idx]
                eff = eff_read(e.name, loc, e.srcinfo)

                # x[...], x
                buf_typ = type_env[e.name]
                if isinstance(buf_typ, T.Window):
                    eff = self.translate_eff(eff, e.name, buf_typ, type_env)

                return eff
            else:
                return eff_null(e.srcinfo)
        elif isinstance(e, LoopIR.BinOp):
            return eff_concat(
                self.eff_e(e.lhs, type_env),
                self.eff_e(e.rhs, type_env),
                srcinfo=e.srcinfo,
            )
        elif isinstance(e, LoopIR.USub):
            return self.eff_e(e.arg, type_env)
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

    def translate_eff(self, eff, buf_name, win_typ, type_env):
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
                buf = typ.src_buf
                idx = typ.idx
                typ = type_env[buf]
                loc_i = 0
                new_loc = []
                for w_acc in idx:
                    if isinstance(w_acc, LoopIR.Point):
                        new_loc.append(lift_expr(w_acc.pt))
                    elif isinstance(w_acc, LoopIR.Interval):
                        j = E.BinOp(
                            "+",
                            loc[loc_i],
                            lift_expr(w_acc.lo),
                            T.index,
                            w_acc.lo.srcinfo,
                        )
                        new_loc.append(j)
                        loc_i += 1
                assert loc_i == len(loc)
                loc = new_loc

            return E.effset(buf, loc, es.names, es.pred, es.srcinfo)

        return E.effect(
            [translate_set(es) for es in eff.reads],
            [translate_set(es) for es in eff.writes],
            [translate_set(es) for es in eff.reduces],
            eff.config_reads,
            eff.config_writes,
            eff.srcinfo,
        )
