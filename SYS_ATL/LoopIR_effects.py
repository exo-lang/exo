from .LoopIR import LoopIR, T, Effects
from .prelude import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Printing Functions

op_prec = {
    "ternary": 5,
    #
    "or":     10,
    #
    "and":    20,
    #
    "<":      30,
    ">":      30,
    "<=":     30,
    ">=":     30,
    "==":     30,
    #
    "+":      40,
    "-":      40,
    #
    "*":      50,
    "/":      50,
    "%":      50,
    #
    # unary - 60
}

@extclass(Effects.Var)
@extclass(Effects.Not)
@extclass(Effects.Const)
@extclass(Effects.BinOp)
@extclass(Effects.Stride)
@extclass(Effects.Select)
def __str__(self):
    return _exprstr(self)
del __str__

def _exprstr(e, prec=0):
    if isinstance(e, Effects.Var):
        return str(e.name)
    elif isinstance(e, Effects.Not):
        return f"(not {e.arg})"
    elif isinstance(e, Effects.Const):
        return str(e.val)
    elif isinstance(e, Effects.BinOp):
        local_prec = op_prec[e.op]
        lhs = _exprstr(e.lhs, prec=local_prec)
        rhs = _exprstr(e.rhs, prec=local_prec + 1)
        if local_prec < prec:
            return f"({lhs} {e.op} {rhs})"
        else:
            return f"{lhs} {e.op} {rhs}"
    elif isinstance(e, Effects.Stride):
        return f"stride({e.name},{e.dim})"
    elif isinstance(e, Effects.Select):
        local_prec = op_prec["ternary"]
        cond = _exprstr(e.cond)
        tcase = _exprstr(e.tcase, prec=local_prec + 1)
        fcase = _exprstr(e.fcase, prec=local_prec + 1)
        if local_prec < prec:
            return f"(({cond})? {tcase} : {fcase})"
        else:
            return f"({cond})? {tcase} : {fcase}"
    elif isinstance(e, Effects.ConfigField):
        return f"{e.config.name()}.{e.field}"
    else:
        assert False, "bad case"


@extclass(Effects.effect)
def __str__(self):
    return _effect_as_str(self)
del __str__

def _effect_as_str(e):
    assert isinstance(e, Effects.effect)

    def name(sym):
        return str(sym)

    def esstr(es, tab="  "):
        lines = []
        buf = name(es.buffer)
        loc = "(" + ','.join([str(l) for l in es.loc]) + ")"
        if len(es.names) == 0:
            names = ""
        else:
            names = f"for ({','.join([name(n) for n in es.names])}) in Z"

        if es.pred is None:
            lines.append(f"{tab}{{ {buf} : {loc} {names} }}")
        else:
            lines.append(f"{tab}{{ {buf} : {loc} {names} if")
            tab += "  "
            pred = str(es.pred)
            lines.append(f"{tab}{pred} }}")

        return '\n'.join(lines)

    def cestr(ce, tab="  "):
        val, pred = "",""
        if ce.value:
            val = f" = {ce.value}"
        if ce.pred:
            pred = f" if {ce.pred}"
        return f"{ce.config.name()}.{ce.field}{val}{pred}"

    eff_str = ""
    if len(e.reads) > 0:
        eff_str += "Reads:\n"
        eff_str += '\n'.join([esstr(es) for es in e.reads])
        eff_str += "\n"
    if len(e.writes) > 0:
        eff_str += f"Writes:\n  "
        eff_str += '\n'.join([esstr(es) for es in e.writes])
        eff_str += "\n"
    if len(e.reduces) > 0:
        eff_str += f"Reduces:\n  "
        eff_str += '\n'.join([esstr(es) for es in e.reduces])
        eff_str += "\n"
    if len(e.config_reads) > 0:
        eff_str += f"Config Reads:\n"
        eff_str += '\n'.join([cestr(ce) for ce in e.config_reads])
        eff_str += "\n"
    if len(e.config_writes) > 0:
        eff_str += f"Config Writes:\n"
        eff_str += '\n'.join([cestr(ce) for ce in e.config_writes])
        eff_str += "\n"

    return eff_str

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Substitution of Effect Variables in an effect

@extclass(Effects.expr)
def negate(self):
    return negate_expr(self)
del negate

def negate_expr(e):
    Tbool = T.bool
    assert e.type == Tbool, "can only negate predicates"
    if isinstance(e, Effects.Const):
        return Effects.Const(not e.val, e.type, e.srcinfo)
    elif isinstance(e, Effects.Var) or isinstance(e, Effects.ConfigField):
        return Effects.Not(e, e.type, e.srcinfo)
    elif isinstance(e, Effects.Not):
        return e.arg
    elif isinstance(e, Effects.BinOp):
        def change_op(op, lhs=e.lhs, rhs=e.rhs):
            return Effects.BinOp(op, lhs, rhs, e.type, e.srcinfo)

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
                l = Effects.BinOp("and", e.lhs, negate_expr(e.rhs),
                                  Tbool, e.srcinfo)
                r = Effects.BinOp("and", negate_expr(e.lhs), e.rhs,
                                  Tbool, e.srcinfo)

                return Effects.BinOp("or", l, r, Tbool, e.srcinfo)
            elif e.lhs.type.is_indexable() and e.rhs.type.is_indexable():
                return Effects.BinOp("or", change_op("<"), change_op(">"),
                                     Tbool, e.srcinfo)
            else:
                assert False, "TODO: add != support explicitly..."
    elif isinstance(e, Effects.Select):
        return Effects.Select(e.cond, negate_expr(e.tcase),
                              negate_expr(e.fcase),
                              e.type, e.srcinfo)
    assert False, "bad case"

@extclass(Effects.effect)
@extclass(Effects.effset)
@extclass(Effects.expr)
def subst(self, env):
    return eff_subst(env, self)

del subst


def eff_subst(env, eff):
    if isinstance(eff, Effects.effset):
        assert all(nm not in env for nm in eff.names)
        buf = env[eff.buffer] if eff.buffer in env else eff.buffer
        pred = eff_subst(env, eff.pred) if eff.pred else None
        return Effects.effset(buf,
                              [eff_subst(env, e) for e in eff.loc],
                              eff.names,
                              pred,
                              eff.srcinfo)
    elif isinstance(eff, Effects.config_eff):
        value = eff_subst(env, eff.value) if eff.value else None
        pred = eff_subst(env, eff.pred) if eff.pred else None
        return Effects.config_eff(eff.config, eff.field,
                                  value, pred, eff.srcinfo)
    elif isinstance(eff, Effects.Var):
        return env[eff.name] if eff.name in env else eff
    elif isinstance(eff, Effects.Not):
        return Effects.Not(eff_subst(env, eff.arg), eff.type, eff.srcinfo)
    elif isinstance(eff, Effects.Const):
        return eff
    elif isinstance(eff, Effects.BinOp):
        return Effects.BinOp(eff.op, eff_subst(env, eff.lhs),
                             eff_subst(env, eff.rhs),
                             eff.type, eff.srcinfo)
    elif isinstance(eff, Effects.Stride):
        name = env[eff.name] if eff.name in env else eff
        return Effects.Stride(name, eff.dim, eff.type, eff.srcinfo)
    elif isinstance(eff, Effects.Select):
        return Effects.Select(eff_subst(env, eff.cond),
                              eff_subst(env, eff.tcase),
                              eff_subst(env, eff.fcase),
                              eff.type, eff.srcinfo)
    elif isinstance(eff, Effects.ConfigField):
        return eff
    elif isinstance(eff, Effects.effect):
        return Effects.effect([eff_subst(env, es) for es in eff.reads],
                              [eff_subst(env, es) for es in eff.writes],
                              [eff_subst(env, es) for es in eff.reduces],
                              [eff_subst(env, ce)
                               for ce in eff.config_reads],
                              [eff_subst(env, ce)
                               for ce in eff.config_writes],
                              eff.srcinfo)
    else:
        assert False, f"bad case: {type(eff)}"

@extclass(Effects.effect)
@extclass(Effects.effset)
@extclass(Effects.expr)
def config_subst(self, env):
    return _subcfg(env, self)
del config_subst

def _subcfg(env, eff):
    if isinstance(eff, Effects.effset):
        return Effects.effset(eff.buffer,
                              [_subcfg(env, e) for e in eff.loc],
                              eff.names,
                              _subcfg(env, eff.pred) if eff.pred else None,
                              eff.srcinfo)
    elif isinstance(eff, Effects.config_eff):
        value = _subcfg(env, eff.value) if eff.value else None
        pred = _subcfg(env, eff.pred) if eff.pred else None
        return Effects.config_eff(eff.config, eff.field,
                                  value, pred, eff.srcinfo)
    elif isinstance(eff, (Effects.Var, Effects.Const)):
        return eff
    elif isinstance(eff, Effects.Not):
        return Effects.Not(_subcfg(env, eff.arg), eff.type, eff.srcinfo)
    elif isinstance(eff, Effects.BinOp):
        return Effects.BinOp(eff.op, _subcfg(env, eff.lhs),
                             _subcfg(env, eff.rhs),
                             eff.type, eff.srcinfo)
    elif isinstance(eff, Effects.Stride):
        return eff
    elif isinstance(eff, Effects.Select):
        return Effects.Select(_subcfg(env, eff.cond),
                              _subcfg(env, eff.tcase),
                              _subcfg(env, eff.fcase),
                              eff.type, eff.srcinfo)
    elif isinstance(eff, Effects.ConfigField):
        if (eff.config, eff.field) in env:
            return env[(eff.config, eff.field)]
        else:
            return eff
    elif isinstance(eff, Effects.effect):
        return Effects.effect([_subcfg(env, es) for es in eff.reads],
                              [_subcfg(env, es) for es in eff.writes],
                              [_subcfg(env, es) for es in eff.reduces],
                              [_subcfg(env, ce) for ce in eff.config_reads],
                              [_subcfg(env, ce) for ce in eff.config_writes],
                              eff.srcinfo)
    else:
        assert False, f"bad case: {type(eff)}"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Querying of the effect of a block of already annotated statements

def get_effect_of_stmts(body):
    assert len(body) > 0
    eff   = eff_null(body[0].srcinfo)
    for s in reversed(body):
        if isinstance(s, LoopIR.Alloc):
            eff = eff_remove_buf(s.name, eff)
        elif s.eff is not None:
            eff = eff_concat(s.eff, eff)
    return eff

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Construction/Composition Functions

def eff_null(srcinfo = null_srcinfo()):
    return Effects.effect( [], [], [], [], [], srcinfo )

def eff_read(buf, loc, srcinfo = null_srcinfo()):
    read = Effects.effset(buf, loc, [], None, srcinfo)
    return Effects.effect( [read], [], [], [], [], srcinfo )

def eff_write(buf, loc, srcinfo = null_srcinfo()):
    write = Effects.effset(buf, loc, [], None, srcinfo)
    return Effects.effect( [], [write], [], [], [], srcinfo )

def eff_reduce(buf, loc, srcinfo = null_srcinfo()):
    reduction = Effects.effset(buf, loc, [], None, srcinfo)
    return Effects.effect( [], [], [reduction], [], [], srcinfo )

def eff_config_read(config, field, srcinfo = null_srcinfo()):
    read = Effects.config_eff(config, field, None, None, srcinfo)
    return Effects.effect( [], [], [], [read], [], srcinfo )

def eff_config_write(config, field, value, srcinfo = null_srcinfo()):
    write = Effects.config_eff(config, field, value, None, srcinfo)
    return Effects.effect( [], [], [], [], [write], srcinfo )

def _and_preds(a,b):
    return (a   if b is None else
            b   if a is None else
            Effects.BinOp("and", a, b, T.bool, a.srcinfo))

def _or_preds(a,b):
    return (None    if a is None or b is None else
            Effects.BinOp("or", a, b, T.bool, a.srcinfo))

def eff_union(e1, e2, srcinfo=None):
    srcinfo = srcinfo or e1.srcinfo

    return Effects.effect( e1.reads + e2.reads,
                           e1.writes + e2.writes,
                           e1.reduces + e2.reduces,
                           e1.config_reads + e2.config_reads,
                           e1.config_writes + e2.config_writes,
                           srcinfo )

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
            old_val = Effects.ConfigField( ce.config, ce.field, T.bool, srcinfo )
            return Effects.Select(ce.pred, ce.value, old_val, T.bool, srcinfo )

    # substitute on the basis of writes in the first effect
    env = { (ce.config,ce.field) : write_val(ce)
            for ce in e1.config_writes }
    e2 = e2.config_subst(env)

    # Step 2: merge writes from the two effects to the same field
    def merge_writes(config_writes_1, config_writes_2):
        cws1    = { (w.config,w.field) : w for w in config_writes_1 }
        cws2    = { (w.config,w.field) : w for w in config_writes_2 }
        overlap = set(cws1.keys()).intersection(set(cws2.keys()))

        def merge(w1, w2):
            # in the case of the second write being unconditional
            if w2.pred is None:
                return w2
            else:
                typ = w1.config.lookup(w1.field)[1]
                assert typ == w2.config.lookup(w2.field)[1]

                pred    = _or_preds(w1.pred, w2.pred)
                val     = Effects.Select(w2.pred, w2.value, w1.value,
                                         typ, w2.srcinfo)

                return Effects.config_eff(w1.config, w1.field,
                                          val, pred, w2.srcinfo)

        return ([ cws1[w] for w in cws1 if w not in overlap ]+
                [ cws2[w] for w in cws2 if w not in overlap ]+
                [ merge(cws1[key], cws2[key]) for key in overlap ])

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
                results.append(Effects.config_eff(read.config, read.field,
                                                  None, pred, read.srcinfo))

        return results


    def shadow_reads(writes, reads):

        def shadow_read_by_write(write, read):
            def boolop(op, lhs, rhs):
                return Effects.BinOp(op, lhs, rhs, T.bool, write.srcinfo)

            loc = []
            for l1,l2 in zip(write.loc, read.loc):
                loc += [boolop("==", l1, l2).negate()]

            # construct loc predicate
            if loc == []:
                loc_e = Effects.Const(False, T.bool, write.srcinfo)
            else:
                loc_e = loc[0]
            for l in loc[1:]:
                loc_e = boolop("or", loc_e, l)

            # loc_e /\ not write.pred
            if write.pred is None:
                pred = loc_e
            else:
                pred = boolop("or", loc_e, write.pred.negate())

            pred = _and_preds(read.pred, pred)

            read = Effects.effset(read.buffer, read.loc,
                                  read.names + write.names,
                                  pred, read.srcinfo)

            return read

        def shadow_read_by_writes(writes, read):
            for w in writes:
                # find that the write
                if w.buffer == read.buffer:
                    read = shadow_read_by_write(w, read)

            return read

        return [ shadow_read_by_writes(writes, r) for r in reads ]




    config_reads    = (e1.config_reads +
                       shadow_config_reads(e1.config_writes, e2.config_reads))
    config_writes   = merge_writes(e1.config_writes, e2.config_writes)

    # TODO: Fix shadow_reads by introducing Exists
    #reads           = (e1.reads + shadow_reads(e1.writes, e2.reads))
    reads = e1.reads + e2.reads

    return Effects.effect( reads,
                           e1.writes + e2.writes,
                           e1.reduces + e2.reduces,
                           config_reads,
                           config_writes,
                           srcinfo )

def eff_remove_buf(buf, e):
    return Effects.effect( [ es for es in e.reads   if es.buffer != buf ],
                           [ es for es in e.writes  if es.buffer != buf ],
                           [ es for es in e.reduces if es.buffer != buf ],
                           e.config_reads,
                           e.config_writes,
                           e.srcinfo )

# handle conditional
def eff_filter(pred, e):
    def filter_es(es):
        return Effects.effset(es.buffer, es.loc, es.names,
                              _and_preds(pred,es.pred), es.srcinfo)

    def filter_ce(ce):
        return Effects.config_eff(ce.config, ce.field,
                                  ce.value, _and_preds(pred,ce.pred),
                                  ce.srcinfo)

    return Effects.effect( [ filter_es(es) for es in e.reads ],
                           [ filter_es(es) for es in e.writes ],
                           [ filter_es(es) for es in e.reduces ],
                           [ filter_ce(ce) for ce in e.config_reads ],
                           [ filter_ce(ce) for ce in e.config_writes ],
                           e.srcinfo )

# handle for loop
def eff_bind(bind_name, e, pred=None, config_pred=None):
    assert isinstance(bind_name, Sym)
    def bind_es(es):
        return Effects.effset(es.buffer, es.loc, [bind_name]+es.names,
                              _and_preds(pred,es.pred), es.srcinfo)
    def filter_ce(ce):
        return Effects.config_eff(ce.config, ce.field,
                                  ce.value, _and_preds(pred, config_pred),
                                  ce.srcinfo)

    return Effects.effect( [ bind_es(es) for es in e.reads ],
                           [ bind_es(es) for es in e.writes ],
                           [ bind_es(es) for es in e.reduces ],
                           [ filter_ce(ce) for ce in e.config_reads ],
                           [ filter_ce(ce) for ce in e.config_writes ],
                           e.srcinfo )

"""
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Free Variable check

@extclass(Effects.effect)
@extclass(Effects.effset)
@extclass(Effects.config_eff)
@extclass(Effects.expr)
def has_FV(self, x):
    return _is_FV(x, self)
del has_FV

def _is_FV(x, eff):
    if isinstance(eff, Effects.effset):
        if x in eff.names:
            return False
        return ( any( _is_FV(x, e) for e in eff.loc ) or
                 (eff.pred is not None and is_FV(x, eff.pred)) )
    elif isinstance(eff, Effects.config_eff):
        return ( (eff.value is not None and _is_FV(x, eff.value)) or
                 (eff.pred  is not None and _is_FV(x, eff.pred)) )
    elif isinstance(eff, Effects.Var):
        return x == eff.name
    elif isinstance(eff, Effects.Const):
        return False
    elif isinstance(eff, Effects.Not):
        return _is_FV(x, eff.arg)
    elif isinstance(eff, Effects.BinOp):
        return _is_FV(x, eff.lhs) or _is_FV(x, eff.rhs)
    elif isinstance(eff, Effects.Stride):
        return False
    elif isinstance(eff, Effects.Select):
        return (_is_FV(x, eff.cond) or
                _is_FV(x, eff.tcase) or
                _is_FV(x, eff.fcase))
    elif isinstance(eff, Effects.ConfigField):
        return False
    elif isinstance(eff, Effects.effect):
        return ( any( _is_FV(x, es) for es in eff.reads ) or
                 any( _is_FV(x, es) for es in eff.writes ) or
                 any( _is_FV(x, es) for es in eff.reduces ) or
                 any( _is_FV(x, ce) for ce in eff.config_reads ) or
                 any( _is_FV(x, ce) for ce in eff.config_writes ) )
    else:
        assert False, f"bad case: {type(eff)}"
"""
