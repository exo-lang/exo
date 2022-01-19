import attrs

from . import LoopIR
from .asts import Effects
from .prelude import *


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Substitution of Effect Variables in an effect

def eff_negate(e: Effects.expr):
    match e:
        case Effects.Const(val, _, _):
            return attrs.evolve(e, val=not val)
        case Effects.Var() | Effects.ConfigField():
            return Effects.Not(e, e.type, e.srcinfo)
        case Effects.Not(arg, _, _):
            return arg
        case Effects.Select(_, tcase, fcase, _, _):
            return attrs.evolve(e, tcase=eff_negate(tcase),
                                fcase=eff_negate(fcase))
        case Effects.BinOp('and', lhs, rhs, _, _):
            return attrs.evolve(e, op='or', lhs=eff_negate(lhs),
                                rhs=eff_negate(rhs))
        case Effects.BinOp('or', lhs, rhs, _, _):
            return attrs.evolve(e, op='and', lhs=eff_negate(lhs),
                                rhs=eff_negate(rhs))
        case Effects.BinOp(op='>'):
            return attrs.evolve(e, op='<=')
        case Effects.BinOp(op='<'):
            return attrs.evolve(e, op='>=')
        case Effects.BinOp(op='>='):
            return attrs.evolve(e, op='<')
        case Effects.BinOp(op='<='):
            return attrs.evolve(e, op='>')
        case Effects.BinOp('==', lhs, rhs, _, _):
            if lhs.type == LoopIR.T.bool and rhs.type == LoopIR.T.bool:
                # L == R ~> (L && !R) || (!L && R) <~> L != R
                return attrs.evolve(
                    e, op='or',
                    lhs=attrs.evolve(e, op='and', rhs=eff_negate(rhs)),
                    rhs=attrs.evolve(e, op='and', lhs=eff_negate(lhs)))
            elif lhs.type.is_indexable() and rhs.type.is_indexable():
                # L == R ~> (L < R) || (L > R) <~> L != R
                return attrs.evolve(
                    e, op='or',
                    lhs=attrs.evolve(e, op='<'),
                    rhs=attrs.evolve(e, op='>'))
            raise NotImplementedError('add != support explicitly...')
    assert False, f'bad case: {type(e).__name__}'


def eff_subst(env, eff):
    match eff:
        case None:
            return None
        case Effects.effset(buffer, loc, names, pred, _):
            assert all(nm not in env for nm in names)
            return attrs.evolve(eff,
                                buffer=env.get(buffer, buffer),
                                loc=[eff_subst(env, e) for e in loc],
                                pred=eff_subst(env, pred))
        case Effects.config_eff(_, _, value, pred, _):
            return attrs.evolve(eff,
                                value=eff_subst(env, value),
                                pred=eff_subst(env, pred))
        case Effects.Var(name, _, _):
            return env.get(name, eff)
        case Effects.Not(arg, _, _):
            return attrs.evolve(eff, arg=eff_subst(env, arg))
        case Effects.Const() | Effects.ConfigField():
            return eff
        case Effects.BinOp(_, lhs, rhs, _, _):
            return attrs.evolve(eff,
                                lhs=eff_subst(env, lhs),
                                rhs=eff_subst(env, rhs))
        case Effects.Stride(name, _, _, _):
            return attrs.evolve(eff, name=env.get(name, name))
        case Effects.Select(cond, tcase, fcase, _, _):
            return attrs.evolve(eff,
                                cond=eff_subst(env, cond),
                                tcase=eff_subst(env, tcase),
                                fcase=eff_subst(env, fcase))
        case Effects.effect(reads, writes, reduces, config_reads,
                            config_writes, _):
            return attrs.evolve(
                eff,
                reads=[eff_subst(env, e) for e in reads],
                writes=[eff_subst(env, e) for e in writes],
                reduces=[eff_subst(env, e) for e in reduces],
                config_reads=[eff_subst(env, e) for e in config_reads],
                config_writes=[eff_subst(env, e) for e in config_writes]
            )
    assert False, f'bad case: {type(eff).__name__}'


def eff_config_subst(env, eff):
    match eff:
        case None:
            return None
        case Effects.effset(_, loc, _, pred, _):
            return attrs.evolve(eff,
                                loc=[eff_config_subst(env, e) for e in loc],
                                pred=eff_config_subst(env, pred))
        case Effects.config_eff(_, _, value, pred, _):
            return attrs.evolve(eff,
                                value=eff_config_subst(env, value),
                                pred=eff_config_subst(env, pred))
        case Effects.Var() | Effects.Const() | Effects.Stride():
            return eff
        case Effects.Not(arg, _, _):
            return attrs.evolve(eff, arg=eff_config_subst(env, arg))
        case Effects.BinOp(_, lhs, rhs, _, _):
            return attrs.evolve(eff,
                                lhs=eff_config_subst(env, lhs),
                                rhs=eff_config_subst(env, rhs))
        case Effects.Select(cond, tcase, fcase, _, _):
            return attrs.evolve(eff,
                                cond=eff_config_subst(env, cond),
                                tcase=eff_config_subst(env, tcase),
                                fcase=eff_config_subst(env, fcase))
        case Effects.ConfigField(config, field, _, _):
            return env.get((config, field), eff)
        case Effects.effect(reads, writes, reduces, config_reads,
                            config_writes, _):
            return attrs.evolve(
                eff,
                reads=[eff_config_subst(env, e) for e in reads],
                writes=[eff_config_subst(env, e) for e in writes],
                reduces=[eff_config_subst(env, e) for e in reduces],
                config_reads=[eff_config_subst(env, e) for e in config_reads],
                config_writes=[eff_config_subst(env, e) for e in config_writes]
            )

    assert False, f"bad case: {type(eff).__name__}"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Querying of the effect of a block of already annotated statements

def get_effect_of_stmts(body):
    assert len(body) > 0
    eff = eff_null(body[0].srcinfo)
    for s in reversed(body):
        if isinstance(s, LoopIR.LoopIR.Alloc):
            eff = eff_remove_buf(s.name, eff)
        elif s.eff is not None:
            eff = eff_concat(s.eff, eff)
    return eff


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Construction/Composition Functions

def eff_null(srcinfo=null_srcinfo()):
    return Effects.effect([], [], [], [], [], srcinfo)


def eff_read(buf, loc, srcinfo=null_srcinfo()):
    read = Effects.effset(buf, loc, [], None, srcinfo)
    return Effects.effect([read], [], [], [], [], srcinfo)


def eff_write(buf, loc, srcinfo=null_srcinfo()):
    write = Effects.effset(buf, loc, [], None, srcinfo)
    return Effects.effect([], [write], [], [], [], srcinfo)


def eff_reduce(buf, loc, srcinfo=null_srcinfo()):
    reduction = Effects.effset(buf, loc, [], None, srcinfo)
    return Effects.effect([], [], [reduction], [], [], srcinfo)


def eff_config_read(config, field, srcinfo=null_srcinfo()):
    read = Effects.config_eff(config, field, None, None, srcinfo)
    return Effects.effect([], [], [], [read], [], srcinfo)


def eff_config_write(config, field, value, srcinfo=null_srcinfo()):
    write = Effects.config_eff(config, field, value, None, srcinfo)
    return Effects.effect([], [], [], [], [write], srcinfo)


def _and_preds(a, b):
    return (a if b is None else
            b if a is None else
            Effects.BinOp("and", a, b, LoopIR.T.bool, a.srcinfo))


def _or_preds(a, b):
    return (None if a is None or b is None else
            Effects.BinOp("or", a, b, LoopIR.T.bool, a.srcinfo))


def eff_union(e1, e2, srcinfo=None):
    srcinfo = srcinfo or e1.srcinfo

    return Effects.effect(e1.reads + e2.reads,
                          e1.writes + e2.writes,
                          e1.reduces + e2.reduces,
                          e1.config_reads + e2.config_reads,
                          e1.config_writes + e2.config_writes,
                          srcinfo)


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
            old_val = Effects.ConfigField(ce.config, ce.field, LoopIR.T.bool,
                                          srcinfo)
            return Effects.Select(ce.pred, ce.value, old_val, LoopIR.T.bool,
                                  srcinfo)

    # substitute on the basis of writes in the first effect
    env = {(ce.config, ce.field): write_val(ce)
           for ce in e1.config_writes}
    e2 = eff_config_subst(env, e2)

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
                typ = w1.config.lookup(w1.field)
                assert typ == w2.config.lookup(w2.field)

                pred = _or_preds(w1.pred, w2.pred)
                val = Effects.Select(w2.pred, w2.value, w1.value,
                                     typ, w2.srcinfo)

                return Effects.config_eff(w1.config, w1.field,
                                          val, pred, w2.srcinfo)

        return ([cws1[w] for w in cws1 if w not in overlap] +
                [cws2[w] for w in cws2 if w not in overlap] +
                [merge(cws1[key], cws2[key]) for key in overlap])

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
                pred = _and_preds(read.pred, eff_negate(write.pred))
                results.append(Effects.config_eff(read.config, read.field,
                                                  None, pred, read.srcinfo))

        return results

    def shadow_reads(writes, reads):

        def shadow_read_by_write(write, read):
            def boolop(op, lhs, rhs):
                return Effects.BinOp(op, lhs, rhs, LoopIR.T.bool, write.srcinfo)

            loc = []
            for l1, l2 in zip(write.loc, read.loc):
                loc += [eff_negate(boolop("==", l1, l2))]

            # construct loc predicate
            loc_e = (loc[0] if loc
                     else Effects.Const(False, LoopIR.T.bool, write.srcinfo))
            for l in loc[1:]:
                loc_e = boolop("or", loc_e, l)

            # loc_e /\ not write.pred
            if write.pred is None:
                pred = loc_e
            else:
                pred = boolop("or", loc_e, eff_negate(write.pred))

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

        return [shadow_read_by_writes(writes, r) for r in reads]

    config_reads = (e1.config_reads +
                    shadow_config_reads(e1.config_writes, e2.config_reads))
    config_writes = merge_writes(e1.config_writes, e2.config_writes)

    # TODO: Fix shadow_reads by introducing Exists
    # reads           = (e1.reads + shadow_reads(e1.writes, e2.reads))
    reads = e1.reads + e2.reads

    return Effects.effect(reads,
                          e1.writes + e2.writes,
                          e1.reduces + e2.reduces,
                          config_reads,
                          config_writes,
                          srcinfo)


def eff_remove_buf(buf, e):
    return Effects.effect([es for es in e.reads if es.buffer != buf],
                          [es for es in e.writes if es.buffer != buf],
                          [es for es in e.reduces if es.buffer != buf],
                          e.config_reads,
                          e.config_writes,
                          e.srcinfo)


# handle conditional
def eff_filter(pred, e):
    def filter_es(es):
        return Effects.effset(es.buffer, es.loc, es.names,
                              _and_preds(pred, es.pred), es.srcinfo)

    def filter_ce(ce):
        return Effects.config_eff(ce.config, ce.field,
                                  ce.value, _and_preds(pred, ce.pred),
                                  ce.srcinfo)

    return Effects.effect([filter_es(es) for es in e.reads],
                          [filter_es(es) for es in e.writes],
                          [filter_es(es) for es in e.reduces],
                          [filter_ce(ce) for ce in e.config_reads],
                          [filter_ce(ce) for ce in e.config_writes],
                          e.srcinfo)


# handle for loop
def eff_bind(bind_name, e, pred=None, config_pred=None):
    assert isinstance(bind_name, Sym)

    def bind_es(es):
        return Effects.effset(es.buffer, es.loc, [bind_name] + es.names,
                              _and_preds(pred, es.pred), es.srcinfo)

    def filter_ce(ce):
        return Effects.config_eff(ce.config, ce.field,
                                  ce.value, _and_preds(pred, config_pred),
                                  ce.srcinfo)

    return Effects.effect([bind_es(es) for es in e.reads],
                          [bind_es(es) for es in e.writes],
                          [bind_es(es) for es in e.reduces],
                          [filter_ce(ce) for ce in e.config_reads],
                          [filter_ce(ce) for ce in e.config_writes],
                          e.srcinfo)


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
