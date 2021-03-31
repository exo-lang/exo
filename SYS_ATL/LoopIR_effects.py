from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

#from .LoopIR import T
from . import LoopIR

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Effect grammar

front_ops = {
    "+":    True,
    "-":    True,
    "*":    True,
    "/":    True,
    "%":    True,
    #
    "<":    True,
    ">":    True,
    "<=":   True,
    ">=":   True,
    "==":   True,
    #
    "and":  True,
    "or":   True,
}

Effects = ADT("""
module Effects {
    effect  = ( effset*     reads,
                effset*     writes,
                effset*     reduces,
                srcinfo     srcinfo )

    -- JRK: the notation of this comprehension is confusing - maybe just use math:
    -- this corresponds to `{ buffer : loc for *names in int if pred }`
    effset  = ( sym         buffer,
                expr*       loc,    -- e.g. reading at (i+1,j+1)
                sym*        names,
                expr?       pred,
                srcinfo     srcinfo )

    expr    = Var( sym name )
            | Const( object val )
            | BinOp( binop op, expr lhs, expr rhs )
            attributes( type type, srcinfo srcinfo )

} """, {
    'sym':          lambda x: type(x) is Sym,
    'type':         lambda x: LoopIR.T.is_type(x),
    'binop':        lambda x: x in front_ops,
    'srcinfo':      lambda x: type(x) is SrcInfo,
})

#
#   for i in par(0, n):      # eff { WRITE, x : (i,j) for i,j in Z
#                            #            if 0 <= i < n and 0 <= j < n }
#       for j in par(0, n):  # eff { WRITE, x : (i,j) for j in Z
#                            #            if 0 <= j < n }
#           x[i,j] = ...     # eff WRITE, x : (i,j)
#
#



# Unused Proposal

# Effects = ADT("""
# module Effects {
#     effect  = PrimEff( mode mode, expr* loc )
#             | Guard( expr cond, effect* effs )
#             | ForAll( sym name, expr cond, effect* effs )
#             attributes( srcinfo srcinfo )
#
#     mode = READ() | WRITE() | REDUCE()
#
#     expr    = Var( sym name )
#             | Const( object val )
#             | BinOp( binop op, expr lhs, expr rhs )
#             attributes( type type, srcinfo srcinfo )
#
# } """, {
#     'sym':          lambda x: type(x) is Sym,
#     'type':         T.is_type,
#     'binop':        lambda x: x in bin_ops,
#     'srcinfo':      lambda x: type(x) is SrcInfo,
# })

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Printing Functions

op_prec = {
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

@extclass(Effects.effect)
def __str__(self):
    return effect_as_str(self)
del __str__

def effect_as_str(e):
    assert type(e) is Effects.effect

    def name(sym):
        return str(sym)

    def exprstr(e, prec=0):
        if type(e) is Effects.Var:
            return name(e.name)
        elif type(e) is Effects.Const:
            return str(e.val)
        elif type(e) is Effects.BinOp:
            local_prec  = op_prec[e.op]
            lhs         = exprstr(e.lhs, prec=local_prec)
            rhs         = exprstr(e.rhs, prec=local_prec+1)
            if local_prec < prec:
                return f"({lhs} {e.op} {rhs})"
            else:
                return f"{lhs} {e.op} {rhs}"

    def esstr(es, tab="  "):
        lines = []
        buf = name(es.buffer)
        loc = "(" + ','.join([exprstr(l) for l in es.loc]) + ")"
        if len(es.names) == 0:
            names = ""
        else:
            names = f"for ({','.join([name(n) for n in es.names])}) in Z"

        if es.pred is None:
            lines.append(f"{tab}{{ {buf} : {loc} {names} }}")
        else:
            lines.append(f"{tab}{{ {buf} : {loc} {names} if")
            tab += "  "
            pred = exprstr(es.pred)
            lines.append(f"{tab}{pred} }}")

        return '\n'.join(lines)

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

    return eff_str

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Substitution of Effect Variables in an effect

@extclass(Effects.effect)
@extclass(Effects.effset)
@extclass(Effects.expr)
def subst(self, env):
    return eff_subst(env, self)

del subst

def eff_subst(env, eff):
    if type(eff) is Effects.effset:
        assert all( nm not in env for nm in eff.names )
        buf  = env[eff.buffer] if eff.buffer in env else eff.buffer
        pred = eff_subst(env, eff.pred) if eff.pred else None
        return Effects.effset(buf,
                              [ eff_subst(env, e) for e in eff.loc ],
                              eff.names,
                              pred,
                              eff.srcinfo)
    elif type(eff) is Effects.Var:
        return env[eff.name] if eff.name in env else eff
    elif type(eff) is Effects.Const:
        return eff
    elif type(eff) is Effects.BinOp:
        return Effects.BinOp(eff.op, eff_subst(env, eff.lhs),
                                     eff_subst(env, eff.rhs),
                             eff.type, eff.srcinfo)
    elif type(eff) is Effects.effect:
        return Effects.effect( [eff_subst(env, es) for es in eff.reads],
                               [eff_subst(env, es) for es in eff.writes],
                               [eff_subst(env, es) for es in eff.reduces],
                               eff.srcinfo )
    else:
        assert False, f"bad case: {type(eff)}"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Construction/Composition Functions

def eff_null(srcinfo = None):
    return Effects.effect( [],
                           [],
                           [],
                           srcinfo )

def eff_union(e1, e2, srcinfo=None):
    srcinfo = srcinfo or e1.srcinfo

    return Effects.effect( e1.reads + e2.reads,
                           e1.writes + e2.writes,
                           e1.reduces + e2.reduces,
                           srcinfo )

def eff_remove_buf(buf, e):
    return Effects.effect( [ es for es in e.reads   if es.buffer != buf ],
                           [ es for es in e.writes  if es.buffer != buf ],
                           [ es for es in e.reduces if es.buffer != buf ],
                           e.srcinfo )

# handle conditional
def eff_filter(pred, e):
    def filter_es(es):
        if pred is None:
            preds = es.pred
        elif es.pred is None:
            preds = pred
        else:
            preds = Effects.BinOp("and", pred, es.pred,
                                  LoopIR.T.bool, pred.srcinfo)
        return Effects.effset(es.buffer, es.loc, es.names, preds, es.srcinfo)

    return Effects.effect( [ filter_es(es) for es in e.reads ],
                           [ filter_es(es) for es in e.writes ],
                           [ filter_es(es) for es in e.reduces ],
                           e.srcinfo )

# handle for loop
def eff_bind(bind_name, e, pred=None):
    assert type(bind_name) is Sym
    def bind_es(es):
        if pred is None:
            preds = es.pred
        elif es.pred is None:
            preds = pred
        else:
            preds = Effects.BinOp("and", pred, es.pred,
                                  LoopIR.T.bool, pred.srcinfo)
        return Effects.effset(es.buffer, es.loc, [bind_name]+es.names,
                              preds, es.srcinfo)

    return Effects.effect( [ bind_es(es) for es in e.reads ],
                           [ bind_es(es) for es in e.writes ],
                           [ bind_es(es) for es in e.reduces ],
                           e.srcinfo )
