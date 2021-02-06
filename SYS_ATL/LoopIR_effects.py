from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T

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

    -- this corresponds to `{ loc for names in int if pred }`
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
    'type':         T.is_type,
    'binop':        lambda x: x in bin_ops,
    'srcinfo':      lambda x: type(x) is SrcInfo,
})


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
# Helper Functions

def eff_null(srcinfo = None):
    srcinfo = null_srcinfo()


def eff_union(e1, e2, srcinfo=None):
    srcinfo = srcinfo or e1.srcinfo

    return Effects.effect( e1.reads + e2.reads,
                           e1.writes + e2.writes,
                           e1.reduces + e2.reduces,
                           srcinfo )

def eff_filter(pred, e):
    def filter_es(es):
        preds = Effects.BinOp("and", pred, es.pred, T.bool, pred.srcinfo)
        return Effects.effset(es.buffer, es.loc, es.names, preds, es.srcinfo)

    return Effects.effect( [ filter_es(es) for es in e.reads ],
                           [ filter_es(es) for es in e.writes ]
                           [ filter_es(es) for es in e.reduces ]
                           e.srcinfo )

def eff_bind(bind_name, e, pred=None):
    assert type(bind_name) is Sym
    def bind_es(es):
        if pred is None:
            preds = es.pred
        else:
            preds = Effects.BinOp("and", pred, es.pred, T.bool, pred.srcinfo)
        return Effects.effset(es.buffer, es.loc, [bind_name]+es.names,
                              preds, es.srcinfo)

    return Effects.effect( [ bind_es(es) for es in e.reads ],
                           [ bind_es(es) for es in e.writes ]
                           [ bind_es(es) for es in e.reduces ]
                           e.srcinfo )
