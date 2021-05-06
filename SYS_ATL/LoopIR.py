from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from .LoopIR_effects import Effects as E

from .memory import Memory

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Untyped AST

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

UAST = ADT("""
module UAST {
    proc    = ( name?           name,
                fnarg*          args,
                expr*           preds,
                stmt*           body,
                string?         instr,
                srcinfo         srcinfo )

    fnarg   = ( sym             name,
                type            type,
                mem?            mem,
                srcinfo         srcinfo )

    stmt    = Assign  ( sym name, expr* idx, expr rhs )
            | Reduce  ( sym name, expr* idx, expr rhs )
            | Pass    ()
            | If      ( expr cond, stmt* body,  stmt* orelse )
            | ForAll  ( sym iter,  expr cond,   stmt* body )
            | Alloc   ( sym name, type type, mem? mem )
            | Call    ( loopir_proc f, expr* args )
            attributes( srcinfo srcinfo )

    expr    = Read    ( sym name, expr* idx )
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            | WindowExpr( sym base, w_access* idx )
            | StrideAssert( sym name, int idx, int val )
            | ParRange( expr lo, expr hi ) -- only use for loop cond
            attributes( srcinfo srcinfo )

    w_access= Interval( expr? lo, expr? hi )
            | Point( expr pt )
            attributes( srcinfo srcinfo )

    type    = Num   ()
            | F32   ()
            | F64   ()
            | INT8  ()
            | Bool  ()
            | Int   ()
            | Size  ()
            | Index ()
            | Tensor( expr *hi, bool is_window, type type )
} """, {
    'name':         is_valid_name,
    'sym':          lambda x: type(x) is Sym,
    'mem':          lambda x: isinstance(x, Memory),
    'loopir_proc':  lambda x: type(x) is LoopIR.proc,
    'op':           lambda x: x in front_ops,
    'srcinfo':      lambda x: type(x) is SrcInfo
})

ADTmemo(UAST, ['Num', 'F32', 'F64', 'INT8', 'Bool', 'Int', 'Size', 'Index'], {
})


@extclass(UAST.Tensor)
@extclass(UAST.Num)
@extclass(UAST.F32)
@extclass(UAST.F64)
@extclass(UAST.INT8)
def shape(t):
    shp = t.hi if type(t) is UAST.Tensor else []
    return shp
del shape

@extclass(UAST.type)
def basetype(t):
    if type(t) is UAST.Tensor:
        t = t.type
    return t
del basetype

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Pattern AST
#   - used to specify pattern-matches

PAST = ADT("""
module PAST {

    stmt    = Assign  ( name name, expr* idx, expr rhs )
            | Reduce  ( name name, expr* idx, expr rhs )
            | Pass    ()
            | If      ( expr cond, stmt* body,  stmt* orelse )
            | ForAll  ( name iter, expr hi,     stmt* body )
            | Alloc   ( name name ) -- may want to add type & mem back in?
            | Call    ( name f, expr* args )
            | S_Hole  ()
            attributes( srcinfo srcinfo )

    expr    = Read    ( name name, expr* idx )
            | E_Hole  ()
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            attributes( srcinfo srcinfo )

} """, {
    'name':         lambda x: x == '_' or is_valid_name(x),
    'op':           lambda x: x in front_ops,
    'srcinfo':      lambda x: type(x) is SrcInfo,
})

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR

bin_ops = {
    "+":    True,
    "-":    True,
    "*":    True,
    "/":    True,
    "%":    True,

    "and":  True,
    "or":   True,

    "<":    True,
    ">":    True,
    "<=":   True,
    ">=":   True,
    "==":   True,
}

LoopIR = ADT("""
module LoopIR {
    proc    = ( name            name,
                fnarg*          args,
                expr*           preds,
                stmt*           body,
                string?         instr,
                effect?         eff,
                srcinfo         srcinfo )

    fnarg   = ( sym             name,
                type            type,
                mem?            mem,
                srcinfo         srcinfo )

    stmt    = Assign ( sym name, type type, string? cast, expr* idx, expr rhs )
            | Reduce ( sym name, type type, string? cast, expr* idx, expr rhs )
            | Pass   ()
            | If     ( expr cond, stmt* body, stmt* orelse )
            | ForAll ( sym iter, expr hi, stmt* body )
            | Alloc  ( sym name, type type, mem? mem )
            | Free   ( sym name, type type, mem? mem )
            | Call   ( proc f, expr* args )
            | WindowStmt( sym lhs, expr rhs )
            attributes( effect? eff, srcinfo srcinfo )

    expr    = Read( sym name, expr* idx )
            | Const( object val )
            | BinOp( binop op, expr lhs, expr rhs )
            | WindowExpr( sym base, w_access* idx )
            | StrideAssert( sym name, int idx, int val ) -- may only occur
                                                         -- at proc.preds
            attributes( type type, srcinfo srcinfo )

    -- WindowExpr = (base : Sym, idx : [ Pt Expr | Interval Expr Expr ])
    w_access= Interval( expr lo, expr hi )
            | Point( expr pt )
            attributes( srcinfo srcinfo )

    type    = Num   ()
            | F32   ()
            | F64   ()
            | INT8  ()
            | Bool  ()
            | Int   ()
            | Index ()
            | Size  ()
            | Error ()
            | Tensor     ( expr* hi, bool is_window, type type )
            | WindowType ( type base, type as_tensor, expr window )

} """, {
    'name':     is_valid_name,
    'sym':      lambda x: type(x) is Sym,
    'effect':   lambda x: type(x) is E.effect,
    'mem':      lambda x: isinstance(x, Memory),
    'binop':    lambda x: x in bin_ops,
    'srcinfo':  lambda x: type(x) is SrcInfo,
})

ADTmemo(LoopIR, ['Num', 'F32', 'F64', 'INT8', 'Bool', 'Int', 'Index',
                 'Size', 'Error'])

# make proc be a hashable object
@extclass(LoopIR.proc)
def __hash__(self):
    return id(self)
del __hash__


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Types

class T:
    Num     = LoopIR.Num
    F32     = LoopIR.F32
    F64     = LoopIR.F64
    INT8    = LoopIR.INT8
    Bool    = LoopIR.Bool
    Int     = LoopIR.Int
    Index   = LoopIR.Index
    Size    = LoopIR.Size
    Error   = LoopIR.Error
    Tensor  = LoopIR.Tensor
    Window  = LoopIR.WindowType
    R       = Num()
    f32     = F32()
    int8    = INT8()
    f64     = F64()
    bool    = Bool()    # note: accessed as T.bool outside this module
    int     = Int()
    index   = Index()
    size    = Size()
    err     = Error()

    def is_type(obj):
        return isinstance(obj, LoopIR.type)

# --------------------------------------------------------------------------- #
# type helper functions

@extclass(T.Tensor)
@extclass(T.Window)
@extclass(T.Num)
@extclass(T.F32)
@extclass(T.F64)
@extclass(T.INT8)
def shape(t):
    #shape=dimension of a window is number of Interval
    if type(t) is T.Window:
        shp = []
        for i in t.window.idx:
            if type(i) is LoopIR.Interval:
                shp.append(i)
            return shp
    shp = t.hi if type(t) is T.Tensor else []
    return shp
del shape

@extclass(T.Num)
@extclass(T.F32)
@extclass(T.F64)
@extclass(T.INT8)
def ctype(t):
    if type(t) is T.Num:
        return "float"
    elif type(t) is T.F32:
        return "float"
    elif type(t) is T.F64:
        return "double"
    elif type(t) is T.INT8:
        return "int8_t"
del ctype

@extclass(LoopIR.type)
def is_real_scalar(t):
    return (type(t) is T.Num or type(t) is T.F32 or
            type(t) is T.F64 or type(t) is T.INT8)
del is_real_scalar

@extclass(LoopIR.type)
def is_tensor_or_window(t):
    return (type(t) is T.Tensor or type(t) is T.Window)
del is_tensor_or_window

@extclass(LoopIR.type)
def is_numeric(t):
    return t.is_real_scalar() or type(t) is T.Tensor or type(t) is T.Window
del is_numeric

@extclass(LoopIR.type)
def is_indexable(t):
    return type(t) is T.Int or type(t) is T.Index or type(t) is T.Size
del is_indexable

@extclass(LoopIR.type)
def is_sizeable(t):
    return type(t) is T.Int or type(t) is T.Size
del is_sizeable

@extclass(LoopIR.type)
def basetype(t):
    if type(t) is T.Window:
        t = t.as_tensor
    if type(t) is T.Tensor:
        t = t.type
    return t
del basetype

#@extclass(LoopIR.type)
#def subst(t, lookup):
#    raise NotImplementedError("TODO: fix 'range' to 'expr' change")
#    if type(t) is T.Tensor:
#        typ     = t.type.subst(lookup)
#        hi      = t.hi if is_pos_int(t.hi) else lookup[t.hi]
#        return T.Tensor(hi, typ)
#    else:
#        return t
#del subst

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Standard Pass Templates for Loop IR


class LoopIR_Rewrite:
    def __init__(self, proc, instr=None, *args, **kwargs):
        self.orig_proc  = proc

        body = self.map_stmts(self.orig_proc.body)

        self.proc = LoopIR.proc(name    = self.orig_proc.name,
                                args    = self.orig_proc.args,
                                preds   = self.orig_proc.preds,
                                body    = body,
                                instr   = instr,
                                eff     = self.orig_proc.eff,
                                srcinfo = self.orig_proc.srcinfo)

    def result(self):
        return self.proc

    def map_stmts(self, stmts):
        return [ s for b in stmts
                   for s in self.map_s(b) ]

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            return [styp( s.name, s.type, s.cast,
                        [ self.map_e(a) for a in s.idx ],
                          self.map_e(s.rhs), s.eff, s.srcinfo )]
        elif styp is LoopIR.If:
            return [LoopIR.If( self.map_e(s.cond), self.map_stmts(s.body),
                               self.map_stmts(s.orelse), s.eff, s.srcinfo )]
        elif styp is LoopIR.ForAll:
            return [LoopIR.ForAll( s.iter, self.map_e(s.hi),
                                   self.map_stmts(s.body), s.eff, s.srcinfo )]
        else:
            return [s]

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            return LoopIR.Read( e.name, [ self.map_e(a) for a in e.idx ],
                                e.type, e.srcinfo )
        elif etyp is LoopIR.BinOp:
            return LoopIR.BinOp( e.op, self.map_e(e.lhs), self.map_e(e.rhs),
                                 e.type, e.srcinfo )
        else:
            return e

class LoopIR_Do:
    def __init__(self, proc, *args, **kwargs):
        self.proc  = proc

        self.do_stmts(self.proc.body)

    def do_stmts(self, stmts):
        for s in stmts:
            self.do_s(s)

    def do_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            for e in s.idx:
                self.do_e(e)
            self.do_e(s.rhs)
        elif styp is LoopIR.If:
            self.do_e(s.cond)
            self.do_stmts(s.body)
            self.do_stmts(s.orelse)
        elif styp is LoopIR.ForAll:
            self.do_e(s.hi)
            self.do_stmts(s.body)
        elif styp is LoopIR.Call:
            for e in s.args:
                self.do_e(e)
        else:
            pass

    def do_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            for e in e.idx:
                self.do_e(e)
        elif etyp is LoopIR.BinOp:
            self.do_e(e.lhs)
            self.do_e(e.rhs)
        else:
            return e


class Alpha_Rename(LoopIR_Rewrite):
    def __init__(self, node):
        assert isinstance(node, list)
        self.env    = {}
        self.node   = []
        for n in node:
            assert isinstance(n, LoopIR.stmt)  # only case handled for now
            self.node += self.map_s(n)

    def result(self):
        return self.node

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            nm = self.env[s.name] if s.name in self.env else s.name
            return [styp( nm, s.type, s.cast,
                        [ self.map_e(a) for a in s.idx ],
                         self.map_e(s.rhs), s.eff, s.srcinfo )]
        elif styp is LoopIR.ForAll:
            itr = s.iter.copy()
            self.env[s.iter] = itr
            return [LoopIR.ForAll( itr, self.map_e(s.hi),
                                   self.map_stmts(s.body),
                                   s.eff, s.srcinfo )]
        elif styp is LoopIR.Alloc:
            nm = s.name.copy()
            self.env[s.name] = nm
            return [LoopIR.Alloc( nm, s.type, s.mem, s.eff, s.srcinfo )]

        return super().map_s(s)

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.Read( nm, [ self.map_e(a) for a in e.idx ],
                                e.type, e.srcinfo )

        return super().map_e(e)


class SubstArgs(LoopIR_Rewrite):
    def __init__(self, stmts, binding):
        assert isinstance(stmts, list)
        assert isinstance(binding, dict)
        assert all( isinstance(v, LoopIR.expr) for v in binding.values() )
        self.env    = binding
        self.stmts  = []
        for s in stmts:
            assert isinstance(s, LoopIR.stmt)  # only case handled for now
            self.stmts += self.map_s(s)

    def result(self):
        return self.stmts

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if s.name in self.env:
                e = self.env[s.name]
                assert type(e) is LoopIR.Read and len(e.idx) == 0
                return [styp( e.name, s.type, s.cast,
                            [ self.map_e(a) for a in s.idx ],
                              self.map_e(s.rhs), s.eff, s.srcinfo )]

        return super().map_s(s)

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            if e.name in self.env:
                if len(e.idx) == 0:
                    return self.env[e.name]
                else:
                    sub_e = self.env[e.name]
                    assert type(sub_e) is LoopIR.Read and len(sub_e.idx) == 0
                    return LoopIR.Read( sub_e.name,
                                        [ self.map_e(a) for a in e.idx ],
                                        e.type, e.srcinfo )

        return super().map_e(e)
