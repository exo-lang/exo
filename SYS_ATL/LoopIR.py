from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from .LoopIR_effects import Effects as E

from .memory import Memory
from .builtins import BuiltIn

from collections import ChainMap

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
            | FreshAssign( sym name, expr rhs )
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
            | BuiltIn( builtin f, expr* args )
            | WindowExpr( sym name, w_access* idx )
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
            | INT32 ()
            | Bool  ()
            | Int   ()
            | Size  ()
            | Index ()
            | Tensor( expr *hi, bool is_window, type type )
} """, {
    'name':         is_valid_name,
    'sym':          lambda x: type(x) is Sym,
    'mem':          lambda x: isinstance(x, Memory),
    'builtin':  lambda x: isinstance(x, BuiltIn),
    'loopir_proc':  lambda x: type(x) is LoopIR.proc,
    'op':           lambda x: x in front_ops,
    'srcinfo':      lambda x: type(x) is SrcInfo
})

ADTmemo(UAST, ['Num', 'F32', 'F64', 'INT8', 'INT32',
               'Bool', 'Int', 'Size', 'Index'], {
})


@extclass(UAST.Tensor)
@extclass(UAST.Num)
@extclass(UAST.F32)
@extclass(UAST.F64)
@extclass(UAST.INT8)
@extclass(UAST.INT32)
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
            | USub( expr arg )  -- i.e.  -(...)
            | BinOp( binop op, expr lhs, expr rhs )
            | BuiltIn( builtin f, expr* args )
            | WindowExpr( sym name, w_access* idx )
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
            | INT32 ()
            | Bool  ()
            | Int   ()
            | Index ()
            | Size  ()
            | Error ()
            | Tensor     ( expr* hi, bool is_window, type type )
            -- src          - type of the tensor
            --                from which the window was created
            -- as_tensor    - tensor type as if this window were simply
            --                a tensor itself
            -- window       - the expression that created this window
            | WindowType ( type src_type, type as_tensor,
                           sym src_buf, w_access *idx )

} """, {
    'name':     is_valid_name,
    'sym':      lambda x: type(x) is Sym,
    'effect':   lambda x: type(x) is E.effect,
    'mem':      lambda x: isinstance(x, Memory),
    'builtin':  lambda x: isinstance(x, BuiltIn),
    'binop':    lambda x: x in bin_ops,
    'srcinfo':  lambda x: type(x) is SrcInfo,
})

ADTmemo(LoopIR, ['Num', 'F32', 'F64', 'INT8', 'INT32' 'Bool', 'Int', 'Index',
                 'Size', 'Error'])

# make proc be a hashable object
@extclass(LoopIR.proc)
def __hash__(self):
    return id(self)
del __hash__


"""
TODO: Delete this once change verified
# break recursion...
@extclass(LoopIR.WindowType)
def __hash__(self):
    return hash([type(self), self.src, self.as_tensor, id(self.window)])
del __hash__

@extclass(LoopIR.WindowType)
def __repr__(self):
    return (f"WindowType(src={repr(self.src)},"+
                       f"as_tensor={repr(self.as_tensor)},"+
                       f"window=(name={repr(self.window.name)},"+
                               f"idx={repr(self.window.idx)}))")
del __repr__

@extclass(LoopIR.WindowType)
def __eq__(lhs,rhs):
    return (type(lhs) == type(rhs) and lhs.src == rhs.src and
            lhs.as_tensor == rhs.as_tensor and
            id(lhs.window) == id(rhs.window))
del __eq__
"""




# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Types

class T:
    Num     = LoopIR.Num
    F32     = LoopIR.F32
    F64     = LoopIR.F64
    INT8    = LoopIR.INT8
    INT32   = LoopIR.INT32
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
    i8      = INT8()
    int32   = INT32()
    i32     = INT32()
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
@extclass(T.INT32)
def shape(t):
    if type(t) is T.Window:
        return t.as_tensor.shape()
    elif type(t) is T.Tensor:
        assert type(t.type) is not T.Tensor, "expect no nesting"
        return t.hi
    else:
        return []
del shape

@extclass(T.Num)
@extclass(T.F32)
@extclass(T.F64)
@extclass(T.INT8)
@extclass(T.INT32)
def ctype(t):
    if type(t) is T.Num:
        return "float"
    elif type(t) is T.F32:
        return "float"
    elif type(t) is T.F64:
        return "double"
    elif type(t) is T.INT8:
        return "int8_t"
    elif type(t) is T.INT32:
        return "int32_t"
del ctype

@extclass(LoopIR.type)
def is_real_scalar(t):
    return (type(t) is T.Num or type(t) is T.F32 or
            type(t) is T.F64 or type(t) is T.INT8 or type(t) is T.INT32)
del is_real_scalar

@extclass(LoopIR.type)
def is_tensor_or_window(t):
    return (type(t) is T.Tensor or type(t) is T.Window)
del is_tensor_or_window

@extclass(LoopIR.type)
def is_win(t):
    return ( (type(t) is T.Tensor and t.is_window ) or
             (type(t) is T.Window))
del is_win

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
        return t.as_tensor.basetype()
    elif type(t) is T.Tensor:
        assert not t.type.is_tensor_or_window()
        return t.type
    else:
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

# convert from LoopIR.expr to E.expr
def lift_to_eff_expr(e):
    if type(e) is LoopIR.Read:
        assert len(e.idx) == 0
        return E.Var( e.name, e.type, e.srcinfo )
    elif type(e) is LoopIR.Const:
        return E.Const( e.val, e.type, e.srcinfo )
    elif type(e) is LoopIR.BinOp:
        return E.BinOp( e.op,
                        lift_to_eff_expr(e.lhs),
                        lift_to_eff_expr(e.rhs),
                        e.type, e.srcinfo )
    elif isinstance(e, LoopIR.USub):
        return E.BinOp('-', E.Const(0, e.type, e.srcinfo), lift_to_eff_expr(e.arg), e.type, e.srcinfo)

    else: assert False, "bad case, e is " + str(type(e))

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Standard Pass Templates for Loop IR


class LoopIR_Rewrite:
    def __init__(self, proc, instr=None, *args, **kwargs):
        self.orig_proc  = proc

        args = [ self.map_fnarg(a) for a in self.orig_proc.args ]
        preds= [ self.map_e(p) for p in self.orig_proc.preds ]
        body = self.map_stmts(self.orig_proc.body)

        eff  = self.map_eff(self.orig_proc.eff)

        self.proc = LoopIR.proc(name    = self.orig_proc.name,
                                args    = args,
                                preds   = preds,
                                body    = body,
                                instr   = instr,
                                eff     = eff,
                                srcinfo = self.orig_proc.srcinfo)

    def result(self):
        return self.proc

    def map_fnarg(self, a):
        return LoopIR.fnarg(a.name, self.map_t(a.type), a.mem, a.srcinfo)

    def map_stmts(self, stmts):
        return [ s for b in stmts
                   for s in self.map_s(b) ]

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            return [styp( s.name, self.map_t(s.type), s.cast,
                        [ self.map_e(a) for a in s.idx ],
                          self.map_e(s.rhs), self.map_eff(s.eff), s.srcinfo )]
        elif styp is LoopIR.WindowStmt:
            return [LoopIR.WindowStmt( s.lhs, self.map_e(s.rhs),
                                       self.map_eff(s.eff), s.srcinfo )]
        elif styp is LoopIR.If:
            return [LoopIR.If( self.map_e(s.cond), self.map_stmts(s.body),
                               self.map_stmts(s.orelse),
                               self.map_eff(s.eff), s.srcinfo )]
        elif styp is LoopIR.ForAll:
            return [LoopIR.ForAll( s.iter, self.map_e(s.hi),
                                   self.map_stmts(s.body),
                                   self.map_eff(s.eff), s.srcinfo )]
        elif styp is LoopIR.Call:
            return [LoopIR.Call( s.f, [ self.map_e(a) for a in s.args ],
                                 self.map_eff(s.eff), s.srcinfo )]
        elif styp is LoopIR.Alloc:
            return [LoopIR.Alloc( s.name, self.map_t(s.type), s.mem,
                                  self.map_eff(s.eff), s.srcinfo )]
        else:
            return [s]

    def map_e(self, e):
        etyp    = type(e)
        if etyp is LoopIR.Read:
            return LoopIR.Read( e.name, [ self.map_e(a) for a in e.idx ],
                                self.map_t(e.type), e.srcinfo )
        elif etyp is LoopIR.BinOp:
            return LoopIR.BinOp( e.op, self.map_e(e.lhs), self.map_e(e.rhs),
                                 self.map_t(e.type), e.srcinfo )
        elif etyp is LoopIR.BuiltIn:
            return LoopIR.BuiltIn( e.f, [ self.map_e(a) for a in e.args ],
                                   self.map_t(e.type), e.srcinfo )
        elif etyp is LoopIR.USub:
            return LoopIR.USub(self.map_e(e.arg), self.map_t(e.type), e.srcinfo)
        elif etyp is LoopIR.WindowExpr:
            return LoopIR.WindowExpr(e.name,
                                     [ self.map_w_access(w) for w in e.idx ],
                                     self.map_t(e.type), e.srcinfo)

        else:
            # constant case cannot have variable-size tensor type
            # stride assert case has bool type
            return e

    def map_w_access(self, w):
        if type(w) is LoopIR.Interval:
            return LoopIR.Interval(self.map_e(w.lo),
                                   self.map_e(w.hi), w.srcinfo)
        else:
            return LoopIR.Point(self.map_e(w.pt), w.srcinfo)

    def map_t(self, t):
        ttyp = type(t)
        if ttyp is T.Tensor:
            return T.Tensor( [ self.map_e(r) for r in t.hi ],
                             t.is_window, self.map_t(t.type) )
        elif ttyp is T.Window:
            return T.Window( self.map_t(t.src_type), self.map_t(t.as_tensor),
                             t.src_buf,
                             [ self.map_w_access(w) for w in t.idx ] )
        else:
            return t

    def map_eff(self, eff):
        if eff is None:
            return eff
        return E.effect( [ self.map_eff_es(es) for es in eff.reads ],
                         [ self.map_eff_es(es) for es in eff.writes ],
                         [ self.map_eff_es(es) for es in eff.reduces ],
                         eff.srcinfo )

    def map_eff_es(self, es):
        return E.effset( es.buffer,
                         [ self.map_eff_e(i) for i in es.loc ],
                         es.names,
                         self.map_eff_e(es.pred) if es.pred else None,
                         es.srcinfo )

    def map_eff_e(self, e):
        if type(e) is E.BinOp:
            return E.BinOp(e.op, self.map_eff_e(e.lhs),
                                 self.map_eff_e(e.rhs), e.type, e.srcinfo )
        else:
            return e


class LoopIR_Do:
    def __init__(self, proc, *args, **kwargs):
        self.proc      = proc

        for a in self.proc.args:
            self.do_t(a.type)
        for p in self.proc.preds:
            self.do_e(p)

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
            self.do_t(s.type)
        elif styp is LoopIR.WindowStmt:
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
        elif styp is LoopIR.Alloc:
            self.do_t(s.type)
        else:
            pass

        self.do_eff(s.eff)

    def do_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            for e in e.idx:
                self.do_e(e)
        elif etyp is LoopIR.BinOp:
            self.do_e(e.lhs)
            self.do_e(e.rhs)
        elif etyp is LoopIR.BuiltIn:
            for a in e.args:
                self.do_e(a)
        elif etyp is LoopIR.USub:
            self.do_e(e.arg)
        elif etyp is LoopIR.WindowExpr:
            for w in e.idx:
                self.do_w_access(w)
        else:
            pass

        self.do_t(e.type)

    def do_w_access(self, w):
        if type(w) is LoopIR.Interval:
            self.do_e(w.lo)
            self.do_e(w.hi)
        elif type(w) is LoopIR.Point:
            self.do_e(w.pt)
        else: assert False, "bad case"

    def do_t(self, t):
        if type(t) is T.Tensor:
            for i in t.hi:
                self.do_e(i)
        elif type(t) is T.Window:
            self.do_t(t.src_type)
            self.do_t(t.as_tensor)
            for w in t.idx:
                self.do_w_access(w)
        else:
            pass

    def do_eff(self, eff):
        if eff is None:
            return
        for es in eff.reads:
            self.do_eff_es(es)
        for es in eff.writes:
            self.do_eff_es(es)
        for es in eff.reduces:
            self.do_eff_es(es)

    def do_eff_es(self, es):
        for i in es.loc:
            self.do_eff_e(i)
        if es.pred:
            self.do_eff_e(es.pred)

    def do_eff_e(self, e):
        if type(e) is E.BinOp:
            self.do_eff_e(e.lhs)
            self.do_eff_e(e.rhs)


class FreeVars(LoopIR_Do):
    def __init__(self, node):
        assert isinstance(node, list)
        self.env    = ChainMap()
        self.fv     = set()

        for n in node:
            if isinstance(n, LoopIR.stmt):
                self.do_s(n)
            elif isinstance(n, LoopIR.expr):
                self.do_e(n)
            elif isinstance(n, E.effect):
                self.do_eff(n)
            else: assert False, "expected stmt, expr, or effect"

    def result(self):
        return self.fv

    def push(self):
        self.env = self.env.new_child()
    def pop(self):
        self.env = self.env.parents

    def do_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if s.name not in self.env:
                self.fv.add(s.name)
        elif styp is LoopIR.WindowStmt:
            self.env[s.lhs] = True
        elif styp is LoopIR.If:
            self.do_e(s.cond)
            self.push()
            self.do_stmts(s.body)
            self.do_stmts(s.orelse)
            self.pop()
            self.do_eff(s.eff)
            return
        elif styp is LoopIR.ForAll:
            self.do_e(s.hi)
            self.push()
            self.env[s.iter] = True
            self.do_stmts(s.body)
            self.pop()
            self.do_eff(s.eff)
            return
        elif styp is LoopIR.Alloc:
            self.env[s.iter] = True

        super().do_s(s)

    def do_e(self, e):
        etyp = type(e)
        if (etyp is LoopIR.Read or
            etyp is LoopIR.WindowExpr or
            etyp is LoopIR.StrideAssert):
            if e.name not in self.env:
                self.fv.add(e.name)

        super().do_e(e)

    def do_t(self, t):
        if type(t) is T.Window:
            if t.name not in self.env:
                self.fv.add(t.name)

        super().do_t(t)

    def do_eff_es(self, es):
        if es.buffer not in self.env:
            self.fv.add(es.buffer)

        self.push()
        for x in es.names:
            self.env[x] = True
                
        super().do_eff_es(es)
        self.pop()

    def do_eff_e(self, e):
        if type(e) is E.Var and e.name not in self.env:
            self.fv.add(e.name)

        super().do_eff_e(e)

class Alpha_Rename(LoopIR_Rewrite):
    def __init__(self, node):
        assert isinstance(node, list)
        self.env    = ChainMap()
        self.node   = []
        for n in node:
            if isinstance(n, LoopIR.stmt):
                self.node += self.map_s(n)
            elif isinstance(n, LoopIR.expr):
                self.node += [self.map_e(n)]
            elif isinstance(n, E.effect):
                self.node += [self.map_eff(n)]
            else: assert False, "expected stmt or expr or effect"

    def result(self):
        return self.node

    def push(self):
        self.env = self.env.new_child()
    def pop(self):
        self.env = self.env.parents

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            nm = self.env[s.name] if s.name in self.env else s.name
            return [styp( nm, self.map_t(s.type), s.cast,
                        [ self.map_e(a) for a in s.idx ],
                         self.map_e(s.rhs), self.map_eff(s.eff), s.srcinfo )]
        elif styp is LoopIR.WindowStmt:
            rhs = self.map_e(s.rhs)
            lhs = s.lhs.copy()
            self.env[s.lhs] = lhs
            return [LoopIR.WindowStmt( lhs, rhs,
                                       self.map_eff(s.eff), s.srcinfo )]
        elif styp is LoopIR.If:
            self.push()
            stmts = super().map_s(s)
            self.pop()
            return stmts

        elif styp is LoopIR.ForAll:
            hi  = self.map_e(s.hi)
            eff = self.map_eff(s.eff)
            self.push()
            itr = s.iter.copy()
            self.env[s.iter] = itr
            stmts = [LoopIR.ForAll( itr, hi, self.map_stmts(s.body),
                                    eff, s.srcinfo )]
            self.pop()
            return stmts
        elif styp is LoopIR.Alloc:
            nm = s.name.copy()
            self.env[s.name] = nm
            return [LoopIR.Alloc( nm, self.map_t(s.type), s.mem,
                                  self.map_eff(s.eff), s.srcinfo )]

        return super().map_s(s)

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.Read( nm, [ self.map_e(a) for a in e.idx ],
                                self.map_t(e.type), e.srcinfo )
        elif etyp is LoopIR.WindowExpr:
            win_e = super().map_e(e)
            nm = self.env[e.name] if e.name in self.env else e.name
            win_e.name = nm
            return win_e
        elif etyp is LoopIR.StrideAssert:
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.StrideAssert(nm, e.idx, e.val, e.type, e.srcinfo)

        return super().map_e(e)

    def map_eff_es(self, es):
        self.push()
        names = [ nm.copy() for nm in es.names ]
        for orig,new in zip(es.names, names):
            self.env[orig] = new

        buf = self.env[es.buffer] if es.buffer in self.env else es.buffer
        eset = E.effset( buf,
                         [ self.map_eff_e(i) for i in es.loc ],
                         names,
                         self.map_eff_e(es.pred) if es.pred else None,
                         es.srcinfo )
        self.pop()
        return eset

    def map_eff_e(self, e):
        if type(e) is E.Var:
            nm = self.env[e.name] if e.name in self.env else e.name
            return E.Var( nm, e.type, e.srcinfo )

        return super().map_eff_e(e)


class SubstArgs(LoopIR_Rewrite):
    def __init__(self, nodes, binding):
        assert isinstance(nodes, list)
        assert isinstance(binding, dict)
        assert all( isinstance(v, LoopIR.expr) for v in binding.values() )
        assert all( type(v) != LoopIR.WindowExpr for v in binding.values() )
        self.env    = binding
        self.nodes  = []
        for n in nodes:
            if isinstance(n, LoopIR.stmt):
                self.nodes += self.map_s(n)
            elif isinstance(n, LoopIR.expr):
                self.nodes += [self.map_e(n)]
            else: assert False, "expected stmt or expr"

    def result(self):
        return self.nodes

    def map_s(self, s):
        # this substitution could refer to a read or a window expression
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if s.name in self.env:
                e = self.env[s.name]
                assert type(e) is LoopIR.Read and len(e.idx) == 0
                return [styp( e.name, self.map_t(s.type), s.cast,
                            [ self.map_e(a) for a in s.idx ],
                              self.map_e(s.rhs), s.eff, s.srcinfo )]

        return super().map_s(s)

    def map_e(self, e):
        # this substitution could refer to a read or a window expression
        if type(e) is LoopIR.Read:
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

    def map_eff_es(self, es):
        # this substitution could refer to a read or a window expression
        new_es = super().map_eff_es(es)
        if es.buffer in self.env:
            sub_e = self.env[es.buffer]
            assert type(sub_e) is LoopIR.Read and len(sub_e.idx) == 0
            new_es.buffer = sub_e.name
        return new_es

    def map_eff_e(self, e):
        # purely index expressions
        if type(e) is E.Var:
            assert e.type.is_indexable()
            if e.name in self.env:
                sub_e = self.env[e.name]
                assert sub_e.type.is_indexable()
                return lift_to_eff_expr(sub_e)

        return super().map_eff_e(e)
