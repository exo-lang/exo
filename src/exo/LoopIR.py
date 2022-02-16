from collections import ChainMap

import re
from typing import Type

from asdl_adt import ADT, validators
from .builtins import BuiltIn
from .configs import Config
from .memory import Memory
from .prelude import Sym, SrcInfo, extclass


# --------------------------------------------------------------------------- #
# Validated string subtypes
# --------------------------------------------------------------------------- #

class Identifier(str):
    _valid_re = re.compile(r"^(?:_\w|[a-zA-Z])\w*$")

    def __new__(cls, name):
        name = str(name)
        if Identifier._valid_re.match(name):
            return super().__new__(cls, name)
        raise ValueError(f'invalid identifier: {name}')


class IdentifierOrHole(str):
    _valid_re = re.compile(r"^[a-zA-Z_]\w*$")

    def __new__(cls, name):
        name = str(name)
        if IdentifierOrHole._valid_re.match(name):
            return super().__new__(cls, name)
        raise ValueError(f'invalid identifier: {name}')


front_ops = {"+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "and", "or"}


class Operator(str):
    def __new__(cls, op):
        op = str(op)
        if op in front_ops:
            return super().__new__(cls, op)
        raise ValueError(f'invalid operator: {op}')


# --------------------------------------------------------------------------- #
# Loop IR
# --------------------------------------------------------------------------- #


LoopIR = ADT("""
module LoopIR {
    proc = ( name    name,
             fnarg*  args,
             expr*   preds,
             stmt*   body,
             string? instr,
             effect? eff,
             srcinfo srcinfo )

    fnarg  = ( sym     name,
               type    type,
               mem?    mem,
               srcinfo srcinfo )

    stmt = Assign( sym name, type type, string? cast, expr* idx, expr rhs )
         | Reduce( sym name, type type, string? cast, expr* idx, expr rhs )
         | WriteConfig( config config, string field, expr rhs )
         | Pass()
         | If( expr cond, stmt* body, stmt* orelse )
         | ForAll( sym iter, expr hi, stmt* body )
         | Seq( sym iter, expr hi, stmt* body )
         | Alloc( sym name, type type, mem? mem )
         | Free( sym name, type type, mem? mem )
         | Call( proc f, expr* args )
         | WindowStmt( sym lhs, expr rhs )
         attributes( effect? eff, srcinfo srcinfo )

    expr = Read( sym name, expr* idx )
         | Const( object val )
         | USub( expr arg )  -- i.e.  -(...)
         | BinOp( binop op, expr lhs, expr rhs )
         | BuiltIn( builtin f, expr* args )
         | WindowExpr( sym name, w_access* idx )
         | StrideExpr( sym name, int dim )
         | ReadConfig( config config, string field )
         attributes( type type, srcinfo srcinfo )

    -- WindowExpr = (base : Sym, idx : [ Pt Expr | Interval Expr Expr ])
    w_access = Interval( expr lo, expr hi )
             | Point( expr pt )
             attributes( srcinfo srcinfo )

    type = Num()
         | F32()
         | F64()
         | INT8()
         | INT32()
         | Bool()
         | Int()
         | Index()
         | Size()
         | Stride()
         | Error()
         | Tensor( expr* hi, bool is_window, type type )
         -- src       - type of the tensor from which the window was created
         -- as_tensor - tensor type as if this window were simply a tensor 
         --             itself
         -- window    - the expression that created this window
         | WindowType( type src_type, type as_tensor,
                       sym src_buf, w_access *idx )

}""", ext_types={
    'name':    validators.instance_of(Identifier, convert=True),
    'sym':     Sym,
    'effect':  (lambda x: validators.instance_of(Effects.effect)(x)),
    'mem':     Type[Memory],
    'builtin': BuiltIn,
    'config':  Config,
    'binop':   validators.instance_of(Operator, convert=True),
    'srcinfo': SrcInfo,
}, memoize={'Num', 'F32', 'F64', 'INT8', 'INT32' 'Bool', 'Int', 'Index',
            'Size', 'Stride', 'Error'})

# --------------------------------------------------------------------------- #
# Untyped AST
# --------------------------------------------------------------------------- #

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
            | WriteConfig ( config config, string field, expr rhs )
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
            | StrideExpr( sym name, int dim )
            | ParRange( expr lo, expr hi ) -- only use for loop cond
            | SeqRange( expr lo, expr hi ) -- only use for loop cond
            | ReadConfig( config config, string field )
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
            | Stride()
            | Tensor( expr *hi, bool is_window, type type )
} """, ext_types={
    'name':        validators.instance_of(Identifier, convert=True),
    'sym':         Sym,
    'mem':         Type[Memory],
    'builtin':     BuiltIn,
    'config':      Config,
    'loopir_proc': LoopIR.proc,
    'op':          validators.instance_of(Operator, convert=True),
    'srcinfo':     SrcInfo,
}, memoize={'Num', 'F32', 'F64', 'INT8', 'INT32',
            'Bool', 'Int', 'Size', 'Index', 'Stride'})

# --------------------------------------------------------------------------- #
# Pattern AST
#   - used to specify pattern-matches
# --------------------------------------------------------------------------- #

PAST = ADT("""
module PAST {

    stmt    = Assign  ( name name, expr* idx, expr rhs )
            | Reduce  ( name name, expr* idx, expr rhs )
            | Pass    ()
            | If      ( expr cond, stmt* body,  stmt* orelse )
            | ForAll  ( name iter, expr hi,     stmt* body )
            | Seq     ( name iter, expr hi,     stmt* body )
            | Alloc   ( name name, expr* sizes ) -- may want to add mem back in?
            | Call    ( name f, expr* args )
            | WriteConfig ( name config, name field )
            | S_Hole  ()
            attributes( srcinfo srcinfo )

    expr    = Read    ( name name, expr* idx )
            | StrideExpr( name name, int dim )
            | E_Hole  ()
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            attributes( srcinfo srcinfo )

} """, ext_types={
    'name':    validators.instance_of(IdentifierOrHole, convert=True),
    'op':      validators.instance_of(Operator, convert=True),
    'srcinfo': SrcInfo,
})

# --------------------------------------------------------------------------- #
# Effects
# --------------------------------------------------------------------------- #

Effects = ADT("""
module Effects {
    effect      = ( effset*     reads,
                    effset*     writes,
                    effset*     reduces,
                    config_eff* config_reads,
                    config_eff* config_writes,
                    srcinfo     srcinfo )

    -- JRK: the notation of this comprehension is confusing -
    ---     maybe just use math:
    -- this corresponds to `{ buffer : loc for *names in int if pred }`
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

} """, {
    'sym':     Sym,
    'type':    LoopIR.type,
    'binop':   validators.instance_of(Operator, convert=True),
    'config':  Config,
    'srcinfo': SrcInfo,
})

# --------------------------------------------------------------------------- #
# Extension methods
# --------------------------------------------------------------------------- #

@extclass(UAST.Tensor)
@extclass(UAST.Num)
@extclass(UAST.F32)
@extclass(UAST.F64)
@extclass(UAST.INT8)
@extclass(UAST.INT32)
def shape(t):
    shp = t.hi if isinstance(t, UAST.Tensor) else []
    return shp
del shape

@extclass(UAST.type)
def basetype(t):
    if isinstance(t, UAST.Tensor):
        t = t.type
    return t
del basetype


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
    INT32   = LoopIR.INT32
    Bool    = LoopIR.Bool
    Int     = LoopIR.Int
    Index   = LoopIR.Index
    Size    = LoopIR.Size
    Stride  = LoopIR.Stride
    Error   = LoopIR.Error
    Tensor  = LoopIR.Tensor
    Window  = LoopIR.WindowType
    type    = LoopIR.type
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
    stride  = Stride()
    err     = Error()


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
    if isinstance(t, T.Window):
        return t.as_tensor.shape()
    elif isinstance(t, T.Tensor):
        assert not isinstance(t.type, T.Tensor), "expect no nesting"
        return t.hi
    else:
        return []
del shape

@extclass(T.Num)
@extclass(T.F32)
@extclass(T.F64)
@extclass(T.INT8)
@extclass(T.INT32)
@extclass(T.Bool)
@extclass(T.Int)
@extclass(T.Index)
@extclass(T.Size)
@extclass(T.Stride)
def ctype(t):
    if isinstance(t, T.Num):
        assert False, "Don't ask for ctype of Num"
    elif isinstance(t, T.F32):
        return "float"
    elif isinstance(t, T.F64):
        return "double"
    elif isinstance(t, T.INT8):
        return "int8_t"
    elif isinstance(t, T.INT32):
        return "int32_t"
    elif isinstance(t, T.Bool):
        return "bool"
    elif isinstance(t, (T.Int, T.Index, T.Size, T.Stride)):
        return "int_fast32_t"
del ctype


@extclass(LoopIR.type)
def is_real_scalar(t):
    return isinstance(t, (T.Num, T.F32, T.F64, T.INT8, T.INT32))
del is_real_scalar

@extclass(LoopIR.type)
def is_tensor_or_window(t):
    return isinstance(t, (T.Tensor, T.Window))
del is_tensor_or_window


@extclass(LoopIR.type)
def is_win(t):
    return ((isinstance(t, T.Tensor) and t.is_window) or
            isinstance(t, T.Window))
del is_win


@extclass(LoopIR.type)
def is_numeric(t):
    return t.is_real_scalar() or isinstance(t, (T.Tensor, T.Window))
del is_numeric

@extclass(LoopIR.type)
def is_bool(t):
    return isinstance(t, (T.Bool))
del is_bool

@extclass(LoopIR.type)
def is_indexable(t):
    return isinstance(t, (T.Int, T.Index, T.Size))
del is_indexable

@extclass(LoopIR.type)
def is_stridable(t):
    return isinstance(t, (T.Int, T.Stride))

@extclass(LoopIR.type)
def basetype(t):
    if isinstance(t, T.Window):
        return t.as_tensor.basetype()
    elif isinstance(t, T.Tensor):
        assert not t.type.is_tensor_or_window()
        return t.type
    else:
        return t
del basetype

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# convert from LoopIR.expr to E.expr
def lift_to_eff_expr(e):
    if isinstance(e, LoopIR.Read):
        assert len(e.idx) == 0
        return Effects.Var(e.name, e.type, e.srcinfo)
    elif isinstance(e, LoopIR.Const):
        return Effects.Const(e.val, e.type, e.srcinfo)
    elif isinstance(e, LoopIR.BinOp):
        return Effects.BinOp(e.op,
                             lift_to_eff_expr(e.lhs),
                             lift_to_eff_expr(e.rhs),
                             e.type, e.srcinfo)
    elif isinstance(e, LoopIR.USub):
        return Effects.BinOp('-',
                             Effects.Const(0, e.type, e.srcinfo),
                             lift_to_eff_expr(e.arg),
                             e.type, e.srcinfo)
    elif isinstance(e, LoopIR.StrideExpr):
        return Effects.Stride(e.name, e.dim, e.type, e.srcinfo)
    elif isinstance(e, LoopIR.ReadConfig):
        return Effects.ConfigField(e.config, e.field,
                                   e.config.lookup(e.field)[1], e.srcinfo)

    else:
        assert False, "bad case, e is " + str(type(e))

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Standard Pass Templates for Loop IR


class LoopIR_Rewrite:
    def __init__(self, proc, instr=None, *args, **kwargs):
        self.orig_proc  = proc

        args = [self.map_fnarg(a) for a in self.orig_proc.args]
        preds = [self.map_e(p) for p in self.orig_proc.preds]
        preds = [p for p in preds
                 if not (isinstance(p, LoopIR.Const) and p.val)]
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
        elif styp is LoopIR.WriteConfig:
            return [LoopIR.WriteConfig( s.config, s.field, self.map_e(s.rhs),
                                        self.map_eff(s.eff), s.srcinfo )]
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
        elif styp is LoopIR.Seq:
            return [LoopIR.Seq( s.iter, self.map_e(s.hi),
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
        elif etyp is LoopIR.ReadConfig:
            return LoopIR.ReadConfig(e.config, e.field,
                                     self.map_t(e.type), e.srcinfo)
        else:
            # constant case cannot have variable-size tensor type
            # stride expr case has stride type
            return e

    def map_w_access(self, w):
        if isinstance(w, LoopIR.Interval):
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
        return eff.update(
            reads=[self.map_eff_es(es) for es in eff.reads],
            writes=[self.map_eff_es(es) for es in eff.writes],
            reduces=[self.map_eff_es(es) for es in eff.reduces],
            config_reads=[self.map_eff_ce(ce) for ce in eff.config_reads],
            config_writes=[self.map_eff_ce(ce) for ce in eff.config_writes],
        )

    def map_eff_es(self, es):
        return es.update(loc=[self.map_eff_e(i) for i in es.loc],
                         pred=self.map_eff_e(es.pred) if es.pred else None)

    def map_eff_ce(self, ce):
        return ce.update(value=self.map_eff_e(ce.value) if ce.value else None,
                         pred=self.map_eff_e(ce.pred) if ce.pred else None)

    def map_eff_e(self, e):
        if isinstance(e, Effects.BinOp):
            return e.update(lhs=self.map_eff_e(e.lhs),
                            rhs=self.map_eff_e(e.rhs))
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
        elif styp is LoopIR.WriteConfig:
            self.do_e(s.rhs)
        elif styp is LoopIR.WindowStmt:
            self.do_e(s.rhs)
        elif styp is LoopIR.If:
            self.do_e(s.cond)
            self.do_stmts(s.body)
            self.do_stmts(s.orelse)
        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
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
        if isinstance(w, LoopIR.Interval):
            self.do_e(w.lo)
            self.do_e(w.hi)
        elif isinstance(w, LoopIR.Point):
            self.do_e(w.pt)
        else: assert False, "bad case"

    def do_t(self, t):
        if isinstance(t, T.Tensor):
            for i in t.hi:
                self.do_e(i)
        elif isinstance(t, T.Window):
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
        if isinstance(e, Effects.BinOp):
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
            elif isinstance(n, Effects.effect):
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
        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            self.do_e(s.hi)
            self.push()
            self.env[s.iter] = True
            self.do_stmts(s.body)
            self.pop()
            self.do_eff(s.eff)
            return
        elif styp is LoopIR.Alloc:
            self.env[s.name] = True

        super().do_s(s)

    def do_e(self, e):
        etyp = type(e)
        if (etyp is LoopIR.Read or
            etyp is LoopIR.WindowExpr or
            etyp is LoopIR.StrideExpr):
            if e.name not in self.env:
                self.fv.add(e.name)

        super().do_e(e)

    def do_t(self, t):
        if isinstance(t, T.Window):
            if t.src_buf not in self.env:
                self.fv.add(t.src_buf)

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
        if isinstance(e, Effects.Var) and e.name not in self.env:
            self.fv.add(e.name)

        super().do_eff_e(e)

class Alpha_Rename(LoopIR_Rewrite):
    def __init__(self, node):
        self.env    = ChainMap()
        self.node   = []

        if isinstance(node, LoopIR.proc):
            self.node = self.map_proc(node)
        else:
            assert isinstance(node, list)
            for n in node:
                if isinstance(n, LoopIR.stmt):
                    self.node += self.map_s(n)
                elif isinstance(n, LoopIR.expr):
                    self.node += [self.map_e(n)]
                elif isinstance(n, Effects.effect):
                    self.node += [self.map_eff(n)]
                else: assert False, "expected stmt or expr or effect"

    def result(self):
        return self.node

    def push(self):
        self.env = self.env.new_child()
    def pop(self):
        self.env = self.env.parents

    def map_proc(self, proc):
        args    = [ self.map_fnarg(fa) for fa in proc.args ]
        preds   = [ self.map_e(e) for e in proc.preds ]
        body    = self.map_stmts(proc.body)
        eff     = self.map_eff(proc.eff)

        return LoopIR.proc(proc.name, args, preds, body,
                           proc.instr, eff, proc.srcinfo)

    def map_fnarg(self, fa):
        nm  = fa.name.copy()
        self.env[fa.name] = nm
        typ = self.map_t(fa.type)
        return LoopIR.fnarg(nm, typ, fa.mem, fa.srcinfo)

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

        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            hi  = self.map_e(s.hi)
            eff = self.map_eff(s.eff)
            self.push()
            itr = s.iter.copy()
            self.env[s.iter] = itr
            stmts = [styp( itr, hi, self.map_stmts(s.body),
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
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.WindowExpr(nm,
                               [self.map_w_access(a) for a in e.idx],
                               self.map_t(e.type), e.srcinfo)

        elif etyp is LoopIR.StrideExpr:
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.StrideExpr(nm, e.dim, e.type, e.srcinfo)

        return super().map_e(e)

    def map_eff_es(self, es):
        self.push()
        names = [nm.copy() for nm in es.names]
        for orig, new in zip(es.names, names):
            self.env[orig] = new

        eset = es.update(
            buffer=self.env.get(es.buffer, es.buffer),
            loc=[self.map_eff_e(i) for i in es.loc],
            names=names,
            pred=self.map_eff_e(es.pred) if es.pred else None,
        )
        self.pop()
        return eset

    def map_eff_e(self, e):
        if isinstance(e, Effects.Var):
            return e.update(name=self.env.get(e.name, e.name))

        return super().map_eff_e(e)

    def map_t(self, t):
        ttyp = type(t)
        if ttyp is T.Window:
            src_buf = t.src_buf
            if t.src_buf in self.env:
                src_buf = self.env[t.src_buf]

            return T.Window( self.map_t(t.src_type), self.map_t(t.as_tensor),
                             src_buf,
                             [ self.map_w_access(w) for w in t.idx ] )

        return super().map_t(t)


class SubstArgs(LoopIR_Rewrite):
    def __init__(self, nodes, binding):
        assert isinstance(nodes, list)
        assert isinstance(binding, dict)
        assert all(isinstance(v, LoopIR.expr) for v in binding.values())
        assert not any(
            isinstance(v, LoopIR.WindowExpr) for v in binding.values())
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
                assert isinstance(e, LoopIR.Read) and len(e.idx) == 0
                return [styp( e.name, self.map_t(s.type), s.cast,
                            [ self.map_e(a) for a in s.idx ],
                              self.map_e(s.rhs), s.eff, s.srcinfo )]

        return super().map_s(s)

    def map_e(self, e):
        # this substitution could refer to a read or a window expression
        if isinstance(e, LoopIR.Read):
            if e.name in self.env:
                if len(e.idx) == 0:
                    return self.env[e.name]
                else:
                    sub_e = self.env[e.name]
                    assert (isinstance(sub_e, LoopIR.Read) and
                            len(sub_e.idx) == 0)
                    return LoopIR.Read(sub_e.name,
                                       [self.map_e(a) for a in e.idx],
                                       e.type, e.srcinfo)
        elif isinstance(e, LoopIR.WindowExpr):
            if e.name in self.env:
                if len(e.idx) == 0:
                    return self.env[e.name]
                else:
                    sub_e = self.env[e.name]
                    assert (isinstance(sub_e, LoopIR.Read) and len(sub_e.idx) == 0)
                    return LoopIR.WindowExpr(sub_e.name,
                                       [self.map_w_access(a) for a in e.idx],
                                       self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.StrideExpr):
            if e.name in self.env:
                sub_e = self.env[e.name]
                return LoopIR.StrideExpr(sub_e.name, e.dim, e.type, e.srcinfo)

        return super().map_e(e)

    def map_eff_es(self, es):
        # this substitution could refer to a read or a window expression
        new_es = super().map_eff_es(es)
        if es.buffer in self.env:
            sub_e = self.env[es.buffer]
            assert isinstance(sub_e, LoopIR.Read) and len(sub_e.idx) == 0
            new_es = new_es.update(buffer=sub_e.name)
        return new_es

    def map_eff_e(self, e):
        if isinstance(e, Effects.Var):
            if e.name in self.env:
                if e.type.is_indexable():
                    sub_e = self.env[e.name]
                    assert sub_e.type.is_indexable()
                    return lift_to_eff_expr(sub_e)
                else: # Could be config value (e.g. f32)
                    sub_e = self.env[e.name]
                    return lift_to_eff_expr(sub_e)

        return super().map_eff_e(e)

    def map_t(self, t):
        ttyp = type(t)
        if ttyp is T.Window:
            src_buf = t.src_buf
            if t.src_buf in self.env:
                src_buf = self.env[t.src_buf].name

            return T.Window( self.map_t(t.src_type), self.map_t(t.as_tensor),
                             src_buf,
                             [ self.map_w_access(w) for w in t.idx ] )

        return super().map_t(t)
