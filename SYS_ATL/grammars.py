import re
from typing import Type

from asdl_adt import ADT, validators
from .builtins import BuiltIn
from .configs import Config
from .memory import Memory
from .prelude import Sym, SrcInfo


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
            | Alloc   ( name name ) -- may want to add type & mem back in?
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
