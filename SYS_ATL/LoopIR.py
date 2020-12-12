from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T

from .instruction_type import Instruction

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
                sym*            sizes,
                fnarg*          args,
                stmt*           body,
                srcinfo         srcinfo )

    fnarg   = ( sym             name,
                type            type,
                effect          effect,
                string?         mem,
                srcinfo         srcinfo )

    stmt    = Assign  ( sym name, expr* idx, expr rhs )
            | Reduce  ( sym name, expr* idx, expr rhs )
            | Pass    ()
            | If      ( expr cond, stmt* body,  stmt* orelse )
            | ForAll  ( sym iter,  expr cond,   stmt* body )
            | Alloc   ( sym name, type type, string? mem )
            | Instr   ( instr op, stmt body )
            | Call    ( proc f, expr* args )
            attributes( srcinfo srcinfo )

    expr    = Read    ( sym name, expr* idx )
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            | ParRange( expr lo, expr hi ) -- only use for loop cond
            attributes( srcinfo srcinfo )

} """, {
    'name':     is_valid_name,
    'sym':      lambda x: type(x) is Sym,
    'type':     T.is_type,
    'effect':   T.is_effect,
    'instr':    lambda x: isinstance(x, Instruction),
    'op':       lambda x: x in front_ops,
    'srcinfo':  lambda x: type(x) is SrcInfo,
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
    proc    = ( name?           name,
                fnarg*          args,
                stmt*           body,
                srcinfo         srcinfo )

    fnarg   = ( sym             name,
                type            type,
                effect?         effect,
                mem?            mem,
                srcinfo         srcinfo )

    stmt    = Assign ( sym name, expr* idx, expr rhs)
            | Reduce ( sym name, expr* idx, expr rhs )
            | Pass()
            | If     ( expr cond, stmt* body, stmt* orelse )
            | ForAll ( sym iter, expr hi, stmt* body )
            | Alloc  ( sym name, type type, mem? mem )
            | Free   ( sym name, type type, mem? mem )
            | Instr  ( instr op, stmt body )
            | Call   ( proc f, expr* args )
            attributes( srcinfo srcinfo )

    expr    = Read( sym name, expr* idx )
            | Const( object val )
            | BinOp( binop op, expr lhs, expr rhs )
            attributes( type type, srcinfo srcinfo )

} """, {
    'name':     is_valid_name,
    'sym':      lambda x: type(x) is Sym,
    'type':     T.is_type,
    'effect':   T.is_effect,
    'instr':    lambda x: isinstance(x, Instruction),
    'mem':      lambda x: type(x) is str,
    'binop':    lambda x: x in bin_ops,
    'srcinfo':  lambda x: type(x) is SrcInfo,
})
