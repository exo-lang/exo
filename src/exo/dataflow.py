# import re
from collections import OrderedDict, ChainMap
from enum import Enum
from itertools import chain

# from typing import Type

from asdl_adt import ADT, validators

from .builtins import BuiltIn
from .configs import Config
from .memory import Memory
from .prelude import Sym, SrcInfo, extclass

from .LoopIR import Alpha_Rename, SubstArgs, LoopIR_Do

# --------------------------------------------------------------------------- #
# Top Level Call
# --------------------------------------------------------------------------- #

# Probably actually make this a class (pass) so it can inherit from LoopIR_Do
def dataflow_analysis(proc):  # returns a (see below)
    pass


# Big Question: How do we represent the result of dataflow analysis?

# Option 1: Take LoopIR and transform into a similar but different IR
#           which includes dataflow annotations and "desugars"/eliminates
#           constructs we don't care about for analysis purposes
#           (e.g. just inline all windowing)

LoopIR = ADT(
    """
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
         | Seq( sym iter, expr lo, expr hi, stmt* body )
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
         | F16()
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

}""",
    ext_types={
        "name": validators.instance_of(Identifier, convert=True),
        "sym": Sym,
        "effect": (lambda x: validators.instance_of(Effects.effect)(x)),
        "mem": Type[Memory],
        "builtin": BuiltIn,
        "config": Config,
        "binop": validators.instance_of(Operator, convert=True),
        "srcinfo": SrcInfo,
    },
    memoize={
        "Num",
        "F16",
        "F32",
        "F64",
        "INT8",
        "INT32" "Bool",
        "Int",
        "Index",
        "Size",
        "Stride",
        "Error",
    },
)

# Option 2: Leave the input LoopIR as is, and create auxiliary datastructures
#           which allow us to "lookup" dataflow results for different variables
#           at different points in the code.
#
#           For instance, use Python dictionaries to hold the annotations
