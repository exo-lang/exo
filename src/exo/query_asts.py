"""Query AST Classes

This module contains a reflection of Exo's internal AST structures.

They are organized into a class hierarchy of Python dataclasses as
follows.

QueryAST
  Proc  ( name : str, args : list[FnArg], assertions : list[Expr],
                      body : list[Stmt],  instruction : Optional[str] )
  FnArg ( name : str, type : Type, memory : Optional[Memory] )
  Stmt
    Assign    ( name : str,   lhs_type : Type,    idx : list[Expr],
                              rhs  : Expr )
    Reduce    ( name : str,   lhs_type : Type,    idx : list[Expr],
                              rhs  : Expr )
    WriteConfig ( config : Config, field : str,
                              rhs  : Expr )
    Pass      ()
    If        ( cond : Expr,  body : list[Stmt],  orelse : list[Stmt] )
    For       ( name : str,   lo   : Expr,        hi : Expr,
                              body : list[Stmt],  is_par : bool )
    Alloc     ( name : str,   type : Type,        memory : Optional[Memory] )
    Call      ( proc : str,   args : list[Expr] )
    WindowStmt( name : str,   rhs  : Expr )
  Expr
    Read    ( name : str,   idx  : list[Expr],    type : Type )
    Const   ( val  : Any,   type : Type )
    USub    ( arg  : Expr,  type : Type )
    BinOp   ( op   : str,   lhs  : Expr,
                            rhs  : Expr,          type : Type )
    BuiltIn ( func : str,   args : list[Expr],    type : Type )
    WindowExpr( name : str, idx : list[WAccess],  type : Type )
    StrideExpr( name : str, dim : int,            type : Type )
    ReadConfig( config : Config, field : str,     type : Type )
  WAccess
    Interval( lo : Expr, hi : Expr )
    Point( pt : Expr )
  Type
    R()
    f32()
    f64()
    i8()
    i32()
    bool()
    int()
    index()
    size()
    stride()
    tensor( hi : list[Expr], is_window : bool, type : Type )

"""

from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Optional as _Optional

from .configs import Config as _Config
from .memory import Memory as _Memory


# ----------------------------------------------------------------------
# -- base classes for all asts returned from the reflection interface --

class QueryAST:
  def __init__(self):
    raise Exception("Should never try to instantiate QueryAST")

class Type(QueryAST):
  def __init__(self):
    raise Exception("Should never try to instantiate Type")

class Expr(QueryAST):
  def __init__(self):
    raise Exception("Should never try to instantiate Expr")

class WAccess(QueryAST):
  def __init__(self):
    raise Exception("Should never try to instantiate WAccess")

class Stmt(QueryAST):
  def __init__(self):
    raise Exception("Should never try to instantiate Stmt")

# -----------------------------------
# -- QueryAST --> Type --> _______ --

@_dataclass
class R(Type):
  pass

@_dataclass
class f32(Type):
  pass

@_dataclass
class f64(Type):
  pass

@_dataclass
class i8(Type):
  pass

@_dataclass
class i32(Type):
  pass

@_dataclass
class bool(Type):
  pass

@_dataclass
class int(Type):
  pass

@_dataclass
class index(Type):
  pass

@_dataclass
class size(Type):
  pass

@_dataclass
class stride(Type):
  pass

@_dataclass
class tensor(Type):
  hi          : list[Expr]
  is_window   : bool
  type        : Type

# --------------------------------------
# -- QueryAST --> WAccess --> _______ --

@_dataclass
class Interval(WAccess):
  lo  : Expr
  hi  : Expr

@_dataclass
class Point(WAccess):
  pt  : Expr

# -----------------------------------
# -- QueryAST --> Expr --> _______ --

@_dataclass
class Read(Expr):
  name  : str
  idx   : list[Expr]
  type  : Type

@_dataclass
class Const(Expr):
  val   : _Any
  type  : Type

@_dataclass
class USub(Expr):
  arg   : Expr
  type  : Type

@_dataclass
class BinOp(Expr):
  op    : str
  lhs   : Expr
  rhs   : Expr
  type  : Type

@_dataclass
class BuiltIn(Expr):
  func  : str
  args  : list[Expr]
  type  : Type

@_dataclass
class WindowExpr(Expr):
  name  : str
  idx   : list[WAccess]
  type  : Type

@_dataclass
class StrideExpr(Expr):
  name  : str
  dim   : int
  type  : Type

@_dataclass
class ReadConfig(Expr):
  config  : _Config
  field   : str
  type    : Type

# -----------------------------------
# -- QueryAST --> Stmt --> _______ --

@_dataclass
class Assign(Stmt):
  name      : str
  lhs_type  : Type
  idx       : list[Expr]
  rhs       : Expr

@_dataclass
class Reduce(Stmt):
  name      : str
  lhs_type  : Type
  idx       : list[Expr]
  rhs       : Expr

@_dataclass
class WriteConfig(Stmt):
  config    : _Config
  field     : str
  rhs       : Expr

@_dataclass
class Pass(Stmt):
  pass

@_dataclass
class If(Stmt):
  cond      : Expr
  body      : list[Expr]
  orelse    : list[Expr]

@_dataclass
class For(Stmt):
  name      : str
  lo        : Expr
  hi        : Expr
  body      : list[Expr]
  is_par    : bool

@_dataclass
class Alloc(Stmt):
  name      : str
  type      : Type
  memory    : _Optional[_Memory]

@_dataclass
class Call(Stmt):
  proc      : str
  args      : list[Expr]

@_dataclass
class WindowStmt(Stmt):
  name      : str
  rhs       : Expr

# --------------------------
# -- QueryAST --> _______ --

@_dataclass
class FnArg(QueryAST):
  name      : str
  type      : Type
  memory    : _Optional[_Memory]

@_dataclass
class Proc(QueryAST):
  name        : str
  args        : list[FnArg]
  assertions  : list[Expr]
  body        : list[Stmt]
  instruction : _Optional[str]


