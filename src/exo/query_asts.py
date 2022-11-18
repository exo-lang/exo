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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .configs import Config
from .memory import Memory

__all__ = [
    "QueryAST",
    "Type",
    "Expr",
    "WAccess",
    "Stmt",
    "R",
    "f32",
    "f64",
    "i8",
    "i32",
    "bool",
    "int",
    "index",
    "size",
    "stride",
    "tensor",
    "Interval",
    "Point",
    "Read",
    "Const",
    "USub",
    "BinOp",
    "BuiltIn",
    "WindowExpr",
    "StrideExpr",
    "ReadConfig",
    "Assign",
    "Reduce",
    "WriteConfig",
    "Pass",
    "If",
    "For",
    "Alloc",
    "Call",
    "WindowStmt",
    "FnArg",
    "Proc",
]


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


@dataclass
class R(Type):
    pass


@dataclass
class f32(Type):
    pass


@dataclass
class f64(Type):
    pass


@dataclass
class i8(Type):
    pass


@dataclass
class i32(Type):
    pass


@dataclass
class bool(Type):
    pass


@dataclass
class int(Type):
    pass


@dataclass
class index(Type):
    pass


@dataclass
class size(Type):
    pass


@dataclass
class stride(Type):
    pass


@dataclass
class tensor(Type):
    hi: list[Expr]
    is_window: bool
    type: Type


# --------------------------------------
# -- QueryAST --> WAccess --> _______ --


@dataclass
class Interval(WAccess):
    lo: Expr
    hi: Expr


@dataclass
class Point(WAccess):
    pt: Expr


# -----------------------------------
# -- QueryAST --> Expr --> _______ --


@dataclass
class Read(Expr):
    name: str
    idx: list[Expr]
    type: Type


@dataclass
class Const(Expr):
    val: Any
    type: Type


@dataclass
class USub(Expr):
    arg: Expr
    type: Type


@dataclass
class BinOp(Expr):
    op: str
    lhs: Expr
    rhs: Expr
    type: Type


@dataclass
class BuiltIn(Expr):
    func: str
    args: list[Expr]
    type: Type


@dataclass
class WindowExpr(Expr):
    name: str
    idx: list[WAccess]
    type: Type


@dataclass
class StrideExpr(Expr):
    name: str
    dim: int
    type: Type


@dataclass
class ReadConfig(Expr):
    config: Config
    field: str
    type: Type


# -----------------------------------
# -- QueryAST --> Stmt --> _______ --


@dataclass
class Assign(Stmt):
    name: str
    lhs_type: Type
    idx: list[Expr]
    rhs: Expr


@dataclass
class Reduce(Stmt):
    name: str
    lhs_type: Type
    idx: list[Expr]
    rhs: Expr


@dataclass
class WriteConfig(Stmt):
    config: Config
    field: str
    rhs: Expr


@dataclass
class Pass(Stmt):
    pass


@dataclass
class If(Stmt):
    cond: Expr
    body: list[Expr]
    orelse: list[Expr]


@dataclass
class For(Stmt):
    name: str
    lo: Expr
    hi: Expr
    body: list[Expr]
    is_par: bool


@dataclass
class Alloc(Stmt):
    name: str
    type: Type
    memory: Optional[Memory]


@dataclass
class Call(Stmt):
    proc: str
    args: list[Expr]


@dataclass
class WindowStmt(Stmt):
    name: str
    rhs: Expr


# --------------------------
# -- QueryAST --> _______ --


@dataclass
class FnArg(QueryAST):
    name: str
    type: Type
    memory: Optional[Memory]


@dataclass
class Proc(QueryAST):
    name: str
    args: list[FnArg]
    assertions: list[Expr]
    body: list[Stmt]
    instruction: Optional[str]
