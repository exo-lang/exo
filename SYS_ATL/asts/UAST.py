import typing
from typing import List, Optional, Type

import attrs
from attrs import validators

from . import OP_STRINGS
from .configs import Config
from ..memory import Memory
from ..prelude import Sym, is_valid_name, SrcInfo
from ..query_asts import BuiltIn as QueryBuiltIn


# Put common types first for subsequent type annotations

@attrs.frozen
class type:
    def shape(self):
        return []

    def basetype(self):
        return self


@attrs.frozen
class w_access:
    pass


@attrs.frozen
class expr:
    pass


@attrs.frozen
class stmt:
    pass


# Procedures

@attrs.frozen
class fnarg:
    name: Sym
    type: type
    mem: Optional[Type[Memory]]
    srcinfo: SrcInfo


@attrs.frozen
class proc:
    name: Optional[str] = attrs.field(validator=lambda *x: is_valid_name(x[2]))
    args: List[fnarg]
    preds: List[expr]
    body: List[stmt]
    instr: Optional[str]
    srcinfo: SrcInfo


# Types

@attrs.frozen
class Num(type):
    pass


@attrs.frozen
class F32(type):
    pass


@attrs.frozen
class F64(type):
    pass


@attrs.frozen
class INT8(type):
    pass


@attrs.frozen
class INT32(type):
    pass


@attrs.frozen
class Bool(type):
    pass


@attrs.frozen
class Int(type):
    pass


@attrs.frozen
class Size(type):
    pass


@attrs.frozen
class Index(type):
    pass


@attrs.frozen
class Stride(type):
    pass


@attrs.frozen
class Tensor(type):
    hi: List[expr]
    is_window: bool
    type: type

    def shape(self):
        return self.hi

    def basetype(self):
        return self.type


# Window accesses

@attrs.frozen
class Interval(w_access):
    lo: Optional[expr]
    hi: Optional[expr]
    srcinfo: SrcInfo


@attrs.frozen
class Point(w_access):
    pt: expr
    srcinfo: SrcInfo


# Expressions

@attrs.frozen
class Read(expr):
    name: Sym
    idx: List[expr]
    srcinfo: SrcInfo


@attrs.frozen
class Const(expr):
    val: object
    srcinfo: SrcInfo


@attrs.frozen
class USub(expr):
    arg: expr
    srcinfo: SrcInfo


@attrs.frozen
class BinOp(expr):
    op: str = attrs.field(validator=validators.in_(OP_STRINGS))
    lhs: expr
    rhs: expr
    srcinfo: SrcInfo


@attrs.frozen
class BuiltIn(expr):
    f: QueryBuiltIn
    args: List[expr]
    srcinfo: SrcInfo


@attrs.frozen
class WindowExpr(expr):
    name: Sym
    idx: List[w_access]
    srcinfo: SrcInfo


@attrs.frozen
class StrideExpr(expr):
    name: Sym
    dim: int
    srcinfo: SrcInfo


@attrs.frozen
class ParRange(expr):
    lo: expr
    hi: expr
    srcinfo: SrcInfo


@attrs.frozen
class SeqRange(expr):
    lo: expr
    hi: expr
    srcinfo: SrcInfo


@attrs.frozen
class ReadConfig(expr):
    config: Config
    field: str
    srcinfo: SrcInfo


# Statements

@attrs.frozen
class Assign(stmt):
    name: Sym
    idx: List[expr]
    rhs: expr
    srcinfo: SrcInfo


@attrs.frozen
class Reduce(stmt):
    name: Sym
    idx: List[expr]
    rhs: expr
    srcinfo: SrcInfo


@attrs.frozen
class WriteConfig(stmt):
    config: Config
    field: str
    rhs: expr
    srcinfo: SrcInfo


@attrs.frozen
class FreshAssign(stmt):
    name: Sym
    rhs: expr
    srcinfo: SrcInfo


@attrs.frozen
class Pass(stmt):
    srcinfo: SrcInfo


@attrs.frozen
class If(stmt):
    cond: expr
    body: List[stmt]
    orelse: List[stmt]
    srcinfo: SrcInfo


@attrs.frozen
class ForAll(stmt):
    iter: Sym
    cond: expr
    body: List[stmt]
    srcinfo: SrcInfo


@attrs.frozen
class Alloc(stmt):
    name: Sym
    type: type
    mem: Optional[typing.Type[Memory]]
    srcinfo: SrcInfo


@attrs.frozen
class Call(stmt):
    f: proc
    args: List[expr]
    srcinfo: SrcInfo
