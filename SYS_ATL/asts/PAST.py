from typing import List

import attrs
from attrs import validators

from . import OP_STRINGS
from ..prelude import *


def _name_validator(_0, _1, name):
    return is_valid_name(name)


@attrs.frozen
class expr:
    pass


@attrs.frozen
class stmt:
    pass


# Expressions

@attrs.frozen
class Read(expr):
    name: str = attrs.field(validator=_name_validator)
    idx: List[expr]
    srcinfo: SrcInfo


@attrs.frozen
class StrideExpr(expr):
    name: str = attrs.field(validator=_name_validator)
    dim: int
    srcinfo: SrcInfo


@attrs.frozen
class E_Hole(expr):
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


# Statements

@attrs.frozen
class Assign(stmt):
    name: str = attrs.field(validator=_name_validator)
    idx: List[expr]
    rhs: List[expr]
    srcinfo: SrcInfo


@attrs.frozen
class Reduce(stmt):
    name: str = attrs.field(validator=_name_validator)
    idx: List[expr]
    rhs: List[expr]
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
    iter: str = attrs.field(validator=_name_validator)
    hi: expr
    body: List[stmt]
    srcinfo: SrcInfo


@attrs.frozen
class Seq(stmt):
    iter: str = attrs.field(validator=_name_validator)
    hi: expr
    body: List[stmt]
    srcinfo: SrcInfo


@attrs.frozen
class Alloc(stmt):
    # may want to add type & mem back in?
    name: str = attrs.field(validator=_name_validator)
    srcinfo: SrcInfo


@attrs.frozen
class Call(stmt):
    f: str = attrs.field(validator=_name_validator)
    args: List[expr]
    srcinfo: SrcInfo


@attrs.frozen
class WriteConfig(stmt):
    config: str = attrs.field(validator=_name_validator)
    field: str = attrs.field(validator=_name_validator)
    srcinfo: SrcInfo


@attrs.frozen
class S_Hole(stmt):
    srcinfo: SrcInfo
