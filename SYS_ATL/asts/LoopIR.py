from typing import List, Optional, Type

import attrs
from attrs import validators

from . import _name_validator, OP_STRINGS
from .configs import Config
from ..LoopIR_effects import Effects as E
from ..memory import Memory
from ..prelude import SrcInfo, Sym
from ..query_asts import BuiltIn as QueryBuiltIn


################################################################################
# AST node types

@attrs.frozen
class expr:
    pass


@attrs.frozen
class stmt:
    pass


@attrs.frozen
class w_access:
    pass


@attrs.frozen
class type:
    def shape(self):
        return []

    def ctype(self):
        raise ValueError(f'{self.__class__.__name__} has no ctype')

    def is_real_scalar(self):
        return False

    def is_tensor_or_window(self):
        return False

    def is_win(self):
        return False

    def is_numeric(self):
        return True

    def is_bool(self):
        return False

    def is_indexable(self):
        return False

    def is_stridable(self):
        return False

    def basetype(self):
        return self


################################################################################
# Procedures

@attrs.frozen
class fnarg:
    name: Sym
    type: type
    mem: Optional[Type[Memory]]
    srcinfo: SrcInfo


@attrs.frozen
class proc:
    name: str = attrs.field(validator=_name_validator)
    args: List[fnarg]
    preds: List[expr]
    body: List[stmt]
    instr: Optional[str]
    eff: Optional[E.effect]
    srcinfo: SrcInfo

    def __hash__(self):
        return id(self)


################################################################################
# Statements

@attrs.frozen
class Assign(stmt):
    name: Sym
    type: type
    cast: Optional[str]
    idx: List[expr]
    rhs: expr
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class Reduce(stmt):
    name: Sym
    type: type
    cast: Optional[str]
    idx: List[expr]
    rhs: expr
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class WriteConfig(stmt):
    config: Config
    field: str
    rhs: expr
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class Pass(stmt):
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class If(stmt):
    cond: expr
    body: List[stmt]
    orelse: List[stmt]
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class ForAll(stmt):
    iter: Sym
    hi: expr
    body: List[stmt]
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class Seq(stmt):
    iter: Sym
    hi: expr
    body: List[stmt]
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class Alloc(stmt):
    name: Sym
    type: type
    mem: Optional[Type[Memory]]
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class Free(stmt):
    name: Sym
    type: type
    mem: Optional[Type[Memory]]
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class Call(stmt):
    f: proc
    args: List[expr]
    eff: Optional[E.effect]
    srcinfo: SrcInfo


@attrs.frozen
class WindowStmt(stmt):
    lhs: Sym
    rhs: expr
    eff: Optional[E.effect]
    srcinfo: SrcInfo


################################################################################
# Expressions

@attrs.frozen
class Read(expr):
    name: Sym
    idx: List[expr]
    type: type
    srcinfo: SrcInfo


@attrs.frozen
class Const(expr):
    val: object
    type: type
    srcinfo: SrcInfo


@attrs.frozen
class USub(expr):
    arg: expr
    type: type
    srcinfo: SrcInfo


@attrs.frozen
class BinOp(expr):
    op: str = attrs.field(validator=validators.in_(OP_STRINGS))
    lhs: expr
    rhs: expr
    type: type
    srcinfo: SrcInfo


@attrs.frozen
class BuiltIn(expr):
    f: QueryBuiltIn
    args: List[expr]
    type: type
    srcinfo: SrcInfo


@attrs.frozen
class WindowExpr(expr):
    name: Sym
    idx: List[w_access]
    type: type
    srcinfo: SrcInfo


@attrs.frozen
class StrideExpr(expr):
    name: Sym
    dim: int
    type: type
    srcinfo: SrcInfo


@attrs.frozen
class ReadConfig(expr):
    config: Config
    field: str
    type: type
    srcinfo: SrcInfo


################################################################################
# Window expressions

@attrs.frozen
class Interval(w_access):
    lo: expr
    hi: expr
    srcinfo: SrcInfo


@attrs.frozen
class Point(w_access):
    pt: expr
    srcinfo: SrcInfo


################################################################################
# Types

@attrs.frozen
class Num(type):
    def is_real_scalar(self):
        return True


@attrs.frozen
class F32(type):
    def ctype(self):
        return 'float'

    def is_real_scalar(self):
        return True


@attrs.frozen
class F64(type):
    def ctype(self):
        return 'double'

    def is_real_scalar(self):
        return True


@attrs.frozen
class INT8(type):
    def ctype(self):
        return 'int8_t'

    def is_real_scalar(self):
        return True


@attrs.frozen
class INT32(type):
    def ctype(self):
        return 'int32_t'

    def is_real_scalar(self):
        return True


@attrs.frozen
class Bool(type):
    def ctype(self):
        return 'bool'

    def is_numeric(self):
        return False

    def is_bool(self):
        return True


@attrs.frozen
class Int(type):
    def ctype(self):
        return 'int_fast32_t'

    def is_numeric(self):
        return False

    def is_indexable(self):
        return True

    def is_stridable(self):
        return True


@attrs.frozen
class Index(type):
    def ctype(self):
        return 'int_fast32_t'

    def is_numeric(self):
        return False

    def is_indexable(self):
        return True


@attrs.frozen
class Size(type):
    def ctype(self):
        return 'int_fast32_t'

    def is_numeric(self):
        return False

    def is_indexable(self):
        return True


@attrs.frozen
class Stride(type):
    def ctype(self):
        return 'int_fast32_t'

    def is_numeric(self):
        return False

    def is_stridable(self):
        return True


@attrs.frozen
class Error(type):
    def is_numeric(self):
        return False


def _tensor_type_validator(_0, _1, ty: type):
    return not ty.is_tensor_or_window()


@attrs.frozen
class Tensor(type):
    hi: List[expr]
    is_window: bool
    type: type = attrs.field(validator=_tensor_type_validator)

    def shape(self):
        return self.hi

    def is_tensor_or_window(self):
        return True

    def is_win(self):
        return self.is_window

    def basetype(self):
        return self.type


@attrs.frozen
class WindowType(type):
    src_type: type
    as_tensor: type
    src_buf: Sym
    idx: List[w_access]

    def shape(self):
        return self.as_tensor.shape()

    def is_tensor_or_window(self):
        return True

    def is_win(self):
        return True

    def basetype(self):
        return self.as_tensor.basetype()
