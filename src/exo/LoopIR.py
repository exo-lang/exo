import re
from collections import ChainMap
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
        raise ValueError(f"invalid identifier: {name}")


class IdentifierOrHole(str):
    _valid_re = re.compile(r"^[a-zA-Z_]\w*$")

    def __new__(cls, name):
        name = str(name)
        if IdentifierOrHole._valid_re.match(name):
            return super().__new__(cls, name)
        raise ValueError(f"invalid identifier: {name}")


comparision_ops = {"<", ">", "<=", ">=", "=="}
arithmetic_ops = {"+", "-", "*", "/", "%"}
logical_ops = {"and", "or"}

front_ops = comparision_ops | arithmetic_ops | logical_ops


class Operator(str):
    def __new__(cls, op):
        op = str(op)
        if op in front_ops:
            return super().__new__(cls, op)
        raise ValueError(f"invalid operator: {op}")


# --------------------------------------------------------------------------- #
# Loop IR
# --------------------------------------------------------------------------- #


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
         | For( sym iter, expr lo, expr hi, stmt* body, loop_mode loop_mode )
         | Alloc( sym name, type type, mem? mem )
         | Free( sym name, type type, mem? mem )
         | Call( proc f, expr* args )
         | WindowStmt( sym lhs, expr rhs )
         attributes( effect? eff, srcinfo srcinfo )

    loop_mode = Seq()
                | Par()

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
         | UINT8()
         | UINT16()
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
        "UINT16",
        "UINT8",
        "INT32" "Bool",
        "Int",
        "Index",
        "Size",
        "Stride",
        "Error",
    },
)

# --------------------------------------------------------------------------- #
# Untyped AST
# --------------------------------------------------------------------------- #

UAST = ADT(
    """
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
            | For     ( sym iter,  expr cond,   stmt* body )
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
            | F16   ()
            | F32   ()
            | F64   ()
            | INT8  ()
            | UINT8  ()
            | UINT16 ()
            | INT32 ()
            | Bool  ()
            | Int   ()
            | Size  ()
            | Index ()
            | Stride()
            | Tensor( expr *hi, bool is_window, type type )
} """,
    ext_types={
        "name": validators.instance_of(Identifier, convert=True),
        "sym": Sym,
        "mem": Type[Memory],
        "builtin": BuiltIn,
        "config": Config,
        "loopir_proc": LoopIR.proc,
        "op": validators.instance_of(Operator, convert=True),
        "srcinfo": SrcInfo,
    },
    memoize={
        "Num",
        "F16",
        "F32",
        "F64",
        "INT8",
        "UINT8",
        "UINT16",
        "INT32",
        "Bool",
        "Int",
        "Size",
        "Index",
        "Stride",
    },
)

# --------------------------------------------------------------------------- #
# Pattern AST
#   - used to specify pattern-matches
# --------------------------------------------------------------------------- #

PAST = ADT(
    """
module PAST {

    stmt    = Assign  ( name name, expr* idx, expr rhs )
            | Reduce  ( name name, expr* idx, expr rhs )
            | Pass    ()
            | If      ( expr cond, stmt* body, stmt* orelse )
            | For     ( name iter, expr lo, expr hi, stmt* body )
            | Alloc   ( name name, expr* sizes ) -- may want to add mem back in?
            | Call    ( name f, expr* args )
            | WriteConfig ( name config, name field )
            | S_Hole  ()
            attributes( srcinfo srcinfo )

    expr    = Read    ( name name, expr* idx )
            | StrideExpr( name name, int? dim )
            | E_Hole  ()
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            | BuiltIn ( builtin f, expr* args )
            | ReadConfig( string config, string field )
            attributes( srcinfo srcinfo )

} """,
    ext_types={
        "name": validators.instance_of(IdentifierOrHole, convert=True),
        "builtin": BuiltIn,
        "op": validators.instance_of(Operator, convert=True),
        "srcinfo": SrcInfo,
    },
)


# --------------------------------------------------------------------------- #
# C Codegen AST
# --------------------------------------------------------------------------- #

CIR = ADT(
    """
module CIR {

    expr    = Read    ( sym name, bool is_non_neg )
            | Stride  ( sym name, int dim )
            | Const   ( object val )
            | BinOp   ( op op, expr lhs, expr rhs, bool is_non_neg )
            | USub    ( expr arg, bool is_non_neg )

} """,
    ext_types={
        "bool": bool,
        "int": int,
        "sym": Sym,
        "op": validators.instance_of(Operator, convert=True),
    },
)


# --------------------------------------------------------------------------- #
# Effects
# --------------------------------------------------------------------------- #

Effects = ADT(
    """
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

} """,
    {
        "sym": Sym,
        "type": LoopIR.type,
        "binop": validators.instance_of(Operator, convert=True),
        "config": Config,
        "srcinfo": SrcInfo,
    },
)


# --------------------------------------------------------------------------- #
# Extension methods
# --------------------------------------------------------------------------- #


@extclass(UAST.Tensor)
@extclass(UAST.Num)
@extclass(UAST.F16)
@extclass(UAST.F32)
@extclass(UAST.F64)
@extclass(UAST.INT8)
@extclass(UAST.UINT8)
@extclass(UAST.UINT16)
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
    Num = LoopIR.Num
    F16 = LoopIR.F16
    F32 = LoopIR.F32
    F64 = LoopIR.F64
    INT8 = LoopIR.INT8
    UINT8 = LoopIR.UINT8
    UINT16 = LoopIR.UINT16
    INT32 = LoopIR.INT32
    Bool = LoopIR.Bool
    Int = LoopIR.Int
    Index = LoopIR.Index
    Size = LoopIR.Size
    Stride = LoopIR.Stride
    Error = LoopIR.Error
    Tensor = LoopIR.Tensor
    Window = LoopIR.WindowType
    type = LoopIR.type
    R = Num()
    f16 = F16()
    f32 = F32()
    int8 = INT8()
    uint8 = UINT8()
    uint16 = UINT16()
    i8 = INT8()
    ui8 = UINT8()
    ui16 = UINT16()
    int32 = INT32()
    i32 = INT32()
    f64 = F64()
    bool = Bool()  # note: accessed as T.bool outside this module
    int = Int()
    index = Index()
    size = Size()
    stride = Stride()
    err = Error()


# --------------------------------------------------------------------------- #
# type helper functions


@extclass(T.Tensor)
@extclass(T.Window)
@extclass(T.Num)
@extclass(T.F16)
@extclass(T.F32)
@extclass(T.F64)
@extclass(T.INT8)
@extclass(T.UINT8)
@extclass(T.UINT16)
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
@extclass(T.F16)
@extclass(T.F32)
@extclass(T.F64)
@extclass(T.INT8)
@extclass(T.UINT8)
@extclass(T.UINT16)
@extclass(T.INT32)
@extclass(T.Bool)
@extclass(T.Int)
@extclass(T.Index)
@extclass(T.Size)
@extclass(T.Stride)
def ctype(t):
    if isinstance(t, T.Num):
        assert False, "Don't ask for ctype of Num"
    elif isinstance(t, T.F16):
        return "_Float16"
    elif isinstance(t, T.F32):
        return "float"
    elif isinstance(t, T.F64):
        return "double"
    elif isinstance(t, T.INT8):
        return "int8_t"
    elif isinstance(t, T.UINT8):
        return "uint8_t"
    elif isinstance(t, T.UINT16):
        return "uint16_t"
    elif isinstance(t, T.INT32):
        return "int32_t"
    elif isinstance(t, T.Bool):
        return "bool"
    elif isinstance(t, (T.Int, T.Index, T.Size, T.Stride)):
        return "int_fast32_t"


del ctype


@extclass(LoopIR.type)
def is_real_scalar(t):
    return isinstance(
        t, (T.Num, T.F16, T.F32, T.F64, T.INT8, T.UINT8, T.UINT16, T.INT32)
    )


del is_real_scalar


@extclass(LoopIR.type)
def is_tensor_or_window(t):
    return isinstance(t, (T.Tensor, T.Window))


del is_tensor_or_window


@extclass(LoopIR.type)
def is_win(t):
    return (isinstance(t, T.Tensor) and t.is_window) or isinstance(t, T.Window)


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

# Install string printing functions on LoopIR, UAST and T
# This must be imported after those objects are defined to
# prevent circular inclusion problems
# TODO: FIX THIS!!!
# noinspection PyUnresolvedReferences
from . import LoopIR_pprint


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
        lhs = lift_to_eff_expr(e.lhs)
        rhs = lift_to_eff_expr(e.rhs)
        return Effects.BinOp(e.op, lhs, rhs, e.type, e.srcinfo)

    elif isinstance(e, LoopIR.USub):
        zero = Effects.Const(0, e.type, e.srcinfo)
        arg = lift_to_eff_expr(e.arg)
        return Effects.BinOp("-", zero, arg, e.type, e.srcinfo)

    elif isinstance(e, LoopIR.StrideExpr):
        return Effects.Stride(e.name, e.dim, e.type, e.srcinfo)

    elif isinstance(e, LoopIR.ReadConfig):
        cfg_val = e.config.lookup(e.field)[1]
        return Effects.ConfigField(e.config, e.field, cfg_val, e.srcinfo)

    assert False, f"bad case, e is {type(e)}"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Standard Pass Templates for Loop IR


class LoopIR_Rewrite:
    def apply_proc(self, old):
        return self.map_proc(old) or old

    def apply_fnarg(self, old):
        return self.map_fnarg(old) or old

    def apply_stmts(self, old):
        if (new := self.map_stmts(old)) is not None:
            return new
        return old

    def apply_exprs(self, old):
        if (new := self.map_exprs(old)) is not None:
            return new
        return old

    def apply_s(self, old):
        if (new := self.map_s(old)) is not None:
            return new
        return [old]

    def apply_e(self, old):
        return self.map_e(old) or old

    def apply_w_access(self, old):
        return self.map_w_access(old) or old

    def apply_t(self, old):
        return self.map_t(old) or old

    def map_proc(self, p):
        new_args = self._map_list(self.map_fnarg, p.args)
        new_preds = self.map_exprs(p.preds)
        new_body = self.map_stmts(p.body)

        if any((new_args is not None, new_preds is not None, new_body is not None)):
            new_preds = new_preds or p.preds
            new_preds = [
                p for p in new_preds if not (isinstance(p, LoopIR.Const) and p.val)
            ]
            return p.update(
                args=new_args or p.args, preds=new_preds, body=new_body or p.body
            )

        return None

    def map_fnarg(self, a):
        if t := self.map_t(a.type):
            return a.update(type=t)

        return None

    def map_stmts(self, stmts):
        return self._map_list(self.map_s, stmts)

    def map_exprs(self, exprs):
        return self._map_list(self.map_e, exprs)

    def map_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            new_type = self.map_t(s.type)
            new_idx = self.map_exprs(s.idx)
            new_rhs = self.map_e(s.rhs)
            if any((new_type, new_idx is not None, new_rhs)):
                return [
                    s.update(
                        type=new_type or s.type,
                        idx=new_idx or s.idx,
                        rhs=new_rhs or s.rhs,
                    )
                ]
        elif isinstance(s, (LoopIR.WriteConfig, LoopIR.WindowStmt)):
            new_rhs = self.map_e(s.rhs)
            if new_rhs:
                return [s.update(rhs=new_rhs or s.rhs)]
        elif isinstance(s, LoopIR.If):
            new_cond = self.map_e(s.cond)
            new_body = self.map_stmts(s.body)
            new_orelse = self.map_stmts(s.orelse)
            if any((new_cond, new_body is not None, new_orelse is not None)):
                return [
                    s.update(
                        cond=new_cond or s.cond,
                        body=new_body or s.body,
                        orelse=new_orelse or s.orelse,
                    )
                ]
        elif isinstance(s, LoopIR.For):
            new_lo = self.map_e(s.lo)
            new_hi = self.map_e(s.hi)
            new_body = self.map_stmts(s.body)
            if any((new_lo, new_hi, new_body is not None)):
                return [
                    s.update(
                        lo=new_lo or s.lo, hi=new_hi or s.hi, body=new_body or s.body
                    )
                ]
        elif isinstance(s, LoopIR.Call):
            new_args = self.map_exprs(s.args)
            if new_args is not None:
                return [s.update(args=new_args or s.args)]
        elif isinstance(s, LoopIR.Alloc):
            new_type = self.map_t(s.type)
            if new_type:
                return [s.update(type=new_type or s.type)]
        elif isinstance(s, LoopIR.Pass):
            return None
        else:
            raise NotImplementedError(f"bad case {type(s)}")
        return None

    def map_e(self, e):
        if isinstance(e, LoopIR.Read):
            new_type = self.map_t(e.type)
            new_idx = self.map_exprs(e.idx)
            if any((new_type, new_idx is not None)):
                return e.update(
                    idx=new_idx or e.idx,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.BinOp):
            new_lhs = self.map_e(e.lhs)
            new_rhs = self.map_e(e.rhs)
            new_type = self.map_t(e.type)
            if any((new_lhs, new_rhs, new_type)):
                return e.update(
                    lhs=new_lhs or e.lhs,
                    rhs=new_rhs or e.rhs,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.BuiltIn):
            new_type = self.map_t(e.type)
            new_args = self.map_exprs(e.args)
            if any((new_type, new_args is not None)):
                return e.update(
                    args=new_args or e.args,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.USub):
            new_arg = self.map_e(e.arg)
            new_type = self.map_t(e.type)
            if any((new_arg, new_type)):
                return e.update(
                    arg=new_arg or e.arg,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.WindowExpr):
            new_idx = self._map_list(self.map_w_access, e.idx)
            new_type = self.map_t(e.type)
            if any((new_idx is not None, new_type)):
                return e.update(
                    idx=new_idx or e.idx,
                    type=new_type or e.type,
                )
        elif isinstance(e, LoopIR.ReadConfig):
            if new_type := self.map_t(e.type):
                return e.update(type=new_type or e.type)
        elif isinstance(e, (LoopIR.Const, LoopIR.StrideExpr)):
            return None
        else:
            raise NotImplementedError(f"bad case {type(e)}")
        return None

    def map_w_access(self, w):
        if isinstance(w, LoopIR.Interval):
            new_lo = self.map_e(w.lo)
            new_hi = self.map_e(w.hi)
            if new_lo or new_hi:
                return w.update(
                    lo=new_lo or w.lo,
                    hi=new_hi or w.hi,
                )
        else:
            if new_pt := self.map_e(w.pt):
                return w.update(pt=new_pt or w.pt)
        return None

    def map_t(self, t):
        if isinstance(t, T.Tensor):
            new_hi = self.map_exprs(t.hi)
            new_type = self.map_t(t.type)
            if (new_hi is not None) or new_type:
                return t.update(hi=new_hi or t.hi, type=new_type or t.type)
        elif isinstance(t, T.Window):
            new_src_type = self.map_t(t.src_type)
            new_as_tensor = self.map_t(t.as_tensor)
            new_idx = self._map_list(self.map_w_access, t.idx)
            if new_src_type or new_as_tensor or (new_idx is not None):
                return t.update(
                    src_type=new_src_type or t.src_type,
                    as_tensor=new_as_tensor or t.as_tensor,
                    idx=new_idx or t.idx,
                )
        return None

    @staticmethod
    def _map_list(fn, nodes):
        new_stmts = []
        needs_update = False

        for s in nodes:
            s2 = fn(s)
            if s2 is None:
                new_stmts.append(s)
            else:
                needs_update = True
                if isinstance(s2, list):
                    new_stmts.extend(s2)
                else:
                    new_stmts.append(s2)

        if not needs_update:
            return None

        return new_stmts


class LoopIR_Do:
    def __init__(self, proc, *args, **kwargs):
        self.proc = proc

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
        elif styp is LoopIR.For:
            self.do_e(s.lo)
            self.do_e(s.hi)
            self.do_stmts(s.body)
        elif styp is LoopIR.Call:
            for e in s.args:
                self.do_e(e)
        elif styp is LoopIR.Alloc:
            self.do_t(s.type)
        else:
            pass

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
        else:
            assert False, "bad case"

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


class LoopIR_Compare:
    def __init__(self):
        pass

    def match_stmts(self, stmts1, stmts2):
        return all(self.match_s(s1, s2) for s1, s2 in zip(stmts1, stmts2))

    def match_s(self, s1, s2):
        if type(s1) is not type(s2):
            return False

        if isinstance(s1, (LoopIR.Assign, LoopIR.Reduce)):
            return (
                self.match_name(s1.name, s2.name)
                and self.match_t(s1.type, s2.type)
                and all(self.match_e(i1, i2) for i1, i2 in zip(s1.idx, s2.idx))
                and self.match_e(s1.rhs, s2.rhs)
            )
        elif isinstance(s1, LoopIR.WriteConfig):
            # TODO: check config and field equality
            return (
                s1.config == s2.config
                and s1.field == s2.field
                and self.match_e(s1.rhs, s2.rhs)
            )
        elif isinstance(s1, LoopIR.Pass):
            return True
        elif isinstance(s1, LoopIR.If):
            return (
                self.match_e(s1.cond, s2.cond)
                and self.match_stmts(s1.body, s2.body)
                and self.match_stmts(s1.orelse, s2.orelse)
            )
        elif isinstance(s1, LoopIR.For):
            return (
                self.match_name(s1.iter, s2.iter)
                and self.match_e(s1.lo, s2.lo)
                and self.match_e(s1.hi, s2.hi)
                and self.match_stmts(s1.body, s2.body)
            )
        elif isinstance(s1, LoopIR.Alloc):
            return self.match_name(s1.name, s2.name) and self.match_t(s1.type, s2.type)
        elif isinstance(s1, LoopIR.Call):
            return s1.f == s2.f and all(
                self.match_e(a1, a2) for a1, a2 in zip(s1.args, s2.args)
            )
        elif isinstance(s1, LoopIR.WindowStmt):
            return self.match_name(s1.lhs, s2.lhs) and self.match_e(s1.rhs, s2.rhs)
        else:
            assert False, f"bad case: {type(s1)}"

    def match_e(self, e1, e2):
        if type(e1) is not type(e2):
            return False

        if isinstance(e1, LoopIR.Read):
            return self.match_name(e1.name, e2.name) and all(
                self.match_e(i1, i2) for i1, i2 in zip(e1.idx, e2.idx)
            )
        elif isinstance(e1, LoopIR.Const):
            return e1.val == e2.val
        elif isinstance(e1, LoopIR.USub):
            return self.match_e(e1.arg, e2.arg)
        elif isinstance(e1, LoopIR.BinOp):
            return (
                e1.op == e2.op
                and self.match_e(e1.lhs, e2.lhs)
                and self.match_e(e1.rhs, e2.rhs)
            )
        elif isinstance(e1, LoopIR.BuiltIn):
            # TODO: check f equality
            return e1.f is e2.f and all(
                self.match_e(a1, a2) for a1, a2 in zip(e1.args, e2.args)
            )
        elif isinstance(e1, LoopIR.WindowExpr):
            return self.match_name(e1.name, e2.name) and all(
                self.match_w_access(w1, w2) for w1, w2 in zip(e1.idx, e2.idx)
            )
        elif isinstance(e1, LoopIR.StrideExpr):
            return self.match_name(e1.name, e2.name) and e1.dim == e2.dim
        elif isinstance(e1, LoopIR.ReadConfig):
            # TODO: check configfield equality
            return e1.config == e2.config and e1.field == e2.field
        else:
            assert False, "bad case"

    def match_name(self, n1, n2):
        # TODO: if its a free var, check for exact match using ID. This
        # doesn't matter for join_loops, but in general if we use this
        # anywhere else, we should reason about that.
        return n1.name() == n2.name()

    def match_w_access(self, w1, w2):
        if isinstance(w1, LoopIR.Interval):
            return self.match_e(w1.lo, w2.lo) and self.match_e(w1.hi, w2.hi)
        elif isinstance(w1, LoopIR.Point):
            return self.match_e(w1.pt, w2.pt)
        else:
            assert False, "bad case"

    def match_t(self, t1, t2):
        if isinstance(t1, LoopIR.Tensor):
            return (
                t1.is_window == t2.is_window
                and self.match_t(t1.type, t2.type)
                and all(self.match_e(i1, i2) for i1, i2 in zip(t1.hi, t2.hi))
            )
        else:  # scalar
            return type(t1) == type(t2)


class GetReads(LoopIR_Do):
    def __init__(self):
        self.reads = []

    def do_e(self, e):
        if isinstance(e, LoopIR.Read):
            self.reads.append((e.name, e.type))
        super().do_e(e)


def get_reads_of_expr(e):
    gr = GetReads()
    gr.do_e(e)
    return gr.reads


def get_reads_of_stmts(stmts):
    gr = GetReads()
    for stmt in stmts:
        gr.do_s(stmt)
    return gr.reads


class GetWrites(LoopIR_Do):
    def __init__(self):
        self.writes = []

    def do_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            self.writes.append((s.name, s.type))
        elif isinstance(s, LoopIR.Call):
            writes_in_subproc = [a for a, _ in get_writes_of_stmts(s.f.body)]
            for arg, call_arg in zip(s.args, s.f.args):
                if call_arg.name in writes_in_subproc:
                    if isinstance(
                        arg, (LoopIR.Read, LoopIR.WindowExpr, LoopIR.StrideExpr)
                    ):
                        self.writes.append((arg.name, arg.type))

        super().do_s(s)

    # early exit
    def do_e(self, e):
        return


def get_writes_of_stmts(stmts):
    gw = GetWrites()
    gw.do_stmts(stmts)
    return gw.writes


def is_const_zero(e):
    return isinstance(e, LoopIR.Const) and e.val == 0


class FreeVars(LoopIR_Do):
    def __init__(self, node):
        assert isinstance(node, list)
        self.env = ChainMap()
        self.fv = set()

        for n in node:
            if isinstance(n, LoopIR.stmt):
                self.do_s(n)
            elif isinstance(n, LoopIR.expr):
                self.do_e(n)
            else:
                assert False, "expected stmt or expr"

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
            return
        elif styp is LoopIR.For:
            self.do_e(s.lo)
            self.do_e(s.hi)
            self.push()
            self.env[s.iter] = True
            self.do_stmts(s.body)
            self.pop()
            return
        elif styp is LoopIR.Alloc:
            self.env[s.name] = True

        super().do_s(s)

    def do_e(self, e):
        etyp = type(e)
        if (
            etyp is LoopIR.Read
            or etyp is LoopIR.WindowExpr
            or etyp is LoopIR.StrideExpr
        ):
            if e.name not in self.env:
                self.fv.add(e.name)

        super().do_e(e)

    def do_t(self, t):
        if isinstance(t, T.Window):
            if t.src_buf not in self.env:
                self.fv.add(t.src_buf)

        super().do_t(t)


class Alpha_Rename(LoopIR_Rewrite):
    def __init__(self, node):
        self.env = ChainMap()
        self.node = []

        if isinstance(node, LoopIR.proc):
            self.node = self.apply_proc(node)
        else:
            assert isinstance(node, list)
            for n in node:
                if isinstance(n, LoopIR.stmt):
                    self.node += self.apply_s(n)
                elif isinstance(n, LoopIR.expr):
                    self.node += [self.apply_e(n)]
                else:
                    assert False, "expected stmt or expr"

    def result(self):
        return self.node

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    def map_fnarg(self, fa):
        nm = fa.name.copy()
        self.env[fa.name] = nm
        return fa.update(name=nm, type=self.map_t(fa.type) or fa.type)

    def map_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            s2 = super().map_s(s)
            if new_name := self.env.get(s.name):
                return [((s2 and s2[0]) or s).update(name=new_name)]
            else:
                return s2
        elif isinstance(s, LoopIR.Alloc):
            s2 = super().map_s(s)
            assert s.name not in self.env
            new_name = s.name.copy()
            self.env[s.name] = new_name
            return [((s2 and s2[0]) or s).update(name=new_name)]
        elif isinstance(s, LoopIR.WindowStmt):
            rhs = self.map_e(s.rhs) or s.rhs
            lhs = s.lhs.copy()
            self.env[s.lhs] = lhs
            return [s.update(lhs=lhs, rhs=rhs)]
        elif isinstance(s, LoopIR.If):
            self.push()
            stmts = super().map_s(s)
            self.pop()
            return stmts
        elif isinstance(s, LoopIR.For):
            lo = self.map_e(s.lo) or s.lo
            hi = self.map_e(s.hi) or s.hi

            self.push()
            itr = s.iter.copy()
            self.env[s.iter] = itr
            body = self.map_stmts(s.body) or s.body
            self.pop()

            return [s.update(iter=itr, lo=lo, hi=hi, body=body)]

        return super().map_s(s)

    def map_e(self, e):
        if isinstance(e, (LoopIR.Read, LoopIR.WindowExpr, LoopIR.StrideExpr)):
            e2 = super().map_e(e)
            if new_name := self.env.get(e.name):
                return (e2 or e).update(name=new_name)
            else:
                return e2

        return super().map_e(e)

    def map_t(self, t):
        t2 = super().map_t(t)

        if isinstance(t, T.Window):
            if src_buf := self.env.get(t.src_buf):
                return (t2 or t).update(src_buf=src_buf)

        return t2


class SubstArgs(LoopIR_Rewrite):
    def __init__(self, nodes, binding):
        assert isinstance(nodes, list)
        assert isinstance(binding, dict)
        assert all(isinstance(v, LoopIR.expr) for v in binding.values())
        assert not any(isinstance(v, LoopIR.WindowExpr) for v in binding.values())
        self.env = binding
        self.nodes = []
        for n in nodes:
            if isinstance(n, LoopIR.stmt):
                self.nodes += self.apply_s(n)
            elif isinstance(n, LoopIR.expr):
                self.nodes += [self.apply_e(n)]
            else:
                assert False, "expected stmt or expr"

    def result(self):
        return self.nodes

    def map_s(self, s):
        s2 = super().map_s(s)
        s_new = s2[0] if s2 is not None else s

        # this substitution could refer to a read or a window expression
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if s.name in self.env:
                sym = self.env[s.name]
                assert isinstance(sym, LoopIR.Read) and len(sym.idx) == 0
                return [s_new.update(name=sym.name)]

        return s2

    def map_e(self, e):
        # this substitution could refer to a read or a window expression
        if isinstance(e, LoopIR.Read):
            if e.name in self.env:
                sub_e = self.env[e.name]

                if not e.idx:
                    return sub_e

                assert isinstance(sub_e, LoopIR.Read) and len(sub_e.idx) == 0
                return e.update(name=sub_e.name, idx=self.apply_exprs(e.idx))

        elif isinstance(e, LoopIR.WindowExpr):
            if e.name in self.env:
                sub_e = self.env[e.name]

                if not e.idx:
                    return sub_e

                assert isinstance(sub_e, LoopIR.Read) and len(sub_e.idx) == 0
                return (super().map_e(e) or e).update(name=sub_e.name)

        elif isinstance(e, LoopIR.StrideExpr):
            if e.name in self.env:
                return e.update(name=self.env[e.name].name)

        return super().map_e(e)

    def map_t(self, t):
        t2 = super().map_t(t)

        if isinstance(t, T.Window):
            if src_buf := self.env.get(t.src_buf):
                return (t2 or t).update(src_buf=src_buf.name)

        return t2
