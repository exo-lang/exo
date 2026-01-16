import atexit
import functools
import re
import sys
from collections import ChainMap, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type, Optional, Union, Set

from asdl_adt import ADT, validators

from ..API_types import ExoType
from .extern import Extern
from .configs import Config
from .instr_info import InstrInfo
from .memory import (
    DRAM,
    MemWin,
    AllocableMemWin,
    Memory,
    SpecialWindow,
    MemGlobalC,
    MemIncludeC,
)

from .prelude import (
    Sym,
    SrcInfo,
    extclass,
    Identifier,
    IdentifierOrHole,
    Operator,
    comparison_ops,
    arithmetic_ops,
    logical_ops,
    front_ops,
    ScalarInfo,
)

from ..spork.timelines import Instr_tl, cpu_in_order_instr, Sync_tl
from ..spork.base_with_context import BaseWithContext
from ..spork.coll_algebra import CollUnit, standalone_thread
from ..spork.loop_modes import LoopMode
from ..spork.sync_types import SyncType, fence_type


# TODO fix typo...
comparision_ops = comparison_ops


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
             instr?  instr,
             srcinfo srcinfo )

    fnarg  = ( sym     name,
               type    type,
               memwin? mem,
               srcinfo srcinfo )

    stmt = Assign( sym name, type type, expr* idx, expr rhs )
         | Reduce( sym name, type type, expr* idx, expr rhs )
         | WriteConfig( config config, string field, expr rhs )
         | Pass()
           -- Fence: barriers[0] is internal name of fence
           -- Arrive: barriers: List[BarrierExpr]
           -- Await: barriers = List[BarrierExpr] of length 1
         | SyncStmt( sync_type sync_type, expr* barriers )
         | If( expr cond, stmt* body, stmt* orelse )
         | For( sym iter, expr lo, expr hi, stmt* body, loop_mode loop_mode )
         | Alloc( sym name, type type, allocable mem )
         | Free( sym name, type type, allocable mem )
         | Call( proc f, expr* args, expr? trailing_barrier_expr )
         | WindowStmt( sym name, expr rhs, special_window? special_window )
         attributes( srcinfo srcinfo )

    expr = Read( sym name, expr* idx )
         | Const( object val )
         | USub( expr arg )  -- i.e.  -(...)
         | BinOp( binop op, expr lhs, expr rhs )
         | Extern( extern f, expr* args )
         | BarrierExpr( sym name, w_access* idx ) -- Should we replace with WindowExpr alone?
         | WindowExpr( sym name, w_access* idx )
         | StrideExpr( sym name, int dim )
         | ReadConfig( config config, string field )
         attributes( type type, srcinfo srcinfo )

    -- WindowExpr = (base : Sym, idx : [ Pt Expr | Interval Expr Expr ])
    w_access = Interval( expr lo, expr hi )
             | Point( expr pt )
             attributes( srcinfo srcinfo )

    type = Num()
         | BF16()
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
         -- src_type  - type of the Tensor from which the window was created
         -- as_tensor - tensor type as if this window were simply a tensor
         --             itself
         -- src_buf   - sym for the Tensor from which the window was created
         -- idx       - the expression that created this window
         -- NB: when creating a derived window from another derived window,
         -- we must "chain" the two window exprs so that src_type, src_buf
         -- still refer to the original Tensor
         | WindowType( type src_type, type as_tensor,
                       sym src_buf, w_access *idx )
         -- Spork (Exo-GPU) extensions
         | WithContext()
         | Barrier( sym? guarded_by, expr* hi )

    -- Dense tensor: Tensor(is_window = False)
    -- Window parameter (of proc): Tensor(is_window = True)
    -- Derived window (from WindowExpr): WindowType / T.Window

    -- First two are both "tensors" although imprecisely sometimes "tensor"
    -- refers only to "dense tensor" -- we should be more clear about that.
    -- Latter two are both "windows" (allows strides), but have separate
    -- types since derived windows (WindowType) requires aliasing reasoning
}""",
    ext_types={
        "name": validators.instance_of(Identifier, convert=True),
        "instr": InstrInfo,
        "sym": Sym,
        "memwin": Type[MemWin],
        "allocable": Type[AllocableMemWin],
        "special_window": Type[SpecialWindow],
        "extern": Extern,
        "config": Config,
        "binop": validators.instance_of(Operator, convert=True),
        "srcinfo": SrcInfo,
        "loop_mode": LoopMode,
        "sync_type": SyncType,
    },
    memoize={
        "Num",
        "BF16",
        "F16",
        "F32",
        "F64",
        "INT8",
        "UINT8",
        "UINT16",
        "INT32",
        "Bool",
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
                srcinfo         srcinfo )

    fnarg   = ( sym             name,
                type            type,
                memwin?         mem,
                srcinfo         srcinfo )

    stmt    = Assign  ( sym name, expr* idx, expr rhs )
            | Reduce  ( sym name, expr* idx, expr rhs )
            | WriteConfig ( config config, string field, expr rhs )
            | FreshAssign( sym name, expr rhs )
            | Pass    ()
            | SyncStmt( sync_type sync_type, expr* barriers )
            | If      ( expr cond, stmt* body,  stmt* orelse )
            | For     ( sym iter,  expr cond,   stmt* body )
            | Alloc   ( sym name, type type, allocable? mem )
            | Call    ( loopir_proc f, expr* args, expr? trailing_barrier_expr )
            attributes( srcinfo srcinfo )

    expr    = Read    ( sym name, expr* idx )
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            | Extern( extern f, expr* args )
            | BarrierExpr( sym name, w_access* idx )
            | WindowExpr( sym name, w_access* idx, special_window? special_window )
            | StrideExpr( sym name, int dim )
            | LoopRange( expr lo, expr hi, loop_mode loop_mode ) -- only use for loop cond
            | ReadConfig( config config, string field )
            attributes( srcinfo srcinfo )

    w_access= Interval( expr? lo, expr? hi )
            | Point( expr pt )
            attributes( srcinfo srcinfo )

    type    = Num   ()
            | BF16()
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
            | WithContext()
            | Barrier( sym? guarded_by, expr *hi )
} """,
    ext_types={
        "name": validators.instance_of(Identifier, convert=True),
        "sym": Sym,
        "memwin": Type[MemWin],
        "allocable": Type[AllocableMemWin],
        "special_window": Type[SpecialWindow],
        "extern": Extern,
        "config": Config,
        "loopir_proc": LoopIR.proc,
        "op": validators.instance_of(Operator, convert=True),
        "srcinfo": SrcInfo,
        "loop_mode": LoopMode,
        "sync_type": SyncType,
    },
    memoize={
        "Num",
        "BF16",
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

# TODO Exo-GPU concepts basically don't exist in PAST
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
            | Extern ( name f, expr* args )
            | ReadConfig( string config, string field )
            attributes( srcinfo srcinfo )

} """,
    ext_types={
        "name": validators.instance_of(IdentifierOrHole, convert=True),
        "op": validators.instance_of(Operator, convert=True),
        "srcinfo": SrcInfo,
        "sync_type": SyncType,
    },
)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Types


class T:
    Num = LoopIR.Num
    BF16 = LoopIR.BF16
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
    Barrier = LoopIR.Barrier
    WithContextT = LoopIR.WithContext
    type = LoopIR.type
    R = Num()
    bf16 = BF16()
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
    # Spork extensions
    with_context = WithContextT()
    barrier = LoopIR.Barrier(None, [])


# str to UAST type instance (see ScalarInfo, it adds to this)
uast_prim_types = {
    "R": UAST.Num(),
}


# UAST to LoopIR non-parameterized types (see ScalarInfo, it adds to this)
loopir_from_uast_metatype_table = {
    UAST.Num: T.R,
    UAST.Int: T.int,
    UAST.Size: T.size,
    UAST.Index: T.index,
    UAST.Stride: T.stride,
    UAST.Bool: T.bool,
}

# ScalarInfo.extclass adds to this
uast_concrete_scalar_metatypes: Type[UAST.type] = []
loopir_concrete_scalar_metatypes: Type[LoopIR.type] = []


# ScalarInfo.extclass will override this for concrete scalar types
@extclass(LoopIR.type)
def scalar_info(t):
    raise TypeError(f"No scalar_info for {t}")


del scalar_info


# To add new concrete scalar types, you have to add more entries
# here, then unfortunately manually edit the LoopIR and UAST and T
# and ExoType class definitions to add the type to the grammar.
# fmt: off
ScalarInfo.extclass(UAST.BF16(),        T.bf16,         ExoType.BF16,   "bf16",         "exo_bf16",     16)
ScalarInfo.extclass(UAST.F16(),         T.f16,          ExoType.F16,    "f16",          "exo_f16",      16)
ScalarInfo.extclass(UAST.F32(),         T.f32,          ExoType.F32,    "f32",          "float",        32)
ScalarInfo.extclass(UAST.F64(),         T.f64,          ExoType.F64,    "f64",          "double",       64)
ScalarInfo.extclass(UAST.INT8(),        T.i8,           ExoType.I8,     "i8",           "int8_t",       8)
ScalarInfo.extclass(UAST.UINT8(),       T.ui8,          ExoType.UI8,    "ui8",          "uint8_t",      8)
ScalarInfo.extclass(UAST.UINT16(),      T.ui16,         ExoType.UI16,   "ui16",         "uint16_t",     16)
ScalarInfo.extclass(UAST.INT32(),       T.i32,          ExoType.I32,    "i32",          "int32_t",      32)
# fmt: on


# extclass for all concrete scalar types
# Only define this after ScalarInfo.extclass populated needed tables.


def extclass_LoopIR_concrete_scalars(f):
    for t in loopir_concrete_scalar_metatypes:
        f = extclass(t)(f)
    return f


def extclass_UAST_concrete_scalars(f):
    for t in uast_concrete_scalar_metatypes:
        f = extclass(t)(f)
    return f


# MemGlobalC will double-duty as a way to inject optional f16/bf16 typedefs.
# This is not very well thought out.
# Can be improved if we have to add more not-so-portable types besides f16/bf16.


@extclass(LoopIR.type)
def scalar_mem_global(t):
    return None


@extclass(T.BF16)
def scalar_mem_global(t):
    code = """#ifndef exo_bf16  /* Define before inclusion to override exo_bf16 */
#ifdef __CUDACC__
using exo_bf16 = __nv_bfloat16;
#else
typedef struct { short bits; } exo_bf16;
#endif
#endif
"""
    return MemGlobalC("exo_bf16", code, ())
    # Crappy issue: we don't include cuda_fp16.h or cuda_bf16.h here, because
    # MemGlobalC appears in extern "C", and using MemIncludeC will force the include
    # even if cuda isn't used (which would force dependence on cuda toolkit)
    #
    # Further issue, could be host/device C++ mangling issues due to not
    # having the same underlying type for host and device code.


@extclass(T.F16)
def scalar_mem_global(t):
    code = """#ifndef exo_f16  /* Define before inclusion to override exo_f16 */
#ifdef __CUDACC__
using exo_f16 = __half;
#elif defined(__STDCPP_FLOAT16_T__)
typedef _Float16 exo_f16;
#else
typedef struct { short bits; } exo_f16;
#endif
#endif
"""
    return MemGlobalC("exo_f16", code, ())


@extclass(T.Tensor)
@extclass(T.Window)
def scalar_mem_global(t):
    return t.basetype().scalar_mem_global()


# --------------------------------------------------------------------------- #
# Extension methods
# --------------------------------------------------------------------------- #


@extclass(UAST.Tensor)
@extclass(UAST.Num)
@extclass(UAST.Barrier)
@extclass_UAST_concrete_scalars
def shape(t):
    shp = t.hi if isinstance(t, (UAST.Tensor, UAST.Barrier)) else []
    return shp


del shape


@extclass(UAST.type)
def basetype(t):
    if isinstance(t, UAST.Tensor):
        t = t.type
    elif isinstance(t, UAST.Barrier):
        t = UAST.Barrier([])
    return t


del basetype


# make proc be a hashable object
@extclass(LoopIR.proc)
def __hash__(self):
    return id(self)


del __hash__


# --------------------------------------------------------------------------- #
# type helper functions


@extclass(LoopIR.proc)
@functools.cache  # Must cache before extclass
def get_cached_const_param_dict(proc: LoopIR.proc) -> Dict[Sym, bool]:
    # NB the proc becomes un-garbage-collectable due to the cache?
    mut_syms = set(x for x, _ in get_writes_of_stmts(proc.body))
    return {a.name: a.name not in mut_syms for a in proc.args}


@extclass(LoopIR.proc)
def is_const_param(proc, sym: Sym):
    assert isinstance(sym, Sym)
    return proc.get_cached_const_param_dict()[sym]


@extclass(T.Tensor)
def as_tensor_type(t):
    return t


@extclass(T.Window)
def as_tensor_type(t):
    return t.as_tensor


del as_tensor_type


@extclass(T.Tensor)
def shape(t):
    assert not isinstance(t.type, T.Tensor), "expect no nesting"
    return t.hi


@extclass(T.Barrier)
def shape(t):
    return t.hi


@extclass(T.Window)
def shape(t):
    return t.as_tensor.shape()


@extclass(T.Num)
@extclass_LoopIR_concrete_scalars
def shape(t):
    return []


del shape


@extclass_LoopIR_concrete_scalars
def ctype(t):
    return t.scalar_info().ctype


@extclass(T.Bool)
def ctype(t):
    return "bool"


@extclass(T.Num)
def ctype(t):
    assert False, "Don't ask for ctype of Num"


@extclass(T.Int)
@extclass(T.Index)
@extclass(T.Size)
@extclass(T.Stride)
def ctype(t):
    return "int_fast32_t"


del ctype


def scalar_bits(ctype):
    return ScalarInfo(ctype).bits


@extclass(LoopIR.type)
def is_real_scalar(t):
    return False


@extclass(LoopIR.Num)
@extclass_LoopIR_concrete_scalars
def is_real_scalar(t):
    return True


del is_real_scalar


@extclass(LoopIR.type)
def is_tensor_or_window(t):
    return isinstance(t, (T.Tensor, T.Window))


del is_tensor_or_window


@extclass(LoopIR.type)
def is_win(t):
    # T.Tensor and t.is_window: window parameter
    # T.Window: derived window
    return (isinstance(t, T.Tensor) and t.is_window) or isinstance(t, T.Window)


del is_win


@extclass(LoopIR.type)
def is_dense_tensor(t):
    return isinstance(t, T.Tensor) and not t.is_window


del is_dense_tensor


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
    return isinstance(t, (T.Int, T.Stride, T.Size, T.Index))


@extclass(LoopIR.type)
def basetype(t):
    if isinstance(t, T.Window):
        return t.as_tensor.basetype()
    elif isinstance(t, T.Tensor):
        assert not t.type.is_tensor_or_window()
        return t.type
    elif isinstance(t, T.Barrier):
        return T.barrier
    else:
        return t


del basetype


def LoopIR_Fence(L1: Sync_tl, L2: Sync_tl, srcinfo: SrcInfo):
    name = Sym("Fence")  # Sym as internal unique ID for Fence.
    barriers = [LoopIR.BarrierExpr(name, [], T.barrier, srcinfo)]
    return LoopIR.SyncStmt(fence_type(L1, L2), barriers, srcinfo)


@extclass(LoopIR.BarrierExpr)
def multicast_flags(e):
    return tuple(isinstance(w, LoopIR.Interval) for w in e.idx)


del multicast_flags


@extclass(LoopIR.SyncStmt)
def multicasts(s):
    return tuple(e.multicast_flags() for e in s.barriers)


del multicasts


@extclass(LoopIR.SyncStmt)
def forbid_multicast(s, reason):
    for e in s.barriers:
        for w in e.idx:
            if isinstance(w, LoopIR.Interval):
                raise ValueError(
                    f"{s.srcinfo}: Unsupported multicast ({w}) in {e}; {reason}"
                )


@extclass(LoopIR.SyncStmt)
def home_barrier_expr(s) -> LoopIR.BarrierExpr:
    """Give expression for the home barrier, e.g.

    Arrive(...) >> foo[a, :] >> foo[:, b]

    becomes foo[a, b]"""
    if not s.barriers:
        raise ValueError(f"{s.srcinfo}: {s} missing >> trailing barrier exprs")

    e0 = s.barriers[0]
    nm = e0.name
    dim = len(e0.idx)
    idx = [None] * dim

    for expr_idx in range(len(s.barriers)):
        e = s.barriers[expr_idx]
        if e.name != nm:
            raise ValueError(
                f"{s.srcinfo}: cannot arrive on different queue barrier arrays {e} and {e0}"
            )
        for dim_idx in range(dim):
            this_idx = e.idx[dim_idx]
            if isinstance(this_idx, LoopIR.Point):
                pt = this_idx.pt
                if not isinstance(pt, LoopIR.Read):
                    raise ValueError(
                        f"{s.srcinfo}: expected a plain variable, not {this_idx}, in {e}"
                    )
                if old_idx := idx[dim_idx]:
                    if old_idx.pt.name != pt.name:
                        raise ValueError(
                            f"{s.srcinfo}: {e} has idx[{dim_idx}] = {pt.name}; mismatches idx[{dim_idx}] in previous trailing barrier expressions of {s}"
                        )
                else:
                    idx[dim_idx] = this_idx

    for dim_idx, w in enumerate(idx):
        if w is None:
            raise ValueError(
                f"{s.srcinfo}: at least one trailing barrier expression must have idx[{dim_idx}] be a point, not an interval {s.barriers[0].idx[dim_idx]} (in {s})"
            )

    return LoopIR.BarrierExpr(nm, idx, T.barrier, s.srcinfo)


del home_barrier_expr


@extclass(LoopIR.type)
def is_barrier(t):
    return False


@extclass(LoopIR.Barrier)
def is_barrier(t):
    return True


del is_barrier


@extclass(LoopIR.stmt)
def is_loop(s):
    return False


@extclass(LoopIR.For)
def is_loop(s):
    return True


del is_loop


@extclass(LoopIR.proc)
def proc_instr_tl(f) -> Instr_tl:
    """Return instr-tl in scope needed to call a proc.

    For now, any non-instr procs are assumed to require 1 CPU thread.
    """
    if f.instr:
        return f.instr.instr_tl
    return cpu_in_order_instr


del proc_instr_tl


@extclass(LoopIR.proc)
def proc_coll_unit(f):
    """Return collective unit needed to call a proc.

    For now, any non-instr procs are assumed to require 1 CPU thread.
    """
    if f.instr:
        return f.instr.coll_unit
    return standalone_thread


del proc_coll_unit


@extclass(LoopIR.proc)
def proc_name_with_args(p):
    arg_list = [str(a.name) for a in p.args]
    instr: InstrInfo
    if instr := p.instr:
        if kwargs := instr._formatted_tparam_kwargs:
            arg_list.append(kwargs)
    argstr = ",".join(arg_list)
    return f"{p.name}({argstr})"


def chain_window_idx(idx0, idx1):
    """Given

    window_0 = tensor[idx0]
    window_1 = window_0[idx1]

    Return chained_idx such that window_1 = tensor[chained_idx]
    """

    def add_e(scalar_0, scalar_1):
        if isinstance(scalar_0, LoopIR.Const) and scalar_0.val == 0:
            return scalar_1
        if isinstance(scalar_1, LoopIR.Const) and scalar_1.val == 0:
            return scalar_0
        return LoopIR.BinOp("+", scalar_0, scalar_1, T.index, scalar_1.srcinfo)

    assert sum(isinstance(e0, LoopIR.Interval) for e0 in idx0) == len(idx1)
    chained_idx = [None] * len(idx0)
    i1 = 0
    for i0, e0 in enumerate(idx0):
        if isinstance(e0, LoopIR.Point):
            chained_idx[i0] = e0
        else:
            assert isinstance(e0, LoopIR.Interval)
            e1 = idx1[i1]
            i1 += 1
            srcinfo = e1.srcinfo  # newer srcinfo likely more relevant
            if isinstance(e1, LoopIR.expr):
                chained_idx[i0] = add_e(e0.lo, e1)
            elif isinstance(e1, LoopIR.Point):
                chained_idx[i0] = LoopIR.Point(add_e(e0.lo, e1.pt), srcinfo)
            else:
                # Note e0.hi unused ... not responsibility here to do
                # bounds checking.
                chained_idx[i0] = LoopIR.Interval(
                    add_e(e0.lo, e1.lo), add_e(e0.lo, e1.hi), srcinfo
                )
    return chained_idx


def build_window_shape(ws: List[LoopIR.w_access]):
    def subtract(hi, lo):
        if isinstance(lo, LoopIR.Const) and lo.val == 0:
            return hi
        else:
            return LoopIR.BinOp("-", hi, lo, T.index, hi.srcinfo)

    return [subtract(w.hi, w.lo) for w in ws if isinstance(w, LoopIR.Interval)]


def create_window_type(in_name: Sym, in_typ: LoopIR.type, idx):
    """Construct a derived window type from any tensor or window type"""
    assert isinstance(in_name, Sym)
    window_shape = build_window_shape(idx)
    as_tensor = T.Tensor(window_shape, True, in_typ.basetype())

    if isinstance(in_typ, T.Tensor):
        # in_typ is dense tensor or window parameter
        w_typ = T.Window(in_typ, as_tensor, in_name, idx)
    else:
        # in_typ is another derived window
        # we need to "inline" through to get the underlying Tensor
        assert isinstance(in_typ, T.Window)
        chained_idx = chain_window_idx(in_typ.idx, idx)
        w_typ = T.Window(in_typ.src_type, as_tensor, in_typ.src_buf, chained_idx)

    return w_typ


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Compiler debug logging
# This functionality is intended to dump formatted LoopIR to the compiler
# output directory, with remarks (likely errors) inserted in-place.
# unsafe_hash needed for Python 3.11 dataclasses changes.
@dataclass(slots=True, unsafe_hash=True)
class ProcDebugRemarks:
    # Maps stmt_id to list of lines to insert.
    stmt_id_lines: Dict[int, List[str]] = field(default_factory=dict)
    # Set of expr id to comment.
    # This is to help contextualize expr IDs in remarks.
    expr_id_comment_set: Set[int] = field(default_factory=set)

    def get_stmt_id_lines(self, stmt_id: Optional[int]) -> List[str]:
        if stmt_id is None:
            return ()
        assert isinstance(stmt_id, int)
        return self.stmt_id_lines.get(stmt_id, ())

    def is_expr_id_commented(self, expr_id: Optional[int]) -> bool:
        if expr_id is None:
            return False
        assert isinstance(expr_id, int)
        return expr_id in self.expr_id_comment_set

    def get_all_stmt_id_lines(self) -> List[Tuple[int, List[str]]]:
        return sorted(self.stmt_id_lines.items())


ProcDebugRemarks.empty = ProcDebugRemarks()


class BaseCompilerDebugLog:
    __slots__ = []

    def get_path(self):
        return None

    def log(self, proc_name: str, suffix: str, subtree, preferred=False):
        pass

    def remark(self, proc_name: str, remark: str):
        pass

    def get_proc_debug_remarks(self, proc_name: str) -> ProcDebugRemarks:
        return ProcDebugRemarks.empty

    def enable_notify_user(self):
        pass


@dataclass(slots=True)
class CompilerDebugLogImpl(BaseCompilerDebugLog):
    """Don't create this directly; use get_debug_log."""

    _path: Path
    _names_to_subtree: Dict[Tuple[str, str], Union[LoopIR.stmt, LoopIR.proc]] = field(
        default_factory=dict
    )
    _proc_debug_remarks: Dict[str, ProcDebugRemarks] = field(default_factory=dict)
    _enable_notify_user: bool = False
    _preferred_names: Set[Tuple[str, str]] = field(default_factory=set)

    def get_path(self):
        return self._path

    def log(
        self,
        proc_name: str,
        suffix: str,
        subtree: Union[LoopIR.stmt, LoopIR.proc, str],
        preferred=False,
    ):
        names = (proc_name, suffix)
        # This assert was too fragile in pytest!
        # assert names not in self._names_to_subtree, names
        assert isinstance(subtree, (LoopIR.proc, LoopIR.stmt, str))
        self._names_to_subtree[names] = subtree
        if preferred:
            self._preferred_names.add(names)

    def remark(self, proc_name: str, remark: str):
        # This is rather hacky but I do what I must to retrofit this logging
        # to existing Exo code. We search the remark (likely error message)
        # for the stmt_id/expr_id formatting pattern that str(SrcInfo) uses,
        # and associate the remark lines with all stmt/expr named.
        # If no stmt_id was found, we associate the lines with the fake stmt_id -1
        # so the remark doesn't just get sent to /dev/null
        #
        # In the future, we could investigate more "structured"
        # exception handling that embeds the stmt_id/expr_id in the
        # error object but this is not that important.
        remarks = self._proc_debug_remarks.get(proc_name)
        if remarks is None:
            remarks = self._proc_debug_remarks.setdefault(proc_name, ProcDebugRemarks())
        lines = [line for line in remark.split("\n") if line]
        stmt_ids = [int(m) for m in re.findall(SrcInfo.stmt_id_pattern, remark)]
        expr_ids = [int(m) for m in re.findall(SrcInfo.expr_id_pattern, remark)]
        if not stmt_ids:
            stmt_ids = (-1,)
        for s_id in stmt_ids:
            lst = remarks.stmt_id_lines.setdefault(s_id, [])
            if lst:
                lst.append("")
            lst.extend(lines)
        for e_id in expr_ids:
            remarks.expr_id_comment_set.add(e_id)

    def get_proc_debug_remarks(self, proc_name: str) -> ProcDebugRemarks:
        return self._proc_debug_remarks.get(proc_name, ProcDebugRemarks.empty)

    def write_all_impl(self):
        debug_path = self._path / "debug"
        debug_path.mkdir(exist_ok=True, parents=True)
        for names, subtree in self._names_to_subtree.items():
            proc_name, suffix = names
            out_path = debug_path / f"{proc_name}-{suffix}.py"
            if isinstance(subtree, (LoopIR.proc, LoopIR.stmt)):
                # str_with_remarks is part of the LoopIR pretty print infra
                remarks = self.get_proc_debug_remarks(proc_name)
                out_path.write_text(subtree.str_with_remarks(remarks))
            else:
                out_path.write_text(str(subtree))
            if self._enable_notify_user:
                color_prefix = ""
                if names in self._preferred_names:
                    color_prefix = "\x1b[1m\x1b[35m"
                # We want this to appear prominently underneath the Python traceback.
                # Currently this only works since write_all_impl is called atexit.
                print(
                    f"{color_prefix}Debug output:\x1b[0m",
                    str(out_path),
                    file=sys.stderr,
                )

    def enable_notify_user(self):
        self._enable_notify_user = True


_debug_log_dict = {}


def get_debug_log(path: Optional[Path]) -> BaseCompilerDebugLog:
    if path is None:
        return BaseCompilerDebugLog()
    assert isinstance(path, Path)
    try:
        log = _debug_log_dict[path]
    except KeyError:
        log = CompilerDebugLogImpl(path)
        _debug_log_dict[path] = log
    return log


@atexit.register
def _atexit_debug_log_write():
    # Log isn't written until program exit, because we can't write
    # remarks inline with an output file (based on printed LoopIR)
    # until all future remarks are collected.
    # This design will be a problem if one of our C modules segfaults.
    for log in _debug_log_dict.values():
        log.write_all_impl()


# Global state; the exocc frontend or pytest config will set up the
# debug directory, but we don't have a great way to communicate this to
# the user's module or the large lib of existing test functions,
# which may raise errors that we now wish to have nicely logged.
_global_debug_log_path = None


def get_global_debug_log():
    return get_debug_log(_global_debug_log_path)


def get_global_debug_log_path():
    return _global_debug_log_path


def set_global_debug_log_path(path: Optional[Path]):
    assert path is None or isinstance(path, Path)
    global _global_debug_log_path
    _global_debug_log_path = path


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
# Standard Pass Templates for Loop IR


class LoopIR_Rewrite:
    __slots__ = []

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
            if new_lo is not None or new_hi is not None or new_body is not None:
                return [
                    s.update(
                        lo=new_lo or s.lo, hi=new_hi or s.hi, body=new_body or s.body
                    )
                ]
        elif isinstance(s, LoopIR.Call):
            new_args = self.map_exprs(s.args)
            old_bar = s.trailing_barrier_expr
            new_bar = None
            if old_bar is not None:
                new_bar = self.map_e(old_bar)
                assert new_bar is None or isinstance(new_bar, LoopIR.BarrierExpr)
            if new_args is not None or new_bar is not None:
                return [
                    s.update(
                        args=new_args or s.args,
                        trailing_barrier_expr=new_bar or old_bar,
                    )
                ]

        elif isinstance(s, (LoopIR.Alloc, LoopIR.Free)):
            new_type = self.map_t(s.type)
            if new_type:
                return [s.update(type=new_type or s.type)]
        elif isinstance(s, LoopIR.SyncStmt):
            new_barriers = self._map_list(self.map_e, s.barriers)
            if new_barriers:
                return [s.update(barriers=new_barriers)]
            return None
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
        elif isinstance(e, LoopIR.Extern):
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
        elif isinstance(e, (LoopIR.WindowExpr, LoopIR.BarrierExpr)):
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
        elif isinstance(t, T.Barrier):
            new_hi = self.map_exprs(t.hi)
            if new_hi is not None:
                return t.update(hi=new_hi)
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
    __slots__ = ["proc"]

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
            if e := s.trailing_barrier_expr:
                self.do_e(e)
        elif styp is LoopIR.Alloc:
            self.do_t(s.type)
        elif styp is LoopIR.SyncStmt:
            for e in s.barriers:
                self.do_e(e)
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
        elif etyp is LoopIR.Extern:
            for a in e.args:
                self.do_e(a)
        elif etyp is LoopIR.USub:
            self.do_e(e.arg)
        elif etyp in (LoopIR.WindowExpr, LoopIR.BarrierExpr):
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
        if isinstance(t, (T.Tensor, T.Barrier)):
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
        elif isinstance(s1, LoopIR.SyncStmt):
            # TODO test this
            return (
                s1.sync_type == s2.sync_type
                and len(s1.barriers) == len(s2.barriers)
                and all(
                    self.match_e(i1, i2) for i1, i2 in zip(s1.barriers, s2.barriers)
                )
            )
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
            return (
                s1.f == s2.f
                and all(self.match_e(a1, a2) for a1, a2 in zip(s1.args, s2.args))
                and self.match_e(s1.trailing_barrier_expr, s2.trailing_barrier_expr)
            )
        elif isinstance(s1, LoopIR.WindowStmt):
            return self.match_name(s1.name, s2.name) and self.match_e(s1.rhs, s2.rhs)
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
        elif isinstance(e1, LoopIR.Extern):
            # TODO: check f equality
            return e1.f is e2.f and all(
                self.match_e(a1, a2) for a1, a2 in zip(e1.args, e2.args)
            )
        elif isinstance(e1, LoopIR.BarrierExpr):
            return self.match_name(e1.name, e2.name) and all(
                self.match_w_access(w1, w2) for w1, w2 in zip(e1.idx, e2.idx)
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
        elif e1 is None:
            return e2 is None
        else:
            assert False, "bad case"

    def match_name(self, n1, n2):
        # TODO: if it's a free var, check for exact match using ID. This
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
        if isinstance(t1, LoopIR.Tensor) and isinstance(t2, LoopIR.Tensor):
            return (
                t1.is_window == t2.is_window
                and self.match_t(t1.type, t2.type)
                and all(self.match_e(i1, i2) for i1, i2 in zip(t1.hi, t2.hi))
            )
        elif isinstance(t1, LoopIR.Barrier) and isinstance(t2, LoopIR.Barrier):
            return all(self.match_e(i1, i2) for i1, i2 in zip(t1.hi, t2.hi))
        else:  # scalar
            return type(t1) == type(t2)


class GetReads(LoopIR_Do):
    def __init__(self):
        self.reads = []

    def do_e(self, e):
        # XXX this is an over-approximation for Call.
        # If a parameter is write-only, it's still counted as a read here.
        if hasattr(e, "name"):
            self.reads.append((e.name, e.type))
        super().do_e(e)


class GetReadsWithReduce(GetReads):
    def do_s(self, s):
        if isinstance(s, LoopIR.Reduce):
            self.reads.append((s.name, s.type))
        super().do_s(s)


class GetReadConfigs(LoopIR_Do):
    def __init__(self):
        self.readconfigs = []

    def do_e(self, e):
        if isinstance(e, LoopIR.ReadConfig):
            self.readconfigs.append((e.config, e.field))
        super().do_e(e)


def get_reads_of_expr(e):
    gr = GetReads()
    gr.do_e(e)
    return gr.reads


def get_reads_of_stmts(stmts, include_reduce=False):
    # XXX David Zhao Akeley 2026-01-15 WindowStmt not handled specially.
    # I think currently, w = x[:] will result in x always being
    # counted as a read, even if x and w are only ever written to.
    gr = GetReadsWithReduce() if include_reduce else GetReads()
    for stmt in stmts:
        gr.do_s(stmt)
    return gr.reads


def get_readconfigs(stmts):
    gr = GetReadConfigs()
    for stmt in stmts:
        gr.do_s(stmt)
    return gr.readconfigs


class GetWrites(LoopIR_Do):
    def __init__(self):
        self.writes = []
        # Translates access through T.Window to underlying T.Tensor
        self.window_dict = {}

    def do_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            sym = s.name
            self.writes.append((self.window_dict.get(sym, sym), s.type))
        elif isinstance(s, LoopIR.Call):
            for arg, call_arg in zip(s.args, s.f.args):
                if not s.f.is_const_param(call_arg.name):
                    if isinstance(
                        arg, (LoopIR.Read, LoopIR.WindowExpr, LoopIR.StrideExpr)
                    ):
                        sym = arg.name
                        self.writes.append((self.window_dict.get(sym, sym), arg.type))
        elif isinstance(s, LoopIR.WindowStmt):
            w_sym, base_sym = s.name, s.rhs.name
            while base_sym in self.window_dict:
                base_sym = self.window_dict[base_sym]
            self.window_dict[w_sym] = base_sym

        super().do_s(s)

    # early exit
    def do_e(self, e):
        return


def get_writes_of_stmts(stmts):
    gw = GetWrites()
    gw.do_stmts(stmts)
    return gw.writes


class GetWriteConfigs(LoopIR_Do):
    def __init__(self):
        self.writeconfigs = []

    def do_s(self, s):
        if isinstance(s, LoopIR.WriteConfig):
            self.writeconfigs.append((s.config, s.field))
        elif isinstance(s, LoopIR.Call):
            self.writeconfigs += get_writeconfigs(s.f.body)

        super().do_s(s)

    # early exit
    def do_e(self, e):
        return


def get_writeconfigs(stmts):
    gw = GetWriteConfigs()
    gw.do_stmts(stmts)
    return gw.writeconfigs


class GetLoopIters(LoopIR_Do):
    def __init__(self):
        self.loop_iters = []

    def do_s(self, s):
        if isinstance(s, LoopIR.For):
            self.loop_iters.append(s.iter)
        super().do_s(s)

    # early exit
    def do_e(self, e):
        return


def get_loop_iters(stmts):
    gw = GetLoopIters()
    gw.do_stmts(stmts)
    return gw.loop_iters


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
            self.env[s.name] = True
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
            or etyp is LoopIR.BarrierExpr
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
        elif isinstance(s, LoopIR.SyncStmt):
            if s.sync_type.is_split():
                return super().map_s(s)
            else:
                # Fence(...) stmt does not refer to allocated barrier variable
                # and we must unique-ify its internal barrier name regardless
                # of self.env; hence we handle this here specially.
                assert len(s.barriers) == 1
                bar_expr = s.barriers[0]
                new_name = bar_expr.name.copy()
                self.env[bar_expr.name] = new_name
                return s.update(barriers=[bar_expr.update(name=new_name)])
        elif isinstance(s, LoopIR.Alloc):
            s2 = super().map_s(s)
            assert s.name not in self.env
            new_name = s.name.copy()
            self.env[s.name] = new_name
            return [((s2 and s2[0]) or s).update(name=new_name)]
        elif isinstance(s, LoopIR.WindowStmt):
            rhs = self.map_e(s.rhs) or s.rhs
            name = s.name.copy()
            self.env[s.name] = name
            return [s.update(name=name, rhs=rhs)]
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
        if isinstance(
            e, (LoopIR.Read, LoopIR.BarrierExpr, LoopIR.WindowExpr, LoopIR.StrideExpr)
        ):
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
            elif isinstance(n, LoopIR.fnarg):
                t = self.map_t(n.type)
                if t:
                    n = n.update(type=t)
                self.nodes.append(n)
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


class LoopIR_Add_ID(LoopIR_Rewrite):
    __slots__ = ["s_id", "e_id"]
    s_id: int
    e_id: int

    def __init__(self):
        self.s_id = 10
        self.e_id = 1

    def map_s(self, s):
        # Allocate stmt_id as multiples of 10 to make room for
        # e.g. MemAnalysis giving a Free a different stmt id from an Alloc.
        stmts = super().map_s(s)
        if stmts:
            assert len(stmts) == 1
            s = stmts[0]
        info = s.srcinfo.update(stmt_id=self.s_id)
        self.s_id += 10
        return s.update(srcinfo=info)

    def map_e(self, e):
        e = super().map_e(e) or e
        info = e.srcinfo.update(stmt_id=self.s_id, expr_id=self.e_id)
        self.e_id += 1
        return e.update(srcinfo=info)


# Data-flow dependencies between variable names
# TODO: Refactor this using new AI based analysis

# So, what is dependency analysis?
# Or to put it another way, what extensional property(s)
# does dependency analysis guarantee?
#
# Let B be a block of statements,
#     s be a store, and
#     x, y, … be names/symbols.
# Let FV(B) be the set of names that are free in B
#
# Then, first observe that the "meaning" of B is
#
#   Exec[[B]] : (FV(B) -> Value) -> Store -> Store
#
# (note that (FV(B) -> Value) is a valuation/mapping specifying the values
#       of all free variables)
# (further note that Store = (Name -> Maybe Value) is a valuation/mapping
#       of variables that models the heap/store)
#
# Then, (not x DependsOn y in B) for some y in FV(B) implies that
#
#   (Exec[[B]] (env[ y := v1 ]) s)[x] =
#   (Exec[[B]] (env[ y := v2 ]) s)[x]
#
# for all v1, v2
#
# Or in other words, the meaning of B
# w.r.t. its effect on x
# is invariant to the value of y
# when x does not depend on y in B


class LoopIR_Dependencies(LoopIR_Do):
    def __init__(self, buf_sym, stmts):
        self._buf_sym = buf_sym
        self._lhs = None
        self._depends = defaultdict(set)
        self._alias = dict()

        # If `lhs` is not None, then `lhs` will become dependent
        # on anything read.
        self._lhs = None

        # variables that affect whether or not the
        # currently examined code is even running
        self._context = set()

        # If `control` is True, then anything read will be added
        # to `context`.
        self._control = False

        self.do_stmts(stmts)

    def result(self):
        depends = self._depends[self._buf_sym]
        new = list(depends)
        done = []
        while True:
            if len(new) == 0:
                break
            sym = new.pop()
            done.append(sym)
            d = self._depends[sym]
            depends.update(d)
            new.extend(s for s in d if s not in done)

        return depends

    def do_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            lhs = self._alias.get(s.name, s.name)
            self._lhs = lhs
            self._depends[lhs].add(lhs)
            self._depends[lhs].update(self._context)
            for i in s.idx:
                self.do_e(i)
            self.do_e(s.rhs)
            self._lhs = None
        elif isinstance(s, LoopIR.WriteConfig):
            lhs = (s.config, s.field)
            self._lhs = lhs
            self._depends[lhs].add(lhs)
            self._depends[lhs].update(self._context)
            self.do_e(s.rhs)
            self._lhs = None
        elif isinstance(s, LoopIR.WindowStmt):
            rhs_buf = self._alias.get(s.rhs.name, s.rhs.name)
            self._alias[s.name] = rhs_buf
            self._lhs = rhs_buf
            self._depends[rhs_buf].add(rhs_buf)
            self.do_e(s.rhs)
            self._lhs = None

        elif isinstance(s, LoopIR.If):
            old_context = self._context
            self._context = old_context.copy()

            self._control = True
            self.do_e(s.cond)
            self._control = False

            self.do_stmts(s.body)
            self.do_stmts(s.orelse)

            self._context = old_context

        elif isinstance(s, LoopIR.For):
            old_context = self._context
            self._context = old_context.copy()

            self._control = True
            self._lhs = s.iter
            self._depends[s.iter].add(s.iter)
            self.do_e(s.lo)
            self.do_e(s.hi)
            self._lhs = None
            self._control = False

            self.do_stmts(s.body)

            self._context = old_context

        elif isinstance(s, LoopIR.Call):

            def process_reads():
                # now handle dependencies on buffers that might
                # be read from in the sub-procedure
                # and dependencies on other arguments
                for faa, aa in zip(s.f.args, s.args):
                    if faa.type.is_numeric():
                        maybe_read = any(
                            t[0] == faa.name for t in get_reads_of_stmts(s.f.body)
                        )
                    else:
                        maybe_read = True

                    if maybe_read:
                        self.do_e(aa)

                # additionally, we need to handle dependencies
                # on configuration fields
                for name in get_readconfigs(s.f.body):
                    if self._lhs:
                        self._depends[self._lhs].add(name)

            # for every argument that represents a buffer being
            # written to
            for fa, a in zip(s.f.args, s.args):
                maybe_write = fa.type.is_numeric() and any(
                    t[0] == fa.name for t in get_writes_of_stmts(s.f.body)
                )
                if maybe_write:
                    name = self._alias.get(a.name, a.name)
                    self._lhs = name
                    self._depends[name].add(name)
                    self._depends[name].update(self._context)
                    process_reads()
                    self._lhs = None

            # secondly, for every configuration field being written to
            # by this sub-procedure, we need to determine dependencies
            for name in get_writeconfigs(s.f.body):
                self._lhs = name
                self._depends[name].add(name)
                self._depends[name].update(self._context)
                process_reads()
                self._lhs = None

        elif isinstance(s, (LoopIR.Pass, LoopIR.Alloc, LoopIR.SyncStmt)):
            pass
        else:
            assert False, "bad case"

    def do_e(self, e):
        if isinstance(e, (LoopIR.Read, LoopIR.BarrierExpr, LoopIR.WindowExpr)):

            def visit_idx(e):
                if isinstance(e, LoopIR.Read):
                    for i in e.idx:
                        self.do_e(i)
                else:
                    for w in e.idx:
                        if isinstance(w, LoopIR.Interval):
                            self.do_e(w.lo)
                            self.do_e(w.hi)
                        else:
                            self.do_e(w.pt)

            name = self._alias.get(e.name, e.name)
            if self._lhs:
                self._depends[self._lhs].add(name)
            if self._control:
                self._context.add(name)

            visit_idx(e)

        elif isinstance(e, LoopIR.ReadConfig):
            name = (e.config, e.field)
            if self._lhs:
                self._depends[self._lhs].add(name)
            if self._control:
                self._context.add(name)

        else:
            super().do_e(e)

    def do_t(self, t):
        pass
