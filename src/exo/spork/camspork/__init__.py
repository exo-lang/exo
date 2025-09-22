# fmt: off
import os
from ctypes import *
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

lib = cdll.LoadLibrary(os.path.join(os.path.split(__file__)[0], "bin/libexospork.so"))
extent_t = c_uint32
value_t = c_int32


class BuilderExpr:
    __slots__ = []

    @staticmethod
    def typecheck(item):
        if isinstance(item, int):
            return BuilderConst(item)
        elif isinstance(item, ExprRef):
            return item
        else:
            assert isinstance(item, BuilderExpr), f"{type(item)}, expected int, ExprRef, Varname, or BuilderExpr (consider tuple(...) if you intended multidimensional indexing)"
            return item

    def __add__(self, other):
        return BuilderBinOp(binop_Add, self, self.typecheck(other))

    def __radd__(self, other):
        return BuilderBinOp(binop_Add, self.typecheck(other), self)

    def __sub__(self, other):
        return BuilderBinOp(binop_Sub, self, self.typecheck(other))

    def __rsub__(self, other):
        return BuilderBinOp(binop_Sub, self.typecheck(other), self)

    def __mul__(self, other):
        return BuilderBinOp(binop_Mul, self, self.typecheck(other))

    def __rmul__(self, other):
        return BuilderBinOp(binop_Mul, self, self.typecheck(other))

    def __truediv__(self, other):
        return BuilderBinOp(binop_Div, self, self.typecheck(other))

    def __floordiv__(self, other):
        return BuilderBinOp(binop_Div, self, self.typecheck(other))

    def __mod__(self, other):
        return BuilderBinOp(binop_Mod, self, self.typecheck(other))

    def __lt__(self, other):
        return BuilderBinOp(binop_Less, self, self.typecheck(other))

    def __le__(self, other):
        return BuilderBinOp(binop_Leq, self, self.typecheck(other))

    def __gt__(self, other):
        return BuilderBinOp(binop_Greater, self, self.typecheck(other))

    def __ge__(self, other):
        return BuilderBinOp(binop_Geq, self, self.typecheck(other))

    def __eq__(self, other):
        return BuilderBinOp(binop_Eq, self, self.typecheck(other))

    def __neq__(self, other):
        return BuilderBinOp(binop_Neq, self, self.typecheck(other))

    def __neg__(self):
        return BuilderUSub(self)


class CamsporkError(ValueError):
    pass


def check_return(code):
    if not code:
        raise CamsporkError(str(_thread_local_message_c_str(), "utf-8"))
    return code

class VoidPtr(c_void_p):
    pass


# These can only be passed to the builder that produced them!
class ExprRef(Structure, BuilderExpr):
    _fields_ = [("raw_data", c_uint32)]
    def __bool__(self):
        return self.raw_data != 0  # 0 used to signal error (use check_return)

    def build_expr(self, builder):
        return self

class TrailingBarrierExprRef(Structure):
    _fields_ = [("raw_data", c_uint32)]
    def __bool__(self):
        return self.raw_data != 0  # 0 used to signal error (use check_return)
    def __repr__(self):
        return "camspork.TrailingBarrierExprRef(%i)" % self.raw_data

null_trailing_barrier_expr = TrailingBarrierExprRef(0)

class StmtRef(Structure):
    _fields_ = [("raw_data", c_uint32)]
    def __bool__(self):
        return self.raw_data != 0  # 0 used to signal error (use check_return)
    def __repr__(self):
        return "camspork.StmtRef(%i)" % self.raw_data

class Varname(Structure, BuilderExpr):
    _fields_ = [("slot_1_index", c_uint32)]
    def __bool__(self):
        return self.slot_1_index != 0  # 0 used to signal error (use check_return)

    def c_var_dim_idxs(self, builder):
        return self, 0, ptr_ExprRef()

    def build_expr(self, builder):
        return BuilderIndexExpr(self, ()).build_expr(builder)

    def __getitem__(self, idxs):
        return BuilderIndexExpr(self, ())[idxs]

class OffsetExtentExpr(Structure):
    _fields_ = [("offset_e", ExprRef), ("extent_e", ExprRef)]

class ArriveIdx(Structure):
    _fields_ = [("idx", ExprRef), ("multicast_per_expr", c_uint32)]

# binop enum values are always the same for a given operator (_binop_from_str)
class binop(Structure):
    _fields_ = [("enum_value", c_uint32)]
    def __bool__(self):
        return self.enum_value != 0  # 0 used to signal error (use check_return)


ptr_uint32 = POINTER(c_uint32)
ptr_StmtRef = POINTER(StmtRef)
ptr_ExprRef = POINTER(ExprRef)
ptr_OffsetExtentExpr = POINTER(OffsetExtentExpr)

_thread_local_message_c_str = lib.camspork_thread_local_message_c_str
_thread_local_message_c_str.restype = c_char_p
_thread_local_message_c_str.argtypes = ()

_thread_local_print_program = lib.camspork_thread_local_print_program
_thread_local_print_program.restype = c_int
_thread_local_print_program.argtypes = (c_size_t, c_void_p)

_thread_local_print_program_with_remarks = lib.camspork_thread_local_print_program_with_remarks
_thread_local_print_program_with_remarks.restype = c_int
_thread_local_print_program_with_remarks.argtypes = (c_void_p, )

_new_ProgramBuilder = lib.camspork_new_ProgramBuilder
_new_ProgramBuilder.restype = VoidPtr
_new_ProgramBuilder.argtypes = ()

_delete_ProgramBuilder = lib.camspork_delete_ProgramBuilder
_delete_ProgramBuilder.restype = None
_delete_ProgramBuilder.argtypes = (c_void_p, )

_finish_ProgramBuilder = lib.camspork_finish_ProgramBuilder
_finish_ProgramBuilder.restype = c_int
_finish_ProgramBuilder.argtypes = (c_void_p, )

_ProgramBuilder_is_finished = lib.camspork_ProgramBuilder_is_finished
_ProgramBuilder_is_finished.restype = c_int
_ProgramBuilder_is_finished.argtypes = (c_void_p, )

_ProgramBuilder_size = lib.camspork_ProgramBuilder_size
_ProgramBuilder_size.restype = c_size_t
_ProgramBuilder_size.argtypes = (c_void_p, )

_ProgramBuilder_data = lib.camspork_ProgramBuilder_data
_ProgramBuilder_data.restype = VoidPtr
_ProgramBuilder_data.argtypes = (c_void_p, )

_add_variable = lib.camspork_add_variable
_add_variable.restype = Varname
_add_variable.argtypes = (c_void_p, c_char_p)

_add_ReadValue = lib.camspork_add_ReadValue
_add_ReadValue.restype = ExprRef
_add_ReadValue.argtypes = (c_void_p, Varname, c_uint32, ptr_ExprRef)

_add_Const = lib.camspork_add_Const
_add_Const.restype = ExprRef
_add_Const.argtypes = (c_void_p, c_int32)

_add_USub = lib.camspork_add_USub
_add_USub.restype = ExprRef
_add_USub.argtypes = (c_void_p, ExprRef)

_add_BinOp = lib.camspork_add_BinOp
_add_BinOp.restype = ExprRef
_add_BinOp.argtypes = (c_void_p, binop, ExprRef, ExprRef)

_add_TrailingBarrierExpr = lib.camspork_add_TrailingBarrierExpr
_add_TrailingBarrierExpr.restype = TrailingBarrierExprRef
_add_TrailingBarrierExpr.argtypes = (c_void_p, Varname, c_uint32, POINTER(ArriveIdx))

_add_SyncEnvAccessSingle = lib.camspork_add_SyncEnvAccessSingle
_add_SyncEnvAccessSingle.restype = StmtRef
_add_SyncEnvAccessSingle.argtypes = (c_void_p, Varname, c_uint32, ptr_ExprRef, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, TrailingBarrierExprRef)

_add_SyncEnvAccessWindow = lib.camspork_add_SyncEnvAccessWindow
_add_SyncEnvAccessWindow.restype = StmtRef
_add_SyncEnvAccessWindow.argtypes = (c_void_p, Varname, c_uint32, ptr_OffsetExtentExpr, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, TrailingBarrierExprRef)

_add_SyncEnvFreeShard = lib.camspork_add_SyncEnvFreeShard
_add_SyncEnvFreeShard.restype = StmtRef
_add_SyncEnvFreeShard.argtypes = (c_void_p, Varname, c_uint32, ptr_ExprRef, c_uint32)

_add_MutateValue = lib.camspork_add_MutateValue
_add_MutateValue.restype = StmtRef
_add_MutateValue.argtypes = (c_void_p, Varname, c_uint32, ptr_ExprRef, binop, ExprRef)

_add_Fence = lib.camspork_add_Fence
_add_Fence.restype = StmtRef
_add_Fence.argtypes = (c_void_p, c_uint32, c_uint32, c_uint32, c_uint32)

_add_Arrive = lib.camspork_add_Arrive
_add_Arrive.restype = StmtRef
_add_Arrive.argtypes = (c_void_p, c_uint32, c_uint32, Varname, c_uint32, POINTER(ArriveIdx))

_add_Await = lib.camspork_add_Await
_add_Await.restype = StmtRef
_add_Await.argtypes = (c_void_p, Varname, c_uint32, ptr_ExprRef, c_uint32, c_uint32, c_int32)

_add_ValueEnvAlloc = lib.camspork_add_ValueEnvAlloc
_add_ValueEnvAlloc.restype = StmtRef
_add_ValueEnvAlloc.argtypes = (c_void_p, Varname, c_uint32, ptr_ExprRef)

_add_SyncEnvAlloc = lib.camspork_add_SyncEnvAlloc
_add_SyncEnvAlloc.restype = StmtRef
_add_SyncEnvAlloc.argtypes = (c_void_p, Varname, c_uint32, ptr_ExprRef)

_add_BarrierEnvAlloc = lib.camspork_add_BarrierEnvAlloc
_add_BarrierEnvAlloc.restype = StmtRef
_add_BarrierEnvAlloc.argtypes = (c_void_p, Varname, c_uint32, ptr_ExprRef)

_add_BarrierEnvFree = lib.camspork_add_BarrierEnvFree
_add_BarrierEnvFree.restype = StmtRef
_add_BarrierEnvFree.argtypes = (c_void_p, Varname)

_push_If = lib.camspork_push_If
_push_If.restype = StmtRef
_push_If.argtypes = (c_void_p, ExprRef)

_begin_orelse = lib.camspork_begin_orelse
_begin_orelse.restype = c_int
_begin_orelse.argtypes = (c_void_p,)

_push_SeqFor = lib.camspork_push_SeqFor
_push_SeqFor.restype = StmtRef
_push_SeqFor.argtypes = (c_void_p, Varname, ExprRef, ExprRef)

_push_TasksFor = lib.camspork_push_TasksFor
_push_TasksFor.restype = StmtRef
_push_TasksFor.argtypes = (c_void_p, Varname, ExprRef, ExprRef)

_push_ThreadsFor = lib.camspork_push_ThreadsFor
_push_ThreadsFor.restype = StmtRef
_push_ThreadsFor.argtypes = (c_void_p, Varname, ExprRef, ExprRef, c_uint32, c_uint32, c_uint32)

_push_ParallelBlock = lib.camspork_push_ParallelBlock
_push_ParallelBlock.restype = StmtRef
_push_ParallelBlock.argtypes = (c_void_p, c_uint32, ptr_uint32)

_push_DomainReshape = lib.camspork_push_DomainReshape
_push_DomainReshape.restype = StmtRef
_push_DomainReshape.argtypes = (c_void_p, c_uint32, ptr_uint32)

_pop_body = lib.camspork_pop_body
_pop_body.restype = c_int
_pop_body.argtypes = (c_void_p, ptr_StmtRef, ptr_StmtRef)

_binop_from_str = lib.camspork_binop_from_str
_binop_from_str.restype = binop
_binop_from_str.argtypes = (c_char_p,)

_binop_to_str = lib.camspork_binop_to_str
_binop_to_str.restype = c_char_p
_binop_to_str.arg_types = (binop,)


binop_Assign = check_return(_binop_from_str(b"="))
binop_Add = check_return(_binop_from_str(b"+"))
binop_Sub = check_return(_binop_from_str(b"-"))
binop_Mul = check_return(_binop_from_str(b"*"))
binop_Div = check_return(_binop_from_str(b"/"))
binop_Mod = check_return(_binop_from_str(b"%"))
binop_Less = check_return(_binop_from_str(b"<"))
binop_Leq = check_return(_binop_from_str(b"<="))
binop_Greater = check_return(_binop_from_str(b">"))
binop_Geq = check_return(_binop_from_str(b">="))
binop_Eq = check_return(_binop_from_str(b"=="))
binop_Neq = check_return(_binop_from_str(b"!="))


_new_ProgramEnv = lib.camspork_new_ProgramEnv
_new_ProgramEnv.restype = VoidPtr
_new_ProgramEnv.argtypes = (c_void_p,)

_copy_ProgramEnv = lib.camspork_copy_ProgramEnv
_copy_ProgramEnv.restype = VoidPtr
_copy_ProgramEnv.argtypes = (c_void_p,)

_delete_ProgramEnv = lib.camspork_delete_ProgramEnv
_delete_ProgramEnv.restype = None
_delete_ProgramEnv.argtypes = (c_void_p,)

_exec_top = lib.camspork_exec_top
_exec_top.restype = c_int
_exec_top.argtypes = (c_void_p, c_char_p)

_exec_stmt = lib.camspork_exec_stmt
_exec_stmt.restype = c_int
_exec_stmt.argtypes = (c_void_p, StmtRef, c_char_p)

_alloc_values = lib.camspork_alloc_values
_alloc_values.restype = c_int
_alloc_values.argtypes = (c_void_p, Varname, c_uint32, POINTER(extent_t))

_alloc_scalar_value = lib.camspork_alloc_scalar_value
_alloc_scalar_value.restype = c_int
_alloc_scalar_value.argtypes = (c_void_p, Varname, value_t)

_alloc_sync = lib.camspork_alloc_sync
_alloc_sync.restype = c_int
_alloc_sync.argtypes = (c_void_p, Varname, c_uint32, POINTER(extent_t))

_read_value = lib.camspork_read_value
_read_value.restype = c_int
_read_value.argtypes = (c_void_p, Varname, c_uint32, POINTER(value_t), POINTER(value_t))

_set_value = lib.camspork_set_value
_set_value.restype = c_int
_set_value.argtypes = (c_void_p, Varname, c_uint32, POINTER(value_t), value_t)

_set_debug_validation_enable = lib.camspork_set_debug_validation_enable
_set_debug_validation_enable.restype = c_int
_set_debug_validation_enable.argtypes = (c_void_p, c_uint32)


def to_binop(op):
    if isinstance(op, binop):
        return op
    elif isinstance(op, str):
        return check_return(_binop_from_str(bytes(op, "utf8")))
    else:
        assert isinstance(op, bytes)
        return check_return(_binop_from_str(op))


class BodyCtx:
    __slots__ = ["_builder", "_on_enter", "node", "body", "orelse"]
    _builder: VoidPtr
    _on_enter: Callable[[VoidPtr], None]
    node: StmtRef
    body: StmtRef
    orelse: StmtRef

    def __init__(self, builder, on_enter):
        """Note this doesn't actually do anything until __enter__. This way

        x = b.foo()
        y = b.bar()
        with y:
            with x:
                ...

        works correctly (i.e. y is built second, but scoped first).

        You may also use begin() and end() explicitly as an alternative to with:

        """
        self._builder = builder
        self._on_enter = on_enter

    def begin(self, *a):
        node = check_return(self._on_enter(self._builder))
        assert isinstance(node, StmtRef)
        self.node = node
        return self

    def end(self, *a):
        body = StmtRef()
        orelse = StmtRef()
        check_return(_pop_body(self._builder, byref(body), byref(orelse)))
        self.body = body
        self.orelse = orelse

    __enter__ = begin
    __exit__ = end


@dataclass(slots=True)
class BuilderIndexExpr(BuilderExpr):
    _varname: Varname
    _idx: Tuple[BuilderExpr | ExprRef]

    def c_var_dim_idxs(self, builder):
        dim = len(self._idx)
        if dim == 0:
            return self._varname, 0, ptr_ExprRef()
        else:
            e = (ExprRef * dim)()
            for i, tmp in enumerate(self._idx):
                e[i] = tmp.build_expr(builder)
            return self._varname, dim, e

    def build_expr(self, builder) -> ExprRef:
        """When interpreted as an expression, generate ReadValue"""
        varname, dim, e = self.c_var_dim_idxs(builder)
        return check_return(_add_ReadValue(builder, varname, dim, e))

    def __getitem__(self, a):
        if isinstance(a, tuple):
            a = tuple(self.typecheck(v) for v in a)
        else:
            a = (self.typecheck(a),)
        return BuilderIndexExpr(self._varname, self._idx + a)


@dataclass(slots=True)
class BuilderConst(BuilderExpr):
    _value: int

    def build_expr(self, builder) -> ExprRef:
        return check_return(_add_Const(builder, self._value))


@dataclass(slots=True)
class BuilderUSub(BuilderExpr):
    _arg: BuilderExpr | ExprRef

    def build_expr(self, builder) -> ExprRef:
        return check_return(_add_USub(builder, self._arg.build_expr(builder)))


@dataclass(slots=True)
class BuilderBinOp(BuilderExpr):
    _binop: binop
    _lhs: BuilderExpr | ExprRef
    _rhs: BuilderExpr | ExprRef

    def __init__(self, op, lhs, rhs):
        self._binop = to_binop(op)
        self._lhs = lhs
        self._rhs = rhs

    def build_expr(self, builder) -> ExprRef:
        return check_return(_add_BinOp(builder, self._binop, self._lhs.build_expr(builder), self._rhs.build_expr(builder)))


class ProgramBuilder:
    __slots__ = ["_builder", "_varname_dict"]

    _builder: VoidPtr
    _varname_dict: Dict[object, Varname]

    def __init__(self):
        self._builder = check_return(_new_ProgramBuilder())
        self._varname_dict = {}

    def __del__(self):
        _delete_ProgramBuilder(self._builder)
        self._builder = 0

    def __repr__(self):
        if self.is_finished():
            check_return(_thread_local_print_program(_ProgramBuilder_size(self._builder), _ProgramBuilder_data(self._builder)))
            return str(_thread_local_message_c_str(), "utf-8")
        else:
            return "ProgramBuilder()"

    def finish(self):
        check_return(_finish_ProgramBuilder(self._builder))

    def is_finished(self):
        return bool(_ProgramBuilder_is_finished(self._builder))

    def add_variable(self, name, to_ascii=lambda name: bytes(str(name), "utf8")) -> Varname:
        assert name not in self._varname_dict, f"Duplicate variable name {name!r}"
        varname = check_return(_add_variable(self._builder, to_ascii(name)))
        self._varname_dict[name] = varname
        return varname

    def add_variables(self, names, to_ascii=lambda name: bytes(str(name), "utf8")) -> List[Varname]:
        return [self.add_variable(nm, to_ascii) for nm in names]

    def __getitem__(self, var):
        if isinstance(var, Varname):
            return var
        else:
            return self._varname_dict[var]

    get_varname = __getitem__

    def build_expr(self, e) -> ExprRef:
        return BuilderExpr.typecheck(e).build_expr(self._builder)

    def SyncEnvAccess(
            self, dst: BuilderIndexExpr | Varname, initial_qual_bit: int, extended_qual_bits: int, *,
            is_mutate: bool, is_ooo: bool, extent: Optional[List[BuilderExpr]] = None,
            atomic_qual_bits: int = 0,
            access_multicasts: Tuple[Tuple[bool]] = (),
            barrier: Optional[BuilderIndexExpr | Varname] = None,
            barrier_multicasts: Tuple[Tuple[bool]] = ()) -> StmtRef:
        if barrier is not None:
            barrier_name, barrier_dim, barrier_idx = self._unpack_multicast(barrier, barrier_multicasts)
            trailing_barrier_expr = check_return(_add_TrailingBarrierExpr(self._builder, barrier_name, barrier_dim, barrier_idx))
        else:
            trailing_barrier_expr = null_trailing_barrier_expr
        if access_multicasts:
            assert not extent, "Can't have extent and multicasts (without barrier)"
            c_func = _add_SyncEnvAccessMulticast
            var, dim, idxs = self._unpack_multicast(dst, access_multicasts)
        elif extent:
            # Window variant -- have to interleave offsets and extents (of window)
            var, dim, offsets = dst.c_var_dim_idxs(self._builder)
            assert len(extent) == dim
            c_func = _add_SyncEnvAccessWindow
            idxs = (OffsetExtentExpr * dim)()
            for i in range(dim):
                idxs[i].offset_e = offsets[i]
                idxs[i].extent_e = self.build_expr(extent[i])
        else:
            # Single value variant
            var, dim, offsets = dst.c_var_dim_idxs(self._builder)
            c_func = _add_SyncEnvAccessSingle
            idxs = offsets
        return check_return(c_func(self._builder, var, dim, idxs, initial_qual_bit, extended_qual_bits, atomic_qual_bits, bool(is_mutate), bool(is_ooo), trailing_barrier_expr))

    def SyncEnvFreeShard(self, dst: BuilderIndexExpr | Varname, extended_qual_bits: int):
        var, dim, offsets = dst.c_var_dim_idxs(self._builder)
        return check_return(_add_SyncEnvFreeShard(self._builder, var, dim, offsets, extended_qual_bits))

    def MutateValue(self, dst: BuilderIndexExpr | Varname, op, rhs) -> StmtRef:
        var, dim, idxs = dst.c_var_dim_idxs(self._builder)
        return check_return(_add_MutateValue(self._builder, var, dim, idxs, to_binop(op), self.build_expr(rhs)))

    def Fence(self, V1_transitive: bool, L1_qual_bits: int, L2_full_qual_bits: int, L2_temporal_qual_bits: int) -> StmtRef:
        return check_return(_add_Fence(self._builder, V1_transitive, L1_qual_bits, L2_full_qual_bits, L2_temporal_qual_bits))

    def Arrive(self, V1_transitive: bool, L1_qual_bits: int, dst: BuilderIndexExpr | Varname, barrier_multicasts: Tuple[Tuple[bool]]):
        var, dim, arrive_idx = self._unpack_multicast(dst, barrier_multicasts)
        return check_return(_add_Arrive(self._builder, V1_transitive, L1_qual_bits, var, dim, arrive_idx))

    def Await(self, dst: BuilderIndexExpr, L2_full_qual_bits: int, L2_temporal_qual_bits: int, N: int):
        var, dim, idxs = dst.c_var_dim_idxs(self._builder)
        return check_return(_add_Await(self._builder, var, dim, idxs, L2_full_qual_bits, L2_temporal_qual_bits, N))

    def ValueEnvAlloc(self, e: Varname | BuilderIndexExpr) -> StmtRef:
        return self._add_alloc(_add_ValueEnvAlloc, e)

    def SyncEnvAlloc(self, e: Varname | BuilderIndexExpr) -> StmtRef:
        return self._add_alloc(_add_SyncEnvAlloc, e)

    def BarrierEnvAlloc(self, e: Varname | BuilderIndexExpr) -> StmtRef:
        return self._add_alloc(_add_BarrierEnvAlloc, e)

    def _add_alloc(self, c_adder, e) -> StmtRef:
        var, dim, idxs = e.c_var_dim_idxs(self._builder)
        return check_return(c_adder(self._builder, var, dim, idxs))

    def BarrierEnvFree(self, name) -> StmtRef:
        return check_return(_add_BarrierEnvFree(self._builder, self[name]))

    def If(self, cond, allow_bool=False) -> BodyCtx:
        # Catches expressions like "not var" which Python reduces to constant bool.
        assert allow_bool or not isinstance(cond, bool), "Literal bool passed"
        cond = self.build_expr(cond)
        return BodyCtx(self._builder, lambda builder: _push_If(builder, cond))

    def begin_orelse(self):
        check_return(_begin_orelse(self._builder))

    def SeqFor(self, var, lo, hi) -> BodyCtx:
        var = self.get_varname(var)
        lo = self.build_expr(lo)
        hi = self.build_expr(hi)
        return BodyCtx(self._builder, lambda builder: _push_SeqFor(builder, var, lo, hi))

    def TasksFor(self, var, lo, hi) -> BodyCtx:
        var = self.get_varname(var)
        lo = self.build_expr(lo)
        hi = self.build_expr(hi)
        return BodyCtx(self._builder, lambda builder: _push_TasksFor(builder, var, lo, hi))

    def ThreadsFor(self, var, lo, hi, dim_idx: int, offset: int, box: int) -> BodyCtx:
        var = self.get_varname(var)
        lo = self.build_expr(lo)
        hi = self.build_expr(hi)
        assert isinstance(dim_idx, int)
        assert isinstance(offset, int)
        assert isinstance(box, int)
        return BodyCtx(self._builder, lambda builder: _push_ThreadsFor(builder, var, lo, hi, dim_idx, offset, box))

    def ParallelBlock(self, *coords) -> BodyCtx:
        dim = len(coords)
        array = (c_uint32 * dim)(*coords)
        return BodyCtx(self._builder, lambda builder: _push_ParallelBlock(builder, dim, array))

    def DomainReshape(self, *coords) -> BodyCtx:
        dim = len(coords)
        array = (c_uint32 * dim)(*coords)
        return BodyCtx(self._builder, lambda builder: _push_DomainReshape(builder, dim, array))

    def _unpack_multicast(self, dst: BuilderIndexExpr | Varname, multicasts: Tuple[Tuple[bool]]):
        dst = BuilderExpr.typecheck(dst)
        var, dim, e_idxs = dst.c_var_dim_idxs(self._builder)
        arrive_idx = (ArriveIdx * dim)()
        for dim_idx in range(dim):
            arrive_idx[dim_idx].idx = e_idxs[dim_idx]
            arrive_idx[dim_idx].multicast_per_expr = 0
        # Pack multicast_flags into multicast_per_expr flags ("transposed bits")
        assert len(multicasts) < 32
        for barrier_expr_idx, multicast_flags in enumerate(multicasts):
            assert len(multicast_flags) == dim
            for dim_idx, f in enumerate(multicast_flags):
                if f:
                    arrive_idx[dim_idx].multicast_per_expr |= 1 << barrier_expr_idx
        return var, dim, arrive_idx


def program(pyfunc):
    b = ProgramBuilder()
    pyfunc(b)
    b.finish()
    return b


class Camspork:
    pass


# camspork.program will still work even if the user imports *
camspork = Camspork()
camspork.program = program
camspork.ProgramBuilder = ProgramBuilder


class ProgramEnv:
    __slots__ = ["_program", "_env", "get_varname"]

    _program: ProgramBuilder
    _env: VoidPtr
    get_varname: Callable[[object], Varname]

    def __init__(self, arg):
        if isinstance(arg, ProgramEnv):
            self._program = arg._program
            self._env = check_return(_copy_ProgramEnv(arg._env))
        else:
            assert isinstance(arg, ProgramBuilder), "Expect ProgramBuilder or ProgramEnv"
            self._program = arg
            self._env = check_return(_new_ProgramEnv(arg._builder))
        self.get_varname = arg.get_varname

    def __del__(self):
        _delete_ProgramEnv(self._env)
        self._env = 0

    def __copy__(self):
        return ProgramEnv(self)

    def __deepcopy__(self, memo):
        return ProgramEnv(self)

    def get_program(self) -> ProgramBuilder:
        return self._program

    def exec(self, stmt: Optional[StmtRef] = None, *, excut_filename=None):
        excut_filename_bytes = bytes(excut_filename, "utf-8") if excut_filename else None
        if stmt is None:
            check_return(_exec_top(self._env, excut_filename_bytes))
        else:
            assert isinstance(stmt, StmtRef)
            check_return(_exec_stmt(self._env, stmt, excut_filename_bytes))

    def alloc_scalar_value(self, var, value: int):
        check_return(_alloc_scalar_value(self._env, self.get_varname(var), value))

    def alloc_values(self, var, *extent):
        self._alloc_impl(var, extent, _alloc_values)

    def alloc_sync(self, var, *extent):
        self._alloc_impl(var, extent, _alloc_sync)

    def _alloc_impl(self, var, extent_tuple, c_func):
        c_var = self.get_varname(var)
        c_dim = len(extent_tuple)
        extent_tuple = tuple(extent_tuple)
        for n in extent_tuple:
            assert n >= 0
        c_extent = (extent_t * c_dim)(*extent_tuple)
        check_return(c_func(self._env, c_var, c_dim, c_extent))

    def read_value(self, var, *idxs):
        c_dim = len(idxs)
        c_idxs = (value_t * c_dim)(*idxs)
        c_out = value_t(0)
        check_return(_read_value(self._env, self.get_varname(var), c_dim, c_idxs, byref(c_out)))
        return c_out.value

    def set_value(self, arg, var, *idxs):
        c_dim = len(idxs)
        c_idxs = (value_t * c_dim)(*idxs)
        check_return(_set_value(self._env, self.get_varname(var), c_dim, c_idxs, arg))

    def set_debug_validation_enable(self, flag):
        check_return(_set_debug_validation_enable(self._env, bool(flag)))

    def program_with_remarks(self):
        check_return(_thread_local_print_program_with_remarks(self._env))
        return str(_thread_local_message_c_str(), "utf-8")


if __name__ == "__main__":
    b_validation = False

    @camspork.program
    def foo_barrier(b: camspork.ProgramBuilder):
        bars = b.add_variable("bars")
        m = b.add_variable("m")
        n = b.add_variable("n")
        k = b.add_variable("k")
        buf = b.add_variable("buf")
        b.SyncEnvAlloc(buf[128])
        with b.ParallelBlock(64):
            task = b.add_variable("task")
            warp = b.add_variable("warp")
            with b.TasksFor(task, 0, 2):
                with b.ThreadsFor(warp, 0, 2, 0, 0, 32):
                    # This program is BS
                    b.BarrierEnvAlloc(bars[4, 2, 2])
                    tid = b.add_variable("tid")
                    with b.ThreadsFor(tid, 0, 24, 0, 0, 1):
                        b.SyncEnvAccess(buf[tid + 32*warp + 64*task], 2, 2, is_mutate=True, is_ooo=True, atomic_qual_bits=8192, barrier=bars[m, n, k], barrier_multicasts=((True, False, False),))
                    with b.ThreadsFor(tid, 0, 1, 0, 0, 14):
                        b.Arrive(True, 3, bars[m, n, k], ((True, False, True), (True, True, False)))
                        b.Await(bars[m, n, k], 3, 3, N=0)
                    with b.ThreadsFor(tid, 0, 14, 0, 0, 1):
                        b.SyncEnvAccess(buf[tid + 32*warp + 64*task], 1, 1, is_mutate=False, is_ooo=False)
    print(foo_barrier)
    env = ProgramEnv(foo_barrier)
    env.set_debug_validation_enable(b_validation)
    env.alloc_scalar_value("m", 0)
    env.alloc_scalar_value("n", 1)
    env.alloc_scalar_value("k", 0)
    env.exec(excut_filename="foo_barrier_excut.json")
    env.set_debug_validation_enable(True)  # defer to later

    @camspork.program
    def fib(b):
        fib_size = b.add_variable("fib_size")
        _fib = b.add_variable("fib")
        _iter = b.add_variable("iter")
        b.ValueEnvAlloc(_fib[fib_size,])
        b.MutateValue(_fib[0,], "=", 0)
        b.MutateValue(_fib[1,], "=", 1)
        with b.SeqFor(_iter, 2, fib_size):
            b.MutateValue(_fib[_iter,], "=", _fib[_iter-1,] + _fib[_iter-2,])

        _dst = b.add_variable("dst")
        b.ValueEnvAlloc(_dst[fib_size,])
        with b.SeqFor(_iter, 0, fib_size):
          with b.If(_iter % 5):
            b.MutateValue(_fib[_iter,], "=", -_fib[_iter,])
            b.MutateValue(_fib[_iter,], "*", 10000)
            b.begin_orelse()
            b.MutateValue(_fib[_iter,], "/", 5)

    env = ProgramEnv(fib)
    env.set_debug_validation_enable(b_validation)
    print(fib)
    env.alloc_scalar_value("fib_size", 22)
    env.exec()
    for i in range(0, env.read_value("fib_size")):
        print("%2i %i" % (i, env.read_value("fib", i)))
    env.set_debug_validation_enable(True)  # defer to later

    @camspork.program
    def extent_test(b: camspork.ProgramBuilder):
        buf = b.add_variable("buf")
        b.SyncEnvAlloc(buf[10, 16])
        with b.ParallelBlock(4):
            tid = b.add_variable("tid")
            with b.ThreadsFor(tid, 0, 4, 0, 0, 1):
                with b.If(tid == 0):
                    b.SyncEnvAccess(buf[0, 1], 1, 1, is_mutate=False, is_ooo=False)
                    b.SyncEnvAccess(buf[0, 2], 1, 1, is_mutate=False, is_ooo=False)
                    b.SyncEnvAccess(buf[0, 3], 1, 1, is_mutate=False, is_ooo=False)
                    b.SyncEnvAccess(buf[0, 4], 1, 1, is_mutate=False, is_ooo=False)
                b.SyncEnvAccess(buf[tid, 2 * tid], 1, 1, is_mutate=False, is_ooo=False, extent=[6, 5])
            b.Fence(True, 1, 1, 1)
            m = b.add_variable("m")
            n = b.add_variable("n")
            with b.SeqFor(m, 0, 10):
                with b.SeqFor(n, 0, 16):
                    b.SyncEnvAccess(buf[m, n], 1, 1, is_mutate=True, is_ooo=False)
    print(extent_test)
    env = ProgramEnv(extent_test)
    env.set_debug_validation_enable(b_validation)
    try:
        env.exec(excut_filename="extent_excut.json")
    except:
        print(env.program_with_remarks())
        raise
    env.set_debug_validation_enable(True)  # defer to later

    @camspork.program
    def atomic_test(b: camspork.ProgramBuilder):
        buf = b.add_variable("buf")
        use_atomics = b.add_variable("use_atomics")
        wrong_tl = b.add_variable("wrong_tl")
        fence_enable = b.add_variable("fence_enable")
        b.SyncEnvAlloc(buf[8])
        with b.ParallelBlock(8):
            tid = b.add_variable("tid")
            with b.ThreadsFor(tid, 0, 8, 0, 0, 1):
                s = b.add_variable("s")
                with b.SeqFor(s, 0, 8):
                    with b.If(use_atomics):
                        b.SyncEnvAccess(buf[s], 1, 1, is_mutate=True, is_ooo=False, atomic_qual_bits=1)
                        with b.If(wrong_tl):
                            b.SyncEnvAccess(buf[s], 2, 2, is_mutate=True, is_ooo=False, atomic_qual_bits=2)
                            b.begin_orelse()
                            b.SyncEnvAccess(buf[s], 1, 1, is_mutate=True, is_ooo=False, atomic_qual_bits=1)
                        b.begin_orelse()
                        b.SyncEnvAccess(buf[s], 1, 1, is_mutate=True, is_ooo=False, atomic_qual_bits=0)
            with b.If(fence_enable):
                b.Fence(True, 1, 5, 5)
            with b.ThreadsFor(tid, 0, 8, 0, 0, 1):
                b.SyncEnvAccess(buf[tid], 1, 1, is_mutate=False, is_ooo=False)

    env = ProgramEnv(atomic_test)
    env.set_debug_validation_enable(b_validation)
    env.alloc_scalar_value("use_atomics", 1)
    env.alloc_scalar_value("wrong_tl", 0)
    env.alloc_scalar_value("fence_enable", 1)
    try:
        env.exec(excut_filename="atomic_excut.json")
    except:
        print(env.program_with_remarks())
        raise
    env.set_debug_validation_enable(True)  # defer to later


    @camspork.program
    def fence_test(b: camspork.ProgramBuilder):
        num_tasks = b.add_variable("num_tasks")
        fence_enable = b.add_variable("fence_enable")
        buf = b.add_variable("buf")
        b.SyncEnvAlloc(buf[64])
        with b.ParallelBlock(64):
            task = b.add_variable("task")
            tid = b.add_variable("tid")
            global tasks_for
            with b.TasksFor(task, 0, num_tasks) as tasks_for:
                with b.ThreadsFor(tid, 0, 64, 0, 0, 1):
                    b.SyncEnvAccess(buf[tid], 1, 1, is_mutate=True, is_ooo=False)
                with b.If(fence_enable):
                    b.Fence(True, 1, 1, 511)
                with b.ThreadsFor(tid, 0, 64, 0, 0, 1):
                    s = b.add_variable("s")
                    with b.SeqFor(s, 0, 64):
                        b.SyncEnvAccess(buf[s], 1, 1, is_mutate=False, is_ooo=False)
    print(fence_test)
    print(tasks_for.node)
    print(tasks_for.body)
    print(tasks_for.orelse)
    env = ProgramEnv(fence_test)
    env.set_debug_validation_enable(b_validation)
    env.alloc_scalar_value("num_tasks", 1)
    env.alloc_scalar_value("fence_enable", 1)
    env.exec(excut_filename="fence_excut.json")
    env.set_debug_validation_enable(True)  # defer to later


    @camspork.program
    def realloc_test(b: camspork.ProgramBuilder):
        with b.ParallelBlock(2):
            task = b.add_variable("task")
            tid = b.add_variable("tid")
            with b.TasksFor(task, 0, 3):
                buf = b.add_variable("buf")
                scalar = b.add_variable("scalar")
                b.SyncEnvAlloc(buf[2])
                b.SyncEnvAlloc(scalar)
                with b.ThreadsFor(tid, 0, 2, 0, 0, 1):
                    b.SyncEnvAccess(buf[tid], 1, 1, is_mutate=True, is_ooo=False)
                    b.SyncEnvAccess(buf[tid], 1, 1, is_mutate=True, is_ooo=False)
                    with b.If(tid == 0):
                      b.SyncEnvAccess(scalar, 1, 1, is_mutate=True, is_ooo=False)
                # b.Fence(True, 1, 1, 1)
                with b.ThreadsFor(tid, 0, 2, 0, 0, 1):
                    b.SyncEnvAccess(buf[tid], 1, 1, is_mutate=True, is_ooo=False)
    print(realloc_test)
    env = ProgramEnv(realloc_test)
    env.exec(excut_filename="realloc_excut.json")
