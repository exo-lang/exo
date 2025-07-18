from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable, Dict, Optional, Type, List
from warnings import warn

from ..core.memory import MemGenError, memwin_template, DRAM, BarrierType
from ..core.instr_info import AccessInfo, InstrInfo
from ..core.prelude import Sym, SrcInfo
from ..core.LoopIR import (
    LoopIR,
    T,
    LoopIR_Do,
    LoopIR_Rewrite,
    GetReads,
)

from .distributed_memory import ThreadIter, DistributedIdxFsm, DistributedAllocState
from .timelines import Instr_tl, Sync_tl
from . import timelines
from .async_config import CudaDeviceFunction, CudaAsync
from .barrier_usage import BarrierUsage, SyncInfo
from .base_with_context import is_if_holding_with
from .ext_with_context import ExtWithContext
from .coll_algebra import (
    CollParam,
    CollUnit,
    clusterDim_param,
    blockDim_param,
    CollIndexExpr,
    CollTiling,
    cuda_thread,
    cuda_warp,
    cuda_warpgroup,
    cuda_cta_in_cluster,
    cuda_agnostic_sub_cta,
    cuda_agnostic_intact_cta,
)
from .cuda_memory import (
    CudaBasicDeviceVisible,
    CudaBasicSmem,
    SmemConfigInputs,
    CudaGridConstant,
    CudaRmem,
)
from .lowered_barrier import LoweredBarrierType, LoweredBarrier
from .cuda_sync_state import SyncStateBuilder
from .cuda_warp_config import WarpLayoutInfo
from .loop_modes import CudaTasks, CudaThreads, Seq, seq, _CodegenPar
from .sync_types import SyncType
from .with_cuda_warps import CudaWarps


# No BarrierExpr here; handled specially as part of SyncStmt.
idx_e_types = (LoopIR.Read, LoopIR.WindowExpr)
idx_s_types = (LoopIR.Assign, LoopIR.Reduce)


def loopir_lower_cuda(s, ctx: SporkLoweringCtx):
    """Top level function to call.

    Transforms with-statement node holding CudaDeviceFunction to
    with-statement node holding ExtWithContext, ready for final
    code lowering with the main LoopIR-to-C compiler.
    """
    scan = SubtreeScan(s, ctx)
    # Scanner validates correctness and passes advice from "global analysis"
    # to the subtree rewriter on how to substitute certain stmts/expressions.
    return SubtreeRewrite(s, scan, ctx).result()


# ========================   PHASE 1: subtree scan   ========================
# Just collect information about the subtree corresponding to the
# CUDA device function.


class SubtreeScan(LoopIR_Do):
    __slots__ = [
        "ctx",
        "sync_state_builder",
        "distributed_alloc_states",
        "thread_iters",
        "cuda_warps_dfs_codegen",
        "fmt_dict",
        "named_warp_used_syms",
        "task_loop_depth",
        "task_iter_syms",
        "device_args_syms",
        "grid_constant_syms",
        "scalar_ref_syms",
        #
        "_local_envtyp",
        "_syms_needed",
        "_stmt_stack",
        "_coll_env",
        "_coll_tiling",
        "_current_warp_name",
        "named_warps",
        "setmaxnreg_is_inc",
    ]

    ctx: SporkLoweringCtx

    # We will have to substitute some LoopIR nodes in the SubtreeRewrite phase.
    # During the scan, for a node that needs to be rewritten, we will stash
    # needed info for the rewrites here.
    sync_state_builder: SyncStateBuilder
    distributed_alloc_states: Dict[Sym, DistributedAllocState]
    thread_iters: Dict[Sym, ThreadIter]  # Info on iterators of cuda_threads loops

    # _CodegenPar needed for rewrites of CudaWarps blocks
    # as encountered in DFS order (typical traversal order).
    # This is a bit fragile, but CudaWarps blocks have no
    # unique information (id(...) is not unique after scheduling).
    cuda_warps_dfs_codegen: List[_CodegenPar]

    fmt_dict: Dict

    # For each warp name, record the set of Syms used when executing the code
    # path for that warp. Needed to remove unused variables.
    named_warp_used_syms: Dict[str, Set[Sym]]

    task_loop_depth: int  # Depth of if stmts + cuda_tasks loops
    task_iter_syms: List[Sym]
    device_args_syms: List[Sym]
    grid_constant_syms: Set[Sym]
    scalar_ref_syms: Set[Sym]

    _local_envtyp: Dict[Sym, LoopIR.type]
    _syms_needed: Set[Sym]
    _stmt_stack: List[LoopIR.stmt]
    _coll_env: Dict[CollParam, int]
    _coll_tiling: CollTiling
    _current_warp_name: Optional[str]
    named_warps: Dict[str, WarpLayoutInfo]
    setmaxnreg_is_inc: Optional[Dict[int, bool]]

    def __init__(self, s, ctx: SporkLoweringCtx):
        assert is_if_holding_with(s, LoopIR)
        cuda_device_function: CudaDeviceFunction = s.cond.val
        assert isinstance(cuda_device_function, CudaDeviceFunction)

        blockDim = cuda_device_function.blockDim
        clusterDim = cuda_device_function.clusterDim
        coll_env = {clusterDim_param: clusterDim, blockDim_param: blockDim}

        self.ctx = ctx
        self.sync_state_builder = SyncStateBuilder(coll_env)
        self.distributed_alloc_states = {}
        self.thread_iters = {}
        self.cuda_warps_dfs_codegen = []
        self.fmt_dict = {
            "proc": ctx.proc_name(),
            "lib_name": ctx.lib_name(),
            "N": ctx.kernel_index(),
            "blockDim": blockDim,
            "clusterDim": clusterDim,
            "launchConfig_clusterDim_snippet": "",
            "blocks_per_sm": cuda_device_function.blocks_per_sm,
        }
        self.named_warps = cuda_device_function.named_warps
        self.setmaxnreg_is_inc = cuda_device_function.setmaxnreg_is_inc
        self._current_warp_name = None
        self.named_warp_used_syms = {nm: set() for nm in self.named_warps}

        # Only set clusterDim if not 1, not only for pre-H100 compatibility,
        # but also this avoids mysterious performance loss.
        if clusterDim != 1:
            self.fmt_dict[
                "launchConfig_clusterDim_snippet"
            ] = launchConfig_clusterDim_snippet

        # Validate top-level form of cuda kernel
        # Must be nest of 1+ cuda_tasks loops, and optional if statements
        # with no else block.
        self.task_iter_syms = []
        task_iter_strs = set()
        valid_sync = False

        if len(s.body) != 1:
            raise ValueError(f"{s.srcinfo}: expected cuda_tasks loop alone")

        self.task_loop_depth = 0
        task_loop_body = s.body
        found_task_loop = False
        first_stmt = s
        while True:
            if len(task_loop_body) == 0:
                break
            first_stmt = task_loop_body[0]

            # single cuda_tasks loop
            if isinstance(first_stmt, LoopIR.For):
                if isinstance(first_stmt.loop_mode, CudaTasks):
                    # Record cuda_tasks iteration variable
                    if str(first_stmt.iter) in task_iter_strs:
                        raise ValueError(
                            f"{s.srcinfo}: unsupported, duplicate cuda_tasks iter variable name {first_stmt.iter}"
                        )
                    task_iter_strs.add(str(first_stmt.iter))
                    self.task_iter_syms.append(first_stmt.iter)
                    # Validate no extra statements, then recurse in
                    if len(task_loop_body) != 1:
                        raise ValueError(
                            f"{task_loop_body[1].srcinfo}: invalid statement after cuda_tasks loop"
                        )
                    else:
                        found_task_loop = True
                        self.task_loop_depth += 1
                        task_loop_body = task_loop_body[0].body
                        continue

            # single if stmt with no orelse
            elif not is_if_holding_with(first_stmt, LoopIR) and isinstance(
                first_stmt, LoopIR.If
            ):
                if len(task_loop_body) == 1 and not first_stmt.orelse:
                    self.task_loop_depth += 1
                    task_loop_body = task_loop_body[0].body
                    continue

            # End when encountering first non-cuda_tasks, non-simple if stmt.
            break

        if not found_task_loop:
            raise ValueError(f"{first_stmt.srcinfo}: missing cuda_tasks loop")

        # Prepare exo_Task struct (struct of task loop iteration variables)
        # They will be named exo_task_* in deviceMainLoop
        # and exo_task.* in deviceTask.
        self.fmt_dict["task_args"] = ", ".join(
            "exo_task_" + str(sym) for sym in self.task_iter_syms
        )
        self.fmt_dict["task_struct_body"] = "\n".join(
            f"    int_fast32_t {str(sym)};" for sym in self.task_iter_syms
        )

        # Scan the subtree
        # We seed the analysis of the collective units with the tiling
        # for the top-level collective (clusterDim x blockDim,
        # with redundant clusterDim removed if clusterDim = 1).
        self._local_envtyp = {}
        self._syms_needed = set()
        self._stmt_stack = []
        self._coll_env = coll_env
        assert clusterDim > 0 and isinstance(clusterDim, int)
        threadIdx_expr = CollIndexExpr("threadIdx.x", blockDim)
        if clusterDim == 1:
            tlc_offset = (0,)
            tlc_box = (blockDim,)
            intra_box_exprs = (threadIdx_expr,)
        else:
            tlc_offset = (0, 0)
            tlc_box = (clusterDim, blockDim)
            cta_expr = CollIndexExpr("blockIdx.x") % clusterDim
            intra_box_exprs = (cta_expr, threadIdx_expr)
        self._coll_tiling = CollTiling(
            None,  # parent
            None,  # _iter
            tlc_box,
            tlc_box,
            tlc_offset,
            tlc_box,
            intra_box_exprs,
            1,
            CollIndexExpr(0),
        )
        self.do_stmts(s.body)

        # Prepare the device args struct
        # These are all the syms that appear in the subtree that were
        # defined by the outside (CPU function) environment.
        #
        # Additionally, we have special handling for grid constants
        # (force const) and scalar parameters (scalar_ref if not grid constant).
        self.device_args_syms = []
        self.grid_constant_syms = set()
        self.scalar_ref_syms = set()
        for sym in tuple(self._syms_needed):
            # For Tensors, we need to pass the sizes explicitly to the device
            try:
                typ = self.sym_type(sym)
                if typ.is_tensor_or_window():
                    getter = GetReads()
                    getter.do_t(typ)
                    for nm, _ in getter.reads:
                        self._syms_needed.add(nm)

            except KeyError:
                continue
        for sym in self._syms_needed:
            try:
                cpu_nm = ctx.sym_c_name(sym)
            except KeyError:
                continue
            self.device_args_syms.append(sym)
            if issubclass(ctx.sym_mem(sym), CudaGridConstant):
                self.grid_constant_syms.add(sym)
            elif self.sym_type(sym).is_real_scalar():
                # elif ensures not added if grid constant
                self.scalar_ref_syms.add(sym)

        # The device args struct will be sorted in the order the variables were
        # created in Python code
        self.device_args_syms.sort(key=lambda s: s.id_number())

        # Assemble the exo_DeviceArgs struct definition
        # (device_args_struct_lines) and the syntax for
        # aggregate-initialization of exo_DeviceArgs in C code
        # (device_args_values).

        device_args_decls = []
        device_args_comments = []
        device_args_values = []

        for sym in self.device_args_syms:
            c_name = ctx.sym_c_name(sym)
            mem = ctx.sym_mem(sym)
            if sym not in self.grid_constant_syms:
                # Non-grid-constant, passed as in Exo C code.
                # They will appear as exo_deviceArgs.{c_name} in CUDA code.
                fnarg = LoopIR.fnarg(sym, self.sym_type(sym), mem, s.srcinfo)
                ctx.append_fnarg_decl(
                    fnarg, c_name, device_args_decls, device_args_comments
                )
                e = LoopIR.Read(sym, [], self.sym_type(sym), s.srcinfo)
                device_args_values.extend(ctx.fnarg_values(e, ctx.is_const(sym), False))
            else:
                # Grid constants are passed as array or scalar by-value
                c_arg = ctx.sym_c_name(sym)
                typ = self.sym_type(sym)
                if typ.is_win():
                    raise TypeError(
                        f"{s.srcinfo}: grid constant parameter {sym} "
                        f"cannot be a window"
                    )
                elif typ.is_dense_tensor():
                    n = prod(type_const_shape(typ, "grid constant", sym, s.srcinfo))
                    device_args_decls.append(f"{typ.basetype().ctype()} {c_name}[{n}]")
                    # We have to manually pass each array element by value ...
                    arg_fragments = ["{"]
                    for i in range(n):
                        if i != 0:
                            arg_fragments.append(", ")
                        arg_fragments.append(f"{c_arg}[{i}]")
                    arg_fragments.append("}")
                    device_args_values.append("".join(arg_fragments))
                else:
                    # Scalar grid constant
                    device_args_decls.append(f"{typ.ctype()} {c_name}")
                    if ctx.sym_is_scalar_ref(sym):
                        c_arg = f"*{c_arg}"
                    device_args_values.append(c_arg)
                device_args_comments.append(f"{sym}: {typ} @{mem.name()}")

        device_args_struct_lines = []
        assert len(device_args_decls) == len(device_args_comments)
        for i in range(len(device_args_decls)):
            device_args_struct_lines.append(
                f"    {device_args_decls[i]};  // {device_args_comments[i]}"
            )
        # exo_ExcutDeviceLog is only defined in the supplemental exo_excut.h file.
        # This used to be an empty struct, but this caused crazy C/C++ ABI issues.
        # Must be the last arg, as exo_excut_get_device_log() is defined to nothing.
        # Fortunately, C seems to allow a trailing comma here.
        device_args_struct_lines.append(
            "    EXO_EXCUT_DEVICE_LOG_MEMBER  // for Exo pytest (exo_excut.h)"
        )
        device_args_values.append("exo_excut_get_device_log()")

        self.fmt_dict["device_args"] = ", ".join(device_args_values)
        self.fmt_dict["device_args_struct_body"] = "\n".join(device_args_struct_lines)

    def sym_type(self, sym: Sym):
        return self.ctx.sym_type(sym, self._local_envtyp)

    def do_s(self, s):
        # Save state
        old_coll_tiling = self._coll_tiling
        old_warp_name = self._current_warp_name
        self._stmt_stack.append(s)

        if isinstance(s, LoopIR.Call):
            self.do_call_stmt(s)
            # do_call_stmt cannot use super().do_s(s) due to window handling
        else:
            # Modify state, then recurse with super()
            # (order is important so recursion sees updated state!)
            self.apply_s(s)
            super().do_s(s)

        # Special case (after recursion) for handling prologue/epilogue sync
        if is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, CudaAsync):
                self.post_inspect_cuda_async(s)

        # Restore state
        self._stmt_stack.pop()
        self._coll_tiling = old_coll_tiling
        self._current_warp_name = old_warp_name

    def do_e(self, e, distributed_coll_units=()):
        self.apply_e(e, distributed_coll_units)
        super().do_e(e)

    def apply_e(self, e, distributed_coll_units):
        if isinstance(e, idx_e_types):
            # BarrierExpr not handled here; part of SyncStmt handling.
            self.mark_sym_used(e.name)
            self.apply_idx(e, self._stmt_stack[-1], distributed_coll_units)
        elif not isinstance(e, (LoopIR.BarrierExpr, LoopIR.StrideExpr)):
            assert not hasattr(e, "name"), "Add handling for array indexing"

    def apply_s(self, s):
        if isinstance(s, idx_s_types):
            self.mark_sym_used(s.name)
            self.apply_idx(s, s, ())
        elif not isinstance(s, (LoopIR.WindowStmt, LoopIR.Alloc, LoopIR.Free)):
            assert not hasattr(s, "name"), "Add handling for array indexing"

        if is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, CudaWarps):
                self.apply_with_cuda_warps(s)
        elif isinstance(s, LoopIR.For):
            loop_mode = s.loop_mode
            if isinstance(loop_mode, Seq):
                pass
            elif isinstance(loop_mode, CudaTasks):
                if s.iter not in self.task_iter_syms:
                    raise ValueError(
                        f"{s.srcinfo}: cuda_tasks loop must appear only in top level nest of CudaDeviceFunction"
                    )
            elif isinstance(loop_mode, CudaThreads):
                self.apply_cuda_threads_loop(s)
            else:
                raise TypeError(
                    f"{s.srcinfo}: unexpected loop mode {s.loop_mode.loop_mode_name()} in CudaDeviceFunction"
                )
        elif isinstance(s, LoopIR.WindowStmt):
            # Unlike for Calls, the WindowExpr here do not allow intervals for
            # any distributed dimensions ... this would be very hard to support.
            # Basically the dimensionality of the WindowStmt will never change!
            # See WindowExpr case for remove_distributed_idx.
            self._local_envtyp[s.name] = s.rhs.type
        elif isinstance(s, LoopIR.Alloc):
            self._local_envtyp[s.name] = s.type
            if s.type.is_barrier():
                native_unit = None
            else:
                if not issubclass(s.mem, CudaBasicDeviceVisible):
                    raise TypeError(
                        f"{s.srcinfo}: For cuda code, memory type "
                        f"({s.mem.name()}) must subclass CudaBasicDeviceVisible"
                    )
                native_unit = s.mem.native_unit()
            self.distributed_alloc_states[s.name] = DistributedAllocState(
                self._coll_tiling, native_unit
            )

        elif isinstance(s, LoopIR.Free):
            if s.type.is_barrier():
                self.sync_state_builder.add_barrier(
                    s.name,
                    self.get_barrier_usage(s.name),
                    self.distributed_alloc_states[s.name],
                    self.thread_iters,
                )

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if (n_threads := self._coll_tiling.box_num_threads()) != 1:
                raise ValueError(
                    f"{s.srcinfo}: write must be executed by one "
                    f"thread only (current: {n_threads} threads)\n"
                    f"stmt: {s}"
                )
        elif isinstance(s, LoopIR.SyncStmt):
            # Distributed memory analysis and CollTiling for Fence/Arrive/Await
            if s.sync_type.is_split():
                assert len(s.barriers) >= 1
                name = s.barriers[0].name
                self.mark_sym_used(name)
                usage: BarrierUsage = self.get_barrier_usage(name)
                state = self.distributed_alloc_states.get(name)
                assert isinstance(state, DistributedAllocState)

                fsm = DistributedIdxFsm(
                    s,
                    state,
                    "cuda_threads",
                    self.thread_iters,  # May be modified
                    self._coll_env,
                    self._coll_tiling,
                    (),
                )
                # There is no native_unit; we parse all indices as distributed
                assert state.optional_native_unit is None
                e0 = s.barriers[0]
                for i in range(len(e0.idx)):
                    fsm.consume_SyncStmt_idx(
                        self._stmt_stack, s, self.sym_type(e0.name), i
                    )

                # We now have the distributed indices in distributed_iters.
                # Store in DistributedAllocState if this is the first use, or check
                # consistency (index equality) with prior uses.
                fsm.check_store_state(s, state)
                fsm.inspect_arrive_await(s, self._coll_tiling, usage, state)
            else:
                assert len(s.barriers) == 1
                e = s.barriers[0]
                assert isinstance(e, LoopIR.BarrierExpr)
                assert e.name not in self.distributed_alloc_states
                state = DistributedAllocState.from_fence(s, self._coll_tiling)
                self.distributed_alloc_states[e.name] = state
                self.sync_state_builder.add_barrier(
                    e.name,
                    self.get_barrier_usage(e.name),
                    state,
                    self.thread_iters,
                )

        elif isinstance(s, LoopIR.Call):
            assert 0, "Was supposed to be handled specially with do_call_stmt"

    def apply_with_cuda_warps(self, s):
        ctx: CudaWarps = s.cond.val
        assert isinstance(ctx, CudaWarps)
        coll_tiling = self._coll_tiling
        is_top_level = self._current_warp_name is None

        # Top-level CudaWarps: adjust CollTiling to account for offset of named warps.
        # We ignore the codegen here ... because of how the deviceTask is specialized
        # per named-warp set, we already can assume the physical code is executed
        # only by the subset of warps that are part of the named warp set.
        #
        # NB it's important that this is skipped when the user doesn't
        # use named warps (fallback len-1 case) because the (***)
        # restriction must not be enforced.
        if is_top_level and len(self.named_warps) > 1:
            name = "" if ctx.name is None else ctx.name
            if (info := self.named_warps.get(name)) is None:
                known_names = sorted(self.named_warps)
                raise ValueError(
                    f"{s.srcinfo}: top-level CudaWarps must provide valid warp name, not {ctx.name!r}; your CudaDeviceFunction defines: {known_names}"
                )

            # (***) Named warps won't work if the CTA has already been
            # subdivided by a cuda_threads loop.
            if detail := self._coll_tiling.unit_mismatch(
                cuda_agnostic_intact_cta, self._coll_env
            ):
                raise ValueError(
                    f"{s.srcinfo}: named {ctx} requires CTA not to be subdivided by parent cuda_threads loop (detail: {detail})"
                )

            # Extract lo/hi offsets (with defaulted values allowed).
            # This gets handled towards the end of the function.
            warps_lo = 0 if ctx.lo is None else ctx.lo
            warps_hi = info.count if ctx.hi is None else ctx.hi
            if warps_hi > info.count:
                raise ValueError(
                    f"{s.srcinfo}: CudaWarps.hi={warps_hi} out-of-range for {name!r}-named warps (only have {info.count})"
                )

            # (1/2) adjust CollTiling for named warps offset. Codegen discarded.
            coll_tiling = coll_tiling.specialized(
                cuda_warp, info.offset, info.offset + info.count, self._coll_env
            )

        # Nested CudaWarps: interpret lo/hi literally as the higher-level
        # CudaWarps will have already handled the named warp offset adjustment.
        # Can't request different named warps now.
        else:
            name = self._current_warp_name if ctx.name is None else ctx.name
            if name != self._current_warp_name:
                raise ValueError(
                    f"{s.srcinfo}: nested CudaWarps cannot change warp name from {self._current_warp_name!r} to {name!r}"
                )
            warps_lo = ctx.lo
            warps_hi = ctx.hi
            if warps_lo is None or warps_hi is None:
                raise ValueError(
                    f"{s.srcinfo}: nested CudaWarps must define lo and hi explicitly"
                )

        self._current_warp_name = name

        # (2/2) Ajdust CollTiling for lo/hi offset.
        try:
            coll_tiling = coll_tiling.specialized(
                cuda_warp, warps_lo, warps_hi, self._coll_env
            )
        except AssertionError:
            raise
        except Exception as e:
            raise ValueError(f"{s.srcinfo}: failed to compile {ctx}: {e}") from e

        self._coll_tiling = coll_tiling
        self.cuda_warps_dfs_codegen.append(
            _CodegenPar(
                coll_tiling.codegen_expr.codegen(),
                str(ctx),
                (coll_tiling.codegen_lo, coll_tiling.codegen_hi),
                name,
            )
        )

    def expect_SyncStmt(self, async_block, is_epilogue, first_sync_tl, second_sync_tl):
        # This is really strict, requires equality with expected sync-tl
        # instead of just implements_first/implements_second(...)
        ctx = async_block.cond.val
        sync = async_block.body[-1] if is_epilogue else async_block.body[0]
        verb = "missing"
        if isinstance(sync, LoopIR.SyncStmt):
            verb = "wrong"
            sync_type = sync.sync_type
            if sync_type.first_sync_tl == first_sync_tl:
                if sync_type.second_sync_tl == second_sync_tl:
                    return sync
        noun = "epilogue" if is_epilogue else "prologue"
        expected = SyncType(first_sync_tl, second_sync_tl, 1).format_stmt(["..."])
        raise ValueError(
            f"{async_block.srcinfo}: {verb} {noun} sync in {ctx} block; "
            f"expect {expected}"
        )

    def post_inspect_cuda_async(self, s):
        # Must be run after inspecting the body of the CudaAsync block
        # since the barriers must have been scanned.
        # We detect required prologue/epilogue sync here.
        ctx = s.cond.val
        assert isinstance(ctx, CudaAsync)
        instr_tl = ctx.get_instr_tl()
        assert instr_tl in timelines.cuda_async_instr_tl
        assert s.body

        def inspect(is_epilogue, L1, L2):
            sync_stmt = self.expect_SyncStmt(s, is_epilogue, L1, L2)

        # wgmma_async_instr requires prologue wgmma fence, epilogue Arrive(wgmma_async)
        if instr_tl == timelines.wgmma_async_instr:
            inspect(False, timelines.wgmma_fence_1, timelines.wgmma_fence_2)
            inspect(True, timelines.wgmma_async, None)
        # Sm80_cp_async, tma_to_smem_async, tma_to_gmem_async have no prologue/epilogue

    def apply_cuda_threads_loop(self, s):
        def get_const(e, name):
            expected = "literal int value"
            if isinstance(e, LoopIR.Const):
                if e.type.is_indexable():
                    v = int(e.val)
                    if v != 0 and name == "lo":
                        expected = "0"
                    else:
                        return v
            raise ValueError(
                f"{e.srcinfo}: expected {expected} for {name} of {s.iter} loop (rewrite with simplify(...) if needed)"
            )

        lo_int = get_const(s.lo, "lo")
        hi_int = get_const(s.hi, "hi")
        assert lo_int == 0

        # Update stored CollTiling
        try:
            new_tiling = self._coll_tiling.tiled(
                s.iter, s.loop_mode.unit, hi_int, self._coll_env
            )
        except AssertionError:
            raise
        except Exception as e:
            loop_str = f"for {s.iter} in {s.loop_mode.format_loop_cond(s.lo, s.hi)}"
            raise ValueError(f"{s.srcinfo}: Failed to compile {loop_str}: {e}") from e
        self._coll_tiling = new_tiling

        # We will advise replacing the loop mode with _CodegenPar
        assert s.iter not in self.thread_iters
        self.thread_iters[s.iter] = ThreadIter(
            self._coll_tiling, s.loop_mode.format_loop_cond(lo_int, hi_int)
        )

    def apply_idx(self, node, context_stmt, distributed_coll_units):
        """Consistent distributed memory analysis"""
        assert isinstance(context_stmt, LoopIR.stmt)
        state: DistributedAllocState
        state = self.distributed_alloc_states.get(node.name)
        if state is None:
            return  # Allocated outside, or not numeric

        assert state.optional_native_unit is not None

        fsm = DistributedIdxFsm(
            context_stmt,
            state,
            "cuda_threads",
            self.thread_iters,  # May be modified
            self._coll_env,
            self._coll_tiling,
            distributed_coll_units,
        )
        for i in range(len(node.idx)):
            if fsm.is_done(node):
                break
            fsm.consume_idx(node, self.sym_type(node.name), i)

        # We only got the correct number of threads, not shape/alignment.
        # Check that the leaf tiling has the correct collective unit.
        fsm.check_native_unit(node)

        # We now have the distributed indices in distributed_iters.
        # Store in DistributedAllocState if this is the first use, or check
        # consistency (index equality) with prior uses.
        fsm.check_store_state(node, state)

    def do_call_stmt(self, s: LoopIR.Call):
        # Check collective unit.
        callee = s.f
        instr_info: InstrInfo = callee.instr
        assert isinstance(instr_info, InstrInfo), "Unimplemented: CUDA function calls"
        needed = callee.proc_coll_unit()
        if msg := self._coll_tiling.unit_mismatch(needed, self._coll_env):
            raise TypeError(
                f"{s.srcinfo}: wrong collective unit for {callee.name}(): {msg}, need {needed}"
            )

        # Inspect distributed indices of arguments (safer after above check)
        assert len(callee.args) == len(s.args)
        for decl, e in zip(callee.args, s.args):
            arg_name_str = str(decl.name)
            coll_units = ()
            if e.type.is_tensor_or_window():
                access_info: AccessInfo = instr_info.access_info[arg_name_str]
                coll_units = access_info.distributed_coll_units
            self.do_e(e, coll_units)

        # Inspect trailing barrier expression
        if bar_e := s.trailing_barrier_expr:
            name = bar_e.name
            self.mark_sym_used(name)
            state = self.distributed_alloc_states.get(name)
            barrier_loopir_type = self.sym_type(name)
            assert barrier_loopir_type.is_barrier()
            assert isinstance(state, DistributedAllocState)

            # Inspect intervals (as opposed to points) in BarrierExpr
            coll_units = instr_info.barrier_coll_units
            interval_count = 0
            for coord in bar_e.idx:
                if isinstance(coord, LoopIR.Interval):
                    interval_count += 1
            if interval_count != len(coll_units):
                raise ValueError(
                    f"{s.srcinfo}: {callee.name} #intervals in barrier {bar_e} wrong; "
                    f"have {interval_count}, need {len(coll_units)}"
                )

            # Distributed memory deduction
            fsm = DistributedIdxFsm(
                s,
                state,
                "cuda_threads",
                self.thread_iters,  # May be modified
                self._coll_env,
                self._coll_tiling,
                coll_units,
            )
            # There is no native_unit; we parse all indices as distributed
            assert state.optional_native_unit is None
            for i in range(len(bar_e.idx)):
                fsm.consume_idx(bar_e, barrier_loopir_type, i)

            # We now have the distributed indices in distributed_iters.
            # Store in DistributedAllocState if this is the first use, or check
            # consistency (index equality) with prior uses.
            fsm.check_store_state(s, state)

    def mark_sym_used(self, name: Sym):
        self._syms_needed.add(name)
        warp_name = self._current_warp_name
        if warp_name is None:
            for syms in self.named_warp_used_syms.values():
                syms.add(name)
        else:
            self.named_warp_used_syms[warp_name].add(name)

    def get_barrier_usage(self, name: Sym) -> BarrierUsage:
        return self.ctx.get_barrier_usage(name)


# End class SubtreeScan


# ========================   PHASE 2: subtree rewrite   ========================
# Rewrite the CUDA device function subtree with nodes the outer LoopIR C compiler
# understands. In particular, we lower barriers, and rewrite parallel loops.
#
# The rewrite happens in two sub-phases:
#   A. main lowering replacing spork constructs with basic LoopIR constructs
#        -> most of SubtreeRewrite
#   B. specialize by named warps; we generate one deviceTask per warp name.
#        -> MainLoopRewrite


def wrap_with_context(with_context, body, srcinfo):
    cond = LoopIR.Const(with_context, T.with_context, srcinfo)
    node = LoopIR.If(cond, body, [], srcinfo)
    assert is_if_holding_with(node, LoopIR)
    return node


def wrap_codegen_par(codegen_par, body, srcinfo):
    assert isinstance(codegen_par, _CodegenPar)
    return LoopIR.For(
        Sym("tmp"),
        LoopIR.Const(0, T.int, srcinfo),
        LoopIR.Const(1, T.int, srcinfo),
        body,
        codegen_par,
        srcinfo,
    )


class MainLoopRewrite(LoopIR_Rewrite):
    __slots__ = [
        "named_warp_used_syms",
        "lowered_body",
        "result_stmts",
        "_current_warp_name",
    ]

    named_warp_used_syms: Dict[str, Set[Sym]]
    lowered_body: List[LoopIR.stmt]
    result_stmts: List[LoopIR.stmt]
    _current_warp_name: str

    def __init__(self, scan, device_function_stmt, lowered_body, make_task_context):
        assert is_if_holding_with(device_function_stmt, LoopIR)
        assert isinstance(device_function_stmt.cond.val, CudaDeviceFunction)
        task_loop = device_function_stmt.body[0]

        self.named_warp_used_syms = scan.named_warp_used_syms
        self.lowered_body = lowered_body

        # Manually rewrite the cuda_tasks loops to use seq(...) mode,
        # and rely on LoopIR_Rewrite to filter the per-warp-name task body,
        # which is wrapped with the task_context to put the code into
        # exo_deviceTask{warp_cname}.
        assert scan.task_loop_depth > 0

        def rewrite_task_loop(loop, warp_name, depth_left=scan.task_loop_depth):
            if depth_left == 1:
                # Phase B: filter rewritten CUDA task body down to per-named-warp code
                self._current_warp_name = warp_name
                cname = scan.named_warps[warp_name].cname
                filtered_body = self.map_stmts(lowered_body)
                if filtered_body is None:
                    filtered_body = lowered_body
                body = [
                    wrap_with_context(
                        make_task_context(cname), filtered_body, loop.srcinfo
                    )
                ]
            else:
                body = [rewrite_task_loop(loop.body[0], warp_name, depth_left - 1)]

            if isinstance(loop, LoopIR.For):
                assert isinstance(loop.loop_mode, CudaTasks)
                return loop.update(loop_mode=seq, body=body)
            else:
                assert isinstance(loop, LoopIR.If)
                assert not loop.orelse
                return loop.update(body=body)

        # Assemble body of exo_deviceMainLoop
        #
        # 1. Decide register count [0 if not adjusted]
        # 2. Case by register count
        #      * {setmaxnreg.inc/dec regcount}
        #      * Device loops with matching register count
        nreg_nm = Sym("nreg")
        stmts = []
        srcinfo = task_loop.srcinfo
        i32 = T.i32

        # nreg = 0
        stmts.append(LoopIR.Alloc(nreg_nm, i32, CudaRmem, srcinfo))
        stmts.append(
            LoopIR.Assign(nreg_nm, i32, [], LoopIR.Const(0, i32, srcinfo), srcinfo)
        )

        def wrap_if_nreg(imm, body):
            var = LoopIR.Read(nreg_nm, [], i32, srcinfo)
            const = LoopIR.Const(imm, i32, srcinfo)
            cond = LoopIR.BinOp("==", var, const, T.bool, srcinfo)
            return LoopIR.If(cond, body, [], srcinfo)

        def wrap_if_threadIdx(lo, hi, body):
            loop_mode = _CodegenPar("threadIdx.x", None, (lo, hi))
            return wrap_codegen_par(loop_mode, body, srcinfo)

        named_warp_tuples = sorted(scan.named_warps.items())

        # if (lo <= threadIdx.x && threadIdx.x < hi) {
        #   nreg = ...nonzero;
        # }
        for name, info in named_warp_tuples:
            if not info.setmaxnreg:
                continue
            lo = info.offset * 32
            hi = (info.offset + info.count) * 32
            nreg = info.setmaxnreg

            asn = LoopIR.Assign(
                nreg_nm, i32, [], LoopIR.Const(nreg, i32, srcinfo), srcinfo
            )
            stmts.append(wrap_if_threadIdx(lo, hi, [asn]))

        # if (ntid == ...) {
        #   if (ntid != 0) setmaxnreg.{inc/dec} ntid
        #   for each named warp with that register count...
        #     if (threadIdx.x in range) {
        #        main loop for that warp name
        #     }
        # }
        from .setmaxnreg import unsafe_setmaxnreg

        for (nreg, is_inc) in [(0, False)] + sorted(scan.setmaxnreg_is_inc.items()):
            body = []
            if nreg != 0:
                instr = unsafe_setmaxnreg(
                    imm_reg_count=nreg, is_inc=is_inc
                )._loopir_proc
                body.append(LoopIR.Call(instr, [], None, srcinfo))
            for name, info in named_warp_tuples:
                if nreg != info.setmaxnreg:
                    continue
                lo = info.offset * 32
                hi = (info.offset + info.count) * 32
                body.append(
                    wrap_if_threadIdx(lo, hi, [rewrite_task_loop(task_loop, name)])
                )
            if body:
                stmts.append(wrap_if_nreg(nreg, body))

        stmts.append(LoopIR.Free(nreg_nm, i32, CudaRmem, srcinfo))

        self.result_stmts = stmts

    def map_s(self, s):
        # Phase B: filter rewritten CUDA task body down to per-named-warp code

        if is_if_holding_with(s, LoopIR):
            assert not isinstance(s.cond.val, CudaWarps), "Phase A not done?"

        if isinstance(s, LoopIR.For):
            # Remove branches of code corresponding to different warp name than
            # what is currently being compiled.
            if isinstance(s.loop_mode, _CodegenPar):
                name = s.loop_mode.warp_name_filter
                if name is not None and name != self._current_warp_name:
                    return [LoopIR.Pass(s.srcinfo)]
        elif isinstance(s, (LoopIR.Alloc, LoopIR.Free)):
            # Remove unused variables to shut the CUDA compiler up.
            if s.name not in self.named_warp_used_syms[self._current_warp_name]:
                return [LoopIR.Pass(s.srcinfo)]

        return super().map_s(s)


class SubtreeRewrite(LoopIR_Rewrite):
    __slots__ = [
        "scan",
        "fmt_dict",
        "distributed_alloc_states",
        "thread_iters",
        "cuda_warps_dfs_codegen",
        "cuda_warps_idx",
        "sync_state_builder",
        "live_solitary_barrier_names",
        "live_smem_ends",  # SMEM stack allocator
        "smem_data_usage",  # SMEM stack allocator
        "codegen_smem",  # SMEM stack allocator helper
        "_result",
    ]

    def __init__(self, s, scan: SubtreeScan, ctx: SporkLoweringCtx):
        fmt_dict = scan.fmt_dict
        self.scan = scan
        self.fmt_dict = fmt_dict
        self.distributed_alloc_states = scan.distributed_alloc_states
        self.thread_iters = scan.thread_iters
        self.cuda_warps_dfs_codegen = scan.cuda_warps_dfs_codegen
        self.cuda_warps_idx = 0
        self.sync_state_builder = scan.sync_state_builder
        fmt_dict["SyncState_body"] = scan.sync_state_builder.generate_SyncState_body()

        # Prepare mbarriers in SMEM
        (
            fmt_dict["device_setup_body"],
            mbarrier_smem_bytes,
        ) = scan.sync_state_builder.generate_device_setup()

        # Dict mapping LoweredBarrierType -> Sym
        # only includes live lowered barriers with solitary flag set.
        self.live_solitary_barrier_names = {}

        # Prepare SMEM stack allocator
        # Base of SMEM allocation is reserved for mbarriers
        self.codegen_smem = {}
        self.smem_data_usage = 0
        # self.live_smem_ends = {8 * num_mbarriers}
        self.live_smem_ends = {mbarrier_smem_bytes}
        # HACK: align mbarriers to 128 bytes for now

        # We override the C names of variables that appear in the
        # exo_DeviceArgs or exo_Task structs, or cuda_threads iterators.
        main_loop_force_names = {}
        task_force_names = {}
        for sym in scan.task_iter_syms:
            main_loop_force_names[sym] = "exo_task_" + str(sym)
            task_force_names[sym] = "exo_task." + str(sym)
        for sym in scan.device_args_syms:
            new_name = "exo_deviceArgs." + ctx.sym_c_name(sym)
            main_loop_force_names[sym] = new_name
            task_force_names[sym] = new_name
        for sym, info in self.thread_iters.items():
            task_force_names[sym] = info.cname(sym.name())

        deviceTask_decls = "".join(
            deviceTask_decl_fmt.format(warp_cname=scan.named_warps[nm].cname)
            for nm in sorted(scan.named_warps)
        )

        # ExtWithContext objects for diverting lowered code into
        # exo_deviceTask{warp_cname}().
        format = lambda fmt_string, **extra: fmt_string.format(**fmt_dict, **extra)

        def make_task_context(warp_cname):
            return ExtWithContext(
                format(task_launch_fmt, warp_cname=warp_cname),
                format(device_task_prefix_fmt, warp_cname=warp_cname),
                "}",
                "cuh",
                {},
                task_force_names,
                scan.grid_constant_syms,  # force_const
                scan.scalar_ref_syms,
                {},  # lowered_barriers
            )

        # Phase A: Extract and rewrite the body of the CUDA task (body of
        # inner-most cuda_tasks loop), except for named cuda warps filtering.
        self.cuda_warps_idx = 0
        task_loop = s
        for i in range(scan.task_loop_depth):
            task_loop = task_loop.body[0]
        rewritten_task_body = self.map_stmts(task_loop.body) or task_loop.body
        assert self.cuda_warps_idx == len(self.cuda_warps_dfs_codegen)

        # Phase B, assemble main loops, specialized per warp name.
        main_loop_stmts = MainLoopRewrite(
            scan, s, rewritten_task_body, make_task_context
        ).result_stmts

        # ExoWithContext object for diverting lowered code into
        # exo_deviceMainLoop(), and putting the required strings
        # into the .cu, .cuh, .h files.
        # Only at this point do we know the SMEM usage of the kernel,
        # which we load into fmt_dict at the last moment.
        assert (
            len(self.live_smem_ends) == 1
        ), "SMEM stack allocator should have returned to initial state"
        fmt_dict["smem_bytes"] = self.smem_data_usage
        main_loop_context = ExtWithContext(
            format(cuda_launch_fmt),
            format(device_main_loop_prefix_fmt),
            "}",
            "cuh",
            {
                "h": format(h_snippet_fmt),
                "c": format(c_snippet_fmt),
                "cu": format(cu_snippet_fmt),
                "cuh": format(cuh_snippet_fmt, deviceTask_decls=deviceTask_decls),
            },
            main_loop_force_names,
            scan.grid_constant_syms,  # force_const
            scan.scalar_ref_syms,
            scan.sync_state_builder.lowered,
        )

        # Finally wrap the per-warp-name main loops into exo_deviceMainLoop
        self._result = wrap_with_context(
            main_loop_context, main_loop_stmts, task_loop.srcinfo
        )

    def result(self):
        assert is_if_holding_with(self._result, LoopIR)
        return self._result

    def updated_stmt(self, s):
        if is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, CudaWarps):
                # Replace with CudaWarps block with _CodegenPar "loop"
                # that the scanner has prepared. NB (0, 1) isn't the same
                # as the indices encoded in _CodegenPar.
                loop_mode = self.cuda_warps_dfs_codegen[self.cuda_warps_idx]
                self.cuda_warps_idx += 1
                s = wrap_codegen_par(loop_mode, s.body, s.srcinfo)
            else:
                assert isinstance(ctx, CudaAsync)
        elif isinstance(s, LoopIR.For):
            # Replace CudaThreads loop with _CodegenPar loop that the
            # scanner has prepared.
            if isinstance(s.loop_mode, CudaThreads):
                new_loop_mode = self.thread_iters[s.iter].codegen_par
                s = s.update(loop_mode=new_loop_mode)
            else:
                assert isinstance(s.loop_mode, (Seq, _CodegenPar))

        elif isinstance(s, LoopIR.Alloc):
            if s.type.is_numeric():
                s = self.update_numeric_alloc_free(s)
            elif s.type.is_barrier():
                self.on_barrier_alloc(s)

        elif isinstance(s, LoopIR.Free):
            if s.type.is_numeric():
                s = self.update_numeric_alloc_free(s)
            elif s.type.is_barrier():
                self.on_barrier_free(s)

        elif isinstance(s, idx_s_types):
            # Remove distributed dimensions for tensor indexing expression
            s = self.remove_distributed_idx(s)

        elif isinstance(s, LoopIR.SyncStmt):
            s = self.update_check_sync_stmt(s)

        return s

    def map_s(self, s):
        s_rewrite = self.updated_stmt(s)

        # Use superclass to recurse and rewrite subtree
        # We have to have logic to handle None being used to indicate
        # "no change"; if the superclass makes no changes, we still have
        # to preserve any rewrites of our own.
        if s_rewrite is s or s_rewrite is None:
            out_stmts = super().map_s(s)
        else:
            super_rewritten = super().map_s(s_rewrite)
            if super_rewritten is None:
                out_stmts = [s_rewrite]
            else:
                out_stmts = super_rewritten

        return out_stmts

    def map_e(self, e):
        e_rewrite = None

        # Remove distributed dimensions
        # HACK: for instructions that take windows with distributed dimensions,
        # the resulting program will no longer typecheck, since the
        # dimensionality of the passed window won't match the fnarg anymore!
        if isinstance(e, idx_e_types):
            e_rewrite = self.remove_distributed_idx(e)

        # Use superclass to recurse and rewrite subtree
        # We have to have logic to handle None being used to indicate
        # "no change"; if the superclass makes no changes, we still have
        # to preserve any rewrites of our own.
        if e_rewrite is None:
            return super().map_e(e)
        else:
            super_rewritten = super().map_e(e_rewrite)
            if super_rewritten is None:
                return e_rewrite
            else:
                return super_rewritten

    def remove_distributed_idx(self, node):
        alloc_state = self.distributed_alloc_states.get(node.name)
        if alloc_state is not None:
            assert isinstance(alloc_state, DistributedAllocState)
            n = alloc_state.n_distributed_dims()
            if n > 0:
                old_idx = node.idx
                new_idx = node.idx[n:]
                if isinstance(node, LoopIR.WindowExpr):
                    # Remove the first n coordinates of the idx expression.
                    # If any removed coordinates were intervals, this reduces
                    # the dimensionality of the resulting window type.
                    n_intervals_removed = sum(
                        isinstance(coord, LoopIR.Interval) for coord in old_idx[:n]
                    )
                    old_type = node.type
                    old_src_type = old_type.src_type
                    old_as_tensor = old_type.as_tensor
                    assert (
                        old_type.src_buf == node.name
                    ), "See WindowStmt case for SubtreeScan.apply_s"
                    assert isinstance(old_type, LoopIR.WindowType)
                    new_hi = old_as_tensor.hi[n_intervals_removed:]
                    if not new_hi:
                        # Decayed to scalar
                        return LoopIR.Read(
                            node.name,
                            [coord.pt for coord in new_idx],
                            node.type.basetype(),
                            node.srcinfo,
                        )
                    new_type = old_type.update(
                        src_type=old_src_type.update(hi=old_src_type.hi[n:]),
                        as_tensor=old_as_tensor.update(hi=new_hi),
                        idx=new_idx,
                    )
                    return node.update(idx=new_idx, type=new_type)
                else:
                    assert isinstance(node, (LoopIR.Read, LoopIR.stmt))
                    return node.update(idx=new_idx)
        return None

    def update_numeric_alloc_free(self, s):
        alloc_state = self.distributed_alloc_states[s.name]
        assert isinstance(alloc_state, DistributedAllocState)
        if not alloc_state.first_usage_stmt:
            # Distributed memory analysis isn't run for unused variables...
            warn(
                f"{s.srcinfo}: Unused allocation {s.name} @ {(s.mem or DRAM).name()} "
                f"in CUDA code may not lower correctly"
            )

        # Remove distributed dimensions
        n = alloc_state.n_distributed_dims()
        typ = s.type
        if n > 0:
            if len(typ.hi) == n:
                # All dimensions removed; reduce to scalar
                typ = typ.basetype()
            else:
                assert n < len(typ.hi)
                typ = typ.update(hi=typ.hi[n:])
            s = s.update(type=typ)

        # SMEM offset lowering (crucially after removing distributed dimensions)
        if issubclass(s.mem, CudaBasicSmem):
            if isinstance(s, LoopIR.Alloc):
                # Get SMEM memory config
                inputs = smem_config_inputs(s)
                config = s.mem.smem_config(inputs)
                offset = max(self.live_smem_ends)

                # Allocate at current offset, rounded up for alignment
                alignment = config.alignment
                element_bits = inputs.element_bits()
                assert element_bits % 8 == 0, "TODO sub-byte scalar types"
                if alignment * 8 < element_bits:
                    alignment = element_bits // 8
                assert 0 == (
                    alignment & (alignment - 1)
                ), "SMEM alignment must be power of 2"
                offset = (offset + alignment - 1) & ~(alignment - 1)

                # Stack allocator reserves space for this allocation
                # It's not truly a "stack" allocator because of how LoopIR
                # can free an alloc as soon as it's dead (and so lifetimes
                # maybe are not strictly nested). Hence the max logic, a
                # dead SMEM allocation stays reserved until all
                # higher-on-the-stack SMEMs are also dead.
                smem_bytes = element_bits // 8
                for n in inputs.const_shape:
                    smem_bytes *= n
                smem_end = offset + smem_bytes
                self.smem_data_usage = max(smem_end, self.smem_data_usage)
                assert smem_end not in self.live_smem_ends
                self.live_smem_ends.add(smem_end)

                # Wrap user-specified memory type with SMEM offset,
                # C++ reference type.
                assert isinstance(config.reftype, str)
                mem = CodegenSmem(offset, smem_end, config.reftype, s.mem)
                self.codegen_smem[s.name] = mem  # for rewriting Free, below
            else:
                # Rewrite Free Memory type to match corresponding Alloc
                # and restore stack allocator state
                mem = self.codegen_smem.get(s.name)
                assert mem
                self.live_smem_ends.remove(mem.smem_end())
            s = s.update(mem=mem)
        return s

    def on_barrier_alloc(self, s):
        lowered = self.sync_state_builder.lowered[s.name]
        if lowered.solitary:
            self.check_solitary_barrier(s, lowered)
            self.live_solitary_barrier_names[lowered.type_enum] = s.name

    def on_barrier_free(self, s):
        lowered = self.sync_state_builder.lowered[s.name]
        if lowered.solitary:
            del self.live_solitary_barrier_names[lowered.type_enum]

    def update_check_sync_stmt(
        self,
        s: LoopIR.SyncStmt,
    ):
        lowered = self.sync_state_builder.lowered[s.barriers[0].name]
        if lowered.solitary and not s.sync_type.is_split():
            # Fence must pass solitary barrier check
            self.check_solitary_barrier(s, lowered)
        assert lowered.codegen_sync_stmt is not None
        return s

    def check_solitary_barrier(self, s, lowered):
        sus = self.live_solitary_barrier_names.get(lowered.type_enum)
        if sus is not None:
            raise TypeError(
                f'{s.srcinfo}: Invalid "{s}" of lowered '
                f"barrier type {lowered.type_enum} due to another "
                f'such live barrier "{sus}" in scope'
            )


# End class SubtreeRewrite


def type_const_shape(t: LoopIR.type, usage_str, name, srcinfo: SrcInfo):
    assert isinstance(t, LoopIR.type)
    assert isinstance(srcinfo, SrcInfo)
    shape = t.shape()

    def as_int(c):
        if isinstance(c, LoopIR.Const):
            val = c.val
            if isinstance(val, int):
                return val
        raise TypeError(
            f"{srcinfo}: {usage_str} {name} requires "
            f"constant shape, not {shape}; simplify() if needed"
        )

    return [as_int(c) for c in shape]


def smem_config_inputs(s: LoopIR.Alloc):
    assert isinstance(s, LoopIR.Alloc)
    scalar_info = s.type.basetype().scalar_info()
    const_shape = type_const_shape(s.type, "SMEM allocation", s.name, s.srcinfo)
    return SmemConfigInputs(scalar_info, const_shape, s.srcinfo, s.mem)


def CodegenSmem(byte_offset, byte_end, reftype, wrapped_smem_type):
    """When rewriting the subtree for the CUDA device function,
    wrap all SMEM memory types with this, which includes the
    exact byte [offset,end) for the allocation in the SMEM segment"""

    assert issubclass(wrapped_smem_type, CudaBasicSmem)

    class Impl(wrapped_smem_type):
        @classmethod
        def alloc(cls, new_name, prim_type, shape, srcinfo):
            # We call the wrapped alloc() method to allow the memory class to raise errors.
            wrapped_alloc = wrapped_smem_type.alloc(new_name, prim_type, shape, srcinfo)
            assert wrapped_alloc == ""
            return f"auto& {new_name} = reinterpret_cast<{reftype}>(exo_smem[{byte_offset}]);"

        @classmethod
        def smem_end(cls):
            return byte_end

        @classmethod
        def wrapped_smem_type(cls):
            return wrapped_smem_type

    return Impl


# HACK: avoid showing to users that we added another level of templatization.
CodegenSmem = memwin_template(CodegenSmem, is_smem_wrapper=True)


h_snippet_fmt = """\
struct exo_CudaDeviceArgs{N}_{proc};

#ifdef __CUDACC__
__global__ void exo_deviceFunction{N}_{proc}(__grid_constant__ const struct exo_CudaDeviceArgs{N}_{proc} exo_deviceArgs);
#endif
void exo_cudaLaunch{N}_{proc}(cudaStream_t exo_cudaStream, const struct exo_CudaDeviceArgs{N}_{proc}* exo_deviceArgs);
"""

# Note: the duplication of the device args struct in .c and .cuh is because the
# common .h file may not have the MemWin code needed for the struct to compile.

c_snippet_fmt = """\
// CUDA device function args -- duplicated in .cuh file
struct exo_CudaDeviceArgs{N}_{proc}
{{
{device_args_struct_body}
}};
"""

deviceTask_decl_fmt = """
  static __device__ __forceinline__ void
  exo_deviceTask{warp_cname}(
      char* exo_smem,
      exo_SyncState& exo_syncState,
      const exo_DeviceArgs& exo_deviceArgs,
      exo_Task exo_task,
      exo_ExcutThreadLog exo_excutLog={{}});
"""

launchConfig_clusterDim_snippet = """
  cudaLaunchAttribute exo_clusterDim_attr{};
  exo_clusterDim_attr.id = cudaLaunchAttributeClusterDimension;
  // For some reason setting a cluster size of (1, 1, 1) tanks performance even though it should do nothing!
  static_assert(exo_clusterDim >= 2, "exo codegen should have elided explicit clusterDim = 1");
  exo_clusterDim_attr.val.clusterDim.x = exo_clusterDim;
  exo_clusterDim_attr.val.clusterDim.y = 1;
  exo_clusterDim_attr.val.clusterDim.z = 1;
  exo_launchConfig.attrs = &exo_clusterDim_attr;
  exo_launchConfig.numAttrs = 1;
"""

cuh_snippet_fmt = """\
// CUDA device function args -- duplicated in .c file
struct exo_CudaDeviceArgs{N}_{proc}
{{
{device_args_struct_body}
}};

struct exo_Cuda{N}_{proc}
{{
  using exo_DeviceArgs = exo_CudaDeviceArgs{N}_{proc};

  static constexpr uint32_t exo_blockDim = {blockDim};
  static constexpr uint32_t exo_clusterDim = {clusterDim};

  static constexpr unsigned exo_smemBytes = {smem_bytes};

  struct exo_Task
  {{
{task_struct_body}
  }};

  struct exo_SyncState
  {{
{SyncState_body}
  }};

  static void
  exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_ExcutThreadLog exo_excutLog={{}});

  static __device__ __forceinline__ void
  exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_ExcutThreadLog exo_excutLog={{}});
{deviceTask_decls}}};

inline void
exo_Cuda{N}_{proc}::exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs)
{{
  namespace exo_CudaUtil = exo_CudaUtil_{lib_name};
  cudaFuncSetAttribute(exo_deviceFunction{N}_{proc}, cudaFuncAttributeMaxDynamicSharedMemorySize, exo_smemBytes);
  // TODO how expensive is it to query this every time?
  int exo_cudaDevice;
  cudaGetDevice(&exo_cudaDevice);
  int exo_SMs;
  cudaDeviceGetAttribute(&exo_SMs, cudaDevAttrMultiProcessorCount, exo_cudaDevice);
  const unsigned exo_gridDim = (unsigned(exo_SMs) & ~(exo_clusterDim - 1)) * {blocks_per_sm}u;

  cudaLaunchConfig_t exo_launchConfig = {{}};
  exo_launchConfig.gridDim = dim3(exo_gridDim, 1, 1);
  exo_launchConfig.blockDim = dim3(exo_blockDim, 1, 1);
  exo_launchConfig.dynamicSmemBytes = exo_smemBytes;
  exo_launchConfig.stream = exo_cudaStream;
{launchConfig_clusterDim_snippet}
  cudaLaunchKernelEx(&exo_launchConfig, exo_deviceFunction{N}_{proc}, exo_deviceArgs);

  [[maybe_unused]] static const char* filename = __FILE__;
  exo_excut_flush_device_log(
      exo_cudaStream, exo_gridDim, exo_blockDim,
      exo_CudaUtil::exo_excut_str_id_count, exo_CudaUtil::exo_excut_str_table,
      1, &filename);
}}

__device__ __forceinline__ void
exo_Cuda{N}_{proc}::exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_ExcutThreadLog exo_excutLog)
{{
{device_setup_body}
}}
"""

cu_snippet_fmt = """\
__launch_bounds__({blockDim}, {blocks_per_sm})
__global__ void
exo_deviceFunction{N}_{proc}(__grid_constant__ const struct exo_CudaDeviceArgs{N}_{proc} exo_deviceArgs)
{{
  extern __shared__ char exo_smem[];
  exo_ExcutThreadLog exo_excutLog = exo_excut_begin_thread_log(exo_deviceArgs.exo_excutDeviceLog);
  exo_Cuda{N}_{proc}::exo_deviceSetup(exo_smem, exo_deviceArgs, exo_excutLog);
  exo_Cuda{N}_{proc}::exo_deviceMainLoop(exo_smem, exo_deviceArgs, exo_excutLog);
}}

void
exo_cudaLaunch{N}_{proc}(cudaStream_t exo_cudaStream, const struct exo_CudaDeviceArgs{N}_{proc}* exo_deviceArgs)
{{
  exo_Cuda{N}_{proc}::exo_cudaLaunch(exo_cudaStream, *exo_deviceArgs);
}}
"""

device_main_loop_prefix_fmt = """__device__ __forceinline__ void
exo_Cuda{N}_{proc}::exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_ExcutThreadLog exo_excutLog)
{{
  namespace exo_CudaUtil = exo_CudaUtil_{lib_name};
  exo_SyncState exo_syncState{{}};
  unsigned exo_taskIndex = 0;"""

device_task_prefix_fmt = """__device__ __forceinline__ void
exo_Cuda{N}_{proc}::exo_deviceTask{warp_cname}(
    char* exo_smem,
    exo_SyncState& exo_syncState,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_Task exo_task,
    exo_ExcutThreadLog exo_excutLog)
{{
  namespace exo_CudaUtil = exo_CudaUtil_{lib_name};"""

# We used to pass exo_deviceArgs by value, now we don't due to bad experiences with ABI.
cuda_launch_fmt = """{{
  struct exo_CudaDeviceArgs{N}_{proc} exo_deviceArgs = {{
    {device_args}
  }};
  exo_cudaLaunch{N}_{proc}(exo_cudaStream, &exo_deviceArgs);
}}"""

task_launch_fmt = """if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) {{
    exo_deviceTask{warp_cname}(exo_smem, exo_syncState, exo_deviceArgs,
        (struct exo_Task) {{ {task_args} }},
        exo_excutLog);
}}"""

# Paste this into the C header (.h) if any proc uses cuda.
h_snippet_for_cuda = r"""
#ifndef EXO_CUDA_HEADER_COMMON
#define EXO_CUDA_HEADER_COMMON
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define EXO_CUDA_INLINE __device__ __forceinline__
EXO_CUDA_INLINE unsigned exo_smemU32(const void* smem_ptr)
{
    return (unsigned)__cvta_generic_to_shared(smem_ptr);
}
EXO_CUDA_INLINE unsigned exo_mapa_shared_cluster(unsigned addr_u32, unsigned cta_rank)
{
#if __CUDA_ARCH__ >= 900
    asm("mapa.shared::cluster.u32 %0, %1, %2;": "=r"(addr_u32) : "r"(addr_u32), "r"(cta_rank));
#endif
    return addr_u32;
}
#endif  // __CUDACC__

#ifndef EXO_EXCUT_bENABLE_LOG
#define EXO_EXCUT_bENABLE_LOG 0
#endif

#if EXO_EXCUT_bENABLE_LOG
#include "exo_excut.h"  // Used for exo excut tests (tracing)
#else
// Do-nothing replacements for exo_excut.h
#define exo_excut_log_file_enabled() 0
#define exo_excut_begin_log_action(action_name)
#define exo_excut_log_str_arg(str)
#define exo_excut_log_int_arg(bytes, binary)
#define exo_excut_log_ptr_arg(ptr)
#define exo_excut_end_log_action(device_name, _blockIdx, _threadIdx, file, line)
#define exo_excut_get_device_log()
#define exo_excut_flush_device_log(stream, _gridDim, _blockDim, string_id_count, string_table, file_id_count, file_table)
#define EXO_EXCUT_DEVICE_LOG_MEMBER
#define EXO_EXCUT_STR_ID(c) 0
#ifdef __CUDACC__
struct exo_ExcutThreadLog {
    EXO_CUDA_INLINE void log_action(uint32_t, uint32_t, uint32_t) {}
    EXO_CUDA_INLINE void log_str_id_arg(uint32_t) {}
    EXO_CUDA_INLINE void log_u32_arg(uint32_t) {}
    EXO_CUDA_INLINE void log_u64_arg(uint32_t) {}
    EXO_CUDA_INLINE void log_ptr_arg(const void*) {}
    template <typename T>
    EXO_CUDA_INLINE void log_ptr_data_arg(const T*, uint32_t = 0) {}
};
#define exo_excut_begin_thread_log(log) {}
#endif
#endif // EXO_EXCUT_bENABLE_LOG

#endif // EXO_CUDA_HEADER_COMMON

#ifndef EXO_CUDA_STREAM_GUARD
#define EXO_CUDA_STREAM_GUARD
static const cudaStream_t exo_cudaStream = 0;
#endif"""
