from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable, Dict, Optional, Type, List
from warnings import warn

from ..core.memory import MemGenError, memwin_template, DRAM, BarrierMechanism
from ..core.prelude import Sym, SrcInfo
from ..core.LoopIR import (
    LoopIR,
    T,
    LoopIR_Do,
    LoopIR_Rewrite,
    GetReads,
)

from .distributed_memory import ThreadIter, DistributedIdxFsm, DistributedAllocState
from .timelines import Instr_tl, Sync_tl, cuda_in_order_instr
from . import timelines
from .async_config import CudaDeviceFunction
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
from .coll_analysis import (
    coll_idx_e_types as idx_e_types,
    coll_idx_s_types as idx_s_types,
    wrap_codegen_par,
)
from .cuda_device_setup_builder import (
    CudaDeviceSetupBuilder,
    CudaDeviceSetupInfo,
)
from .cuda_memory import (
    CudaBasicDeviceVisible,
    CudaBasicSmem,
    SmemConfigInputs,
    CudaGridConstant,
    CudaRmem,
    SmemConfig,
)
from .lowered_barrier import LoweredBarrierType, LoweredBarrier
from .cuda_sync_state import SyncStateBuilder
from .cuda_warp_config import WarpLayoutInfo
from .loop_modes import CudaTasks, CudaThreads, Seq, seq, _CodegenPar, cuda_tasks
from .sync_types import SyncType
from .with_cuda_warps import CudaWarps

from ..backend.compiler_fwd import (
    SporkLoweringCtx,
    cuda_tasks_lo_cname,
    cuda_tasks_hi_cname,
    cuda_tasks_num_cname,
)


def loopir_lower_cuda(s, ctx: SporkLoweringCtx):
    """Top level function to call.

    Transforms with-statement node holding CudaDeviceFunction to
    with-statement node holding ExtWithContext, ready for final
    code lowering with the main LoopIR-to-C compiler.
    """

    dim_rewrite = DimensionRewrite(ctx.coll_analysis().distributed_alloc_states)
    if s_rewrite := dim_rewrite.map_s(s):
        assert len(s_rewrite) == 1
        s = s_rewrite[0]

    scan = SubtreeScan(s, ctx)
    # Scanner validates correctness and passes advice from "global analysis"
    # to the subtree rewriter on how to substitute certain stmts/expressions.
    return SubtreeRewrite(s, dim_rewrite, scan, ctx).result()


# =========== PHASE 0: distributed & managed ring buffer rewrites ===========
# Erase distributed dimensions, and rewrite managed ring buffer dimensions (todo)


@dataclass(slots=True)
class ManagedRingBufferEntry:
    ring_depth: int
    syncState_varname: str


class DimensionRewrite(LoopIR_Rewrite):
    distributed_alloc_states: Dict[Sym, DistributedAllocState]
    managed_ring_buffer_entries: Dict[Sym, ManagedRingBufferEntry]
    varname_counter: int
    ring_buffer_consumption_varnames: List[str]

    def __init__(self, distributed_alloc_states):
        assert isinstance(distributed_alloc_states, dict)
        self.distributed_alloc_states = distributed_alloc_states
        self.managed_ring_buffer_entries = {}
        self.varname_counter = 0
        self.ring_buffer_consumption_varnames = []

    def map_s(self, s):
        s_rewrite = super().map_s(s)
        if s_rewrite:
            assert len(s_rewrite) == 1
            s = s_rewrite[0]
        else:
            s_rewrite = None
        if isinstance(s, idx_s_types):
            if tmp := self.rewrite_idx(s):
                s_rewrite = [tmp]
        if isinstance(s, (LoopIR.Alloc, LoopIR.Free)):
            s_rewrite = self.map_alloc_free(s) or [s]
        return s_rewrite

    def map_e(self, e):
        # Remove distributed dimensions
        # HACK: for instructions that take windows with distributed dimensions,
        # the resulting program will no longer typecheck, since the
        # dimensionality of the passed window won't match the fnarg anymore!
        e = super().map_e(e) or e
        e_rewrite = None
        if isinstance(e, LoopIR.BarrierExpr):
            e_rewrite = self.rewrite_idx(e)
        if isinstance(e, idx_e_types):
            e_rewrite = self.rewrite_idx(e)
        return e_rewrite

    def map_alloc_free(self, s):
        s = s.update(type=self.distributed_alloc_states[s.name].get_shard_type())

        if isinstance(s, LoopIR.Alloc):
            entry = None
            ring_depth = s.mem.managed_ring_buffer_depth()
            if ring_depth is not None:
                assert ring_depth > 0
                assert isinstance(ring_depth, int)
                count = self.varname_counter
                varname = f"ring_consumption_{count}_{s.name}"
                self.varname_counter += 1
                entry = ManagedRingBufferEntry(ring_depth, varname)
                self.managed_ring_buffer_entries[s.name] = entry
                self.ring_buffer_consumption_varnames.append(varname)
        else:
            entry = self.managed_ring_buffer_entries.get(s.name)
        if entry:
            ring_depth = entry.ring_depth
            typ = s.type
            if not typ.hi:
                raise ValueError(
                    f"{s.srcinfo}: After removing distributed dimensions, "
                    f"{s} had no dimensions left for ring buffering by {ring_depth}."
                )
            consumption_e = typ.hi[0]
            # Replace 0th extent with constant ring buffer depth,
            # but still increment the ring buffer consumption (after the Free)
            # by the original 0th extent.
            hi = [LoopIR.Const(ring_depth, T.plain_size, s.srcinfo)] + typ.hi[1:]
            s = s.update(type=typ.update(hi=hi))
            if isinstance(s, LoopIR.Free):
                from .codegen_instr import IncrementRingBuffer

                incr = IncrementRingBuffer(syncState_varname=entry.syncState_varname)
                return [s, incr.ProcCallGen_make_call([consumption_e], s.srcinfo)]

        return [s]

    def rewrite_idx(self, node):
        node = self.remove_distributed_idx(node) or node

        if entry := self.managed_ring_buffer_entries.get(node.name):
            # Rewrite the 0th non-distributed index (so [0], as distributed dimensions
            # were removed) to be (idx[0] + consumption) % ring_depth.
            ring_depth = entry.ring_depth
            assert ring_depth > 0
            assert isinstance(ring_depth, int)
            assert len(node.idx) > 0
            ring_idx = node.idx[0]
            ring_idx = LoopIR.ManagedRingBufferIdx(
                ring_idx,
                ring_depth,
                "exo_syncState." + entry.syncState_varname,
                ring_idx.type,
                ring_idx.srcinfo,
            )
            node = node.update(idx=[ring_idx] + node.idx[1:])

        return node

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
                    # fmt: off
                    assert isinstance(node, (LoopIR.Read, LoopIR.stmt, LoopIR.BarrierExpr)), node
                    return node.update(idx=new_idx)
        return None


# ========================   PHASE 1: subtree scan   ========================
# Just collect information about the subtree corresponding to the
# CUDA device function. We already have the CollTiling from CollAnalysis
# and BarrierUsage from BarrierUsageAnalysis; build on that.


class SubtreeScan(LoopIR_Do):
    __slots__ = [
        "ctx",
        "cuda_device_function",
        "sync_state_builder",
        "device_setup_builder",
        "distributed_alloc_states",
        "codegen_smem",
        "thread_iters",
        "fmt_dict",
        "named_warp_used_syms",
        "task_loop_bounds",
        "task_iter_syms",
        "device_args_syms",
        "grid_constant_syms",
        "scalar_ref_syms",
        #
        "_local_envtyp",
        "_syms_needed",
        "_coll_tiling",
        "_current_warp_name",
        "named_warps",
        "setmaxnreg_is_inc",
    ]

    ctx: SporkLoweringCtx

    cuda_device_function: CudaDeviceFunction
    sync_state_builder: SyncStateBuilder
    device_setup_builder: CudaDeviceSetupBuilder
    distributed_alloc_states: Dict[Sym, DistributedAllocState]
    codegen_smem: Dict[Sym, Type[CudaBasicSmem]]
    thread_iters: Dict[Sym, ThreadIter]  # Info on iterators of cuda_threads loops

    fmt_dict: Dict

    # For each warp name, record the set of Syms used when executing the code
    # path for that warp. Needed to remove unused variables.
    named_warp_used_syms: Dict[str, Set[Sym]]

    task_loop_bounds: List[Tuple[LoopIR.expr, LoopIR.expr]]  # (lo, hi)
    task_iter_syms: List[Sym]
    device_args_syms: List[Sym]
    grid_constant_syms: Set[Sym]
    scalar_ref_syms: Set[Sym]

    _local_envtyp: Dict[Sym, LoopIR.type]
    _syms_needed: Set[Sym]
    _coll_tiling: CollTiling
    _current_warp_name: Optional[str]
    named_warps: Dict[str, WarpLayoutInfo]
    setmaxnreg_is_inc: Optional[Dict[int, bool]]

    # Edit __slots__ if you add more attributes

    def __init__(self, s, ctx: SporkLoweringCtx):
        assert is_if_holding_with(s, LoopIR)
        cuda_device_function: CudaDeviceFunction = s.cond.val
        assert isinstance(cuda_device_function, CudaDeviceFunction)

        blockDim = cuda_device_function.blockDim
        clusterDim = cuda_device_function.clusterDim

        self.ctx = ctx
        self.cuda_device_function = cuda_device_function
        self.sync_state_builder = SyncStateBuilder(cuda_device_function.coll_env())
        self.device_setup_builder = CudaDeviceSetupBuilder()
        self.distributed_alloc_states = ctx.coll_analysis().distributed_alloc_states
        self.codegen_smem = {}
        self.thread_iters = ctx.coll_analysis().thread_iters
        self.fmt_dict = {
            "proc": ctx.proc_name(),
            "lib_name": ctx.lib_name(),
            "N": ctx.kernel_index(),
            "blockDim": blockDim,
            "clusterDim": clusterDim,
            "launchConfig_clusterDim_snippet": "",
            "blocks_per_sm": cuda_device_function.blocks_per_sm,
            "exo_smem_align": SmemConfig.opportunistic_alignment,
        }
        self.named_warps = cuda_device_function.named_warps
        self.setmaxnreg_is_inc = cuda_device_function.setmaxnreg_is_inc
        self._current_warp_name = None
        self.named_warp_used_syms = {nm: set() for nm in self.named_warps}

        # Only set clusterDim if not 1, not only for pre-H100 compatibility,
        # but also this avoids mysterious performance loss.
        if clusterDim != 1:
            self.fmt_dict["launchConfig_clusterDim_snippet"] = (
                launchConfig_clusterDim_snippet
            )

        # Validate top-level form of cuda kernel
        # Must be nest of 1+ cuda_tasks loops.
        # Record task_loop_bounds and task_iter_syms
        self.task_iter_syms = []
        task_iter_strs = set()
        valid_sync = False

        if len(s.body) != 1:
            # Usually we rely on cuda_tasks.validate_loop but it has this blind spot.
            assert s.body, "Unexpected empty CudaDeviceFunction"
            raise ValueError(
                f"{s.srcinfo}: Invalid cuda_tasks loop, expected cuda_tasks loop alone in CudaDeviceFunction"
            )

        self.task_loop_bounds = []
        task_loop_body = s.body
        found_device_task = False

        while not found_device_task:
            first_stmt = task_loop_body[0]
            found_device_task = cuda_tasks.validate_loop(first_stmt)

            # Record cuda_tasks iteration variable
            if str(first_stmt.iter) in task_iter_strs:
                raise ValueError(
                    f"{s.srcinfo}: Invalid cuda_tasks loop, duplicate cuda_tasks iter variable name {first_stmt.iter}"
                )
            task_iter_strs.add(str(first_stmt.iter))
            self.task_iter_syms.append(first_stmt.iter)
            # Record cuda_tasks loop bounds
            bounds = (first_stmt.lo, first_stmt.hi)
            self.task_loop_bounds.append(bounds)
            # The CudaTasks loop nest must be a cuboid (all we support for now)
            for bdd in bounds:
                getter = GetReads()
                getter.do_e(bdd)
                for nm, _ in getter.reads:
                    if nm in self.task_iter_syms:
                        txt = f"for {first_stmt.iter} in {first_stmt.loop_mode.format_loop_cond(*bounds)}"
                        raise ValueError(
                            f"{first_stmt.srcinfo}: Invalid cuda_tasks loop,"
                            f"non-cuboid cuda_tasks loop nest unimplemented; "
                            f"{txt} has dependence on {nm} iterator of previous cuda_tasks loop."
                        )

            # Recurse into cuda_tasks loop.
            # If task_loop_body is not itself a cuda_tasks loop,
            # then found_device_task=True and the loop will terminate.
            task_loop_body = first_stmt.body

        # Prepare exo_Task struct (struct of task loop iteration variables)
        # They will be named exo_task.* in deviceTask.
        # In exo_deviceMainLoop, these are represented by lo/hi variable pairs.
        assert len(self.task_iter_syms) == len(self.task_loop_bounds)
        self.fmt_dict["task_cuboid_args"] = "\n".join(
            f"    {cuda_tasks_lo_cname(str(sym))}, {cuda_tasks_hi_cname(str(sym))},"
            for sym in self.task_iter_syms
        )
        self.fmt_dict["task_struct_body"] = "\n".join(
            f"    {T.index.ctype()} {str(sym)};" for sym in self.task_iter_syms
        )

        # Prepare exo_TaskGenerator struct.
        # TODO better algorithms than lexicographical.
        # fmt: off
        idx_t = T.index.ctype()
        task_generator_lines = []
        # Member variables: task index, count, lo/hi for each cuda_tasks iterator.
        task_generator_lines.append("uint32_t exo_taskIndex;")
        task_generator_lines.append("uint32_t exo_numClusters;")
        task_generator_lines.append("uint32_t exo_taskCount;")
        for sym in self.task_iter_syms:
            c_lo = cuda_tasks_lo_cname(str(sym))
            c_num = cuda_tasks_num_cname(str(sym))
            task_generator_lines.append(f"{idx_t} {c_lo};")
            task_generator_lines.append(f"uint32_t {c_num};")
        # Constructor: initialize variables and compute task count.
        task_generator_lines.append("EXO_CUDA_INLINE exo_TaskGenerator(")
        task_generator_lines.append("    uint32_t cluster_index, uint32_t num_clusters,")
        for sym in self.task_iter_syms:
            c_lo = cuda_tasks_lo_cname(str(sym))
            c_hi = cuda_tasks_hi_cname(str(sym))
            task_generator_lines.append(f"    {idx_t} _{c_lo}, {idx_t} _{c_hi},")
        task_generator_lines.append("    const exo_DeviceArgs&)")
        task_generator_lines.append("{")
        task_generator_lines.append("  exo_taskIndex = cluster_index;")
        task_generator_lines.append("  exo_numClusters = num_clusters;")
        task_generator_lines.append("  exo_taskCount = 1;")
        for sym in self.task_iter_syms:
            c_lo = cuda_tasks_lo_cname(str(sym))
            c_hi = cuda_tasks_hi_cname(str(sym))
            c_num = cuda_tasks_num_cname(str(sym))
            task_generator_lines.append(f"  {c_lo} = _{c_lo};")
            task_generator_lines.append(f"  {c_num} = static_cast<uint32_t>(_{c_hi} - _{c_lo});")
            task_generator_lines.append(f"  exo_taskCount *= {c_num};")
        task_generator_lines.append("}")
        # prepare_next_task
        task_generator_lines.append("[[nodiscard]] EXO_CUDA_INLINE bool prepare_next_task()")
        task_generator_lines.append("{")
        task_generator_lines.append("  return exo_taskIndex < exo_taskCount;")
        task_generator_lines.append("}")
        # get_next_task
        task_generator_lines.append("EXO_CUDA_INLINE exo_Task get_next_task()")
        task_generator_lines.append("{")
        task_generator_lines.append("  exo_Task exo_task;")
        task_generator_lines.append("  uint32_t exo_tmp = exo_taskIndex;")
        task_generator_lines.append("  exo_taskIndex += exo_numClusters;")
        for sym in reversed(self.task_iter_syms):
            c_lo = cuda_tasks_lo_cname(str(sym))
            c_num = cuda_tasks_num_cname(str(sym))
            task_generator_lines.append(f"  exo_task.{sym} = {c_lo} + static_cast<{idx_t}>(exo_tmp % {c_num});")
            task_generator_lines.append(f"  exo_tmp /= {c_num};")
        task_generator_lines.append("  return exo_task;")
        task_generator_lines.append("}")
        self.fmt_dict["task_generator_body"] = "\n".join("    " + line for line in task_generator_lines)
        # fmt: on

        # Scan the subtree
        # We seed the analysis of the collective units with the tiling
        # for the top-level collective (clusterDim x blockDim,
        # with redundant clusterDim removed if clusterDim = 1).
        self._local_envtyp = {}
        self._syms_needed = set()
        self._coll_tiling = cuda_device_function.top_level_coll_tiling()
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

        # Modify state, then recurse with super()
        # (order is important so recursion sees updated state!)
        self.apply_s(s)
        super().do_s(s)

        # Restore state
        self._coll_tiling = old_coll_tiling
        self._current_warp_name = old_warp_name

    def do_e(self, e, distributed_coll_units=()):
        self.apply_e(e, distributed_coll_units)
        super().do_e(e)

    def apply_e(self, e, distributed_coll_units):
        if isinstance(e, idx_e_types):
            self.mark_sym_used(e.name)
        if isinstance(e, LoopIR.BarrierExpr):
            self.mark_sym_used(e.name)

    def apply_s(self, s):
        if isinstance(s, idx_s_types):
            self.mark_sym_used(s.name)
        elif not isinstance(s, (LoopIR.WindowStmt, LoopIR.Alloc, LoopIR.Free)):
            assert not hasattr(s, "name"), "Add handling for array indexing"

        if isinstance(s, LoopIR.For):
            loop_mode = s.loop_mode
            if isinstance(loop_mode, Seq):
                pass
            elif isinstance(loop_mode, CudaTasks):
                if s.iter not in self.task_iter_syms:
                    raise ValueError(
                        f"{s.srcinfo}: Invalid cuda_tasks loop, must appear only in top level nest of CudaDeviceFunction"
                    )
            elif isinstance(loop_mode, _CodegenPar):
                self._coll_tiling = self.thread_iters[s.iter].coll_tiling
                if (warp_name := loop_mode.warp_name_filter) is not None:
                    self._current_warp_name = warp_name
            else:
                # CollAnalysis should have rewritten cuda_threads loops.
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
            if issubclass(s.mem, CudaBasicSmem):
                self.device_setup_builder.begin_smem_alloc(s.name)
        elif isinstance(s, LoopIR.Free):
            if s.type.is_barrier():
                self.sync_state_builder.add_barrier(
                    s.name,
                    self.get_barrier_usage,  # Callable[[Sym], BarrierUsage]
                    self.distributed_alloc_states[s.name],
                    self.thread_iters,
                    self.device_setup_builder,
                )
            elif issubclass(s.mem, CudaBasicSmem):
                # End SMEM lifetime.
                offset_name = self.device_setup_builder.end_smem_alloc(s.name)
                if s.mem.managed_ring_buffer_depth() is not None:
                    # Managed ring buffer allocations are persistent.
                    self.device_setup_builder.make_persistent(s.name)

                # Record required alloc size
                inputs: SmemConfigInputs = smem_config_inputs(s)
                config: SmemConfig = s.mem.smem_config(inputs)
                smem_bits = inputs.element_bits() * prod(inputs.const_shape)
                assert smem_bits % 8 == 0, "TODO: error message for this"
                smem_bytes = smem_bits // 8
                self.device_setup_builder.set_smem_alloc_size(
                    s.name, smem_bytes, config.alignment
                )

                # Store wrapped CodegenSmem type for later rewrite.
                mem = CodegenSmem(offset_name, config.reftype, s.mem)
                self.codegen_smem[s.name] = mem

        elif isinstance(s, LoopIR.SyncStmt):
            # Distributed memory analysis and CollTiling for Fence/Arrive/Await
            if s.sync_type.is_split():
                # Arrive/Await
                assert len(s.barriers) >= 1
                name = s.barriers[0].name
                self.mark_sym_used(name)
            else:
                # Fence
                assert len(s.barriers) == 1
                e = s.barriers[0]
                assert isinstance(e, LoopIR.BarrierExpr)
                state = DistributedAllocState.from_fence(s, self._coll_tiling)
                self.sync_state_builder.add_barrier(
                    e.name,
                    self.get_barrier_usage,  # Callable[[Sym], BarrierUsage]
                    state,
                    self.thread_iters,
                    self.device_setup_builder,
                )

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

    # Edit __slots__ if you add more attributes

    def __init__(self, scan, device_function_stmt, lowered_body, make_task_context):
        assert is_if_holding_with(device_function_stmt, LoopIR)
        assert isinstance(device_function_stmt.cond.val, CudaDeviceFunction)
        task_loop = device_function_stmt.body[0]

        self.named_warp_used_syms = scan.named_warp_used_syms
        self.lowered_body = lowered_body

        # Rewrite the body of the inner-most cuda_tasks loop.
        # Rely on LoopIR_Rewrite to filter the per-warp-name task body,
        # which is wrapped with the task_context to put the code into
        # exo_deviceTask{warp_cname}.
        assert scan.task_loop_bounds

        def rewrite_task_loop(loop, warp_name, depth_left=len(scan.task_loop_bounds)):
            assert isinstance(loop, LoopIR.For)
            assert isinstance(loop.loop_mode, CudaTasks)
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
            loop_mode = _CodegenPar("threadIdx.x", None, (lo, hi), None, (), -1, -1, -1)
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
        from .codegen_instr import unsafe_setmaxnreg

        for nreg, is_inc in [(0, False)] + sorted(scan.setmaxnreg_is_inc.items()):
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
        "device_setup_info",
        "fmt_dict",
        "distributed_alloc_states",
        "thread_iters",
        "sync_state_builder",
        "codegen_smem",
        "live_solitary_barrier_names",
        "_result",
    ]

    # Edit __slots__ if you add more attributes

    def __init__(
        self, s, dim_rewrite: DimensionRewrite, scan: SubtreeScan, ctx: SporkLoweringCtx
    ):
        fmt_dict = scan.fmt_dict
        self.scan = scan
        self.device_setup_info = scan.device_setup_builder.make_info(
            scan.cuda_device_function.clusterDim
        )
        self.fmt_dict = fmt_dict
        self.distributed_alloc_states = scan.distributed_alloc_states
        self.thread_iters = scan.thread_iters
        self.sync_state_builder = scan.sync_state_builder
        self.codegen_smem = scan.codegen_smem

        fmt_dict["SyncState_body"] = scan.sync_state_builder.generate_SyncState_body(
            dim_rewrite.ring_buffer_consumption_varnames
        )

        setup = self.device_setup_info
        fmt_dict["smem_bytes"] = setup.smem_bytes
        fmt_dict["device_setup_body"] = "\n".join("  " + ln for ln in setup.setup_lines)
        fmt_dict["device_setup_decls"] = "\n".join(
            "  " + ln for ln in setup.static_decls
        )

        # Dict mapping LoweredBarrierType -> Sym
        # only includes live lowered barriers with solitary flag set.
        self.live_solitary_barrier_names = {}

        # We override the C names of variables that appear in the
        # exo_DeviceArgs or exo_Task structs, or cuda_threads iterators.
        main_loop_force_names = {}
        task_force_names = {}
        for sym in scan.task_iter_syms:
            # Never mangle in main loop
            # so that exo_cudaTasksLo_{nm} and exo_cudaTasksHi_{nm} works.
            main_loop_force_names[sym] = str(sym)
            task_force_names[sym] = "exo_task." + str(sym)
        for sym in scan.device_args_syms:
            new_name = "exo_deviceArgs." + ctx.sym_c_name(sym)
            main_loop_force_names[sym] = new_name
            task_force_names[sym] = new_name
        for sym, info in self.thread_iters.items():
            if info.mangle:
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
        task_loop = s
        for bounds in scan.task_loop_bounds:
            task_loop = task_loop.body[0]
        rewritten_task_body = self.map_stmts(task_loop.body) or task_loop.body

        # Phase B, assemble main loops, specialized per warp name.
        main_loop_stmts = MainLoopRewrite(
            scan, s, rewritten_task_body, make_task_context
        ).result_stmts

        # ExoWithContext object for diverting lowered code into
        # exo_deviceMainLoop(), and putting the required strings
        # into the .cu, .cuh, .h files.
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
            assert isinstance(s.cond.val, CudaAsync)

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

    def update_numeric_alloc_free(self, s):
        # SMEM offset lowering
        if issubclass(s.mem, CudaBasicSmem):
            mem = self.codegen_smem[s.name]
            s = s.update(mem=mem)

        return s

    def on_barrier_alloc(self, s):
        lowered = self.sync_state_builder.lowered[s.name]
        if lowered.solitary:
            alloc_state = self.distributed_alloc_states[s.name]
            shard_type = s.type
            assert isinstance(shard_type, LoopIR.Barrier)
            # TODO test this
            for extent in shard_type.hi:
                if not (isinstance(extent, LoopIR.Const) and extent.val == 1):
                    raise ValueError(
                        f"{s.srcinfo}: {s}, expected all dimensions to be distributed.\n"
                        f"Have shard type {shard_type}, deduced from {alloc_state.native_unit}"
                    )
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
        shape_str = "[" + ", ".join(str(c) for c in shape) + "]"
        raise TypeError(
            f"{srcinfo}: {usage_str} {name} requires "
            f"constant shape, not {shape_str}; simplify() if needed"
        )

    return [as_int(c) for c in shape]


def smem_config_inputs(s: LoopIR.Alloc | LoopIR.Free):
    scalar_info = s.type.basetype().scalar_info()
    const_shape = type_const_shape(s.type, "SMEM allocation", s.name, s.srcinfo)
    return SmemConfigInputs(scalar_info, const_shape, s.srcinfo, s.mem)


def CodegenSmem(offset_name, reftype, wrapped_smem_type):
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
            return f"auto& {new_name} = reinterpret_cast<{reftype}>(exo_smem[{offset_name}]);"

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

# Note about ODR isuse and inline namespace:
#
# I've been having issues with test cases failing when running the
# full test suite but not when run in isolation. The test cases work
# by compiling a shared library, loading it, and executing the
# compiled test proc from the .so. For my CUDA stuff, I use inline
# heavily. It turns out that on Linux, if you have a name collision
# between inline symbols exported by two shared libraries, the second
# shared library transparently replaces its inline objects with
# definitions loaded from the first shared library. This caused my
# tests to fail because the executed C++ library was a chimera of code
# from the current test and spliced-in code from an unrelated earlier
# test case (this took me HOURS to figure out).

cuh_snippet_fmt = """\
// CUDA device function args -- duplicated in .c file
struct exo_CudaDeviceArgs{N}_{proc}
{{
{device_args_struct_body}
}};

// We need this inline namespace to avoid ODR problems in pytest.
inline namespace exo_CudaInline_{lib_name} {{
struct exo_Cuda{N}_{proc}
{{
  using exo_DeviceArgs = exo_CudaDeviceArgs{N}_{proc};

  static constexpr uint32_t exo_blockDim = {blockDim};
  static constexpr uint32_t exo_clusterDim = {clusterDim};

  static constexpr unsigned exo_smemBytes = {smem_bytes};
{device_setup_decls}

  struct exo_Task
  {{
{task_struct_body}
  }};

  struct exo_TaskGenerator
  {{
{task_generator_body}
  }};

  struct exo_SyncState
  {{
{SyncState_body}
  }};

  static inline const char*& exo_FILE()
  {{
    static const char* name = __FILE__;
    return name;
  }}

  static void
  exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_ExcutThreadLog exo_excutLog={{}});

  static __device__ __forceinline__ void
  exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_ExcutThreadLog exo_excutLog={{}});
{deviceTask_decls}}};
}}  // end inline namespace

inline void
exo_CudaInline_{lib_name}::exo_Cuda{N}_{proc}::exo_cudaLaunch(
    cudaStream_t exo_cudaStream,
    const exo_DeviceArgs& exo_deviceArgs)
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

  exo_excut_flush_device_log(
      exo_cudaStream, exo_gridDim, exo_blockDim,
      exo_CudaUtil::exo_excut_str_id_count, exo_CudaUtil::exo_excut_str_table,
      1, &exo_FILE());
}}

__device__ __forceinline__ void
exo_CudaInline_{lib_name}::exo_Cuda{N}_{proc}::exo_deviceSetup(
    char* exo_smem,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_ExcutThreadLog exo_excutLog)
{{
{device_setup_body}
}}
"""

cu_snippet_fmt = """\
__launch_bounds__({blockDim}, {blocks_per_sm})
__global__ void
exo_deviceFunction{N}_{proc}(__grid_constant__ const struct exo_CudaDeviceArgs{N}_{proc} exo_deviceArgs)
{{
  extern __shared__ __align__({exo_smem_align}) char exo_smem[];
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
exo_CudaInline_{lib_name}::exo_Cuda{N}_{proc}::exo_deviceMainLoop(
    char* exo_smem,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_ExcutThreadLog exo_excutLog)
{{
  namespace exo_CudaUtil = exo_CudaUtil_{lib_name};
  exo_SyncState exo_syncState{{}};"""

device_task_prefix_fmt = """__device__ __forceinline__ void
exo_CudaInline_{lib_name}::exo_Cuda{N}_{proc}::exo_deviceTask{warp_cname}(
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

task_launch_fmt = """exo_TaskGenerator exo_taskGenerator(
    blockIdx.x / exo_clusterDim,
    gridDim.x / exo_clusterDim,
{task_cuboid_args}
    exo_deviceArgs);
while (exo_taskGenerator.prepare_next_task()) {{
  exo_deviceTask{warp_cname}(exo_smem, exo_syncState, exo_deviceArgs, exo_taskGenerator.get_next_task(), exo_excutLog);
}}"""

# Paste this into the C header (.h) if any proc uses cuda.
# TODO cuda_fp16.h and cuda_bf16.h always included even if f16/bf16 isn't used
# but this is harder to fix than it may appear at first.
h_snippet_for_cuda = r"""
#ifndef EXO_CUDA_HEADER_COMMON
#define EXO_CUDA_HEADER_COMMON
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
