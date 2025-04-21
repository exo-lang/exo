from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import prod
from typing import Callable, Dict, Optional, Type
from warnings import warn

from ..core.memory import MemGenError, memwin_template
from ..core.prelude import Sym, SrcInfo
from ..core.LoopIR import (
    LoopIR,
    T,
    LoopIR_Do,
    LoopIR_Rewrite,
)

from . import actor_kinds
from .actor_kinds import ActorKind
from .async_config import CudaDeviceFunction, CudaAsync
from .base_with_context import is_if_holding_with, ExtWithContext
from .coll_algebra import (
    CollParam,
    CollUnit,
    clusterDim_param,
    blockDim_param,
    CollIndexExpr,
    CollTiling,
    CollLoweringAdvice,
    cuda_thread,
    cuda_warp,
    cuda_warpgroup,
    cuda_cta_in_cluster,
)
from .cuda_memory import (
    CudaBasicDeviceVisible,
    CudaBasicSmem,
    SmemConfigInputs,
    CudaGridConstant,
)
from .loop_modes import CudaTasks, CudaThreads, Seq, seq, _CodegenPar
from .sync_types import SyncType
from .with_cuda_warps import CudaWarps


# We use the reserved exo_ prefix everywhere, but we still have to reserve
# CUDA builtins we have no control over.
reserved_names = {"gridDim", "blockDim", "blockIdx", "threadIdx"}

idx_e_types = (LoopIR.Read, LoopIR.WindowExpr, LoopIR.StrideExpr)
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


class SubtreeScan(LoopIR_Do):
    __slots__ = [
        "alloc_states",
        "thread_iters",
        "barrier_scans",
        "cuda_warps_dfs_codegen",
        "uses_async_proxy",
        "blockDim",
        "clusterDim",
        "fmt_dict",
        "task_loop_depth",
        "task_iter_syms",
        "device_args_syms",
        "grid_constant_syms",
        "scalar_ref_syms",
        #
        "_syms_needed",
        "_stmt_stack",  # XXX delete me if useless
        "_coll_env",
        "_coll_tiling",
        "_iter_coll_tiling",
        "_body_containing_s",
    ]

    # We will have to substitute some LoopIR nodes in the SubtreeRewrite phase.
    # During the scan, for a node that needs to be rewritten, we will stash
    # needed info for the rewrites here.
    alloc_states: Dict[Sym, AllocState]  # name of non-barrier -> AllocState
    thread_iters: Dict[Sym, ThreadIter]  # Info on iterators of cuda_threads loops
    barrier_scans: Dict[Sym, BarrierScan]  # name of barrier -> BarrierScan

    # _CodegenPar needed for rewrites of CudaWarps blocks
    # as encountered in DFS order (typical traversal order).
    # This is a bit fragile, but CudaWarps blocks have no
    # unique information (id(...) is not unique after scheduling).
    cuda_warps_dfs_codegen: List[_CodegenPar]

    uses_async_proxy: bool

    blockDim: int
    clusterDim: int
    fmt_dict: Dict
    task_loop_depth: int
    task_iter_syms: List[Sym]
    _syms_needed: Set[Sym]
    _stmt_stack: List[LoopIR.stmt]  # XXX delete me if useless
    _coll_env: Dict[CollParam, int]
    _coll_tiling: CollTiling
    # CollTiling created by the cuda_threads loop with the given iter
    _iter_coll_tiling: Dict[Sym, CollTiling]
    # In do_s(s), we will always have s in _body_containing_s
    _body_containing_s: List[LoopIR.stmt]

    def __init__(self, s, ctx: SporkLoweringCtx):
        assert is_if_holding_with(s, LoopIR)
        cuda_device_function = s.cond.val
        assert isinstance(cuda_device_function, CudaDeviceFunction)

        self.alloc_states = {}
        self.thread_iters = {}
        self.barrier_scans = {}
        self.cuda_warps_dfs_codegen = []
        self.uses_async_proxy = False
        self.blockDim = cuda_device_function.blockDim
        self.clusterDim = cuda_device_function.clusterDim
        self.fmt_dict = {
            "proc": ctx.proc_name(),
            "lib_name": ctx.lib_name(),
            "N": ctx.kernel_index(),
            "gridDim": 132 * cuda_device_function.blocks_per_sm,  # TODO
            "blockDim": self.blockDim,
            "clusterDim": self.clusterDim,
            "blocks_per_sm": cuda_device_function.blocks_per_sm,
        }

        # Validate top-level form of cuda kernel
        # Must be nest of 1+ cuda_tasks loops
        self.task_iter_syms = []
        task_iter_strs = set()
        valid_sync = False

        if len(s.body) != 1:
            raise ValueError(f"{s.srcinfo}: expected cuda_tasks loop alone")

        self.task_loop_depth = 0
        task_loop_body = s.body
        while True:
            if len(task_loop_body) == 0:
                break
            # TODO also support if statements, to allow imperfect loop divide!
            first_stmt = task_loop_body[0]
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
                        self.task_loop_depth += 1
                        task_loop_body = task_loop_body[0].body
                        continue
            # End when encountering first non-cuda_tasks stmt.
            break

        if self.task_loop_depth == 0:
            raise ValueError(f"{s.srcinfo}: missing cuda_tasks loop")

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
        self._syms_needed = set()
        self._stmt_stack = []
        self._coll_env = {
            clusterDim_param: self.clusterDim,
            blockDim_param: self.blockDim,
        }
        assert self.clusterDim > 0 and isinstance(self.clusterDim, int)
        if self.clusterDim == 1:
            tlc_offset = (0,)
            tlc_box = (self.blockDim,)
            intra_box_exprs = (CollIndexExpr("threadIdx.x"),)
        else:
            tlc_offset = (0, 0)
            tlc_box = (self.clusterDim, self.blockDim)
            cta_expr = CollIndexExpr("blockIdx.x") % self.clusterDim
            intra_box_exprs = (cta_expr, CollIndexExpr("threadIdx.x"))
        self._coll_tiling = CollTiling(
            None,
            tlc_box,
            tlc_box,
            tlc_offset,
            tlc_box,
            intra_box_exprs,
            1,
            CollIndexExpr(0),
        )
        self._iter_coll_tiling = {}
        self._body_containing_s = s.body
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
        for sym in self._syms_needed:
            try:
                cpu_nm = ctx.sym_c_name(sym)
            except KeyError:
                continue
            self.device_args_syms.append(sym)
            if issubclass(ctx.sym_mem(sym), CudaGridConstant):
                self.grid_constant_syms.add(sym)
            elif ctx.sym_type(sym).is_real_scalar():
                # elif ensures not added if grid constant
                self.scalar_ref_syms.add(sym)

        # The device args struct will be sorted in the order the variables were
        # created in Python code
        self.device_args_syms.sort(key=lambda s: s.id_number())

        device_args_decls = []
        device_args_comments = []
        device_args_values = []

        for sym in self.device_args_syms:
            if sym not in self.grid_constant_syms:
                # Non-grid-constant, passed as in Exo C code.
                # We don't mangle syms in the device args struct.
                # They will appear as exo_deviceArgs.{str(sym)} in CUDA code.
                mem = ctx.sym_mem(sym)
                fnarg = LoopIR.fnarg(sym, ctx.sym_type(sym), mem, s.srcinfo)
                ctx.append_fnarg_decl(
                    fnarg, str(sym), device_args_decls, device_args_comments
                )
                e = LoopIR.Read(sym, [], ctx.sym_type(sym), s.srcinfo)
                device_args_values.extend(ctx.fnarg_values(e, ctx.is_const(sym), False))
            else:
                # Grid constants are passed as array or scalar by-value
                c_arg = ctx.sym_c_name(sym)
                typ = ctx.sym_type(sym)
                if typ.is_win():
                    raise TypeError(
                        f"{s.srcinfo}: grid constant parameter {sym} "
                        f"cannot be a window"
                    )
                elif typ.is_dense_tensor():
                    n = prod(type_const_shape(typ, "grid constant", sym, s.srcinfo))
                    # See "we don't mangle syms" for str(sym) vs c_args
                    device_args_decls.append(
                        f"{typ.basetype().ctype()} {str(sym)}[{n}]"
                    )
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
                    # See "we don't mangle syms" for str(sym) vs c_args
                    device_args_decls.append(f"{typ.ctype()} {str(sym)}")
                    if ctx.sym_is_scalar_ref(sym):
                        c_arg = f"*{c_arg}"
                    device_args_values.append(c_arg)
                device_args_comments.append(f"{sym}: {typ} -- grid constant")

        device_args_struct_lines = []
        assert len(device_args_decls) == len(device_args_comments)
        for i in range(len(device_args_decls)):
            device_args_struct_lines.append(
                f"    {device_args_decls[i]};  // {device_args_comments[i]}"
            )
        self.fmt_dict["device_args"] = ", ".join(device_args_values)
        self.fmt_dict["device_args_struct_body"] = "\n".join(device_args_struct_lines)

    def do_stmts(self, stmts):
        saved = self._body_containing_s
        self._body_containing_s = stmts
        super().do_stmts(stmts)
        self._body_containing_s = saved

    def do_s(self, s):
        # Save state
        old_coll_tiling = self._coll_tiling
        self._stmt_stack.append(s)

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

    def do_e(self, e):
        self.apply_e(e)
        super().do_e(e)

    def apply_e(self, e):
        if isinstance(e, idx_e_types):
            self._syms_needed.add(e.name)
            self.apply_idx(e)
        else:
            assert not hasattr(e, "name"), "Add handling for array indexing"

    def apply_s(self, s):
        if isinstance(s, idx_s_types):
            self._syms_needed.add(s.name)
            self.apply_idx(s)
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
                        f"{s.srcinfo}: cuda_tasks loop must appear only at top level of CudaDeviceFunction"
                    )
            elif isinstance(loop_mode, CudaThreads):
                self.apply_cuda_threads_loop(s)
            else:
                raise TypeError(
                    f"{s.srcinfo}: unexpected loop mode {s.loop_mode.loop_mode_name()} in CudaDeviceFunction"
                )
        elif isinstance(s, LoopIR.Alloc):
            if s.type.is_barrier():
                self.barrier_scans[s.name] = BarrierScan(
                    self, s, self._body_containing_s
                )
            else:
                if not issubclass(s.mem, CudaBasicDeviceVisible):
                    raise TypeError(
                        f"{s.srcinfo}: For cuda code, memory type "
                        f"({s.mem.name()}) must subclass CudaBasicDeviceVisible"
                    )
                native_unit = s.mem.native_unit()
                self.alloc_states[s.name] = AllocState(self._coll_tiling, native_unit)

        elif isinstance(s, LoopIR.Free):
            if s.type.is_barrier():
                self.barrier_scans[s.name].check()

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if (n_threads := self._coll_tiling.box_num_threads()) != 1:
                raise ValueError(
                    f"{s.srcinfo}: write must be executed by one "
                    f"thread only (current: {n_threads} threads)\n"
                    f"stmt: {s}"
                )
        elif isinstance(s, LoopIR.SyncStmt):
            if s.lowered is None:  # backdoor, may remove later
                if s.sync_type.is_split():
                    # arrive/await
                    state = self.barrier_scans.get(s.bar)
                    assert isinstance(state, BarrierScan)
                else:
                    # Fence: each should have its own unique internal-use name (Sym)
                    state = BarrierScan(self, s, self._body_containing_s)
                    assert s.bar not in self.barrier_scans
                    self.barrier_scans[s.bar] = state
                state.inspect_sync_stmt(s, self._coll_tiling)

        elif isinstance(s, LoopIR.Call):
            callee = s.f
            needed = callee.proc_coll_unit()
            if msg := self._coll_tiling.unit_mismatch(needed, self._coll_env):
                raise TypeError(
                    f"{s.srcinfo}: wrong collective unit for " f"{callee.name}(): {msg}"
                )

    def apply_with_cuda_warps(self, s):
        ctx = s.cond.val
        assert isinstance(ctx, CudaWarps)
        self._coll_tiling, advice = self._coll_tiling.specialized(
            cuda_warp, ctx.lo, ctx.hi, self._coll_env
        )
        self.cuda_warps_dfs_codegen.append(
            _CodegenPar(advice.coll_index.codegen(), (advice.lo, advice.hi))
        )

    def expect_SyncStmt(
        self, async_block, is_epilogue, first_actor_kind, second_actor_kind
    ):
        # This is really strict, requires equality with expected actor kind
        # instead of just implements_first/implements_second(...)
        ctx = async_block.cond.val
        sync = async_block.body[-1] if is_epilogue else async_block.body[0]
        verb = "missing"
        if isinstance(sync, LoopIR.SyncStmt):
            verb = "wrong"
            sync_type = sync.sync_type
            if sync_type.first_actor_kind == first_actor_kind:
                if sync_type.second_actor_kind == second_actor_kind:
                    return sync
        noun = "epilogue" if is_epilogue else "prologue"
        expected = SyncType(first_actor_kind, second_actor_kind, False, 1).format_stmt(
            "..."
        )
        raise ValueError(
            f"{async_block.srcinfo}: {verb} {noun} sync in {ctx} block; "
            f"expect {expected}"
        )

    def post_inspect_cuda_async(self, s):
        # Must be run after inspecting the body of the CudaAsync block
        # since the barriers must have been scanned.
        # We detect required prologue/epilogue sync here.
        # The converse requirement (that some sync must be a prologue/epilogue)
        # is instead detected during code lowering.
        ctx = s.cond.val
        assert isinstance(ctx, CudaAsync)
        actor_kind = ctx.get_actor_kind()
        assert actor_kind in actor_kinds.cuda_async_actor_kinds
        assert s.body

        if not actor_kinds.cuda_async_proxy.signatures.isdisjoint(
            actor_kind.signatures
        ):
            self.uses_async_proxy = True

        def inspect(is_epilogue, A1, A2):
            sync_stmt = self.expect_SyncStmt(s, is_epilogue, A1, A2)
            scan = self.barrier_scans[sync_stmt.bar]
            sync_type = sync_stmt.sync_type

        # tma_to_smem_async requires epilogue Arrive(tma_to_smem_async);
        if actor_kind == actor_kinds.tma_to_smem_async:
            inspect(True, actor_kinds.tma_to_smem_async, None)
        # wgmma_async requires prologue wgmma fence, epilogue Arrive(wgmma_async)
        elif actor_kind == actor_kinds.wgmma_async:
            inspect(False, actor_kinds.wgmma_fence_1, actor_kinds.wgmma_fence_2)
            inspect(True, actor_kinds.wgmma_async, None)
        # Sm80_cp_async, tma_to_gmem_async have no prologue/epilogue

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
        self._coll_tiling, advice = self._coll_tiling.tiled(
            s.loop_mode.unit, hi_int, self._coll_env
        )
        self._iter_coll_tiling[s.iter] = self._coll_tiling

        # We will advise replacing the loop mode with _CodegenPar
        assert s.iter not in self.thread_iters
        self.thread_iters[s.iter] = ThreadIter(
            _CodegenPar(advice.coll_index.codegen(), (advice.lo, advice.hi))
        )

    def apply_idx(self, node):
        """Do analysis for one usage of tensor in distributed memory"""
        if not node.idx:
            # XXX early exit needed for Reads that are not from tensors
            # (e.g. index variables), but could hide issues?
            return

        state: AllocState
        state = self.alloc_states.get(node.name)
        if state is None:
            return  # was allocated outside presumably

        assert isinstance(state, AllocState)
        native_threads = state.native_unit.int_threads(self._coll_env)
        assert isinstance(native_threads, int)

        try:
            # Allocation collective tiling is tiled by leading index variables,
            # until the actual tiling matches the native unit of the memory type.
            # Remaining indices are lowered to source code.
            # TODO improve error messages. Maybe less strict usage patterns.
            cur_coll_tiling = state.alloc_coll_tiling
            n_distributed_dims = None
            for dim_i, idx_coord in enumerate(node.idx):
                cur_threads = cur_coll_tiling.box_num_threads()
                if cur_threads == native_threads:
                    n_distributed_dims = dim_i
                    break
                if cur_threads < native_threads:
                    # TODO no one is going to understand this...
                    message = (
                        f"Expected {native_threads} threads to allocate "
                        f"{node.name} but have {cur_threads}"
                    )
                    if dim_i > 0:
                        str_idxs = ", ".join(str(idx) for idx in node.idx[:dim_i])
                        message += f" (deduced after indexing by {str_idxs})"
                    raise ValueError(message)

                def coord_error(expected):
                    raise ValueError(
                        f"Index {dim_i}: expected {expected}, "
                        f"got {idx_coord}, to index distributed "
                        f"memory {node.name}"
                    )

                if isinstance(idx_coord, LoopIR.Read):
                    idx_sym = idx_coord.name
                elif isinstance(idx_coord, LoopIR.Point) and isinstance(
                    idx_coord.pt, LoopIR.Read
                ):
                    idx_sym = idx_coord.pt.name
                else:
                    coord_error("plain variable")

                next_coll_tiling = self._iter_coll_tiling.get(idx_sym)
                if next_coll_tiling is None:
                    coord_error("index from cuda_threads loop")
                if (
                    next_coll_tiling.parent.tile_num_threads()
                    != cur_coll_tiling.tile_num_threads()
                ):
                    coord_error("correct parent tiling (TODO explain)")
                cur_coll_tiling = next_coll_tiling

            # Check thread box shape, alignment; not just thread count.
            if mismatch_reason := cur_coll_tiling.unit_mismatch(
                state.native_unit, self._coll_env
            ):
                raise ValueError(mismatch_reason)

            # Record usage.
            # If not the first usage, check usage pattern matches prior usage.
            if not state.live:
                state.live = True
                state.n_distributed_dims = n_distributed_dims
                state.usage_coll_tiling = cur_coll_tiling
            else:
                if not state.usage_coll_tiling.equiv(cur_coll_tiling):
                    raise ValueError("collective tiling mismatch")

        except ValueError as e:
            # TODO better error messages
            message = f"{node.srcinfo}: {node.name} distributed memory analysis failed (see chained exception)"
            raise MemGenError(message) from e

    def get_cta_count(self, coll_tiling: CollTiling, srcinfo: SrcInfo):
        assert isinstance(srcinfo, SrcInfo)
        if self.clusterDim == 1:
            return 1
        else:
            # Only if the clusterDim is not 1 can we rely on the left-most
            # dimension of the domain to correspond to the CTA-in-cluster axis.
            domain = coll_tiling.full_domain
            box = coll_tiling.box
            assert len(domain) == len(box)
            if domain[0] != self.clusterDim:
                # Unlikely error, only occurs of the user defines their own
                # unit splitting the cluster dimension of the coll tiling.
                raise TypeError(f"{srcinfo}: could not deduce cluster count")
            return box[0]

    def lower_barriers(self) -> Tuple[Dict[LoweredBarrier], List[Tuple[int, int]], str]:
        """Lower barriers

        Returns Dict[Sym, LoweredBarrier], mbarrier pairs, SyncState struct body

        mbarrier pairs are tuples (mbarrier_count, arrive_count) to
        initialize in SMEM, e.g. (8, 64), (2, 384) means initialize an
        array of 10 mbarriers in SMEM with the first 8 having
        arrive_count=64, last 2 arrive_count=384

        This requires hard-wired compiler support to lower, so
        "lower_barriers" doesn't really do its job fully ... not clean,
        but should be OK for now, at least until CUDA introduces even more
        synchronization primitives to construct in weird new ways.

        """
        # Need to assign a name unique-ifying suffix for each barrier
        # This is different than what the main LoopIR->C compiler does because the
        # name needs to be unique throughout the full device function, i.e. it's not
        # enough to be unique just within the barrier's scope in Exo object code.
        sym_suffix = {}
        sym_counters = {}

        def assign_suffix(barrier_name):
            assert isinstance(barrier_name, Sym)
            count = sym_counters.get(barrier_name, 0)
            sym_counters[barrier_name] = count + 1
            suffix = str(count)
            sym_suffix[barrier_name] = suffix
            return suffix

        lowered = {}
        mbarrier_pairs = []
        SyncState_lines = []

        # Sort scanned barriers by Sym ID for output stability.
        key = lambda tup: tup[0].id_number()

        for name, scan in sorted(self.barrier_scans.items(), key=key):
            srcinfo = scan.barrier_srcinfo
            barrier_type = scan.barrier_type
            suffix = assign_suffix(name)
            if not scan.is_split():
                # Fence
                if scan.Arrive.actor_kind == actor_kinds.wgmma_fence_1:
                    lowered[name] = self.lower_wgmma_fence(scan, suffix)
                else:
                    lowered[name] = self.lower_garden_variety_fence(scan, suffix)
            elif isinstance(barrier_type, LoopIR.CudaMbarrier):
                lowered[name] = self.lower_mbarrier(scan, suffix, mbarrier_pairs)
            elif isinstance(barrier_type, LoopIR.CudaCommitGroup):
                lowered[name] = self.lower_commit_group(scan, suffix)
            else:
                raise TypeError(
                    f"{srcinfo}: {barrier_type} must not "
                    f"be used in CUDA device function"
                )

            for line in lowered[name].SyncState_lines:
                if line:
                    SyncState_lines.append("    " + line)

        return lowered, mbarrier_pairs, "\n".join(SyncState_lines)

    def lower_wgmma_fence(self, scan: BarrierScan, suffix):
        Arrive = scan.Arrive
        Await = scan.Await
        A1 = Arrive.actor_kind
        A2 = Await.actor_kind
        srcinfo = Arrive.get_srcinfo()
        assert A1 == actor_kinds.wgmma_fence_1
        if A2 != actor_kinds.wgmma_fence_2:
            raise ValueError(
                f"{srcinfo}: wgmma fence needs second actor kind wgmma_fence_2"
            )

        coll_tiling = Arrive.coll_tiling
        # Should be the case for a Fence
        assert coll_tiling is Await.coll_tiling

        if msg := coll_tiling.unit_mismatch(cuda_warpgroup, self._coll_env):
            raise ValueError(
                f"{srcinfo}: wgmma fence must be executed by a warpgroup: {msg}"
            )

        lowered = CudaLoweredBarrier(False, LoweredBarrierType.wgmma_fence)
        lowered.codegen = lambda _: LoweredPrologueSync(
            actor_kinds.wgmma_async, ['asm("wgmma.fence.sync.aligned;");']
        )
        return lowered

    def lower_garden_variety_fence(self, scan: BarrierScan, suffix):
        """Do up to 3 things

        - wait_all if first actor kind includes Sm80_cp_async
        - barrier arrive/await if more than 1 thread, or special exception (*)
        - fence.proxy.async if second actor kinds includes any async proxy

        (*) special exception, if thread collective is a warpgroup and
        the second actor kind only includes wgmma_async, we can elide
        the barrier. This relies on wgmma_async not being V1-transitive.
        """

        Arrive = scan.Arrive
        Await = scan.Await
        A1 = Arrive.actor_kind
        A2 = Await.actor_kind
        srcinfo = Arrive.get_srcinfo()
        clusterDim = self.clusterDim

        lowered = CudaLoweredBarrier(False, LoweredBarrierType.garden_variety_fence)
        lines = []

        # Insert wait for sm_80 cp.async if needed.
        if actor_kinds.cuda_classic.implements_first(A1):
            pass
        elif actor_kinds.Sm80_generic.implements_first(A1):
            lines.append('asm volatile("cp.async.wait_all;" ::);')
        else:
            raise ValueError(
                f"{srcinfo}: Fence first actor kind "
                f"{A1} not supported (we allow Sm80_generic)"
            )

        coll_tiling = Arrive.coll_tiling
        # Should be the case for a Fence
        assert coll_tiling is Await.coll_tiling

        cta_count = self.get_cta_count(coll_tiling, srcinfo)
        threads = coll_tiling.box_num_threads()
        n_warps = threads // 32
        if threads != 1:
            if threads % 32 != 0:
                raise ValueError(
                    f"{srcinfo}: Fence expects convergent warps "
                    f"(threads = {threads} is not divisible by 32)"
                )
            unit = n_warps * cuda_warp
            if msg := coll_tiling.unit_mismatch(unit, self._coll_env):
                raise ValueError(f"{srcinfo}: Fence expects convergent warps: {msg}")

        # Insert cross-thread sync if needed
        assert not actor_kinds.wgmma_async.V1_transitive
        wgmma_special_case = actor_kinds.wgmma_async.implements_second(
            A2
        ) and not coll_tiling.unit_mismatch(cuda_warpgroup, self._coll_env)
        if n_warps > 0 and not wgmma_special_case:
            if cta_count == 1:
                if n_warps == 1:
                    lines.append("__syncwarp();")
                elif n_warps * 32 == self.blockDim:
                    lines.append("__syncthreads();")  # TODO consider PTX
                else:
                    raise NotImplementedError(
                        "TODO Fence lowering other than warp/CTA/cluster"
                    )
            elif cta_count == clusterDim:
                if msg := coll_tiling.unit_mismatch(
                    cta_count * cuda_cta_in_cluster, self._coll_env
                ):
                    raise ValueError(
                        f"{srcinfo}: expected full cluster " f"or only 1 CTA: {msg}"
                    )
                else:
                    lines.append('asm("barrier.cluster.arrive.aligned;"::);')
                    lines.append('asm("barrier.cluster.wait.aligned;"::);')
            else:
                raise ValueError(
                    f"{srcinfo}: {cta_count}/{clusterDim} CTAs in cluster active for thread collective for Fence; must have 1 or all"
                )

        # Insert fence.proxy.async if needed
        if actor_kinds.Sm80_generic.implements_second(A2):
            pass
        elif actor_kinds.cuda_generic_and_async_proxy.implements_second(A2):
            lines.append('asm("fence.proxy.async;");')
        else:
            raise ValueError(
                f"{srcinfo}: Fence second actor kind {A2} not "
                f"supported (at most CUDA generic+async proxy)"
            )
        lowered.codegen = lambda _: lines
        return lowered

    def lower_mbarrier(self, scan: BarrierScan, suffix, mbarrier_pairs: list):
        lowered = CudaLoweredBarrier(False, LoweredBarrierType.mbarrier)
        mbarrier_offset = sum(c for c, _ in mbarrier_pairs)  # O(n^2)
        nm_suffix = f"{suffix}_{scan.barrier_name}"

        # Calculate the size of the ring buffer (number of mbarriers)
        # TODO we need checking that N is correct for arrive/await
        def max_skips(sync_stmts):
            for s in sync_stmts:
                if s.sync_type.N >= 0:
                    raise ValueError(f"{s.srcinfo}: {s} must have N < 0 for mbarrier")
            return max(~s.sync_type.N for s in sync_stmts)

        max_await_skips = max_skips(scan.Await.sync_stmts)
        if scan.has_reverse():
            max_reverse_await_skips = max_skips(scan.ReverseAwait.sync_stmts)
            ring = max_await_skips + max_reverse_await_skips
        else:
            ring = max_await_skips + 1
        if ring == 0:
            raise ValueError(
                f"{scan.get_srcinfo()}: {scan.barrier_name} must have some await with nonzero skips (e.g. set N = ~1)"
            )

        # Need to be able to store values 0 through (ring-1)
        ring_bits = (ring - 1).bit_length()
        # Need to be able to count 0 to ring (inclusive) skips.
        # This value will not be used if skipping is not actually enabled.
        skip_bits = ring.bit_length()

        # black formatting will ruin the readability of the generated C++ code below
        # fmt: off
        def mbarrier_to_u32(lines, is_reverse, idx):
            byte_offset = 8 * (mbarrier_offset + ring if is_reverse else mbarrier_offset)
            lines.append(f"  const auto mbarrier_u32 = exo_smemU32(exo_smem + {byte_offset} + 8*{idx});")

        def generate_arrive(is_reverse):
            r = "Reverse" if is_reverse else ""
            info = scan.ReverseArrive if is_reverse else scan.Arrive
            actor_kind = info.actor_kind

            if actor_kinds.Sm80_cp_async.implements_first(actor_kind):
                is_Sm80_cp_async = True
                is_tma = False
            elif actor_kinds.cuda_classic.implements_first(actor_kind):
                is_Sm80_cp_async = False
                is_tma = False
            elif actor_kinds.tma_to_smem_async.implements_first(actor_kind):
                is_Sm80_cp_async = False
                is_tma = True
            else:
                raise ValueError(
                    f"{info.get_srcinfo()}: mbarrier Arrive actor kind {actor_kind} "
                    f"not supported: need cuda_classic, Sm80_cp_async, or tma_to_smem_async")

            lines = lowered.SyncState_lines
            idx = f"{r}ArriveIdx{nm_suffix}"
            if ring_bits > 0:
                lines.append(f"unsigned {idx} : {ring_bits} = 0;")
            else:
                lines.append(f"static constexpr unsigned {idx} = 0;  // Trivial size-1 ring buffer")
            lines.append(f"__device__ __forceinline__ uint32_t {r}Arrive{nm_suffix}(char* exo_smem, bool enable) {{")
            mbarrier_to_u32(lines, is_reverse, idx);
            lines.append(f"  if (enable) {{")
            # TODO cluster broadcast if needed
            if is_Sm80_cp_async:
                lines.append(f'    asm("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(mbarrier_u32));');
            else:
                lines.append(f'    asm("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mbarrier_u32));');
            if ring_bits > 0:
                lines.append(f"    // Advance ring buffer state")
                lines.append(f"    {idx} = {idx} == {ring - 1} ? 0 : {idx} + 1;")
            lines.append(f"  }}")
            lines.append(f"  return mbarrier_u32;")
            lines.append(f"}}")
            return is_tma

        def generate_await(is_reverse, A1):
            r = "Reverse" if is_reverse else ""
            info = scan.ReverseAwait if is_reverse else scan.Await
            A2 = info.actor_kind

            if actor_kinds.cuda_async_proxy_wgmma.implements_first(A1):
                # proxy fence always elided if first actor kind includes only
                # async proxy and wgmma register access.
                proxy_fence = False
            elif actor_kinds.Sm80_generic.implements_second(A2):
                proxy_fence = False
            elif actor_kinds.cuda_generic_and_async_proxy.implements_second(A2):
                proxy_fence = True
            else:
                if A2 == actor_kinds.wgmma_async:
                    remark = "consider wgmma_async_smem"
                else:
                    remark = "at most CUDA generic+async proxy"
                raise ValueError(
                    f"{info.get_srcinfo()}: mbarrier Await actor kind {A2} "
                    f"not supported ({remark})")

            lines = lowered.SyncState_lines
            # If we have ReverseAwait/ReverseArrive, the mbarriers for them
            # are allocated right after those for Arrive/Await
            offset = mbarrier_offset + ring if is_reverse else mbarrier_offset
            idx = f"{r}AwaitIdx{nm_suffix}"
            skips = f"{r}Skips{nm_suffix}"
            parity_bits = f"{r}Parity{nm_suffix}"
            max_skips = max_reverse_await_skips if is_reverse else max_await_skips
            enable_skips = max_skips != 0

            # Define (register) exo_SyncState member variables: ring buffer
            # index, parity bitfield, and, if needed, counter for inital skips.
            if ring_bits > 0:
                lines.append(f"unsigned {idx} : {ring_bits} = 0;")
            else:
                lines.append(f"static constexpr unsigned {idx} = 0;  // Trivial size-1 ring buffer")
            lines.append(f"unsigned {parity_bits} : {ring} = 0;")
            if enable_skips:
                lines.append(f"unsigned {skips} : {skip_bits} = 0;")

            # Define Await member function
            # The initial_skips parameter is included iff skipping is enabled,
            # as a last line of defense against future Exo compiler bugs.
            if enable_skips:
                lines.append(f"__device__ __forceinline__ void {r}Await{nm_suffix}(char* exo_smem, int initial_skips = 0) {{")
                mbarrier_to_u32(lines, is_reverse, idx)
                lines.append(f"  const bool enable = {skips} >= initial_skips;")
            else:
                lines.append(f"__device__ __forceinline__ void {r}Await{nm_suffix}(char* exo_smem) {{")
                mbarrier_to_u32(lines, is_reverse, idx)
                lines.append(f"  const bool enable = true;")
            lines.append(f"  if (enable) {{")
            # sm_90 needed for try_wait
            test_or_try = "try" if self.uses_async_proxy else "test"
            lines.append(f"    // Wait for mbarrier ... PTX loop needed for this")
            lines.append(f'    asm volatile("{{.reg.pred P1; EXO_BEFORE_WAIT: mbarrier.{test_or_try}_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1; @P1 bra.uni EXO_WAIT_DONE; bra.uni EXO_BEFORE_WAIT; EXO_WAIT_DONE: }}"::')
            lines.append(f'        "r"(mbarrier_u32), "r"(1u & {parity_bits} >> {idx}));')
            lines.append(f"    // Flip parity")
            lines.append(f"    {parity_bits} ^= 1u << {idx};")
            if ring_bits > 0:
                lines.append(f"    // Advance ring buffer state")
                lines.append(f"    {idx} = {idx} == {ring - 1} ? 0 : {idx} + 1;")
            if proxy_fence:
                lines.append(f'    // Needed for first actor kind {A1}; second actor kind {A2}')
                lines.append(f'    asm("fence.proxy.async;");')
            lines.append(f"  }}")
            if enable_skips:
                lines.append(f"  else {{")
                lines.append(f"    // {r}Await({scan.barrier_name}) returns without waiting for mbarrier first <initial_skips> times")
                lines.append(f"    {skips}++;")
                lines.append(f"  }}")
            lines.append(f"}}")

        # Generate Arrive and Await syntax
        # {Reverse}Awaits must be aware with the actor kind
        # of the matched {Reverse}Arrive
        Arrive_is_tma = generate_arrive(False)
        generate_await(False, scan.Arrive.actor_kind)
        if scan.has_reverse():
            ReverseArrive_is_tma = generate_arrive(True)
            generate_await(True, scan.ReverseArrive.actor_kind)

        # Arrive/Await lowers to call to generated exo_syncState member function.
        # We also record mbarriers to initialize, first those for Arrive/Await,
        # then those for ReverseArrive/ReverseAwait.
        # Finally, for arrives with actor kind tma_to_smem_async, we need to enforce
        # that it is an epilogue sync for tx-count to work correctly.
        Arrive_txt = f"Arrive{nm_suffix}(exo_smem, "
        def codegen(sync_type: SyncType):
            r = "Reverse" if sync_type.is_reversed else ""
            assert sync_type.is_split()
            if sync_type.is_arrive():
                assert sync_type.N == 1, "TODO implement me"
                result = [f"exo_syncState.{r}{Arrive_txt}{sync_type.N});"]
                is_tma = ReverseArrive_is_tma if sync_type.is_reversed else Arrive_is_tma
                if is_tma:
                    result = LoweredEpilogueSync(actor_kinds.tma_to_smem_async, result)
                return result
            else:
                skips_arg = ""
                if skips := ~sync_type.N:
                    assert skips > 0, "should have been caught earlier"
                    skips_arg = f", {skips}"
                return [f"exo_syncState.{r}Await{nm_suffix}(exo_smem{skips_arg});"]
        lowered.codegen = codegen
        lowered.c_Arrive_mbarrier = f"exo_syncState.{Arrive_txt}0)"
        arrive_count = scan.Arrive.coll_tiling.box_num_threads()
        mbarrier_pairs.append((ring, arrive_count))

        if scan.has_reverse():
            lowered.c_ReverseArrive_mbarrier = f"exo_syncState.Reverse{Arrive_txt}0)"
            arrive_count = scan.ReverseArrive.coll_tiling.box_num_threads()
            mbarrier_pairs.append((ring, arrive_count))
        return lowered
        # fmt: on

    def lower_commit_group(self, scan: BarrierScan, suffix):
        # Commit groups
        #
        # Sm80_cp_async -> Sm80_generic; 1 thread
        # tma_to_gmem_async -> cuda_generic_and_async_proxy; 1 thread
        # wgmma_async -> cuda_generic_and_async_proxy; 128 threads
        #
        # Can fail due to
        #   * unsupported first actor kind
        #   * incorrect second actor kind given supported first actor kind
        #   * incorrect collective unit given supported first actor kind
        #   * incorrect N for Arrive/Await
        assert scan.ReverseAwait is None
        assert scan.ReverseArrive is None

        solitary = True
        A1 = scan.Arrive.actor_kind
        A2 = scan.Await.actor_kind
        is_wgmma = False

        def check_coll_unit(thing, action_name, coll_unit):
            srcinfo = thing.get_srcinfo()
            coll_tiling = thing.get_coll_tiling()
            if msg := coll_tiling.unit_mismatch(coll_unit, self._coll_env):
                raise TypeError(
                    f"{srcinfo}: {action_name} of commit group "
                    f"{scan.barrier_name} with Arrive({A1}) "
                    f"expects collective unit {coll_unit}: {msg}"
                )

        def check_A2_coll_unit(expect_A2, coll_unit):
            if not expect_A2.implements_second(A2):
                raise TypeError(
                    f"{scan.barrier_srcinfo}: commit group "
                    f"{scan.barrier_name} with Arrive({A1}) "
                    f"expects Await({expect_A2}), "
                    f"not {A2} (wrong second actor kind)"
                )
            # TODO declaration coll_tiling analysis
            # check_coll_unit(scan, "declaration", coll_unit)
            check_coll_unit(scan.Arrive, "Arrive", coll_unit)
            check_coll_unit(scan.Await, "Await", coll_unit)

        if actor_kinds.Sm80_cp_async.implements_first(A1):
            # sm_80 non-bulk cp.async
            check_A2_coll_unit(actor_kinds.Sm80_generic, cuda_thread)
            lowered = CudaLoweredBarrier(solitary, LoweredBarrierType.Sm80_commit_group)
            lowered_arrive = ['asm("cp.async.commit_group;");']
            await_fmt = 'asm("cp.async.wait_group {N};");'
        elif actor_kinds.tma_to_gmem_async.implements_first(A1):
            # sm_90a bulk cp.async SMEM->GMEM
            check_A2_coll_unit(actor_kinds.cuda_generic_and_async_proxy, cuda_thread)
            lowered = CudaLoweredBarrier(
                solitary, LoweredBarrierType.tma_to_gmem_commit_group
            )
            lowered_arrive = ['asm("cp.async.bulk.commit_group;");']
            await_fmt = 'asm("cp.async.bulk.wait_group {N};");'
        elif actor_kinds.wgmma_async.implements_first(A1):
            # sm_90a wgmma; note unit is now warpgroup and not a single thread;
            # also enforce that this is an epilogue sync of CudaAsync(wgmma_async).
            check_A2_coll_unit(actor_kinds.cuda_generic_and_async_proxy, cuda_warpgroup)
            lowered = CudaLoweredBarrier(
                solitary, LoweredBarrierType.wgmma_commit_group
            )
            lowered_arrive = LoweredEpilogueSync(
                actor_kinds.wgmma_async, ['asm("wgmma.commit_group.sync.aligned;");']
            )
            await_fmt = 'asm("wgmma.wait_group.sync.aligned {N};");'
            is_wgmma = True
        else:
            raise TypeError(
                f"{scan.barrier_srcinfo}: {scan.barrier_name} : "
                f"cuda_commit_group does not support "
                f"Arrive({A1}) (wrong first actor kind)"
            )

        def codegen(sync_type):
            assert sync_type.is_split()
            assert not sync_type.is_reversed
            if sync_type.is_arrive():
                if sync_type.N != 1:
                    raise ValueError("Expect N=1 for Arrive of commit group")
                return lowered_arrive
            else:
                if sync_type.N < 0:
                    raise ValueError("Expect N>=0 for Await of commit group")
                return [await_fmt.format(N=sync_type.N)]

        lowered.codegen = codegen
        return lowered


# End class SubtreeScan


class SubtreeRewrite(LoopIR_Rewrite):
    __slots__ = [
        "scan",
        "fmt_dict",
        "alloc_states",
        "thread_iters",
        "cuda_warps_dfs_codegen",
        "cuda_warps_idx",
        "lowered_barriers",
        "live_solitary_barrier_names",
        "live_smem_ends",  # SMEM stack allocator
        "smem_data_usage",  # SMEM stack allocator
        "_result",
    ]

    def __init__(self, s, scan: SubtreeScan, ctx: SporkLoweringCtx):
        fmt_dict = scan.fmt_dict
        self.scan = scan
        self.fmt_dict = fmt_dict
        self.alloc_states = scan.alloc_states
        self.thread_iters = scan.thread_iters
        self.cuda_warps_dfs_codegen = scan.cuda_warps_dfs_codegen
        self.cuda_warps_idx = 0
        (
            self.lowered_barriers,
            mbarrier_pairs,
            fmt_dict["SyncState_body"],
        ) = scan.lower_barriers()

        # Prepare mbarriers in SMEM
        if mbarrier_pairs:
            fmt_dict["device_setup_body"], num_mbarriers = self.generate_device_setup(
                mbarrier_pairs
            )
        else:
            fmt_dict["device_setup_body"] = "  // No mbarriers used"
            num_mbarriers = 0

        # Dict mapping LoweredBarrierType -> Sym
        # only includes live lowered barriers with solitary flag set.
        self.live_solitary_barrier_names = {}

        # Prepare SMEM stack allocator
        # Base of SMEM allocation is reserved for mbarriers
        self.smem_data_usage = 0
        # self.live_smem_ends = {8 * num_mbarriers}
        self.live_smem_ends = {128 * ((num_mbarriers + 15) // 16)}
        # HACK: align mbarriers to 128 bytes for now

        # We override the C names of variables that appear in the
        # exo_DeviceArgs or exo_Task structs.
        main_loop_force_names = {}
        task_force_names = {}
        for sym in scan.task_iter_syms:
            main_loop_force_names[sym] = "exo_task_" + str(sym)
            task_force_names[sym] = "exo_task." + str(sym)
        for sym in scan.device_args_syms:
            new_name = "exo_deviceArgs." + str(sym)
            main_loop_force_names[sym] = new_name
            task_force_names[sym] = new_name

        # ExtWithContext objects for diverting lowered code into
        # exo_deviceTask().
        format = lambda fmt_string: fmt_string.format(**fmt_dict)
        task_context = ExtWithContext(
            format(task_launch_fmt),
            format(device_task_prefix_fmt),
            "}",
            "cuh",
            {},
            reserved_names,
            task_force_names,
            scan.grid_constant_syms,  # force_const
            scan.scalar_ref_syms,
        )

        def wrap_with_context(with_context, body, srcinfo):
            cond = LoopIR.Const(with_context, T.with_context, srcinfo)
            node = LoopIR.If(cond, body, [], srcinfo)
            assert is_if_holding_with(node, LoopIR)
            return node

        # Manually rewrite the cuda_tasks loops to use seq(...) mode,
        # and rely on LoopIR_Rewrite to rewrite the task body, which
        # is wrapped with the task_context to put the code into
        # exo_deviceTask().
        assert scan.task_loop_depth > 0

        def rewrite_task_loop(loop, depth_left):
            if depth_left == 1:
                mapped_body = self.map_stmts(loop.body) or loop.body
                body = [wrap_with_context(task_context, mapped_body, loop.srcinfo)]
            else:
                body = [rewrite_task_loop(loop.body[0], depth_left - 1)]
            assert isinstance(loop.loop_mode, CudaTasks)
            return loop.update(loop_mode=seq, body=body)

        assert len(s.body) == 1
        task_loop = rewrite_task_loop(s.body[0], scan.task_loop_depth)

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
                "cuh": format(cuh_snippet_fmt),
            },
            reserved_names,
            main_loop_force_names,
            scan.grid_constant_syms,  # force_const
            scan.scalar_ref_syms,
        )

        # Finally wrap the task loops into exo_deviceMainLoop
        self._result = wrap_with_context(
            main_loop_context, [task_loop], task_loop.srcinfo
        )

        assert self.cuda_warps_idx == len(self.cuda_warps_dfs_codegen)

    def result(self):
        assert is_if_holding_with(self._result, LoopIR)
        return self._result

    def updated_stmt(self, s, prologue_of=None, epilogue_of=None):
        if is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, CudaAsync):
                s = self.update_with_cuda_async(s)
            elif isinstance(ctx, CudaWarps):
                # Replace with CudaWarps block with _CodegenPar "loop"
                # that the scanner has prepared. NB (0, 1) isn't the same
                # as the indices encoded in _CodegenPar.
                loop_mode = self.cuda_warps_dfs_codegen[self.cuda_warps_idx]
                self.cuda_warps_idx += 1
                s = LoopIR.For(
                    Sym("tmp"),
                    LoopIR.Const(0, T.int, s.srcinfo),
                    LoopIR.Const(1, T.int, s.srcinfo),
                    s.body,
                    loop_mode,
                    s.srcinfo,
                )
            else:
                raise TypeError(
                    f"{s.srcinfo}: unexpected with context type {type(ctx)} in CUDA device code"
                )
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
            s = self.update_check_sync_stmt(s, prologue_of, epilogue_of)

        return s

    def map_s(self, s, prologue_of=None, epilogue_of=None):
        s_rewrite = self.updated_stmt(s, prologue_of, epilogue_of)

        # Use superclass to recurse and rewrite subtree
        # We have to have logic to handle None being used to indicate
        # "no change"; if the superclass makes no changes, we still have
        # to preserve any rewrites of our own.
        if is_if_holding_with(s, LoopIR) and isinstance(s.cond.val, CudaAsync):
            # Special handling for CudaAsync blocks; see update_with_cuda_async
            # We must bypass the normal superclass behavior in this case.
            out_stmts = [s_rewrite]
        elif s_rewrite is s or s_rewrite is None:
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

        if isinstance(e, idx_e_types):
            e_rewrite = self.remove_distributed_idx(e)

        # Use superclass to recursre and rewrite subtree
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
        alloc_state = self.alloc_states.get(node.name)
        if isinstance(alloc_state, AllocState):
            assert isinstance(alloc_state, AllocState)
            n = alloc_state.n_distributed_dims
            if n > 0:
                return node.update(idx=node.idx[n:])
        return None

    def generate_device_setup(self, mbarrier_pairs):
        # fmt: off
        lines = []
        lines.append("    if (threadIdx.x == 0) {")
        lines.append("      const auto mbarrier_u32 = exo_smemU32(exo_smem);")
        offset = 0
        for mbarrier_count, arrive_count in mbarrier_pairs:
            lines.append(f"      for (int i = 0; i < {mbarrier_count}; ++i) {{")
            lines.append(f'        asm volatile("mbarrier.init.shared::cta.b64 [%0], {arrive_count};"::')
            lines.append(f'          "r"(mbarrier_u32 + {8*offset} + 8*i));')
            lines.append(f"      }}")
            offset += mbarrier_count
        if self.scan.uses_async_proxy:
            lines.append('      asm("fence.proxy.async;");')
        lines.append("    }")
        if self.scan.clusterDim > 1:
            lines.append('    asm("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;\n"::);')
        else:
            lines.append('    __syncthreads();')
        return "\n".join(lines), offset
        # fmt: on

    def update_numeric_alloc_free(self, s):
        alloc_state = self.alloc_states[s.name]
        assert isinstance(alloc_state, AllocState)
        if not alloc_state.live:
            # Distributed memory analysis isn't run for unused variables...
            warn(
                f"{s.srcinfo}: Unused allocation {s.name} in CUDA code may not lower correctly"
            )

        # Remove distributed dimensions
        n = alloc_state.n_distributed_dims
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
                alloc_state.codegen_smem = mem  # for rewriting Free, below
            else:
                # Rewrite Free Memory type to match corresponding Alloc
                # and restore stack allocator state
                mem = alloc_state.codegen_smem
                assert mem
                self.live_smem_ends.remove(mem.smem_end())
            s = s.update(mem=mem)
        return s

    def on_barrier_alloc(self, s):
        lowered = self.lowered_barriers[s.name]
        if lowered.solitary:
            self.check_solitary_barrier(s, lowered)
            self.live_solitary_barrier_names[lowered.type_enum] = s.name

    def on_barrier_free(self, s):
        lowered = self.lowered_barriers[s.name]
        if lowered.solitary:
            del self.live_solitary_barrier_names[lowered.type_enum]

    def update_check_sync_stmt(
        self,
        s: LoopIR.SyncStmt,
        prologue_of: Optional[ActorKind],
        epilogue_of: Optional[ActorKind],
    ):
        if s.lowered is None:
            lowered = self.lowered_barriers[s.bar]
            if lowered.solitary and not s.sync_type.is_split():
                # Fence must pass solitary barrier check
                self.check_solitary_barrier(s, lowered)
            assert lowered.codegen is not None

            # Do codegen, and supply srcinfo if codegen fails
            try:
                lowered = lowered.codegen(s.sync_type)
            except Exception as e:
                raise ValueError(f"{s.srcinfo}: {e}")

            # Enforce prologue/epilogue sync requirements
            if isinstance(lowered, LoweredPrologueSync):
                if lowered.actor_kind != prologue_of:
                    raise ValueError(
                        f"{s.srcinfo}: {s} must be the first stmt of a CudaAsync({lowered.actor_kind}) block"
                    )
                lowered = lowered.lines
            elif isinstance(lowered, LoweredEpilogueSync):
                if lowered.actor_kind != epilogue_of:
                    raise ValueError(
                        f"{s.srcinfo}: {s} must be the last stmt of a CudaAsync({lowered.actor_kind}) block"
                    )
                lowered = lowered.lines
            else:
                assert isinstance(lowered, list)

            # Inject lowered code
            s = s.update(lowered=lowered)
        return s

    def check_solitary_barrier(self, s, lowered):
        sus = self.live_solitary_barrier_names.get(lowered.type_enum)
        if sus is not None:
            raise TypeError(
                f'{s.srcinfo}: Invalid "{s}" of lowered '
                f"barrier type {lowered.type_enum} due to another "
                f'such live barrier "{sus}" in scope'
            )

    def update_with_cuda_async(self, s):
        # We have to customize mapping the body of a CudaAsync block, so as to
        # allow special handling for prologue/epilogue sync.
        #
        # Further, handle the special case of with CudaAsync(tma_to_smem_async)
        # where we need to make the mbarrier used for the /epilogue/
        # mbarrier Arrive available at the /start/ of the async block.
        # The mbarrier's 32-bit address used for this sync will be aliased as
        # exo_tma_mbarrier in the body of the lowered CUDA C++ async block.
        ctx = s.cond.val
        assert isinstance(ctx, CudaAsync)
        actor_kind = ctx.get_actor_kind()
        assert actor_kind in actor_kinds.cuda_async_actor_kinds
        assert s.body

        new_body = []
        for i, child in enumerate(s.body):
            prologue_of = actor_kind if i == 0 else None
            epilogue_of = actor_kind if i + 1 == len(s.body) else None
            new_children = self.map_s(child, prologue_of, epilogue_of)
            if new_children is None:
                new_body.append(child)  # Append unchanged stmt
            else:
                new_body.extend(new_children)

        if actor_kind == actor_kinds.tma_to_smem_async:
            # We will insert the needed uint32_t variable as "lowered" syntax
            # for a do-nothing Fence statement. This will look goofy but works.
            _arrive = self.scan.expect_SyncStmt(
                s, True, actor_kinds.tma_to_smem_async, None
            )
            dummy_sync_type = SyncType(
                actor_kinds.empty_actor_kind, actor_kinds.empty_actor_kind, False, 0
            )
            lowered = self.lowered_barriers[_arrive.bar]
            if _arrive.sync_type.is_reversed:
                mbarrier = lowered.c_ReverseArrive_mbarrier
            else:
                mbarrier = lowered.c_Arrive_mbarrier
            c_alias = f"const uint32_t exo_tma_mbarrier = {mbarrier};"
            alias_stmt = LoopIR.SyncStmt(
                dummy_sync_type,
                Sym("exo_tma_mbarrier"),
                [c_alias],
                s.srcinfo,
            )
            new_body = [alias_stmt] + new_body

        s = s.update(body=new_body)
        return s


# End class SubtreeRewrite


@dataclass(slots=True)
class ThreadIter:
    codegen_par: _CodegenPar


class AllocState(object):
    # Some GPU allocations are "distributed", when the collective unit
    # (e.g. CTA) that allocated a tensor doesn't match the "native unit"
    # of the memory type (e.g. thread for a register; warp for a wmma tile).
    #
    # Some of the leading dimensions of the tensor will be deduced to be
    # "distributed", i.e., correspond to a thread index rather than a
    # (CUDA syntactic) array index. e.g. if the CTA size is 512, something like
    #
    # foo : f32[32,16,4] @ CudaRmem  # Access with 2D grid of 32 x 16 threads
    #
    # may lower to `float foo[4]` since the first 2 dimensions are distributed.
    #
    # We deduce this from the usage of the memory, and enforce that each thread
    # only accesses its own index. TODO explain all this tiling stuff better.
    # Currently, we only support very trivial patterns, but this should be good
    # enough for prototyping.
    #
    # In the rewrite phase, we will strip the leading n_distributed_dims-many
    # indices from all uses of the memory ... this is just a hack for
    # code lowering; ignore that this changes the real meaning of the LoopIR.

    __slots__ = [
        "live",
        "n_distributed_dims",
        "alloc_coll_tiling",
        "usage_coll_tiling",
        "native_unit",
        "codegen_smem",
    ]

    live: bool
    n_distributed_dims: int
    alloc_coll_tiling: CollTiling
    usage_coll_tilings: Optional[CollTiling]
    native_unit: CollUnit
    codegen_smem: Optional[Type[CudaBasicSmem]]

    def __init__(self, alloc_coll_tiling, native_unit):
        assert isinstance(alloc_coll_tiling, CollTiling)
        assert isinstance(native_unit, CollUnit)
        self.live = False
        self.n_distributed_dims = 0
        self.alloc_coll_tiling = alloc_coll_tiling
        self.usage_coll_tiling = None
        self.native_unit = native_unit
        self.codegen_smem = None


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
    ctype = s.type.basetype().ctype()
    const_shape = type_const_shape(s.type, "SMEM allocation", s.name, s.srcinfo)
    return SmemConfigInputs(ctype, const_shape, s.srcinfo, s.mem)


@memwin_template
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

    return Impl


@dataclass(slots=True)
class ArriveAwaitInfo:
    # Information from the first such statement encountered.
    # Others need to be checked as matching.
    actor_kind: ActorKind
    coll_tiling: CollTiling
    sync_stmts: List[LoopIR.SyncStmt]

    def get_srcinfo(self):
        # Matches BarrierScan.get_srcinfo()
        return self.sync_stmts[0].srcinfo if self.sync_stmts else "<no SyncStmt seen>"

    def get_coll_tiling(self):
        # Matches BarrierScan.get_coll_tiling
        return self.coll_tiling

    def expect_prologue_sync_of(self, actor_kind):
        self._expect_impl("prologue", self.prologue_sync_of, actor_kind)

    def expect_epilogue_sync_of(self, actor_kind):
        self._expect_impl("epilogue", self.epilogue_sync_of, actor_kind)

    def _expect_impl(self, name, actual, expected):
        if actual != expected:
            s = self.sync_stmt
            raise ValueError(
                f"{s.srcinfo}: misplaced {s}; must be "
                f"{name} sync of CudaAsync({expected}) block"
            )


class BarrierScan(object):
    """Helper object for collecting all usages of one barrier symbol.

    Call the check() callback at the end to check basic common requirements:

        * At least one Arrive stmt and at least one Await stmt
        * At least one ReverseArrive stmt iff at least one ReverseAwait stmt
        * ReverseArrive/ReverseAwait only allowed for cuda_mbarrier
        * All stmts of one kind must have the same actor kind
        * TODO all stmts of one kind must use the same collective tiling

    """

    __slots__ = [
        "barrier_decl_body",
        "barrier_name",
        "barrier_srcinfo",
        "barrier_coll_tiling",
        "barrier_type",
        "Arrive",
        "Await",
        "ReverseArrive",
        "ReverseAwait",
    ]

    # LoopIR body containing declaration of barrier.
    barrier_decl_body: List[LoopIR.stmt]
    barrier_name: Sym
    barrier_srcinfo: SrcInfo
    barrier_coll_tiling: CollTiling
    barrier_type: Optional[LoopIR.type]  # None iff is Fence
    # Info on [Reverse]Arrive/Await statements encountered
    Arrive: Optional[ArriveAwaitInfo]
    Await: Optional[ArriveAwaitInfo]
    ReverseArrive: Optional[ArriveAwaitInfo]
    ReverseAwait: Optional[ArriveAwaitInfo]

    def __init__(self, scanner: SubtreeScan, s: LoopIR.stmt, body: List[LoopIR.stmt]):
        if isinstance(s, LoopIR.Alloc):
            # Barrier alloc
            self.barrier_type = s.type
            self.barrier_name = s.name
            assert self.barrier_type.is_barrier()
        else:
            # Fence
            assert isinstance(s, LoopIR.SyncStmt)
            assert not s.sync_type.is_split()
            self.barrier_type = None
            self.barrier_name = s.bar
        assert any(s is s1 for s1 in body)
        self.barrier_decl_body = body
        self.barrier_srcinfo = s.srcinfo
        self.barrier_coll_tiling = scanner._coll_tiling
        self.Arrive = None
        self.Await = None
        self.ReverseArrive = None
        self.ReverseAwait = None

    def get_srcinfo(self):
        return self.barrier_srcinfo

    def get_coll_tiling(self):
        return self.barrier_coll_tiling

    def inspect_sync_stmt(
        self,
        s: LoopIR.SyncStmt,
        coll_tiling: CollTiling,
    ):
        assert s.bar is self.barrier_name

        assert isinstance(s, LoopIR.SyncStmt)
        sync_type: SyncType = s.sync_type

        if not sync_type.is_split():
            # Fence, it's like an arrive+await
            # XXX this is maybe too clever, but avoids some special casing
            assert self.Arrive is None
            assert self.Await is None
            assert not self.is_split()
            self.Arrive = ArriveAwaitInfo(sync_type.first_actor_kind, coll_tiling, [s])
            self.Await = ArriveAwaitInfo(sync_type.second_actor_kind, coll_tiling, [s])
        else:
            # Set or modify self.Arrive, self.Await, self.ReverseArrive,
            # or self.ReverseAwait. Enforce identical actor kind
            # and (TODO) collective tiling
            assert self.is_split()
            if sync_type.is_arrive():
                actor_kind = sync_type.first_actor_kind
                attr = "ReverseArrive" if sync_type.is_reversed else "Arrive"
            else:
                assert sync_type.is_await()
                actor_kind = sync_type.second_actor_kind
                attr = "ReverseAwait" if sync_type.is_reversed else "Await"
            assert isinstance(actor_kind, ActorKind)
            info = getattr(self, attr)
            if info is None:
                setattr(self, attr, ArriveAwaitInfo(actor_kind, coll_tiling, [s]))
            else:
                assert isinstance(info, ArriveAwaitInfo)
                canon = info.sync_stmts[0]
                if info.actor_kind != actor_kind:
                    raise ValueError(
                        f"{s.srcinfo}: {s} mismatches actor kind of {canon} at {canon.srcinfo}"
                    )
                # TODO coll_tiling check
                info.sync_stmts.append(s)

    def is_split(self):
        is_none = self.barrier_type is None
        assert is_none or self.barrier_type.is_barrier()
        return not is_none

    def has_reverse(self):
        return self.ReverseArrive is not None

    def check(self):
        srcinfo = self.barrier_srcinfo
        name = self.barrier_name

        def reverse_allowed_check(info, stmt_name):
            # NB it may be argued that we should not be handling the
            # cuda_mbarrier special case in this common code, but this avoids
            # every other barrier type having to check this themselves.
            if not isinstance(self.barrier_type, LoopIR.CudaMbarrier):
                if info is not None:
                    sus = info.sync_stmts[0]
                    raise TypeError(
                        f"{sus.get_srcinfo()}: {stmt_name} not allowed for {name} of type {self.barrier_type}"
                    )

        reverse_allowed_check(self.ReverseArrive, "ReverseArrive")
        reverse_allowed_check(self.ReverseAwait, "ReverseAwait")

        if self.Arrive is None:
            raise ValueError(f"{srcinfo}: missing Arrive({name}) statement")
        if self.Await is None:
            raise ValueError(f"{srcinfo}: missing Await({name}) statement")
        if self.ReverseArrive is None and self.ReverseAwait is not None:
            raise ValueError(f"{srcinfo}: missing ReverseArrive({name}) statement")
        if self.ReverseAwait is None and self.ReverseArrive is not None:
            raise ValueError(f"{srcinfo}: missing ReverseAwait({name}) statement")

        # TODO check arrive/await pairing


class LoweredBarrierType(Enum):
    garden_variety_fence = auto()
    wgmma_fence = auto()
    mbarrier = auto()
    Sm80_commit_group = auto()
    tma_to_gmem_commit_group = auto()
    wgmma_commit_group = auto()


@dataclass(slots=True)
class LoweredPrologueSync:
    actor_kind: ActorKind
    lines: List[str]


@dataclass(slots=True)
class LoweredEpilogueSync:
    actor_kind: ActorKind
    lines: List[str]


class CudaLoweredBarrier:
    __slots__ = [
        "codegen",
        "SyncState_lines",
        "solitary",
        "type_enum",
        "c_Arrive_mbarrier",
        "c_ReverseArrive_mbarrier",
    ]

    # Lower SyncStmt to lines of C++ code (List[str])
    # (you may assume the SyncStmt uses this lowered barrier)
    # If the sync must appear as the prologue/epilogue sync
    # of a CudaAsync(A) block, warp the lines with
    # LoweredPrologueSync(A, lines) or LoweredEpilogueSync(A, lines)
    codegen: Callable[[SyncType], object]

    # Added to SyncState struct
    SyncState_lines: List[str]

    # If set, two barrier objects of the same type_enum (in Exo code)
    # cannot be live at the same time.
    solitary: bool

    # More specific than the LoopIR types (specialized by actor kind).
    # Also applies to Fence(...), which has no associated barrier object.
    type_enum: LoweredBarrierType

    # If applicable, syntax for getting the mbarrier used for
    # an Arrive/ReverseArrive
    c_Arrive_mbarrier: str
    c_ReverseArrive_mbarrier: str

    def __init__(self, solitary, type_enum):
        self.codegen = None
        self.SyncState_lines = []
        self.solitary = solitary
        self.type_enum = type_enum
        assert isinstance(type_enum, LoweredBarrierType)


h_snippet_fmt = """\
struct exo_CudaDeviceArgs{N}_{proc};

#ifdef __CUDACC__
__global__ void exo_deviceFunction{N}_{proc}(__grid_constant__ const struct exo_CudaDeviceArgs{N}_{proc} exo_deviceArgs);
#endif
void exo_cudaLaunch{N}_{proc}(cudaStream_t exo_cudaStream, struct exo_CudaDeviceArgs{N}_{proc} exo_deviceArgs);
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
  exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceTask(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task);
}};

inline void
exo_Cuda{N}_{proc}::exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs)
{{
  const unsigned exo_gridDim = {gridDim};
  cudaFuncSetAttribute(exo_deviceFunction{N}_{proc}, cudaFuncAttributeMaxDynamicSharedMemorySize, exo_smemBytes);
  exo_deviceFunction{N}_{proc}<<<exo_gridDim, exo_blockDim, exo_smemBytes, exo_cudaStream>>>(exo_deviceArgs);
}}

__device__ __forceinline__ void
exo_Cuda{N}_{proc}::exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs)
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
  exo_Cuda{N}_{proc}::exo_deviceSetup(exo_smem, exo_deviceArgs);
  exo_Cuda{N}_{proc}::exo_deviceMainLoop(exo_smem, exo_deviceArgs);
}}

void
exo_cudaLaunch{N}_{proc}(cudaStream_t exo_cudaStream, struct exo_CudaDeviceArgs{N}_{proc} exo_deviceArgs)
{{
  exo_Cuda{N}_{proc}::exo_cudaLaunch(exo_cudaStream, exo_deviceArgs);
}}
"""

device_main_loop_prefix_fmt = """__device__ __forceinline__ void
exo_Cuda{N}_{proc}::exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs)
{{
  exo_SyncState exo_syncState{{}};
  unsigned exo_taskIndex = 0;"""

device_task_prefix_fmt = """__device__ __forceinline__ void
exo_Cuda{N}_{proc}::exo_deviceTask(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task)
{{
  namespace exo_CudaUtil = exo_CudaUtil_{lib_name};"""

cuda_launch_fmt = """exo_cudaLaunch{N}_{proc}(exo_cudaStream, (struct exo_CudaDeviceArgs{N}_{proc}) {{ {device_args} }});"""

task_launch_fmt = """if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) exo_deviceTask(exo_smem, exo_syncState, exo_deviceArgs, (struct exo_Task) {{ {task_args} }});"""

# Paste this into the C header (.h) if any proc uses cuda.
h_snippet_for_cuda = r"""
#ifndef EXO_CUDA_HEADER_COMMON
#define EXO_CUDA_HEADER_COMMON
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef __CUDACC__
#define EXO_CUDA_INLINE __device__ __forceinline__
EXO_CUDA_INLINE unsigned exo_smemU32(const void* smem_ptr)
{
    return (unsigned)__cvta_generic_to_shared(smem_ptr);
}
#endif
#endif

#ifndef EXO_CUDA_STREAM_GUARD
#define EXO_CUDA_STREAM_GUARD
static const cudaStream_t exo_cudaStream = 0;
#endif
"""
