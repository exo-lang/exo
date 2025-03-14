from __future__ import annotations

from math import prod
from typing import Dict, Optional, Type
from warnings import warn

from ..core.memory import MemGenError, memwin_template
from ..core.prelude import Sym, SrcInfo
from ..core.LoopIR import LoopIR, T, LoopIR_Do, LoopIR_Rewrite, ctype_bits

from .actor_kinds import cpu, cpu_cuda_api, cuda_api
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
        "sym_advice",
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
        "_stmt_stack",
        "_coll_env",
        "_coll_tiling",
        "_iter_coll_tiling",
    ]

    # We will have to substitute some LoopIR nodes in the SubtreeRewrite phase.
    # During the scan, for a node that needs to be rewritten, we will stash
    # needed info for the rewrite here, indexed by a relevant Sym.
    # The type of the info depends on the rewrite needed -- see corresponding
    # rewrite in SubtreeRewrite.
    #
    # Allocated Sym: AllocState
    # Index of cuda_threads loop: _CodegenPar
    sym_advice: Dict[Sym, object]

    blockDim: int
    clusterDim: int
    fmt_dict: Dict
    task_loop_depth: int
    task_iter_syms: List[Sym]
    _syms_needed: Set[Sym]
    _stmt_stack: List[LoopIR.stmt]
    _coll_env: Dict[CollParam, int]
    _coll_tiling: CollTiling
    # CollTiling created by the cuda_threads loop with the given iter
    _iter_coll_tiling: Dict[Sym, CollTiling]

    def __init__(self, s, ctx: SporkLoweringCtx):
        assert is_if_holding_with(s, LoopIR)
        cuda_device_function = s.cond.val
        assert isinstance(cuda_device_function, CudaDeviceFunction)

        self.sym_advice = {}
        self.blockDim = cuda_device_function.blockDim
        self.clusterDim = cuda_device_function.clusterDim
        self.fmt_dict = {
            "proc": ctx.proc_name(),
            "N": ctx.kernel_index(),
            "gridDim": 48 * cuda_device_function.blocks_per_sm,  # TODO
            "blockDim": self.blockDim,
            "clusterDim": self.clusterDim,
            "blocks_per_sm": cuda_device_function.blocks_per_sm,
        }

        # Validate top-level form of cuda kernel
        # Must be Fence(cpu_cuda_api, cuda_api) followed by
        # nest of 1+ cuda_tasks loops
        self.task_iter_syms = []
        task_iter_strs = set()
        valid_sync = False
        if len(s.body) >= 1:
            sync_stmt = s.body[0]
            if isinstance(sync_stmt, LoopIR.SyncStmt):
                if sync_stmt.sync_type.first_actor_kind == cpu_cuda_api:
                    if sync_stmt.sync_type.second_actor_kind == cuda_api:
                        valid_sync = True

        if not valid_sync:
            raise ValueError(
                f"{s.srcinfo}: expected Fence(cpu_cuda_api, cuda_api) at start of CudaDeviceFunction"
            )

        if len(s.body) != 2:
            raise ValueError(
                f"{s.srcinfo}: expected cuda_tasks loop alone following Fence"
            )

        self.task_loop_depth = 0
        task_loop_body = s.body[1:]
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
        # for the top-level collective (2D tile clusterDim x blockDim)
        self._syms_needed = set()
        self._stmt_stack = []
        self._coll_env = {
            clusterDim_param: self.clusterDim,
            blockDim_param: self.blockDim,
        }
        tlc_offset = (0, 0)
        tlc_box = (self.clusterDim, self.blockDim)
        cta_expr = CollIndexExpr("blockIdx.x") % self.clusterDim
        thread_expr = CollIndexExpr("threadIdx.x")
        intra_box_exprs = (cta_expr, CollIndexExpr("threadIdx.x"))
        self._coll_tiling = CollTiling(
            None, tlc_box, tlc_box, tlc_offset, tlc_box, intra_box_exprs
        )
        self._iter_coll_tiling = {}
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

    def do_s(self, s):
        # Save state
        old_coll_tiling = self._coll_tiling
        self._stmt_stack.append(s)

        # Modify state, then recurse with super()
        # (order is important so recursion sees updated state!)
        self.apply_s(s)
        super().do_s(s)

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

        if isinstance(s, LoopIR.For):
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
            if not issubclass(s.mem, CudaBasicDeviceVisible):
                raise TypeError(
                    f"{s.srcinfo}: For cuda code, memory type "
                    f"({s.mem.name()}) must subclass CudaBasicDeviceVisible"
                )
            native_unit = s.mem.native_unit()
            self.sym_advice[s.name] = AllocState(self._coll_tiling, native_unit)

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if (n_threads := self._coll_tiling.box_num_threads()) != 1:
                raise ValueError(
                    f"{s.srcinfo}: write must be executed by one "
                    f"thread only (current: {n_threads} threads)\n"
                    f"stmt: {s}"
                )

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
        self.sym_advice[s.iter] = _CodegenPar(
            advice.coll_index.codegen(), (advice.lo, advice.hi)
        )

    def apply_idx(self, node):
        """Do analysis for one usage of tensor in distributed memory"""
        if not node.idx:
            # XXX early exit needed for Reads that are not from tensors
            # (e.g. index variables), but could hide issues?
            return

        state: AllocState
        state = self.sym_advice.get(node.name)
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
                if next_coll_tiling.parent != cur_coll_tiling:
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
                if state.usage_coll_tiling != cur_coll_tiling:
                    raise ValueError("collective tiling mismatch")

        except ValueError as e:
            # TODO better error messages
            message = f"{node.srcinfo}: {node.name} distributed memory analysis failed (see chained exception)"
            raise MemGenError(message) from e


class SubtreeRewrite(LoopIR_Rewrite):
    __slots__ = [
        "sym_advice",
        "fmt_dict",
        "smem_offset",
        "smem_data_usage",
        "_result",
    ]

    def __init__(self, s, scan: SubtreeScan, ctx: SporkLoweringCtx):
        self.sym_advice = scan.sym_advice
        fmt_dict = scan.fmt_dict
        self.fmt_dict = fmt_dict

        # Prepare SMEM stack allocator
        self.smem_offset = 0
        self.smem_data_usage = 0

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

        assert len(s.body) == 2  # SyncStmt, task loop
        task_loop = rewrite_task_loop(s.body[1], scan.task_loop_depth)

        # ExoWithContext object for diverting lowered code into
        # exo_deviceMainLoop(), and putting the required strings
        # into the .cu, .cuh, .h files.
        # Only at this point do we know the SMEM usage of the kernel,
        # which we load into fmt_dict at the last moment.
        assert (
            self.smem_offset == 0
        ), "SMEM stack allocator should have returned to initial state"
        fmt_dict["smem_bytes"] = self.smem_data_usage  # TODO mbarrier
        main_loop_context = ExtWithContext(
            format(cuda_launch_fmt),
            format(device_main_loop_prefix_fmt),
            "}",
            "cuh",
            {
                "h": format(h_snippet_fmt),
                "cu": format(cu_snippet_fmt),
                "cuh": format(cuh_snippet_fmt),
            },
            reserved_names,
            main_loop_force_names,
            scan.grid_constant_syms,  # force_const
            scan.scalar_ref_syms,
        )

        # Finally wrap the task loops into exo_deviceMainLoop
        # The Fence(cpu_cuda_api, cuda_api) is eliminated since
        # its effect comes for free from CUDA kernel launch.
        self._result = wrap_with_context(
            main_loop_context, [task_loop], task_loop.srcinfo
        )

    def result(self):
        assert is_if_holding_with(self._result, LoopIR)
        return self._result

    def updated_stmt(self, s):
        if is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, CudaAsync):
                s = s.update(cond=LoopIR.Const(True, T.bool, s.srcinfo))
            else:
                raise TypeError(
                    f"{s.srcinfo}: unexpected with context type {type(ctx)} in CUDA device code"
                )
        elif isinstance(s, LoopIR.For):
            # Replace CudaThreads loop with _CodegenPar loop that the
            # scanner has prepared.
            if isinstance(s.loop_mode, CudaThreads):
                new_loop_mode = self.sym_advice[s.iter]
                s = s.update(loop_mode=new_loop_mode)
            else:
                assert isinstance(s.loop_mode, (Seq, _CodegenPar))

        elif isinstance(s, (LoopIR.Alloc, LoopIR.Free)):
            alloc_state = self.sym_advice[s.name]
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
                    # TODO consider scalar refs
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
                    offset = self.smem_offset

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
                    alloc_state.smem_offset_before = self.smem_offset
                    smem_bytes = element_bits // 8
                    for n in inputs.const_shape:
                        smem_bytes *= n
                    self.smem_offset = offset + smem_bytes
                    self.smem_data_usage = max(self.smem_offset, self.smem_data_usage)

                    # Wrap user-specified memory type with SMEM offset,
                    # C++ reference type.
                    assert isinstance(config.reftype, str)
                    mem = CodegenSmem(offset, config.reftype, s.mem)
                    alloc_state.codegen_smem = mem  # for rewriting Free, below
                else:
                    # Rewrite Free Memory type to match corresponding Alloc
                    # and restore stack allocator state
                    mem = alloc_state.codegen_smem
                    assert mem
                    assert alloc_state.smem_offset_before < self.smem_offset
                    self.smem_offset = alloc_state.smem_offset_before
                s = s.update(mem=mem)

        elif isinstance(s, idx_s_types):
            # Remove distributed dimensions for tensor indexing expression
            s = self.remove_distributed_idx(s)

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
        alloc_state = self.sym_advice.get(node.name)
        if isinstance(alloc_state, AllocState):
            assert isinstance(alloc_state, AllocState)
            n = alloc_state.n_distributed_dims
            if n > 0:
                return node.update(idx=node.idx[n:])
        return None


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
        "smem_offset_before",
    ]

    live: bool
    n_distributed_dims: int
    alloc_coll_tiling: CollTiling
    usage_coll_tilings: Optional[CollTiling]
    native_unit: CollUnit
    codegen_smem: Optional[Type[CudaBasicSmem]]
    smem_offset_before: Optional[int]

    def __init__(self, alloc_coll_tiling, native_unit):
        assert isinstance(alloc_coll_tiling, CollTiling)
        assert isinstance(native_unit, CollUnit)
        self.live = False
        self.n_distributed_dims = 0
        self.alloc_coll_tiling = alloc_coll_tiling
        self.usage_coll_tiling = None
        self.native_unit = native_unit
        self.codegen_smem = None
        self.smem_offset_before = None


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
def CodegenSmem(byte_offset, reftype, wrapped_smem_type):
    """When rewriting the subtree for the CUDA device function,
    wrap all SMEM memory types with this, which includes the
    exact byte offset for the allocation in the SMEM segment"""

    assert issubclass(wrapped_smem_type, CudaBasicSmem)

    class Impl(wrapped_smem_type):
        @classmethod
        def alloc(cls, new_name, prim_type, shape, srcinfo):
            # We call the wrapped alloc() method to allow the memory class to raise errors.
            wrapped_alloc = wrapped_smem_type.alloc(new_name, prim_type, shape, srcinfo)
            assert wrapped_alloc == ""
            return f"auto& {new_name} = reinterpret_cast<{reftype}>(exo_smem[{byte_offset}]);"

    return Impl


h_snippet_fmt = """\
struct exo_CudaDeviceArgs{N}_{proc}
{{
{device_args_struct_body}
}};

#ifdef __CUDACC__
__global__ void exo_deviceFunction{N}_{proc}(__grid_constant__ const struct exo_CudaDeviceArgs{N}_{proc} exo_deviceArgs);
#endif
void exo_cudaLaunch{N}_{proc}(cudaStream_t exo_cudaStream, struct exo_CudaDeviceArgs{N}_{proc} exo_deviceArgs);
"""

cuh_snippet_fmt = """\
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

  static void
  exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceTask(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task);
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
  // TODO setup
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
  unsigned exo_taskIndex = 0;"""

device_task_prefix_fmt = """__device__ __forceinline__ void
exo_Cuda{N}_{proc}::exo_deviceTask(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task)
{{"""

cuda_launch_fmt = """exo_cudaLaunch{N}_{proc}(exo_cudaStream, (struct exo_CudaDeviceArgs{N}_{proc}) {{ {device_args} }});"""

task_launch_fmt = """if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) exo_deviceTask(exo_smem, exo_deviceArgs, (struct exo_Task) {{ {task_args} }});"""

# Paste this into the C header (.h) if any proc uses cuda.
# TODO this should be minimal.
# cp.async and MMA stuff should be externalized
h_snippet_for_cuda = r"""
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef EXO_CUDA_STREAM_GUARD
#define EXO_CUDA_STREAM_GUARD
static const cudaStream_t exo_cudaStream = 0;
#endif

#ifdef __CUDACC__
#ifndef EXO_CUDA_GLOBAL_DEFS
#define EXO_CUDA_GLOBAL_DEFS

// XXX this is really hacky to fix extern C
}
#include <cuda/std/array>
extern "C" {

__device__ __forceinline__ unsigned exo_smemU32(const void* smem_ptr)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
}

__device__ __forceinline__ void exo_Sm80_cpAsync16B(void *smem_ptr, const void* gmem_ptr) {
    const int BYTES = 16;
    uint32_t smem = exo_smemU32(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;" ::"r"(smem),
        "l"(gmem_ptr),
        "n"(BYTES) : "memory");
}

inline __device__ void exo_Sm80_tmp_load_a(unsigned rmem[4], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
    const unsigned row_stride = element_strides[0];
    const unsigned col_stride = element_strides[1];
    const unsigned warp_lane = threadIdx.x % 32u;
    const float* gmem_thread_baseaddr = &gmem[warp_lane / 4u * row_stride + warp_lane % 4u * col_stride];
    rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
    rmem[1] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride]);
    rmem[2] = __float_as_uint(gmem_thread_baseaddr[4 * col_stride]);
    rmem[3] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride + 4 * col_stride]);
}

inline __device__ void exo_Sm80_tmp_load_b(unsigned rmem[2], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
    const unsigned row_stride = element_strides[0];
    const unsigned col_stride = element_strides[1];
    const unsigned warp_lane = threadIdx.x % 32u;
    const float* gmem_thread_baseaddr = &gmem[warp_lane % 4u * row_stride + warp_lane / 4u * col_stride];
    rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
    rmem[1] = __float_as_uint(gmem_thread_baseaddr[4 * row_stride]);
}

inline __device__ void exo_Sm80_tmp_store_d(float* gmem, const unsigned rmem[4], cuda::std::array<int_fast32_t, 2> element_strides)
{
    const unsigned row_stride = element_strides[0];
    const unsigned col_stride = element_strides[1];
    const unsigned warp_lane = threadIdx.x % 32u;
    float* gmem_thread_baseaddr = &gmem[(warp_lane / 4u) * row_stride + (warp_lane % 4u) * 2u * col_stride];
    gmem_thread_baseaddr[0] = __uint_as_float(rmem[0]);
    gmem_thread_baseaddr[col_stride] = __uint_as_float(rmem[1]);
    gmem_thread_baseaddr[8 * row_stride] = __uint_as_float(rmem[2]);
    gmem_thread_baseaddr[8 * row_stride + col_stride] = __uint_as_float(rmem[3]);
}

inline __device__ void exo_Sm80_tmp_zero_d(unsigned rmem[4])
{
    rmem[0] = 0;
    rmem[1] = 0;
    rmem[2] = 0;
    rmem[3] = 0;
}

inline __device__ void exo_Sm80_tmp_mma(unsigned d[4], const unsigned a[4], const unsigned b[2])
{
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n\t"
        "{%0,%1,%2,%3},\n\t"
        "{%4,%5,%6,%7},\n\t"
        "{%8,%9},\n\t"
        "{%10,%11,%12,%13};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
}

#endif
#endif
"""
