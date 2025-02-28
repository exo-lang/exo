from __future__ import annotations

from typing import Dict, Optional, Type

from ..core.prelude import Sym
from ..core.LoopIR import LoopIR, T, LoopIR_Do, LoopIR_Rewrite

from .actor_kinds import cpu_cuda_api, cuda_api
from .async_config import CudaDeviceFunction, CudaAsync
from .base_with_context import is_if_holding_with, ExtWithContext
from .coll_algebra import (
    CollParam,
    clusterDim_param,
    blockDim_param,
    CollIndexExpr,
    CollTiling,
    CollLoweringAdvice,
)
from .loop_modes import CudaTasks, CudaThreads, Seq, seq, _CodegenPar
from .sync_types import SyncType


# We use the reserved exo_ prefix everywhere, but we still have to reserve
# CUDA builtins we have no control over.
reserved_names = {"gridDim", "blockDim", "blockIdx", "threadIdx"}


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
        #
        "_syms_needed",
        "_stmt_stack",
        "_coll_params",
        "_coll_tiling",
    ]

    sym_advice: Dict[Sym, object]
    blockDim: int
    clusterDim: int
    fmt_dict: Dict
    task_loop_depth: int
    task_iter_syms: List[Sym]
    _syms_needed: Set[Sym]
    _stmt_stack: List[LoopIR.stmt]
    _coll_params: Dict[CollParam, int]
    _coll_tiling: CollTiling

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
            "gridDim": "48",  # TODO
            "blockDim": self.blockDim,
            "clusterDim": self.clusterDim,
            "smem_bytes": 0,  # TODO
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
        self._coll_params = {
            clusterDim_param: self.clusterDim,
            blockDim_param: self.blockDim,
        }
        # TODO need to fix coll_algebra to remove this special casing for clusterDim = 1
        if self.clusterDim == 1:
            tlc_offset = (0,)
            tlc_box = (self.blockDim,)
            intra_box_exprs = (CollIndexExpr("threadIdx.x"),)
        else:
            tlc_offset = (0, 0)
            tlc_box = (self.clusterDim, self.blockDim)
            cta_expr = CollIndexExpr("blockIdx.x") % self.clusterDim
            thread_expr = CollIndexExpr("threadIdx.x")
            intra_box_exprs = (cta_expr, CollIndexExpr("threadIdx.x"))
        self._coll_tiling = CollTiling(
            None, tlc_box, tlc_box, tlc_offset, tlc_box, intra_box_exprs
        )
        self.do_stmts(s.body)

        # Prepare the device args struct
        # These are all the syms that appear in the subtree that were
        # defined by the outside (CPU function) environment.
        self.device_args_syms = []
        for sym in self._syms_needed:
            try:
                cpu_nm = ctx.sym_c_name(sym)
            except KeyError:
                continue
            self.device_args_syms.append(sym)
        self.device_args_syms.sort(key=lambda s: s.id_number())

        device_args_values = []
        for sym in self.device_args_syms:
            # TODO grid constant parameters must be const, pass by value.
            e = LoopIR.Read(sym, [], ctx.sym_type(sym), s.srcinfo)
            device_args_values.extend(ctx.fnarg_values(e, ctx.is_const(sym), False))
        self.fmt_dict["device_args"] = ", ".join(device_args_values)

        device_args_decls = []
        device_args_comments = []
        for sym in self.device_args_syms:
            # We don't mangle syms in the device args struct
            # They will appear as exo_deviceArgs.{str(sym)} in CUDA code.
            # TODO grid constant scalars must be pass-by-value
            mem = ctx.sym_mem(sym)
            fnarg = LoopIR.fnarg(sym, ctx.sym_type(sym), mem, s.srcinfo)
            ctx.append_fnarg_decl(
                fnarg, str(sym), device_args_decls, device_args_comments
            )
        device_args_struct_lines = []
        for i in range(len(device_args_decls)):
            device_args_struct_lines.append(
                f"    {device_args_decls[i]};  // {device_args_comments[i]}"
            )
        self.fmt_dict["device_args_struct_body"] = "\n".join(device_args_struct_lines)

    def do_s(self, s):
        old_coll_tiling = self._coll_tiling
        self._stmt_stack.append(s)

        self.apply_s(s)
        super().do_s(s)

        self._stmt_stack.pop()
        self._coll_tiling = old_coll_tiling

    def do_e(self, e):
        super().do_e(e)
        if isinstance(e, (LoopIR.Read, LoopIR.WindowExpr, LoopIR.StrideExpr)):
            self._syms_needed.add(e.name)
        else:
            assert not hasattr(e, "name")

    def apply_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            self._syms_needed.add(s.name)
        elif not isinstance(s, (LoopIR.WindowStmt, LoopIR.Alloc, LoopIR.Free)):
            assert not hasattr(s, "name")

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
                    f"{s.srcinfo}: unexpected loop mode {s.loop_mode.loop_mode_name()}"
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

        self._coll_tiling, advice = self._coll_tiling.tiled(
            s.loop_mode.unit, hi_int, self._coll_params
        )

        # We will advise replacing the loop mode with _CodegenPar
        self.sym_advice[s.iter] = _CodegenPar(
            advice.coll_index.codegen(), (advice.lo, advice.hi)
        )


class SubtreeRewrite(LoopIR_Rewrite):
    __slots__ = [
        "sym_advice",
        "fmt_dict",
        "_result",
    ]

    def __init__(self, s, scan: SubtreeScan, ctx: SporkLoweringCtx):
        self.sym_advice = scan.sym_advice
        fmt_dict = scan.fmt_dict
        self.fmt_dict = fmt_dict

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
        # exo_deviceMainLoop() and exo_deviceTask(), respectively.
        # We arbitrarily choose one to additionally emit the needed
        # snippets into the .h, .cuh, .cu files.
        format = lambda fmt_string: fmt_string.format(**fmt_dict)
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
            set(),  # TODO force_const
            set(),  # TODO scalar_refs
        )
        task_context = ExtWithContext(
            format(task_launch_fmt),
            format(device_task_prefix_fmt),
            "}",
            "cuh",
            {},
            reserved_names,
            task_force_names,
            set(),  # TODO force_const
            set(),  # TODO scalar_refs
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

        # Finally wrap the task loops into exo_deviceMainLoop
        # The Fence(cpu_cuda_api, cuda_api) is eliminated since
        # its effect comes for free from CUDA kernel launch.
        self._result = wrap_with_context(
            main_loop_context, [task_loop], task_loop.srcinfo
        )

    def result(self):
        assert is_if_holding_with(self._result, LoopIR)
        return self._result

    def map_s(self, s):
        s_rewrite = None

        if isinstance(s, LoopIR.For):
            # Replace CudaThreads loop with _CodegenPar loop that the
            # scanner has prepared.
            if isinstance(s.loop_mode, CudaThreads):
                new_loop_mode = self.sym_advice[s.iter]
                s_rewrite = s.update(loop_mode=new_loop_mode)
            else:
                assert isinstance(s.loop_mode, (Seq, _CodegenPar))

        if s_rewrite is None:
            return super().map_s(s)
        else:
            super_rewritten = super().map_s(s_rewrite)
            if super_rewritten is None:
                return [s_rewrite]
            else:
                return super_rewritten


# Paste this into the C header (.h) if any proc uses cuda.
# TODO this should be minimal.
h_snippet_for_cuda = """\
#include <cuda.h>
#include <cuda_runtime.h>
#ifndef EXO_CUDA_STREAM_GUARD
#define EXO_CUDA_STREAM_GUARD
static const cudaStream_t exo_cudaStream = 0;
#endif
"""

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
