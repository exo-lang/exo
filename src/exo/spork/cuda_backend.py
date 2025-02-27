from __future__ import annotations

from typing import Dict, Optional, Type

from ..core.prelude import Sym
from ..core.LoopIR import LoopIR, T, LoopIR_Do, LoopIR_Rewrite

from .actor_kinds import cpu_cuda_api, cuda_api
from .async_config import CudaDeviceFunction, CudaAsync
from .base_with_context import is_if_holding_with, ExtWithContext
from .loop_modes import CudaTasks, CudaThreads, Seq, seq
from .sync_types import SyncType


# We use the reserved exo_ prefix everywhere, but we still have to reserve
# CUDA builtins we have no control over.
reserved_names = {"gridDim", "blockDim", "blockIdx", "threadIdx"}


def loopir_lower_cuda(s, ctx: SporkLoweringCtx):
    scan = SubtreeScan(s, ctx)
    return SubtreeRewrite(s, scan, ctx).result()


class SubtreeScan(LoopIR_Do):
    __slots__ = [
        "sym_advice",
        "blockDim",
        "clusterDim",
        "fmt_dict",
        "task_loop_depth",
        "task_iter_syms",
    ]

    sym_advice: Dict[Sym, object]
    blockDim: int
    clusterDim: int
    fmt_dict: Dict
    task_loop_depth: int
    task_iter_syms: List[Sym]

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
            "device_args": "M, N, K, A, B, C",  # TODO
            "smem_bytes": 0,  # TODO
            "device_struct_body": """  int_fast32_t M, N, K;
  const float* A;
  const float* B;
  float* C;""",  # TODO
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
        self.fmt_dict["task_struct_decls"] = "\n".join(
            f"    int_fast32_t {str(sym)};" for sym in self.task_iter_syms
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
        # TODO don't hard-wire
        for sym in ctx._compiler.env:
            if str(sym) in "MNKABC":
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
{device_struct_body}
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
{task_struct_decls}
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
