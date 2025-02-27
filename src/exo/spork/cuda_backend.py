from typing import Dict, Optional, Type

from ..core.prelude import Sym
from ..core.LoopIR import LoopIR, T, LoopIR_Do, LoopIR_Rewrite

from .async_config import CudaDeviceFunction, CudaAsync
from .base_with_context import is_if_holding_with, ExtWithContext
from .loop_modes import CudaTasks, CudaThreads, Seq, seq
from .spork_lowering_ctx import SporkLoweringCtx
from .sync_types import SyncType


def loopir_lower_cuda(s, ctx: SporkLoweringCtx):
    scan = SubtreeScan(s, ctx.proc_name(), ctx.kernel_index())
    return SubtreeRewrite(s, scan).result()


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

  struct exo_Task
  {{
{task_struct_body}
  }};

  static constexpr unsigned exo_smemBytes = {smem_bytes};

  static void
  exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceTask(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, const exo_Task& exo_task);
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
exo_Cuda{N}_{proc}::exo_deviceTask(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, const exo_Task& exo_task)
{{"""

cuda_launch_fmt = """exo_cudaLaunch{N}_{proc}(exo_cudaStream, (struct exo_CudaDeviceArgs{N}_{proc}) {braced_device_args});"""

task_launch_fmt = """if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) exo_deviceTask(exo_smem, exo_deviceArgs, exo_task);"""


class SubtreeScan(LoopIR_Do):
    __slots__ = [
        "sym_advice",
        "blockDim",
        "clusterDim",
        "fmt_dict",
        "task_loop_depth",
    ]

    sym_advice: Dict[Sym, object]
    blockDim: int
    clusterDim: int
    fmt_dict: Dict
    task_loop_depth: int

    def __init__(self, s, proc_name, cuda_kernel_index):
        assert is_if_holding_with(s, LoopIR)
        cuda_device_function = s.cond.val
        assert isinstance(cuda_device_function, CudaDeviceFunction)

        self.sym_advice = {}
        self.blockDim = cuda_device_function.blockDim
        self.clusterDim = cuda_device_function.clusterDim
        self.fmt_dict = {
            "proc": proc_name,
            "N": cuda_kernel_index,
            "gridDim": "48",  # TODO
            "blockDim": self.blockDim,
            "clusterDim": self.clusterDim,
            "braced_device_args": "{M, N, K, A, B, C}",  # TODO
            "smem_bytes": 0,  # TODO
            "task_struct_body": "    int_fast32_t m2, n2;",  # TODO
            "device_struct_body": """  int_fast32_t M, N, K;
  const float* A;
  const float* B;
  float* C;""",  # TODO
        }

        self.task_loop_depth = 2  # TODO


class SubtreeRewrite(LoopIR_Rewrite):
    __slots__ = [
        "sym_advice",
        "fmt_dict",
        "_result",
    ]

    def __init__(self, s, scan: SubtreeScan):
        self.sym_advice = scan.sym_advice
        fmt_dict = scan.fmt_dict
        self.fmt_dict = fmt_dict

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
        )
        task_context = ExtWithContext(
            format(task_launch_fmt), format(device_task_prefix_fmt), "}", "cuh", {}
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
        # The Fence(cpu_cuda_all, cuda_all) is eliminated since
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
