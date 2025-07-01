# Public module for using CUDA with Exo
# See Sm80.py and Sm90.py for A100/H100-specific functionality

from __future__ import annotations
from ..API import instr

# Currently we import from the exo.spork directory,
# which users shouldn't import directly.
from ..spork.timelines import (
    cpu_in_order_instr,
    cuda_temporal,
    cuda_in_order,
    cuda_in_order_instr,
    cuda_sync_rmem_usage,
    cuda_ram_usage,
)
from ..spork.async_config import CudaDeviceFunction, CudaAsync
from ..spork.coll_algebra import (
    cuda_thread,
    cuda_quadpair,
    cuda_warp,
    cuda_warpgroup,
    cuda_cluster,
    cuda_cta_in_cluster,
    cuda_warp_in_cluster,
)
from ..spork.cuda_memory import (
    scalar_bits,
    CudaBasicDeviceVisible,
    SmemConfig,
    SmemConfigInputs,
    CudaBasicSmem,
    CudaDeviceVisibleLinear,
    CudaGridConstant,
    CudaGmemLinear,
    CudaSmemLinear,
    CudaRmem,
    CudaEvent,
    CudaMbarrier,
    CudaCommitGroup,
    CudaClusterSync,
    DRAM,
)
from ..spork.coll_algebra import CollUnit, blockDim, clusterDim
from ..spork.cuda_warp_config import CudaWarpConfig
from ..spork.excut import InlinePtxGen
from ..spork.loop_modes import CudaTasks, CudaThreads
from ..spork.with_cuda_warps import CudaWarps

# TODO spork.sync_types, needed for scheduling

# XXX temporary cudaMemcpyAsync: we need this for testing for now.
# We need to do a better job of reasoning about CUDA synchronization for
# CUDA API calls from the CPU (as opposed to device-side code)
@instr(
    "cudaMemcpyAsync(&{dst_data}, &{src_data}, 4 * {n}, cudaMemcpyHostToDevice, exo_cudaStream);"
)
def cudaMemcpyAsync_htod_1f32(
    n: size, dst: [f32][n] @ CudaGmemLinear, src: [f32][n] @ DRAM
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, n):
        dst[i] = src[i]


@instr(
    "cudaMemcpyAsync(&{dst_data}, &{src_data}, 4 * {n}, cudaMemcpyDeviceToHost, exo_cudaStream);"
)
def cudaMemcpyAsync_dtoh_1f32(
    n: size, dst: [f32][n] @ DRAM, src: [f32][n] @ CudaGmemLinear
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, n):
        dst[i] = src[i]


@instr(
    "cudaMemcpyAsync(&{dst_data}, &{src_data}, 4 * {n}, cudaMemcpyHostToDevice, exo_cudaStream);"
)
def cudaMemcpyAsync_htod_1i32(
    n: size, dst: [i32][n] @ CudaGmemLinear, src: [i32][n] @ DRAM
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, n):
        dst[i] = src[i]


@instr(
    "cudaMemcpyAsync(&{dst_data}, &{src_data}, 4 * {n}, cudaMemcpyDeviceToHost, exo_cudaStream);"
)
def cudaMemcpyAsync_dtoh_1i32(
    n: size, dst: [i32][n] @ DRAM, src: [i32][n] @ CudaGmemLinear
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, n):
        dst[i] = src[i]


@instr(
    "cudaMemcpyAsync(&{dst_data}, &{src_data}, 4 * {M} * {N}, cudaMemcpyHostToDevice, exo_cudaStream);"
)
def cudaMemcpyAsync_htod_2f32(
    M: size, N: size, dst: [f32][M, N] @ CudaGmemLinear, src: [f32][M, N] @ DRAM
):
    assert stride(dst, 1) == 1  # TODO stride(..., 0)
    assert stride(src, 1) == 1
    for m in seq(0, M):
        for n in seq(0, N):
            dst[m, n] = src[m, n]


@instr(
    "cudaMemcpyAsync(&{dst_data}, &{src_data}, 4 * {M} * {N}, cudaMemcpyDeviceToHost, exo_cudaStream);"
)
def cudaMemcpyAsync_dtoh_2f32(
    M: size, N: size, dst: [f32][M, N] @ DRAM, src: [f32][M, N] @ CudaGmemLinear
):
    assert stride(dst, 1) == 1  # TODO stride(..., 0)
    assert stride(src, 1) == 1
    for m in seq(0, M):
        for n in seq(0, N):
            dst[m, n] = src[m, n]


@instr(
    "cudaMemcpyAsync(&{dst_data}, &{src_data}, 4 * {M} * {N}, cudaMemcpyHostToDevice, exo_cudaStream);"
)
def cudaMemcpyAsync_htod_2i32(
    M: size, N: size, dst: [i32][M, N] @ CudaGmemLinear, src: [i32][M, N] @ DRAM
):
    assert stride(dst, 1) == 1  # TODO stride(..., 0)
    assert stride(src, 1) == 1
    for m in seq(0, M):
        for n in seq(0, N):
            dst[m, n] = src[m, n]


@instr(
    "cudaMemcpyAsync(&{dst_data}, &{src_data}, 4 * {M} * {N}, cudaMemcpyDeviceToHost, exo_cudaStream);"
)
def cudaMemcpyAsync_dtoh_2i32(
    M: size, N: size, dst: [i32][M, N] @ DRAM, src: [i32][M, N] @ CudaGmemLinear
):
    assert stride(dst, 1) == 1  # TODO stride(..., 0)
    assert stride(src, 1) == 1
    for m in seq(0, M):
        for n in seq(0, N):
            dst[m, n] = src[m, n]
