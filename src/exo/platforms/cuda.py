# Public module for using CUDA with Exo
# See Sm80.py and Sm90.py for A100/H100-specific functionality

from __future__ import annotations
from .cuda_fwd import *
from .cuda_warp_intrin import *

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
