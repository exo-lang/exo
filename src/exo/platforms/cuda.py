# Public module for using CUDA with Exo
# See Sm80.py and Sm90.py for A100/H100-specific functionality

from __future__ import annotations
from typing import List
from .cuda_fwd import *
from .cuda_warp_intrin import *

# TODO spork.sync_types, needed for scheduling

# XXX temporary cudaMemcpyAsync: we need this for testing for now.
class cudaMemcpyAsync_base:
    __slots__ = []

    def instance_impl(self, size_fmt: str, *, htod: bool):
        direction_arg = "cudaMemcpyHostToDevice" if htod else "cudaMemcpyDeviceToHost"
        self.instr_tl = cpu_cuda_stream_instr
        self.instr_format = [
            "cudaMemcpyAsync(&{dst_data}, &{src_data}, %s, %s, exo_cudaStream);"
            % (size_fmt, direction_arg)
        ]
        self.access_info["dst"].out_of_order = False
        self.access_info["src"].out_of_order = False


class cudaMemsetAsync0_base:
    __slots__ = []

    def instance_impl(self, size_fmt: str):
        self.instr_tl = cpu_cuda_stream_instr
        self.instr_format = [
            "cudaMemsetAsync(&{dst_data}, 0, %s, exo_cudaStream);" % size_fmt
        ]
        self.access_info["dst"].out_of_order = False


@instr
class cudaMemcpyAsync_htod_1f32(cudaMemcpyAsync_base):
    def behavior(n: size, dst: [f32][n] @ CudaGmemLinear, src: [f32][n] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, n):
            dst[i] = src[i]

    def instance(self):
        self.instance_impl("4 * {n}", htod=True)


@instr
class cudaMemcpyAsync_dtoh_1f32(cudaMemcpyAsync_base):
    def behavior(n: size, dst: [f32][n] @ DRAM, src: [f32][n] @ CudaGmemLinear):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, n):
            dst[i] = src[i]

    def instance(self):
        self.instance_impl("4 * {n}", htod=False)


@instr
class cudaMemsetAsync0_1f32(cudaMemsetAsync0_base):
    def behavior(n: size, dst: [f32][n] @ CudaGmemLinear):
        assert stride(dst, 0) == 1
        for i in seq(0, n):
            dst[i] = 0

    def instance(self):
        self.instance_impl("4 * {n}")


@instr
class cudaMemcpyAsync_htod_1i32(cudaMemcpyAsync_base):
    def behavior(n: size, dst: [i32][n] @ CudaGmemLinear, src: [i32][n] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, n):
            dst[i] = src[i]

    def instance(self):
        self.instance_impl("4 * {n}", htod=True)


@instr
class cudaMemcpyAsync_dtoh_1i32(cudaMemcpyAsync_base):
    def behavior(n: size, dst: [i32][n] @ DRAM, src: [i32][n] @ CudaGmemLinear):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, n):
            dst[i] = src[i]

    def instance(self):
        self.instance_impl("4 * {n}", htod=False)


@instr
class cudaMemsetAsync0_1i32(cudaMemsetAsync0_base):
    def behavior(n: size, dst: [i32][n] @ CudaGmemLinear):
        assert stride(dst, 0) == 1
        for i in seq(0, n):
            dst[i] = 0

    def instance(self):
        self.instance_impl("4 * {n}", htod=False)


@instr
class cudaMemcpyAsync_htod_2f32(cudaMemcpyAsync_base):
    def behavior(
        M: size, N: size, dst: [f32][M, N] @ CudaGmemLinear, src: [f32][M, N] @ DRAM
    ):
        # assert stride(dst, 0) == N
        # assert stride(src, 0) == N
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = src[m, n]

    def instance(self):
        self.instance_impl("4 * {M} * {N}", htod=True)


@instr
class cudaMemcpyAsync_dtoh_2f32(cudaMemcpyAsync_base):
    def behavior(
        M: size, N: size, dst: [f32][M, N] @ DRAM, src: [f32][M, N] @ CudaGmemLinear
    ):
        # assert stride(dst, 0) == N
        # assert stride(src, 0) == N
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = src[m, n]

    def instance(self):
        self.instance_impl("4 * {M} * {N}", htod=False)


@instr
class cudaMemsetAsync0_2f32(cudaMemsetAsync0_base):
    def behavior(
        M: size,
        N: size,
        dst: [f32][M, N] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == N
        assert stride(dst, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = 0

    def instance(self):
        self.instance_impl("4 * {M} * {N}")


@instr
class cudaMemcpyAsync_htod_2i32(cudaMemcpyAsync_base):
    def behavior(
        M: size, N: size, dst: [i32][M, N] @ CudaGmemLinear, src: [i32][M, N] @ DRAM
    ):
        # assert stride(dst, 0) == N
        # assert stride(src, 0) == N
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = src[m, n]

    def instance(self):
        self.instance_impl("4 * {M} * {N}", htod=True)


@instr
class cudaMemcpyAsync_dtoh_2i32(cudaMemcpyAsync_base):
    def behavior(
        M: size, N: size, dst: [i32][M, N] @ DRAM, src: [i32][M, N] @ CudaGmemLinear
    ):
        # assert stride(dst, 0) == N
        # assert stride(src, 0) == N
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = src[m, n]

    def instance(self):
        self.instance_impl("4 * {M} * {N}", htod=False)


@instr
class cudaMemsetAsync0_2i32(cudaMemsetAsync0_base):
    def behavior(
        M: size,
        N: size,
        dst: [i32][M, N] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == N
        assert stride(dst, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = 0

    def instance(self):
        self.instance_impl("4 * {M} * {N}", htod=True)
