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
class cudaMemcpyAsync_htod_1f16(cudaMemcpyAsync_base):
    def behavior(n: size, dst: [cu_f16][n] @ CudaGmemLinear, src: [cu_f16][n] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, n):
            dst[i] = src[i]

    def instance(self):
        self.instance_impl("2 * {n}", htod=True)


@instr
class cudaMemcpyAsync_dtoh_1f16(cudaMemcpyAsync_base):
    def behavior(n: size, dst: [cu_f16][n] @ DRAM, src: [cu_f16][n] @ CudaGmemLinear):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, n):
            dst[i] = src[i]

    def instance(self):
        self.instance_impl("2 * {n}", htod=False)


@instr
class cudaMemsetAsync0_1f16(cudaMemsetAsync0_base):
    def behavior(n: size, dst: [cu_f16][n] @ CudaGmemLinear):
        assert stride(dst, 0) == 1
        for i in seq(0, n):
            dst[i] = 0

    def instance(self):
        self.instance_impl("2 * {n}")


@instr
class cudaMemcpyAsync_htod_1bf16(cudaMemcpyAsync_base):
    def behavior(n: size, dst: [cu_bf16][n] @ CudaGmemLinear, src: [cu_bf16][n] @ DRAM):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, n):
            dst[i] = src[i]

    def instance(self):
        self.instance_impl("2 * {n}", htod=True)


@instr
class cudaMemcpyAsync_dtoh_1bf16(cudaMemcpyAsync_base):
    def behavior(n: size, dst: [cu_bf16][n] @ DRAM, src: [cu_bf16][n] @ CudaGmemLinear):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, n):
            dst[i] = src[i]

    def instance(self):
        self.instance_impl("2 * {n}", htod=False)


@instr
class cudaMemsetAsync0_1bf16(cudaMemsetAsync0_base):
    def behavior(n: size, dst: [cu_bf16][n] @ CudaGmemLinear):
        assert stride(dst, 0) == 1
        for i in seq(0, n):
            dst[i] = 0

    def instance(self):
        self.instance_impl("2 * {n}")


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
class cudaMemcpyAsync_htod_2f16(cudaMemcpyAsync_base):
    def behavior(
        M: size,
        N: size,
        dst: [cu_f16][M, N] @ CudaGmemLinear,
        src: [cu_f16][M, N] @ DRAM,
    ):
        # assert stride(dst, 0) == N
        # assert stride(src, 0) == N
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = src[m, n]

    def instance(self):
        self.instance_impl("2 * {M} * {N}", htod=True)


@instr
class cudaMemcpyAsync_dtoh_2f16(cudaMemcpyAsync_base):
    def behavior(
        M: size,
        N: size,
        dst: [cu_f16][M, N] @ DRAM,
        src: [cu_f16][M, N] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == N
        # assert stride(src, 0) == N
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = src[m, n]

    def instance(self):
        self.instance_impl("2 * {M} * {N}", htod=False)


@instr
class cudaMemsetAsync0_2f16(cudaMemsetAsync0_base):
    def behavior(
        M: size,
        N: size,
        dst: [cu_f16][M, N] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == N
        assert stride(dst, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = 0

    def instance(self):
        self.instance_impl("2 * {M} * {N}")


@instr
class cudaMemcpyAsync_htod_2bf16(cudaMemcpyAsync_base):
    def behavior(
        M: size,
        N: size,
        dst: [cu_bf16][M, N] @ CudaGmemLinear,
        src: [cu_bf16][M, N] @ DRAM,
    ):
        # assert stride(dst, 0) == N
        # assert stride(src, 0) == N
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = src[m, n]

    def instance(self):
        self.instance_impl("2 * {M} * {N}", htod=True)


@instr
class cudaMemcpyAsync_dtoh_2bf16(cudaMemcpyAsync_base):
    def behavior(
        M: size,
        N: size,
        dst: [cu_bf16][M, N] @ DRAM,
        src: [cu_bf16][M, N] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == N
        # assert stride(src, 0) == N
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = src[m, n]

    def instance(self):
        self.instance_impl("2 * {M} * {N}", htod=False)


@instr
class cudaMemsetAsync0_2bf16(cudaMemsetAsync0_base):
    def behavior(
        M: size,
        N: size,
        dst: [cu_bf16][M, N] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == N
        assert stride(dst, 1) == 1
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = 0

    def instance(self):
        self.instance_impl("2 * {M} * {N}")


# TODO we really need to write a script for generating all possibilities.
@instr
class cudaMemsetAsync0_3f32(cudaMemsetAsync0_base):
    def behavior(
        L: size,
        M: size,
        N: size,
        dst: [f32][L, M, N] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == M * N
        # assert stride(dst, 1) == N
        assert stride(dst, 2) == 1
        for batch in seq(0, L):
            for m in seq(0, M):
                for n in seq(0, N):
                    dst[batch, m, n] = 0

    def instance(self):
        self.instance_impl("4 * {L} * {M} * {N}")


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


class cuda_packed_load_base(InstrInfo):
    __slots__ = []

    def instance(self):
        self.instr_tl = cuda_in_order_instr

    def codegen(self, args):
        dst_ptr = args.dst.index_ptr(ptx_data=True)
        src_ptr = args.src.index_ptr()
        return [f"memcpy({dst_ptr}, {src_ptr}, 4);"]


@instr
class cuda_packed_load_f32(cuda_packed_load_base):
    def behavior(
        dst: [f32][1] @ CudaRmemPacked32, src: [f32][1] @ CudaBasicDeviceVisible
    ):
        for i in seq(0, 1):
            dst[i] = src[i]


@instr
class cuda_packed_load_i32(cuda_packed_load_base):
    def behavior(
        dst: [i32][1] @ CudaRmemPacked32, src: [i32][1] @ CudaBasicDeviceVisible
    ):
        for i in seq(0, 1):
            dst[i] = src[i]


@instr
class cuda_packed_load_f16(cuda_packed_load_base):
    def behavior(
        dst: [cu_f16][2] @ CudaRmemPacked32, src: [cu_f16][2] @ CudaBasicDeviceVisible
    ):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 2):
            dst[i] = src[i]


@instr
class cuda_packed_load_bf16(cuda_packed_load_base):
    def behavior(
        dst: [cu_bf16][2] @ CudaRmemPacked32, src: [cu_bf16][2] @ CudaBasicDeviceVisible
    ):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, 2):
            dst[i] = src[i]
