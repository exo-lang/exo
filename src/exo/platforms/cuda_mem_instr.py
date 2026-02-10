from __future__ import annotations
from typing import List
from .cuda_fwd import *
import exo.scalars as scalars

__all__ = []  # append stuff to this to export it.


########################################################################
# Implementation details for cudaMemcpyAsync and cudaMemsetAsync(0)
########################################################################


def _new_cudaMemcpyAsync_base(dims: int, *, htod: bool):
    direction_arg = "cudaMemcpyHostToDevice" if htod else "cudaMemcpyDeviceToHost"

    class cudaMemcpyAsync_custom_base(InstrInfo):
        def instance(self):
            self.instr_tl = cpu_cuda_stream_instr
            self.access_info["dst"].out_of_order = False
            self.access_info["src"].out_of_order = False

        def codegen(self, args):
            dst_ptr = args.dst.index_ptr()
            src_ptr = args.src.index_ptr()
            scalar_info: scalars.ScalarInfo = args.dst.get_scalar_info()
            assert scalar_info == args.src.get_scalar_info()
            size_expr = " * ".join(str(getattr(args, f"size{i}")) for i in range(dims))
            return [
                "cudaMemcpyAsync(",
                f"  {dst_ptr},",
                f"  {src_ptr},",
                f"  {size_expr}{scalar_info.get_scale_bytes_suffix()},",
                f"  {direction_arg},",
                f"  exo_cudaStream);",
            ]

        valid_num_types = scalars.ScalarInfo.same()

    return cudaMemcpyAsync_custom_base


def _new_cudaMemsetAsync0_base(dims: int):
    class cudaMemsetAsync0_custom_base(InstrInfo):
        def instance(self):
            self.instr_tl = cpu_cuda_stream_instr
            self.access_info["dst"].out_of_order = False

        def codegen(self, args):
            dst_ptr = args.dst.index_ptr()
            scalar_info: scalars.ScalarInfo = args.dst.get_scalar_info()
            size_expr = " * ".join(str(getattr(args, f"size{i}")) for i in range(dims))
            return [
                "cudaMemsetAsync(",
                f"  {dst_ptr},",
                f"  0,",
                f"  {size_expr}{scalar_info.get_scale_bytes_suffix()},",
                f"  exo_cudaStream);",
            ]

        valid_num_types = scalars.ScalarInfo.same()

    return cudaMemsetAsync0_custom_base


########################################################################
# 1D cudaMemcpyAsync and cudaMemsetAsync(0)
########################################################################


@instr
class cudaMemcpyAsync_htod_1d(_new_cudaMemcpyAsync_base(1, htod=True)):
    def behavior(
        size0: size,
        dst: [R][size0] @ CudaGmemLinear,
        src: [R][size0] @ DRAM,
    ):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i0 in seq(0, size0):
            dst[i0] = src[i0]


@instr
class cudaMemcpyAsync_dtoh_1d(_new_cudaMemcpyAsync_base(1, htod=False)):
    def behavior(
        size0: size,
        dst: [R][size0] @ DRAM,
        src: [R][size0] @ CudaGmemLinear,
    ):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i0 in seq(0, size0):
            dst[i0] = src[i0]


@instr
class cudaMemsetAsync0_1d(_new_cudaMemsetAsync0_base(1)):
    def behavior(
        size0: size,
        dst: [R][size0] @ CudaGmemLinear,
    ):
        assert stride(dst, 0) == 1
        for i0 in seq(0, size0):
            dst[i0] = 0


__all__.append("cudaMemcpyAsync_htod_1d")
__all__.append("cudaMemcpyAsync_dtoh_1d")
__all__.append("cudaMemsetAsync0_1d")


########################################################################
# 2D cudaMemcpyAsync and cudaMemsetAsync(0)
########################################################################


@instr
class cudaMemcpyAsync_htod_2d(_new_cudaMemcpyAsync_base(2, htod=True)):
    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1] @ CudaGmemLinear,
        src: [R][size0, size1] @ DRAM,
    ):
        # assert stride(dst, 0) == size1
        # assert stride(src, 0) == size1
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]


@instr
class cudaMemcpyAsync_dtoh_2d(_new_cudaMemcpyAsync_base(2, htod=False)):
    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1] @ DRAM,
        src: [R][size0, size1] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == size1
        # assert stride(src, 0) == size1
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]


@instr
class cudaMemsetAsync0_2d(_new_cudaMemsetAsync0_base(2)):
    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == size1
        assert stride(dst, 1) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = 0


__all__.append("cudaMemcpyAsync_htod_2d")
__all__.append("cudaMemcpyAsync_dtoh_2d")
__all__.append("cudaMemsetAsync0_2d")


########################################################################
# 3D cudaMemcpyAsync and cudaMemsetAsync(0)
########################################################################


@instr
class cudaMemcpyAsync_htod_3d(_new_cudaMemcpyAsync_base(3, htod=True)):
    def behavior(
        size0: size,
        size1: size,
        size2: size,
        dst: [R][size0, size1, size2] @ CudaGmemLinear,
        src: [R][size0, size1, size2] @ DRAM,
    ):
        # assert stride(dst, 0) == size1 * size2
        # assert stride(src, 0) == size1 * size2
        assert stride(dst, 2) == 1
        assert stride(src, 2) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                for i2 in seq(0, size2):
                    dst[i0, i1, i2] = src[i0, i1, i2]


@instr
class cudaMemcpyAsync_dtoh_3d(_new_cudaMemcpyAsync_base(3, htod=False)):
    def behavior(
        size0: size,
        size1: size,
        size2: size,
        dst: [R][size0, size1, size2] @ DRAM,
        src: [R][size0, size1, size2] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == size1 * size2
        # assert stride(src, 0) == size1 * size2
        assert stride(dst, 2) == 1
        assert stride(src, 2) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                for i2 in seq(0, size2):
                    dst[i0, i1, i2] = src[i0, i1, i2]


@instr
class cudaMemsetAsync0_3d(_new_cudaMemsetAsync0_base(3)):
    def behavior(
        size0: size,
        size1: size,
        size2: size,
        dst: [R][size0, size1, size2] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == size1 * size2
        assert stride(dst, 2) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                for i2 in seq(0, size2):
                    dst[i0, i1, i2] = 0


__all__.append("cudaMemcpyAsync_htod_3d")
__all__.append("cudaMemcpyAsync_dtoh_3d")
__all__.append("cudaMemsetAsync0_3d")


########################################################################
# 4D cudaMemcpyAsync and cudaMemsetAsync(0)
########################################################################


@instr
class cudaMemcpyAsync_htod_4d(_new_cudaMemcpyAsync_base(4, htod=True)):
    def behavior(
        size0: size,
        size1: size,
        size2: size,
        size3: size,
        dst: [R][size0, size1, size2, size3] @ CudaGmemLinear,
        src: [R][size0, size1, size2, size3] @ DRAM,
    ):
        # assert stride(dst, 0) == size1 * size2 * size3
        # assert stride(src, 0) == size1 * size2 * size3
        assert stride(dst, 3) == 1
        assert stride(src, 3) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                for i2 in seq(0, size2):
                    for i3 in seq(0, size3):
                        dst[i0, i1, i2, i3] = src[i0, i1, i2, i3]


@instr
class cudaMemcpyAsync_dtoh_4d(_new_cudaMemcpyAsync_base(4, htod=False)):
    def behavior(
        size0: size,
        size1: size,
        size2: size,
        size3: size,
        dst: [R][size0, size1, size2, size3] @ DRAM,
        src: [R][size0, size1, size2, size3] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == size1 * size2 * size3
        # assert stride(src, 0) == size1 * size2 * size3
        assert stride(dst, 3) == 1
        assert stride(src, 3) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                for i2 in seq(0, size2):
                    for i3 in seq(0, size3):
                        dst[i0, i1, i2, i3] = src[i0, i1, i2, i3]


@instr
class cudaMemsetAsync0_4d(_new_cudaMemsetAsync0_base(4)):
    def behavior(
        size0: size,
        size1: size,
        size2: size,
        size3: size,
        dst: [R][size0, size1, size2, size3] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == size1 * size2 * size3
        assert stride(dst, 3) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                for i2 in seq(0, size2):
                    for i3 in seq(0, size3):
                        dst[i0, i1, i2, i3] = 0


__all__.append("cudaMemcpyAsync_htod_4d")
__all__.append("cudaMemcpyAsync_dtoh_4d")
__all__.append("cudaMemsetAsync0_4d")


########################################################################
# 5D cudaMemcpyAsync and cudaMemsetAsync(0)
########################################################################


@instr
class cudaMemcpyAsync_htod_5d(_new_cudaMemcpyAsync_base(5, htod=True)):
    def behavior(
        size0: size,
        size1: size,
        size2: size,
        size3: size,
        size4: size,
        dst: [R][size0, size1, size2, size3, size4] @ CudaGmemLinear,
        src: [R][size0, size1, size2, size3, size4] @ DRAM,
    ):
        # assert stride(dst, 0) == size1 * size2 * size3 * size4
        # assert stride(src, 0) == size1 * size2 * size3 * size4
        assert stride(dst, 4) == 1
        assert stride(src, 4) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                for i2 in seq(0, size2):
                    for i3 in seq(0, size3):
                        for i4 in seq(0, size4):
                            dst[i0, i1, i2, i3, i4] = src[i0, i1, i2, i3, i4]


@instr
class cudaMemcpyAsync_dtoh_5d(_new_cudaMemcpyAsync_base(5, htod=False)):
    def behavior(
        size0: size,
        size1: size,
        size2: size,
        size3: size,
        size4: size,
        dst: [R][size0, size1, size2, size3, size4] @ DRAM,
        src: [R][size0, size1, size2, size3, size4] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == size1 * size2 * size3 * size4
        # assert stride(src, 0) == size1 * size2 * size3 * size4
        assert stride(dst, 4) == 1
        assert stride(src, 4) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                for i2 in seq(0, size2):
                    for i3 in seq(0, size3):
                        for i4 in seq(0, size4):
                            dst[i0, i1, i2, i3, i4] = src[i0, i1, i2, i3, i4]


@instr
class cudaMemsetAsync0_5d(_new_cudaMemsetAsync0_base(5)):
    def behavior(
        size0: size,
        size1: size,
        size2: size,
        size3: size,
        size4: size,
        dst: [R][size0, size1, size2, size3, size4] @ CudaGmemLinear,
    ):
        # assert stride(dst, 0) == size1 * size2 * size3 * size4
        assert stride(dst, 4) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                for i2 in seq(0, size2):
                    for i3 in seq(0, size3):
                        for i4 in seq(0, size4):
                            dst[i0, i1, i2, i3, i4] = 0


__all__.append("cudaMemcpyAsync_htod_5d")
__all__.append("cudaMemcpyAsync_dtoh_5d")
__all__.append("cudaMemsetAsync0_5d")


########################################################################
# Legacy aliases for cudaMemcpyAsync and cudaMemsetAsync(0)
########################################################################
cudaMemcpyAsync_htod_1f32 = cudaMemcpyAsync_htod_1d(dst="f32", src="f32")
cudaMemcpyAsync_dtoh_1f32 = cudaMemcpyAsync_dtoh_1d(dst="f32", src="f32")
cudaMemsetAsync0_1f32 = cudaMemsetAsync0_1d(dst="f32")
cudaMemcpyAsync_htod_1i32 = cudaMemcpyAsync_htod_1d(dst="i32", src="i32")
cudaMemcpyAsync_dtoh_1i32 = cudaMemcpyAsync_dtoh_1d(dst="i32", src="i32")
cudaMemsetAsync0_1i32 = cudaMemsetAsync0_1d(dst="i32")

cudaMemcpyAsync_htod_2f32 = cudaMemcpyAsync_htod_2d(dst="f32", src="f32")
cudaMemcpyAsync_dtoh_2f32 = cudaMemcpyAsync_dtoh_2d(dst="f32", src="f32")
cudaMemsetAsync0_2f32 = cudaMemsetAsync0_2d(dst="f32")
cudaMemcpyAsync_htod_2i32 = cudaMemcpyAsync_htod_2d(dst="i32", src="i32")
cudaMemcpyAsync_dtoh_2i32 = cudaMemcpyAsync_dtoh_2d(dst="i32", src="i32")
cudaMemsetAsync0_2i32 = cudaMemsetAsync0_2d(dst="i32")

cudaMemcpyAsync_htod_3f32 = cudaMemcpyAsync_htod_3d(dst="f32", src="f32")
cudaMemcpyAsync_dtoh_3f32 = cudaMemcpyAsync_dtoh_3d(dst="f32", src="f32")
cudaMemsetAsync0_3f32 = cudaMemsetAsync0_3d(dst="f32")
cudaMemcpyAsync_htod_3i32 = cudaMemcpyAsync_htod_3d(dst="i32", src="i32")
cudaMemcpyAsync_dtoh_3i32 = cudaMemcpyAsync_dtoh_3d(dst="i32", src="i32")
cudaMemsetAsync0_3i32 = cudaMemsetAsync0_3d(dst="i32")

cudaMemcpyAsync_htod_4f32 = cudaMemcpyAsync_htod_4d(dst="f32", src="f32")
cudaMemcpyAsync_dtoh_4f32 = cudaMemcpyAsync_dtoh_4d(dst="f32", src="f32")
cudaMemsetAsync0_4f32 = cudaMemsetAsync0_4d(dst="f32")
cudaMemcpyAsync_htod_4i32 = cudaMemcpyAsync_htod_4d(dst="i32", src="i32")
cudaMemcpyAsync_dtoh_4i32 = cudaMemcpyAsync_dtoh_4d(dst="i32", src="i32")
cudaMemsetAsync0_4i32 = cudaMemsetAsync0_4d(dst="i32")

cudaMemcpyAsync_htod_5f32 = cudaMemcpyAsync_htod_5d(dst="f32", src="f32")
cudaMemcpyAsync_dtoh_5f32 = cudaMemcpyAsync_dtoh_5d(dst="f32", src="f32")
cudaMemsetAsync0_5f32 = cudaMemsetAsync0_5d(dst="f32")
cudaMemcpyAsync_htod_5i32 = cudaMemcpyAsync_htod_5d(dst="i32", src="i32")
cudaMemcpyAsync_dtoh_5i32 = cudaMemcpyAsync_dtoh_5d(dst="i32", src="i32")
cudaMemsetAsync0_5i32 = cudaMemsetAsync0_5d(dst="i32")


for i in range(1, 6):
    for r in "fi":
        __all__.append(f"cudaMemcpyAsync_htod_{i}{r}32")
        __all__.append(f"cudaMemcpyAsync_dtoh_{i}{r}32")
        __all__.append(f"cudaMemsetAsync0_{i}{r}32")
