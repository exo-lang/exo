# Public module for using CUDA with Exo
# See Sm80.py and Sm90.py for A100/H100-specific functionality

from __future__ import annotations
from typing import List
from .cuda_fwd import *
from .cuda_warp_intrin import *

import exo.scalars as scalars


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


########################################################################
# Packed 32-bit register load/store, not vectorized
########################################################################


@instr
class cuda_packed32_load(InstrInfo):
    def behavior(
        pack: size,
        dst: [R][pack] @ CudaRmemPacked32,
        src: [R][pack] @ CudaBasicDeviceVisible,
    ):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, pack):
            dst[i] = src[i]

    def instance(self, pack):
        self.instr_tl = cuda_in_order_instr

        dst_scalar = self.access_info["dst"].scalar_info
        assert dst_scalar == self.access_info["src"].scalar_info
        bits = pack * dst_scalar.bits
        if bits != 32:
            raise ValueError(
                f"Copy of pack={pack} {dst_scalar} is {bits}-bit, not 32-bit"
            )

    def codegen(self, args):
        return _packed32_load_codegen_helper(args)

    valid_num_types = scalars.ScalarInfo.same()


@instr
class cuda_packed32_store(InstrInfo):
    def behavior(
        pack: size,
        dst: [R][pack] @ CudaBasicDeviceVisible,
        src: [R][pack] @ CudaRmemPacked32,
    ):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, pack):
            dst[i] = src[i]

    def instance(self, pack):
        self.instr_tl = cuda_in_order_instr

        dst_scalar = self.access_info["dst"].scalar_info
        assert dst_scalar == self.access_info["src"].scalar_info
        bits = pack * dst_scalar.bits
        if bits != 32:
            raise ValueError(
                f"Copy of pack={pack} {dst_scalar} is {bits}-bit, not 32-bit"
            )

    def codegen(self, args):
        return _packed32_store_codegen_helper(args)

    valid_num_types = scalars.ScalarInfo.same()


# Legacy typed wrappers
cuda_packed_load_f32 = cuda_packed32_load.partial(pack=1, dst="f32", src="f32")
cuda_packed_store_f32 = cuda_packed32_store.partial(pack=1, dst="f32", src="f32")
cuda_packed_load_i32 = cuda_packed32_load.partial(pack=1, dst="i32", src="i32")
cuda_packed_store_i32 = cuda_packed32_store.partial(pack=1, dst="i32", src="i32")
cuda_packed_load_f16 = cuda_packed32_load.partial(pack=2, dst="f16", src="f16")
cuda_packed_store_f16 = cuda_packed32_store.partial(pack=2, dst="f16", src="f16")
cuda_packed_load_bf16 = cuda_packed32_load.partial(pack=2, dst="bf16", src="bf16")
cuda_packed_store_bf16 = cuda_packed32_store.partial(pack=2, dst="bf16", src="bf16")


def _packed32_load_codegen_helper(args):
    dst = args.dst.index(ptx_data=True)
    src_ptr = args.src.index_ptr()
    # We have no choice but to accept the strict aliasing violation.
    # When I use memcpy, I get insanely slow {ld|st}.{shared|global}.u8 usage.
    return [f"{dst} = *reinterpret_cast<const decltype({dst})*>({src_ptr});"]


def _packed32_store_codegen_helper(args):
    # Theoretical issue, src could be const (very unlikely for a register)
    # causing the decltype to be const as well.
    dst_ptr = args.dst.index_ptr()
    src = args.src.index(ptx_data=True)
    # We have no choice but to accept the strict aliasing violation.
    # When I use memcpy, I get insanely slow {ld|st}.{shared|global}.u8 usage.
    return [f"*reinterpret_cast<decltype({src})*>({dst_ptr}) = {src};"]


########################################################################
# Packed 32-bit register load/store, vectorized
# Scheduling: use get_cuda_packed32_load_v, get_cuda_packed32_store_v
########################################################################


def _new_packed32_v_base(type_bits: int, *, store: bool):
    # Unfortunately, we need separate InstrTemplates for 16-bit and 32-bit
    # base types, due to affine indexing restrictions (pack must be constant).
    st_ld = "st" if store else "ld"

    class cuda_packed32_v_base(InstrInfo):
        __slots__ = ["mem"]

        def instance(self, v, *, mem):
            self.instr_tl = cuda_in_order_instr
            self.mem = mem
            typ = self.access_info["dst"].scalar_info

            if mem == "shared":
                exo_mem = CudaSmemAtomicity16B
            elif mem == "global":
                exo_mem = CudaGmemAtomicity16B
            else:
                assert 0, f"mem={mem!r} must be global or shared"

            if store:
                self.access_info["dst"].mem = exo_mem
            else:
                self.access_info["src"].mem = exo_mem

            # fmt: off
            assert v in (2, 4) or (v == 8 and mem == "global"), f"Invalid vector size v={v}"
            assert type_bits == typ.bits, f"Expect {type_bits}-bit type, not {typ}"

        def codegen(self, args: InstrArgs):
            v = args.v
            is_f32 = args.dst.get_scalar_info() == scalars.f32
            rmem_arg = args.src if store else args.dst
            ptr_arg = (args.dst if store else args.src).index_ptr()
            rmem_vec = [rmem_arg.index(i, ptx_data=True) for i in range(v)]
            t = "f32" if is_f32 else "u32"
            constraint = "f" if is_f32 else "r"
            ptx = InlinePtxGen(f"{st_ld}.{self.mem}.v{v}.{t} #0#;", volatile=True)
            ptx.add_arg(ptr_arg, constraint="generic", log_as="bits")
            ptx.add_arg(rmem_vec, constraint=constraint, log_as=None)
            return ptx.as_c_lines()

        valid_num_types = scalars.ScalarInfo.same()

    return cuda_packed32_v_base


@instr
class cuda_packed32_loadv_16b(_new_packed32_v_base(16, store=False)):
    def behavior(
        v: size,
        dst: [R][v, 2] @ CudaRmemPacked32,
        src: [R][v * 2],  # Cuda?memAtomicity16B
    ):
        for iv in seq(0, v):
            for ip in seq(0, 2):
                dst[iv, ip] = src[iv * 2 + ip]


@instr
class cuda_packed32_loadv_32b(_new_packed32_v_base(32, store=False)):
    def behavior(
        v: size,
        dst: [R][v, 1] @ CudaRmemPacked32,
        src: [R][v * 1],  # Cuda?memAtomicity16B
    ):
        for iv in seq(0, v):
            for ip in seq(0, 1):
                dst[iv, ip] = src[iv * 1 + ip]


@instr
class cuda_packed32_storev_16b(_new_packed32_v_base(16, store=True)):
    def behavior(
        v: size,
        dst: [R][v * 2],  # Cuda?memAtomicity16B
        src: [R][v, 2] @ CudaRmemPacked32,
    ):
        for iv in seq(0, v):
            for ip in seq(0, 2):
                dst[iv * 2 + ip] = src[iv, ip]


@instr
class cuda_packed32_storev_32b(_new_packed32_v_base(32, store=True)):
    def behavior(
        v: size,
        dst: [R][v * 1],  # Cuda?memAtomicity16B
        src: [R][v, 1] @ CudaRmemPacked32,
    ):
        for iv in seq(0, v):
            for ip in seq(0, 1):
                dst[iv * 1 + ip] = src[iv, ip]


_cuda_packed32_v_instr_dict = {
    (16, False): (cuda_packed32_loadv_16b, 2),
    (32, False): (cuda_packed32_loadv_32b, 1),
    (16, True): (cuda_packed32_storev_16b, 2),
    (32, True): (cuda_packed32_storev_32b, 1),
}


def get_cuda_packed32_load_v(typ, mem: str):
    """Get partial InstrTemplate suitable for generating a ld.{mem}.v instr

    Returns the partial InstrTemplate and the number of values packed
    per 32-bit register.

    """
    typ = scalars.ScalarInfo(typ)
    bits = typ.bits
    assert bits in (16, 32), "Only support 16-bit and 32-bit for now"
    base, pack = _cuda_packed32_v_instr_dict[(bits, False)]
    return (base.partial(dst=typ, src=typ, mem=mem), pack)


def get_cuda_packed32_store_v(typ, mem: str):
    """Get partial InstrTemplate suitable for generating a st.{mem}.v instr

    Returns the partial InstrTemplate and the number of values packed
    per 32-bit register.

    """
    typ = scalars.ScalarInfo(typ)
    bits = typ.bits
    assert bits in (16, 32), "Only support 16-bit and 32-bit for now"
    base, pack = _cuda_packed32_v_instr_dict[(bits, True)]
    return (base.partial(dst=typ, src=typ, mem=mem), pack)


# TODO remove this
cuda_packed_store_global_f16, pack = get_cuda_packed32_store_v("f16", "global")
del pack
