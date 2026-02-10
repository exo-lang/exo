from __future__ import annotations
from typing import List
from .cuda_fwd import *
import exo.scalars as scalars


__all__ = []  # append stuff to this to export it.


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


__all__.append("cuda_packed32_load")


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


__all__.append("cuda_packed32_store")


# Legacy typed wrappers
cuda_packed_load_f32 = cuda_packed32_load.partial(pack=1, dst="f32", src="f32")
cuda_packed_store_f32 = cuda_packed32_store.partial(pack=1, dst="f32", src="f32")
cuda_packed_load_i32 = cuda_packed32_load.partial(pack=1, dst="i32", src="i32")
cuda_packed_store_i32 = cuda_packed32_store.partial(pack=1, dst="i32", src="i32")
cuda_packed_load_f16 = cuda_packed32_load.partial(pack=2, dst="f16", src="f16")
cuda_packed_store_f16 = cuda_packed32_store.partial(pack=2, dst="f16", src="f16")
cuda_packed_load_bf16 = cuda_packed32_load.partial(pack=2, dst="bf16", src="bf16")
cuda_packed_store_bf16 = cuda_packed32_store.partial(pack=2, dst="bf16", src="bf16")
for t in ("f32", "i32", "f16", "bf16"):
    __all__.append(f"cuda_packed_load_{t}")
    __all__.append(f"cuda_packed_store_{t}")


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
        assert stride(src, 0) == 1
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
        assert stride(src, 0) == 1
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
        assert stride(dst, 0) == 1
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
        assert stride(dst, 0) == 1
        for iv in seq(0, v):
            for ip in seq(0, 1):
                dst[iv * 1 + ip] = src[iv, ip]


_cuda_packed32_v_instr_dict = {
    (16, False): (cuda_packed32_loadv_16b, 2),
    (32, False): (cuda_packed32_loadv_32b, 1),
    (16, True): (cuda_packed32_storev_16b, 2),
    (32, True): (cuda_packed32_storev_32b, 1),
}
# TODO test this stuff
for b in (16, 32):
    __all__.append(f"cuda_packed32_loadv_{b}b")
    __all__.append(f"cuda_packed32_storev_{b}b")


def get_cuda_packed32_loadv(typ, mem: str):
    """Get partial InstrTemplate suitable for generating a ld.{mem}.v instr

    Returns the partial InstrTemplate and the number of values packed
    per 32-bit register.

    """
    typ = scalars.ScalarInfo(typ)
    bits = typ.bits
    assert bits in (16, 32), "Only support 16-bit and 32-bit for now"
    base, pack = _cuda_packed32_v_instr_dict[(bits, False)]
    return (base.partial(dst=typ, src=typ, mem=mem), pack)


def get_cuda_packed32_storev(typ, mem: str):
    """Get partial InstrTemplate suitable for generating a st.{mem}.v instr

    Returns the partial InstrTemplate and the number of values packed
    per 32-bit register.

    """
    typ = scalars.ScalarInfo(typ)
    bits = typ.bits
    assert bits in (16, 32), "Only support 16-bit and 32-bit for now"
    base, pack = _cuda_packed32_v_instr_dict[(bits, True)]
    return (base.partial(dst=typ, src=typ, mem=mem), pack)


# TODO test this stuff
__all__.append("get_cuda_packed32_loadv")
__all__.append("get_cuda_packed32_storev")


# TODO remove this
cuda_packed_store_global_f16, pack = get_cuda_packed32_storev("f16", "global")
del pack


__all__.append("cuda_packed_store_global_f16")
