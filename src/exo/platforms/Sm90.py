# Memory, instructions, and actor kinds specific to CUDA sm_90 and sm_90a (H100)
# Everything exported by this module starts with Sm90_,
# except for actor kinds and actor signatures.
from __future__ import annotations

# Currently we import from the exo.spork directory,
# which users shouldn't import directly.
from ..spork.actor_kinds import (
    sig_tma_to_smem,
    sig_tma_to_gmem,
    sig_wgmma_rmem_a,
    sig_wgmma_rmem_d,
    sig_wgmma_smem,
    tma_to_smem_async,
    tma_to_gmem_async,
    wgmma_async,
    wgmma_async_smem,
    wgmma_fence_1,
    wgmma_fence_2,
    cuda_async_proxy,
    cuda_async_proxy_wgmma,
    cuda_generic_and_async_proxy,
)

__all__ = [
    "sig_tma_to_smem",
    "sig_tma_to_gmem",
    "sig_wgmma_rmem_a",
    "sig_wgmma_rmem_d",
    "sig_wgmma_smem",
    "tma_to_smem_async",
    "tma_to_gmem_async",
    "wgmma_async",
    "wgmma_async_smem",
    "wgmma_fence_1",
    "wgmma_fence_2",
    "cuda_async_proxy",
    "cuda_async_proxy_wgmma",
    "cuda_generic_and_async_proxy",
]

# We use these but don't put them in __all__
from math import prod
from ..API import instr
from ..core.memory import (
    memwin_template,
    Memory,
    SpecialWindow,
    WindowStructCtx,
    SpecialWindowFromMemoryCtx,
)
from ..spork.cuda_memory import *
from ..spork.coll_algebra import cuda_warp, cuda_warpgroup


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Swizzled shared memory, as used by wgmma and TMA (cp.async.bulk).
# The rightmost 2 dimensions correspond to 2, 4, or 8 swizzled matrices.
# Any other dimensions are not swizzled (C order)
@memwin_template
def Sm90_SmemSwizzled(swizzle):
    if swizzle not in (32, 64, 128):
        raise ValueError(f"swizzle must be 32, 64, or 128 bytes, not {swizzle}")

    # As I understand it 2, 4, or 8 "core matrices" (128 bytes per matrix)
    # are swizzled together for 32B, 64B, 128B swizzle mode, respectively.
    # Each core matrix has M or N = 8 (outer dimension) and
    # K = 16 / sizeof(T) (inner dimension).
    #
    # Unfortunately it seems the term "core matrices" was later expunged from
    # the wgmma documentation so none of this makes sense anymore.
    matrix_mn = 8
    num_matrices = swizzle // 16
    matrix_bytes = 128
    c_matrices = f"Sm90_SmemMatrices_SW{swizzle}"

    class SwizzledImpl(CudaBasicSmem):
        @classmethod
        def global_(cls):
            return f"""typedef struct {c_matrices} {{
    char matrix_bytes[{matrix_bytes * num_matrices}];
#ifdef __CUDACC__
    EXO_CUDA_INLINE {c_matrices}& byte_offset(unsigned bytes)
    {{
        return reinterpret_cast<{c_matrices}&>(matrix_bytes[bytes]);
    }}
#endif
}} {c_matrices};"""

        @classmethod
        def can_read(cls):
            return False

        @classmethod
        def smem_config(cls, inputs: SmemConfigInputs) -> SmemConfig:
            matrix_k = cls.get_matrix_k(inputs.ctype)
            inputs.require_shape_tile((matrix_mn, matrix_k * num_matrices))
            return SmemConfig(f"{c_matrices} (&)[]", 128)

        @classmethod
        def window_definition(cls, ctx: WindowStructCtx):
            dataptr_ctype, sdef = ctx.generate_default(
                "Sm90_SmemSwizzled", c_matrices, (swizzle,)
            )
            return dataptr_ctype, sdef

        @classmethod
        def window(cls, basetyp, in_expr, indices, strides, srcinfo):
            matrix_k = cls.get_matrix_k(basetyp)
            assert len(indices) >= 2
            if strides[-1] != "1" or strides[-2] != str(matrix_k * num_matrices):
                raise MemGenError("Cannot stride swizzled dimensions (last 2)")
            # TODO, we should only allow M/N offset (indices[-2]) in multiples of 8.
            # We should allow K offset only for multiples of 256 / element_bits
            # Stride core matrices
            vector_size = num_matrices * matrix_mn * matrix_k
            # In CUDA, the K (inner) dimension can be offset just by adding
            # K_offset * sizeof(element) to the pointer. This survives
            # the swizzling process.
            return (
                cls.default_window(
                    vector_size, basetyp, in_expr, indices, strides, srcinfo
                )
                + f".byte_offset(({indices[-1]}) * {scalar_bits(basetyp) // 8})"
            )

        @classmethod
        def get_matrix_k(cls, ctype):
            return 128 // scalar_bits(ctype)

    return SwizzledImpl


def Sm90_get_mma_smem(swizzle):
    if swizzle == 0:
        return CudaSmemLinear  # XXX TODO 128 byte alignment
    else:
        return Sm90_SmemSwizzled(swizzle)


__all__.append("Sm90_SmemSwizzled")
__all__.append("Sm90_get_mma_smem")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# CUtensorMap
# This is a 128-byte blob created on the CPU and passed to CUDA device functions
# which use it to execute TMA (cp.async.bulk) instructions.
# In Exo, we model this as a SpecialWindow (similar to Memory)
# that is constructed as a window to CudaGmemLinear.
@memwin_template
def Sm90_tensorMap(swizzle, *smem_box):
    if swizzle == 0:
        cu_swizzle = "CU_TENSOR_MAP_SWIZZLE_NONE"
    else:
        assert swizzle in (32, 64, 128)
        cu_swizzle = f"CU_TENSOR_MAP_SWIZZLE_{swizzle}B"
    rank = len(smem_box)
    assert 1 <= rank <= 5

    class CUtensorMap(SpecialWindow):
        @classmethod
        def global_(cls):
            return "#include <assert.h>\n#include <stdio.h>"

        @classmethod
        def window_definition(cls, ctx: WindowStructCtx):
            sname = ctx.struct_name("Sm90_tensorMap", cls.memwin_template_parameters)
            cls.sname = sname
            typ = ctx.type_shorthand()
            try:
                cu_ctype_enum, stride_suffix = CUtensorMap_type_dict[typ]
            except KeyError:
                raise TypeError("CUtensorMap: implement me: " + typ)
            # CUDA boxDim in opposite order as Exo smem_box
            cu_boxDim = "{ " + ", ".join(str(n) for n in smem_box[::-1]) + " }"
            s_def = CUtensorMap_s_def_template.format(
                sname=sname,
                rank=rank,
                cu_swizzle=cu_swizzle,
                cu_boxDim=cu_boxDim,
                cu_ctype_enum=cu_ctype_enum,
                stride_suffix=stride_suffix,
            )
            # Return 2-tuple: enables custom layout for window (obscure).
            return "CUtensorMap", s_def

        @classmethod
        def separate_dataptr(cls):
            return True

        @classmethod
        def window(cls, basetyp, in_expr, indices, strides, srcinfo):
            # Window creation: CUtensorMap (dataptr) is passed through unchanged.
            # Offsets (layout) are modified.
            assert len(indices) == rank
            dataptr, in_layout = in_expr

            out_layout = "{{ "
            out_layout += ", ".join(
                f"{in_layout}.exo_offsets[{r}] + (unsigned)({indices[r]})"
                for r in range(rank)
            )
            out_layout += " }}"

            return dataptr, out_layout

        @classmethod
        def source_memory_type(cls):
            return CudaGmemLinear

        @classmethod
        def from_memory(cls, ctx: SpecialWindowFromMemoryCtx):
            sname = ctx.dst_struct_name()
            shape = ctx.shape_strs()
            clayout = f"(struct {sname}_strides){{ {ctx.src_layout()} }}"
            cdim = f'(struct {sname}_gmem_dim){{ {{ {", ".join(s for s in shape)} }} }}'

            # Dataptr: Encode CUtensorMap from strides and GMEM dimension (shape)
            d_def = f"{sname}_encode(&{ctx.src_data()}, {clayout}, {cdim})"

            # Layout: offsets are initially zero.
            w_def = "{}"
            return d_def, w_def

    return CUtensorMap


__all__.append("Sm90_tensorMap")


# str.format template for CUtensorMap-related Exo window C definition
CUtensorMap_s_def_template = """
struct {sname} {{
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned exo_offsets[{rank}];
}};

struct {sname}_strides {{
    // Stored in reverse-order as the raw CUtensorMap,
    // and in element count, not in bytes.
    // Leftmost stride is most-significant.
    unsigned exo_strides[{rank}];
}};

struct {sname}_gmem_dim {{
    // Stored in the reverse-order as the raw CUtensorMap.
    // Leftmost dimension is the most-significant.
    unsigned exo_dim[{rank}];
}};

static inline CUtensorMap {sname}_encode(
        // Window dataptr, layout
        const void* globalAddress, struct {sname}_strides gmem_stride,
        // Tensor size
        struct {sname}_gmem_dim gmem_dim)
{{
    assert(gmem_stride.exo_strides[{rank} - 1] == 1);

    CUtensorMap tensorMap;
    const CUtensorMapSwizzle swizzle = {cu_swizzle};

    cuuint64_t globalDim[{rank}];
    cuuint64_t allGlobalStrides[{rank}];  // allGlobalStrides[0] unused by CUDA
    cuuint32_t elementStrides[{rank}];

    // We translate from the Exo ordering (leftmost stride is most-significant)
    // to the CUDA ordering (leftmost stride is least-significant).
    for (uint32_t cu_dim = 0; cu_dim < {rank}; ++cu_dim) {{
        const uint32_t exo_dim = {rank} - 1 - cu_dim;
        globalDim[cu_dim] = gmem_dim.exo_dim[exo_dim];
        allGlobalStrides[cu_dim] = ((cuuint64_t)gmem_stride.exo_strides[exo_dim]){stride_suffix};
        elementStrides[cu_dim] = 1;
    }}

    cuuint32_t boxDim[{rank}] = {cu_boxDim};
    const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    const CUresult result = cuTensorMapEncodeTiled(
            &tensorMap,
            {cu_ctype_enum},
            {rank},
            (void*)globalAddress,
            globalDim,
            &allGlobalStrides[1],  // Cuda presumes least-significant dim is tightly-packed
            boxDim,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill);
    if (result != 0) {{
        fprintf(stderr, "{sname}_encode: error %i\\n", (int)result);
        assert(0);
    }}
    return tensorMap;
}}
"""


# Translate type shorthand to CUDA enum + stride suffix
# where f"element_count {stride suffix}" is C syntax for byte count for
# element_count many values.
# NB not all shorthands here are implemented in Exo ... David just implemented
# them anyway so things will "just work" in the future.
CUtensorMap_type_dict = {
    "u8": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", ""),
    "u16": ("CU_TENSOR_MAP_DATA_TYPE_UINT16", " * 2"),
    "u32": ("CU_TENSOR_MAP_DATA_TYPE_UINT32", " * 4"),
    "i32": ("CU_TENSOR_MAP_DATA_TYPE_INT32", " * 4"),
    "u64": ("CU_TENSOR_MAP_DATA_TYPE_UINT64", " * 8"),
    "i64": ("CU_TENSOR_MAP_DATA_TYPE_INT64", " * 8"),
    "f16": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT16", " * 2"),
    "f32": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT32", " * 4"),
    "f64": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT64", " * 8"),
    "bf16": ("CU_TENSOR_MAP_DATA_TYPE_BFLOAT16", " * 2"),
    "u4": ("CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B", " / 2"),
}


def copy_tensor_to_smem_util(rank: int, multicast: bool):
    cache_hint = 1152921504606846976  # copied from cutlass PTX
    vector_fmt = "{" + ", ".join(f"%{r+2}" for r in range(rank)) + "}"
    ptx_fmt = f" [%0], [%1, {vector_fmt}], [%{rank+2}], %{rank+3}"
    if multicast:
        ptx_fmt += f", %{rank+4}"
    vector_values = ", ".join(
        f'"r"(exo_offsets[{rank - 1 - r}])' for r in range(0, rank)
    )

    # fmt: off
    elect_one_prefix = r"""// cute::elect_one_sync
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));"""

    # TODO ensure mbarriers for multicast TMA don't use broadcast, unlike other mbarriers
    expect_tx = f'asm("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" :: "r"(exo_tma_mbarrier), "r"(n_bytes));'

    if multicast:
        return f"""EXO_CUDA_INLINE void
exo_Sm90_tma_to_smem_{rank}d(void* dst, const CUtensorMap& tensorMap, cuda::std::array<unsigned, {rank}> exo_offsets,
                       uint32_t exo_tma_mbarrier, uint32_t n_bytes, uint16_t cta_mask)
{{
    {elect_one_prefix}
    if (pred) {{
        {expect_tx}
        asm volatile(
            "cp.async.bulk.tensor.{rank}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast.L2::cache_hint"
            "{ptx_fmt};"
            :
            : "r"(exo_smemU32(dst)), "l"(&tensorMap),
              {vector_values},
              "r"(exo_tma_mbarrier), "h"(cta_mask), "n"({cache_hint})
            : "memory");
    }}
}}"""
    else:
        return f"""EXO_CUDA_INLINE void
exo_Sm90_tma_to_smem_{rank}d(void* dst, const CUtensorMap& tensorMap, cuda::std::array<unsigned, {rank}> exo_offsets,
                        uint32_t exo_tma_mbarrier, uint32_t n_bytes)

{{
    {elect_one_prefix}
    if (pred) {{
        {expect_tx}
        asm volatile(
            "cp.async.bulk.tensor.{rank}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            "{ptx_fmt};"
            :
            : "r"(exo_smemU32(dst)), "l"(&tensorMap),
              {vector_values},
              "r"(exo_tma_mbarrier), "n"({cache_hint})
            : "memory");
    }}
}}"""
    # fmt: on


class copy_tensor_to_smem_impl:
    def instance_impl(self, smem_box, swizzled, element_bits):
        rank = len(smem_box)
        assert rank > 0
        if swizzled:
            swizzle = smem_box[-1] * element_bits // 8
            if swizzle not in (32, 64, 128):
                raise ValueError(
                    f"Invalid smem_box {smem_box}; "
                    f"last dimension must lead to swizzle of "
                    f"32, 64, or 128; not {swizzle}"
                )
        else:
            swizzle = 0
        self.access_info["dst"].actor_signature = sig_tma_to_smem
        self.access_info["dst"].mem = Sm90_get_mma_smem(swizzle)
        self.access_info["src"].actor_signature = sig_tma_to_smem
        self.access_info["src"].mem = Sm90_tensorMap(swizzle, *smem_box)
        self.actor_kind = tma_to_smem_async
        self.coll_unit = cuda_warp
        self.cu_includes.append("cuda/std/array")
        self.cu_utils.append(copy_tensor_to_smem_util(rank, False))
        # We don't know the real indent; just try to make it less ugly.
        indent = " " * 20
        fmt = f"exo_CudaUtil::exo_Sm90_tma_to_smem_{rank}d("
        fmt += f"\n{indent}&{{dst_data}}"  # Pointer to SMEM
        fmt += f",\n{indent}{{src_data}}"  # CUtensorMap
        fmt += f",\n{indent}{{src_layout}}"  # exo_offsets
        fmt += f",\n{indent}exo_tma_mbarrier"
        fmt += f",\n{indent}{prod(smem_box) * element_bits // 8}"
        fmt += ");"
        self.instr_format = fmt


@instr
class Sm90_copy_tensor_to_smem_linear_2f32(copy_tensor_to_smem_impl):
    def behavior(
        box0: size, box1: size, dst: [f32][box0, box1], src: [f32][box0, box1]
    ):
        assert stride(dst, 1) == 1
        # assert stride(dst, 0) == box1  # TODO why doesn't this work?
        # We need to assert that the dst is densely packed.
        for i0 in seq(0, box0):
            for i1 in seq(0, box1):
                dst[i0, i1] = src[i0, i1]

    def instance(self, box0, box1):
        self.instance_impl((box0, box1), False, 32)


__all__.append("Sm90_copy_tensor_to_smem_linear_2f32")


@instr
class Sm90_copy_tensor_to_smem_swizzled_2f32(copy_tensor_to_smem_impl):
    def behavior(
        box0: size, box1: size, dst: [f32][box0 / 8, 8, box1], src: [f32][box0, box1]
    ):
        assert box0 % 8 == 0
        assert box0 >= 8
        assert stride(dst, 2) == 1
        # assert stride(dst, 0) == box1  # TODO why doesn't this work?
        # We need to assert that the dst is densely packed.
        for i0 in seq(0, box0):
            for i1 in seq(0, box1):
                dst[i0 / 8, i0 % 8, i1] = src[i0, i1]

    def instance(self, box0, box1):
        self.instance_impl((box0, box1), True, 32)


__all__.append("Sm90_copy_tensor_to_smem_swizzled_2f32")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# wgmma: warpgroup matrix multiply-accumulate


class WgmmaRmemImpl(CudaBasicDeviceVisible):
    @classmethod
    def actor_kind_permission(cls, actor_kind, is_instr):
        return cls.device_allocated_impl(actor_kind, is_instr)

    @classmethod
    def native_unit(cls):
        return cuda_warpgroup

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # Right two dimensions correspond to register tile of n_regs-many 32-bit
        # registers; other dimensions are passed through directly as {idxs}.
        element_bits = scalar_bits(prim_type)
        const_shape = cls.as_const_shape(new_name, shape, srcinfo, min_dim=2)
        n_regs = const_shape[-2] * const_shape[-1] * element_bits // 4096
        idxs = "".join(f"[{extent}]" for extent in const_shape[:-2])
        comment = f"{n_regs} registers store {prim_type}[{const_shape[-2]}][{const_shape[-1]}]"
        assert prim_type == "float"
        reg_type = "float"  # TODO
        return f"{reg_type} {new_name}{idxs}[{n_regs}]; // {comment}"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        if basetyp.is_win():
            return f"*{baseptr}.data"

        # TODO enforce last two indices are :,:
        # e.g. we currently accept
        # D : f32[64,64,256]
        # D2 = D[:,:,0]
        if indices[-1] != "0" or indices[-2] != "0":
            raise MemGenError(
                f"{srcinfo}: cannot offset right 2 dimensions of {baseptr} @ {cls.name()}"
            )

        idxs = "".join(f"[{idx}]" for idx in indices[:-2])
        return f"{baseptr}{idxs}[0]"

    @classmethod
    def window_definition(cls, ctx: WindowStructCtx):
        if ctx.n_dims() != 2:
            raise MemGenError(
                f"{ctx.srcinfo():} Only support windows to 2D wgmma tiles"
            )
        return ctx.generate_default("Sm90_RmemMatrix", "float")  # TODO


class Sm90_RmemMatrixA(WgmmaRmemImpl):
    pass


class Sm90_RmemMatrixD(WgmmaRmemImpl):
    pass


__all__.append("Sm90_RmemMatrixA")
__all__.append("Sm90_RmemMatrixD")


matrix_descriptor_util = """\
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor_encode(uint32_t val)
{
    uint64_t enc = (val & 0x3FFFF) >> 4;
    return enc;
}

template <unsigned swizzle_bits>
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor(const void* smem_ptr, uint32_t mn_stride, uint32_t k_stride)
{
    return exo_matrix_descriptor_encode(exo_smemU32(smem_ptr))
           | exo_matrix_descriptor_encode(k_stride) << 16u
           | exo_matrix_descriptor_encode(mn_stride) << 32u
           | uint64_t(swizzle_bits) << 62;
}
"""


def mma_async_fname(m, n, k, ptx_d_type, ptx_ab_type):
    return f"exo_wgmma_mma_async_m{m}n{n}k{k}_{ptx_d_type}_{ptx_ab_type}"


def mma_async_util(m, n, k, ptx_d_type, ptx_ab_type, reg_type, n_regs):
    # fmt: off
    if reg_type == "float":
        r = "f"
    elif reg_type == "unsigned":
        r = "r"
    else:
        assert 0

    fname = mma_async_fname(m, n, k, ptx_d_type, ptx_ab_type)
    instr_name = f"wgmma.mma_async.sync.aligned.m{m}n{n}k{k}.{ptx_d_type}.{ptx_ab_type}.{ptx_ab_type}"
    vector_fmt = "{" + ", ".join(f"%{i}" for i in range(n_regs)) + "}"
    vector_args = ", ".join(f'"+{r}"(d[{i}])' for i in range(n_regs))

    return fr"""template <unsigned swizzle_bits_a, unsigned swizzle_bits_b>
EXO_CUDA_INLINE void {fname}(
        {reg_type}* d, const void* smem_a, const void* smem_b,
        unsigned m_matrix_stride, unsigned n_matrix_stride, unsigned k_matrix_stride, unsigned scale_d)
{{
    auto desc_a = exo_matrix_descriptor<swizzle_bits_a>(smem_a, m_matrix_stride, k_matrix_stride);
    auto desc_b = exo_matrix_descriptor<swizzle_bits_b>(smem_a, m_matrix_stride, k_matrix_stride);
    asm volatile(
                "{{ // {fname} \n"
                  ".reg .pred p;\n"
                  "setp.ne.b32 p, %{n_regs+2}, 0;\n"
                  "{instr_name} "
                  "{vector_fmt}, "
                  " %{n_regs},"
                  " %{n_regs+1},"
                  " p,    1,  1;\n"
                "}}\n"
                  : {vector_args}
                  :  "l"(desc_a), "l"(desc_b), "r"(scale_d));
}}
"""
    # fmt: on


class mma_async_impl:
    def instance_impl(self, m, n, k, ptx_d_type, ptx_ab_type, reg_type, n_regs):
        if n % 8 != 0 or not (8 <= n <= 256):
            raise ValueError(f"n = {n} invalid for wgmma.mma_async")
        self.access_info["d"].actor_signature = sig_wgmma_rmem_d
        self.access_info["a"].actor_signature = sig_wgmma_smem
        self.access_info["b"].actor_signature = sig_wgmma_smem
        self.actor_kind = wgmma_async
        self.coll_unit = cuda_warpgroup
        self.cu_utils.append(matrix_descriptor_util)
        self.cu_utils.append(
            mma_async_util(m, n, k, ptx_d_type, ptx_ab_type, reg_type, n_regs)
        )

        # TODO correct swizzle bits, scale_d, strides (1024 is assuming densely packed)
        scale_d = "false"
        swizzle_bits_a = 1
        swizzle_bits_b = 1
        fname = mma_async_fname(m, n, k, ptx_d_type, ptx_ab_type)

        fmt = f"""exo_CudaUtil::{fname}<{swizzle_bits_a}, {swizzle_bits_b}>(
                    &{{d_data}},
                    &{{a_data}},
                    &{{b_data}},
                    1024, 1024, 0, {{scale_d}});"""
        self.instr_format = fmt


@instr
class Sm90_mma_async_tf32(mma_async_impl):
    def behavior(
        n: size,
        d: [f32][64, n] @ Sm90_RmemMatrixD,
        a: [f32][8, 8, 8] @ Sm90_SmemSwizzled(128),
        b: [f32][n / 8, 8, 8] @ Sm90_SmemSwizzled(128),
        scale_d: index,
    ):
        assert n >= 8
        assert n % 8 == 0
        for mi in seq(0, 64):
            for ni in seq(0, n):
                for ki in seq(0, 8):
                    d[mi, ni] += a[mi / 8, mi % 8, ki] * b[ni / 8, ni % 8, ki]

    def instance(self, n):
        self.instance_impl(64, n, 8, "f32", "tf32", "float", n // 2)


__all__.append("Sm90_mma_async_tf32")
