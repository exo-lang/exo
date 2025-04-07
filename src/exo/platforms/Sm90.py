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
    // Stored in the same order as the raw CUtensorMap.
    // Rightmost dimension is most-significant.
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
    def instance_impl(self, smem_box, swizzle, n_bytes):
        rank = len(smem_box)
        assert swizzle == 0  # TODO
        self.access_info["dst"].actor_signature = sig_tma_to_smem
        self.access_info["dst"].mem = CudaSmemLinear  # TODO
        self.access_info["src"].actor_signature = sig_tma_to_smem
        self.access_info["src"].mem = Sm90_tensorMap(swizzle, *smem_box)
        self.actor_kind = tma_to_smem_async
        self.coll_unit = cuda_warp
        self.cu_includes.append("cuda/std/array")
        self.cu_utils.append(copy_tensor_to_smem_util(rank, False))
        indent = (
            " " * 20
        )  # We don't know the real indent, just try to make it less ugly
        fmt = f"exo_CudaUtil::exo_Sm90_tma_to_smem_{rank}d("
        fmt += f"\n{indent}&{{dst_data}}"  # Pointer to SMEM
        fmt += f",\n{indent}{{src_data}}"  # CUtensorMap
        fmt += f",\n{indent}{{src_layout}}"  # exo_offsets
        fmt += f",\n{indent}exo_tma_mbarrier"
        fmt += f",\n{indent}{n_bytes}"
        fmt += ");"
        self.instr_format = fmt


@instr
class Sm90_copy_tensor_to_smem_linear_2f32(copy_tensor_to_smem_impl):
    def behavior(
        box0: size, box1: size, dst: [f32][box0, box1], src: [f32][box0, box1]
    ):
        # assert stride(dst, 0) == box1  # TODO why doesn't this work?
        assert stride(dst, 1) == 1
        for i0 in seq(0, box0):
            for i1 in seq(0, box1):
                dst[i0, i1] = src[i0, i1]

    def instance(self, box0, box1):
        self.instance_impl((box0, box1), 0, box0 * box1 * 4)


__all__.append("Sm90_copy_tensor_to_smem_linear_2f32")
