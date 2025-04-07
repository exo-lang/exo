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
            # Return 2-tuple: enables separate layout for window (obscure).
            # This is due to the tensormap needing to be stored in
            # grid constant memory, but the offsets are in RMEM.
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
                f"{in_layout}.exo_offsets[{r}] + {indices[r]}" for r in range(rank)
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
        allGlobalStrides[cu_dim] = gmem_stride.exo_strides[exo_dim]{stride_suffix};
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


# static inline struct {sname} {sname}_add_offsets(struct {sname} a, struct {sname} b)
# {{
#     struct {sname} c;
#     for (uint32_t i = 0; i < {rank}; ++i) c.exo_offsets[i] = a.exo_offsets[i] + b.exo_offsets[i];
#     return c;
# }}
