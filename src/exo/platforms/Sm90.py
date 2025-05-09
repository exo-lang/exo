# Memory, instructions, and actor kinds specific to CUDA sm_90 and sm_90a (H100)
# Everything exported by this module starts with Sm90_,
# except for actor kinds and actor signatures.
from __future__ import annotations

# Currently we import from the exo.spork directory,
# which users shouldn't import directly.
from ..spork.actor_kinds import (
    sig_cuda_classic,
    cuda_classic,
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

    swizzle_bits = 1 if swizzle == 128 else 2 if swizzle == 64 else 3

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

    EXO_CUDA_INLINE uint64_t get_swizzle_bits() const
    {{
        return {swizzle_bits};
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


# TODO support more than float
store_d_util = """template <bool ColumnMajor, typename Window, typename Reg>
EXO_CUDA_INLINE void exo_Sm90_store_d_reg(Window dst, Reg value, uint32_t m_offset, uint32_t reg_index)
{
    const uint32_t tid = threadIdx.x % 128u;
    const uint32_t r_base = (tid / 32u) * 16u + (tid % 32u) / 4u;
    const uint32_t c_base = (tid % 4u) * 2u;
    const uint32_t r = m_offset + r_base + ((reg_index % 4u) / 2u) * 8u;
    const uint32_t c = c_base + (reg_index / 4u) * 8 + (reg_index % 2u);
    auto dst_ptr = reinterpret_cast<Reg*>(
            &dst.data[c * dst.strides[!ColumnMajor] + r * dst.strides[ColumnMajor]]);
    *dst_ptr = value;
}
"""

matrix_descriptor_util = """\
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor_encode(uint32_t val)
{
    return (val & 0x3FFFF) >> 4;
}

template <typename Window>
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor(Window window, uint32_t element_size, uint32_t mn_offset = 0)
{
    uint64_t mn_stride = window.strides[0] * element_size;
    return exo_matrix_descriptor_encode(exo_smemU32(window.data) + (mn_offset / 8u) * mn_stride)
           | exo_matrix_descriptor_encode(sizeof(*window.data)) << 16u
           | exo_matrix_descriptor_encode(mn_stride) << 32u
           | uint64_t(window.data->get_swizzle_bits()) << 62;
}
"""


@dataclass(slots=True)
class WgmmaHelper:
    # wgmma.mma_async.sync.aligned.m{M}n{N}k{K}.{ptx_dtype}.{ptx_atype}.{ptx.btype}
    M: int
    N: int
    ptx_dtype: str
    ptx_atype: str
    ptx_btype: str

    def __post_init__(self):
        M = self.M
        N = self.N
        if M % 64 != 0 or M <= 0:
            raise ValueError("Require M to be a positive multiple of 64")
        if N % 8 != 0 or N < 8 or N > 256:
            raise ValueError("Require N to be a multiple of 8 in [8, 256]")

    def ptx_instr_name(self):
        # fmt: off
        K = self.get_K()
        return f"wgmma.mma_async.sync.aligned.m64n{self.N}k{K}.{self.ptx_dtype}.{self.ptx_atype}.{self.ptx_btype}"
        # fmt: on

    def rmem_d_struct_name(self):
        # Qualify with exo_CudaUtil:: in usage in generated Exo function
        return f"exo_Sm90_RmemD_m{self.M}n{self.N}_{self.ptx_dtype}"

    def rmem_a_struct_name(self):
        # Qualify with exo_CudaUtil:: in usage in generated Exo function
        return f"exo_Sm90_RmemA_m{self.M}n{self.N}_{self.ptx_atype}"

    def dreg_ctype(self):
        # TODO
        assert self.ptx_dtype == "f32"
        return "float"

    def areg_ctype(self):
        # TODO
        assert self.ptx_dtype == "tf32"
        return "unsigned"

    def get_K(self):
        assert self.ptx_atype == "tf32", "TODO"
        assert self.ptx_btype == "tf32", "TODO"
        return 8  # TODO

    def dreg_names(self, m=None, n=None):
        result = []
        assert self.ptx_dtype == "f32", "TODO"
        n_stride = 8  # TODO

        m_lo = 0 if m is None else m
        m_hi = self.M if m is None else m + 64
        n_lo = 0 if n is None else n
        n_hi = self.N if n is None else n + n_stride

        for m_ in range(m_lo, m_hi, 64):
            for n_ in range(n_lo, n_hi, n_stride):
                result.append(f"m{m_}n{n_}r0")
                result.append(f"m{m_}n{n_}r1")
                result.append(f"m{m_}n{n_}r2")
                result.append(f"m{m_}n{n_}r3")

        return result

    def areg_names(self, m=None):
        result = []
        m_lo = 0 if m is None else m
        m_hi = self.M if m is None else m + 64

        assert self.ptx_atype == "tf32", "TODO"
        k_divisor = 2  # TODO

        for m_ in range(m_lo, m_hi, 64):
            for r_ in range(0, self.get_K() // k_divisor):
                result.append(f"m{m_}r{r_}")

        return result

    def rmem_d_struct_def(self):
        sname = self.rmem_d_struct_name()
        return f"""struct {sname} {{
    {self.dreg_ctype()} {", ".join(self.dreg_names())};
    int scale_d;
}};"""

    def rmem_a_struct_def(self):
        sname = self.rmem_a_struct_name()
        return """struct {sname} {{
    {self.areg_ctype()} {", ".join(self.areg_names())};
}};"""

    def cu_utils_ss(self):
        return [
            store_d_util,
            matrix_descriptor_util,
            self.rmem_d_struct_def(),
            self.wgmma_ss_function_def(),
        ]

    def cu_utils_rs(self):
        return [
            store_d_util,
            matrix_descriptor_util,
            self.rmem_d_struct_def(),
            self.rmem_a_struct_def(),
            self.wgmma_rs_function_def(),
        ]

    # fmt: off

    def wgmma_ss_function_name(self):
        return f"exo_Sm90_mma_async_ss_m{self.M}n{self.N}_{self.ptx_dtype}_{self.ptx_atype}_{self.ptx_btype}"

    def wgmma_rs_function_name(self):
        return f"exo_Sm90_mma_async_rs_m{self.M}n{self.N}_{self.ptx_dtype}_{self.ptx_atype}_{self.ptx_btype}"

    def wgmma_ss_function_def(self):
        lines = []
        fname = self.wgmma_ss_function_name()
        params = []
        d_reftype = f"{self.dreg_ctype()}&"

        for m in range(0, self.M, 64):
            params.append(f"uint64_t a_descriptor_m{m}")
        params.append("uint64_t b_descriptor")

        for rname in self.dreg_names():
            params.append(f"{d_reftype} {rname}")

        params.append("int scale_d")

        c = "f"
        assert self.ptx_dtype == "f32", "TODO"

        instr = self.ptx_instr_name()
        lines.append(fr'EXO_CUDA_INLINE void {fname}({", ".join(params)})')
        lines.append(r"{")

        for m in range(0, self.M, 64):
            dreg_names = self.dreg_names(m=m)
            dreg_count = len(dreg_names)
            lines.append(r'  asm volatile("{\n"')
            lines.append(r'  ".reg .pred p;\n"')
            lines.append(rf'  "setp.ne.b32 p, %{dreg_count + 2}, 0;\n"')
            lines.append(rf'  "{instr} "');
            d_vec_template = "{" + ", ".join(f"%{n}" for n in range(dreg_count)) + "}"
            lines.append(rf'  "{d_vec_template}, "')
            lines.append(rf'  "%{dreg_count}, %{dreg_count+1}, p, 1, 1;\n"')
            lines.append(r'  "}"')
            d_vec_args = ", ".join(f'"+{c}"({rname})' for rname in dreg_names)
            lines.append(rf'  : {d_vec_args}')
            lines.append(rf'  : "l"(a_descriptor_m{m}), "l"(b_descriptor), "r"(scale_d)')
            lines.append(r"  );")

        lines.append(r"}")
        return "\n".join(lines)


# fmt: on


class WgmmaRmemImpl(CudaBasicDeviceVisible):
    @classmethod
    def actor_kind_permission(cls, actor_kind, is_instr):
        return cls.device_allocated_impl(actor_kind, is_instr)

    @classmethod
    def native_unit(cls):
        return cuda_warpgroup

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # We expect exactly 2D tiles to be lowered.
        # NB if the allocation is not used, the lowering will fail, as we expect
        # an Exo wgmma instr to inject the needed struct definition into exo_CudaUtil.
        element_bits = scalar_bits(prim_type)
        M, N = cls.as_const_shape(new_name, shape, srcinfo, min_dim=2, max_dim=2)
        assert prim_type == "float"  # TODO
        helper = WgmmaHelper(M, N, "f32", None, None)
        sname = helper.rmem_d_struct_name()
        return f"exo_CudaUtil::{sname} {new_name};"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        # This is really broken; we need to adjust the Exo window model
        # to reflect the reality of non-random-accessable register tiles.
        if basetyp.is_win():
            baseptr = f"*{baseptr}.data"

        # TODO enforce last two indices are :,:
        # e.g. we currently accept
        # D : f32[64,64,256]
        # D2 = D[:,:,0]
        if indices[-1] != "0" or indices[-2] != "0":
            raise MemGenError(
                f"{srcinfo}: cannot offset right 2 dimensions of {baseptr} @ {cls.name()}"
            )

        return baseptr

    @classmethod
    def window_definition(cls, ctx: WindowStructCtx):
        # TODO placeholder; this doesn't work at all.
        # We need to make windows "non-materializable" for wgmma
        if ctx.n_dims() != 2:
            raise MemGenError(
                f"{ctx.srcinfo():} Only support windows to 2D wgmma tiles"
            )
        return ctx.generate_default("Sm90_RmemMatrix", "float")


class Sm90_RmemMatrixA(WgmmaRmemImpl):
    # TODO
    pass


class Sm90_RmemMatrixD(WgmmaRmemImpl):
    pass


__all__.append("Sm90_RmemMatrixA")
__all__.append("Sm90_RmemMatrixD")


class mma_async_impl:
    def instance_impl(self, M, N, ptx_dtype, ptx_atype, ptx_btype):
        helper = WgmmaHelper(M, N, ptx_dtype, ptx_atype, ptx_btype)
        self.access_info["d"].actor_signature = sig_wgmma_rmem_d
        self.access_info["a"].actor_signature = sig_wgmma_smem
        self.access_info["b"].actor_signature = sig_wgmma_smem
        self.actor_kind = wgmma_async
        self.coll_unit = cuda_warpgroup
        self.cu_utils = helper.cu_utils_ss()

        element_size = helper.get_K() // 2
        fname = "exo_CudaUtil::" + helper.wgmma_ss_function_name()
        args = []
        for m in range(0, M, 64):
            args.append(
                "exo_CudaUtil::exo_matrix_descriptor({a}, %i, %i)" % (element_size, m)
            )
        args.append("exo_CudaUtil::exo_matrix_descriptor({b}, %i)" % (element_size,))
        for rname in helper.dreg_names():
            args.append("{d_data}.%s" % rname)
        args.append("{d_data}.scale_d")
        self.instr_format = (
            fname + "(" + ", ".join(args) + ");\n" + "{d_data}.scale_d = 1;"
        )


# For a wgmma D-matrix (in RMEM), set the scale-d flag to 0, so
# the NEXT wgmma.mma.async instruction will zero-initialize D.
# This is modelled in Exo as a zero-clear, even though the effect
# does not actually happen unless a subsequent mma.async occurs.
# In the future, I may introduce a "wgmma zero" actor signature to model this.
#
# TODO this still seems to be an issue.
@instr
class Sm90_zero_scale_d_f32:
    def behavior(M: size, N: size, d: [f32][M, N] @ Sm90_RmemMatrixD):
        for m in seq(0, M):
            for n in seq(0, N):
                d[m, n] = 0

    def instance(self):
        # XXX cuda_classic is completely wrong
        self.access_info["d"].actor_signature = sig_cuda_classic
        self.actor_kind = cuda_classic
        self.coll_unit = cuda_warpgroup
        self.instr_format = "{d_data}.scale_d = 0;"


__all__.append("Sm90_zero_scale_d_f32")


@instr
class Sm90_mma_async_tf32(mma_async_impl):
    def behavior(
        M: size,
        N: size,
        d: [f32][M, N] @ Sm90_RmemMatrixD,
        a: [f32][M / 8, 8, 8] @ Sm90_SmemSwizzled(128),
        b: [f32][N / 8, 8, 8] @ Sm90_SmemSwizzled(128),
    ):
        assert M >= 64
        assert M % 64 == 0
        assert N >= 8
        assert N % 8 == 0
        for m in seq(0, M):
            for n in seq(0, N):
                for k in seq(0, 8):
                    d[m, n] += a[m / 8, m % 8, k] * b[n / 8, n % 8, k]

    def instance(self, M, N):
        self.instance_impl(M, N, "f32", "tf32", "tf32")


__all__.append("Sm90_mma_async_tf32")


class Sm90_mma_write_d_impl:
    def instance_impl(self, helper, col_major):
        self.access_info["dst"].actor_signature = sig_cuda_classic
        self.access_info["src"].actor_signature = sig_cuda_classic
        self.actor_kind = cuda_classic
        self.coll_unit = cuda_warpgroup
        self.cu_utils = helper.cu_utils_ss()
        col_major = "true" if col_major else "false"
        lines = []

        for m in range(0, helper.M, 64):
            for reg_index, reg_name in enumerate(helper.dreg_names(m=m)):
                lines.append(
                    "exo_CudaUtil::exo_Sm90_store_d_reg<%s>({dst}, {src_data}.%s, %i, %i);"
                    % (col_major, reg_name, m, reg_index)
                )

        self.instr_format = "\n".join(lines)


@instr
class Sm90_mma_write_d_col_major_tf32(Sm90_mma_write_d_impl):
    def behavior(
        M: size,
        N: size,
        dst: [f32][N, M] @ CudaDeviceVisibleLinear,
        src: [f32][M, N] @ Sm90_RmemMatrixD,
    ):
        for m in seq(0, M):
            for n in seq(0, N):
                dst[n, m] = src[m, n]  # Transposed

    def instance(self, M, N):
        helper = WgmmaHelper(M, N, "f32", "tf32", "tf32")
        self.instance_impl(helper, True)


@instr
class Sm90_mma_write_d_row_major_tf32(Sm90_mma_write_d_impl):
    def behavior(
        M: size,
        N: size,
        dst: [f32][M, N] @ CudaDeviceVisibleLinear,
        src: [f32][M, N] @ Sm90_RmemMatrixD,
    ):
        for m in seq(0, M):
            for n in seq(0, N):
                dst[m, n] = src[m, n]

    def instance(self, M, N):
        helper = WgmmaHelper(M, N, "f32", "tf32", "tf32")
        self.instance_impl(helper, False)


__all__.append("Sm90_mma_write_d_col_major_tf32")
__all__.append("Sm90_mma_write_d_row_major_tf32")
