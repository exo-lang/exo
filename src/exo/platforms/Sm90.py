# Memory, instructions, instr-tl, sync-tl specific to CUDA sm_90 and sm_90a (H100)
# Everything exported by this module starts with Sm90_, except for timelines (tl).
from __future__ import annotations

# Currently we import from the exo.spork directory,
# which users shouldn't import directly.
from ..spork.timelines import (
    cuda_in_order_instr,
    tma_to_smem_async_instr,
    tma_to_gmem_async_instr,
    wgmma_async_instr,
    cuda_in_order,
    tma_to_smem_async,
    tma_to_gmem_async,
    tma_to_gmem_async_qual,
    wgmma_async,
    wgmma_async_smem,
    wgmma_fence_1,
    wgmma_fence_2,
    cuda_async_proxy,
    cuda_async_proxy_wgmma,
    cuda_generic_and_async_proxy,
    wgmma_async_rmem_a_qual,
    wgmma_async_rmem_d_qual,
    cuda_rmem_qual_tl_dict,
    cuda_ram_qual_tl_dict,
    wgmma_zero_instr,
    wgmma_zero_qual,
)

__all__ = [
    "tma_to_smem_async_instr",
    "tma_to_gmem_async_instr",
    "wgmma_async_instr",
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
from ..API import (
    instr,
    memwin_template,
    window_encoder,
    window_indexer,
    Memory,
    MemGlobalC,
    MemIncludeC,
    SpecialWindow,
    WindowFeatures,
    WindowEncoder,
    WindowIndexer,
    ScalarInfo,
    UtilInjector,
    CIR_Wrapper,
    InstrInfo,
    InstrArgs,
    AtomicityInfo,
)
from ..spork.cuda_memory import *
from ..spork.coll_algebra import (
    cuda_warp,
    cuda_warpgroup,
    cuda_warp_in_cluster,
    cuda_warp_in_cluster_strided,
)


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

    @window_encoder(SwizzledEncoder)
    @window_indexer(SwizzledIndexer)
    class SwizzledImpl(CudaSmemAtomicity16B):
        @classmethod
        def global_(cls):
            return f"""typedef struct {c_matrices} {{
    char matrix_bytes[{matrix_bytes * num_matrices}];
#ifdef __CUDACC__
    template <typename T>
    EXO_CUDA_INLINE T& swizzled_element(unsigned linear_offset)
    {{
        // Adapted from ThunderKittens appendix which actually documents CUDA correctly.
        uintptr_t addr = reinterpret_cast<uintptr_t>(this) + sizeof(T) * linear_offset;
        return *reinterpret_cast<T*>(addr ^ (((addr % sizeof(*this)) >> 7) << 4));
    }}

    EXO_CUDA_INLINE {c_matrices}& raw_byte_offset(unsigned byte_offset)
    {{
        // Offset without swizzling, for passing to TMA/wgmma instruction.
        // We further preserve the matrix type, for exo_matrix_descriptor to inspect.
        return reinterpret_cast<{c_matrices}&>(matrix_bytes[byte_offset]);
    }}

    static constexpr uint64_t get_swizzle_bits()
    {{
        return {swizzle_bits};
    }}
#endif
}} {c_matrices};"""

        @classmethod
        def can_read(cls):
            return True

        @classmethod
        def write(cls, s, lhs, rhs):
            return f"{lhs} = {rhs};"

        @classmethod
        def reduce(cls, s, lhs, rhs):
            return f"{lhs} += {rhs};"

        @classmethod
        def packed_tensor_shape(cls, scalar_info: ScalarInfo):
            # 2, 4, or 8 core matrices
            return (matrix_mn, num_matrices * cls.get_matrix_k(scalar_info))

        @classmethod
        def smem_config(cls, inputs: SmemConfigInputs) -> SmemConfig:
            return SmemConfig(f"{c_matrices} (&)[]", 128)

        @classmethod
        def get_matrix_k(cls, scalar_info):
            return 128 // ScalarInfo(scalar_info).bits

        @classmethod
        def get_swizzle_bits(cls):
            return swizzle_bits

        @classmethod
        def get_swizzle(cls):
            return swizzle

    return SwizzledImpl


def Sm90_get_mma_smem(swizzle):
    if swizzle == 0:
        return CudaSmemLinear
    else:
        return Sm90_SmemSwizzled(swizzle)


__all__.append("Sm90_SmemSwizzled")
__all__.append("Sm90_get_mma_smem")


window_struct_template = """\
struct {sname} {{
    {const_keyword}Sm90_SmemMatrices_SW{swizzle}* data;
    int32_t strides[{n_dims}];
}};"""


class SwizzledEncoder(WindowEncoder):
    __slots__ = []

    def define_struct(self, depends_on: list) -> str:
        sname = self.exo_struct_name()
        const_keyword = "const " if self.const else ""
        return window_struct_template.format(
            sname=sname,
            swizzle=self.mem.get_swizzle(),
            n_dims=self.n_dims,
            const_keyword=const_keyword,
        )

    def supports_dim_change(self) -> bool:
        return True

    def encode_window(self, utils: UtilInjector, features: WindowFeatures) -> str:
        sname = self.exo_struct_name()
        mem = features.get_mem()
        n_dims = features.n_array_dims()

        dataptr, filtered_strides = features.strided_window_helper()
        strides = "{" + ", ".join(str(s) for s in filtered_strides) + "}"
        return f"(struct {sname}) {{ {dataptr}, {strides} }}"

    def decode_array_offset(
        self, utils: UtilInjector, window: CIR_Wrapper, n: int
    ) -> int:
        return 0

    def decode_array_stride_as_packed(
        self, utils: UtilInjector, window: CIR_Wrapper, n: int
    ) -> CIR_Wrapper:
        return window.strides[n]


class SwizzledIndexer(WindowIndexer):
    __slots__ = []

    def index(self, utils, features: WindowFeatures, for_wgmma=False):
        ctype = self.ctype()
        mem = features.get_mem()

        dataptr = features.get_dataptr()
        array_offset = 0
        for i in range(features.n_array_dims()):
            array_offset += features.get_array_offset(
                i
            ) * features.get_array_stride_as_packed(i)

        assert features.n_packed_dims() == 2
        assert self.element_bits() >= 8, "TODO implement float4 etc."
        if for_wgmma:
            # This passes a non-swizzled pointer to wgmma or TMA.
            # wgmma cannot offset packed M/N.
            features.get_packed_offset(0).exo_expect_int(0)
            byte_offset = features.get_packed_offset(1)  # Offset packed K
            byte_offset *= self.element_bits() // 8
            code = f"{dataptr}[{array_offset}].raw_byte_offset({byte_offset})"
        else:
            # This path swizzles the pointer.
            # Needed for access with normal scalar code.
            packed_shape = features.packed_tensor_shape()
            linear_offset = features.get_packed_offset(0) * packed_shape[
                1
            ] + features.get_packed_offset(1)
            code = (
                f"{dataptr}[{array_offset}].swizzled_element<{ctype}>({linear_offset})"
            )

        return self.pack_result(code, False)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# CUtensorMap
# This is a 128-byte blob created on the CPU and passed to CUDA device functions
# which use it to execute TMA (cp.async.bulk) instructions.
# In Exo, we model this as a SpecialWindow (similar to Memory)
# that is constructed as a window to CudaGmemLinear.
@memwin_template
def Sm90_tensorMap(swizzle, *smem_box):
    # Minimal SMEM box: we allow copies that reduce dimensionality
    # where the removed dimensions have extent size 1.
    # For example, for batched GEMM, we could have a GMEM window
    # of size [1, Ms, Ks] copied to an SMEM tensor sized [Ms, Ks].
    # The size of the destination window is the minimal SMEM box.
    #
    # To actually use this functionality, you have to pass smem_box
    # to instrs explicitly as a keyword argument.
    #
    # TODO this consumes dims that count against the 5 dim limit.
    # More flexible alternatives exist that don't map 1:1
    # with CUDA (e.g. manually offset the ptr on the device???) but
    # this interacts subtly with TMA's built-in bounds checking.
    rank = len(smem_box)
    assert 1 <= rank <= 5
    assert swizzle in (0, 32, 64, 128)

    @window_encoder(TensorMapEncoder)
    class CUtensorMap(SpecialWindow):
        @classmethod
        def global_(cls):
            return ""

        @classmethod
        def device_permission(cls, device, instr_tl):
            return CudaBasicDeviceVisible.host_allocated_impl(
                device, instr_tl, pinned=False
            )

        qual_tl_dict = cuda_ram_qual_tl_dict

        @classmethod
        def source_memory_type(cls):
            return CudaGmemLinear

        @classmethod
        def swizzle(cls):
            return swizzle

        @classmethod
        def smem_box(cls):
            return smem_box

        @classmethod
        def rank(cls):
            return rank

    return CUtensorMap


__all__.append("Sm90_tensorMap")


class TensorMapEncoder(WindowEncoder):
    def separate_dataptr(self):
        return True

    def define_struct(self, depends_on: list):
        rank = self.mem.rank()
        sdef = CUtensorMap_window_template.format(
            rank=rank, sname=self.exo_struct_name()
        )
        strides_sname = f"exo_Sm90_CUtensorMap_{rank}_strides"
        dim_sname = f"exo_Sm90_CUtensorMap_{rank}_dim"
        depends_on.append(
            MemGlobalC(strides_sname, CUtensorMap_strides_template.format(rank=rank))
        )
        depends_on.append(
            MemGlobalC(dim_sname, CUtensorMap_dim_template.format(rank=rank))
        )
        depends_on.append(MemIncludeC("cuda.h"))
        return sdef

    def supports_dim_change(self):
        return True

    def supports_special_dim_change(self):
        return True

    def dataptr_ctype(self):
        return "CUtensorMap"

    def encode_window(self, utils, features: WindowFeatures):
        """Convert from one window struct to another; just encode offsets

        We have special handling allowing points to substitute for
        intervals of size 1. All of this needs to be tested later.
        For now, we require the input to have dimensionality
        equal to that of the original SMEM box, i.e. if we window a window,
        only the last window expression may have points.

        WindowFeatures get_array_offset() needs to be clarified
        and improved if we wish to lift this restriction.

        """
        mem = features.get_mem()
        dim = features.n_array_dims()
        smem_box = mem.smem_box()
        if dim != len(smem_box):
            raise ValueError(
                f"{features.srcinfo()}: "
                f"taking window of {dim}d tensor not supported given SMEM box {smem_box}; "
                f"NOTE, window-of-window case should have points (non-intervals lo:hi) "
                f"only for the final window expression"
            )

        for i in range(dim):
            cir_size = features.get_array_interval_size(i)
            if cir_size is None:
                box_coord = smem_box[i]
                if box_coord != 1:
                    cir_off = features.get_array_offset(i)
                    raise ValueError(
                        f"{features.srcinfo()}: "
                        f"Unexpected point expression {cir_off}; "
                        f"not allowed for non-1 box size {box_coord} on dimension {dim} (0-indexed)"
                    )

        # This code also requires the above dim != ... check to function.
        init = (
            "{ {"
            + ", ".join(str(features.get_array_offset(i)) for i in range(dim))
            + "} }"
        )
        return f"({self.exo_struct_name()}) {init}"

    def encode_separate_dataptr(self, utils, features: WindowFeatures):
        return features.get_dataptr()

    def encode_special_window(self, utils, features: WindowFeatures):
        """For CudaGmemLinear -> CUtensorMap conversion.

        The window struct is just 0-initialized

        """
        init = "{ {" + ", ".join("0" for i in range(features.n_array_dims())) + "} }"
        return f"({self.exo_struct_name()}) {init}"

    def encode_special_separate_dataptr(
        self, utils: UtilInjector, features: WindowFeatures
    ):
        """For CudaGmemLinear -> CUtensorMap conversion. Make CUtensorMap blob"""
        sname = self.exo_struct_name()
        rank = self.n_dims
        swizzle = self.mem.swizzle()
        if swizzle == 0:
            cu_swizzle = "CU_TENSOR_MAP_SWIZZLE_NONE"
        else:
            cu_swizzle = f"CU_TENSOR_MAP_SWIZZLE_{swizzle}B"
        # CUDA boxDim in opposite order as Exo smem_box
        cu_boxDim = "{ " + ", ".join(str(n) for n in self.mem.smem_box()[::-1]) + " }"
        try:
            cu_type = CUtensorMap_type_dict[self.scalar_info.shorthand]
        except KeyError as e:
            raise TypeError("CUtensorMap: doesn't currently support {e}")

        kwargs = dict(
            sname=sname,
            rank=rank,
            cu_swizzle=cu_swizzle,
            cu_boxDim=cu_boxDim,
            cu_ctype_enum=cu_type[0],
            stride_suffix=cu_type[1],
        )
        utils.add_c_include("stdio.h")
        utils.add_c_include("assert.h")
        utils.add_c_util(CUtensorMap_encode_template.format(**kwargs))

        cw_dataptr, cw_strides = features.strided_window_helper()
        cw_dim = features.array_interval_sizes_without_points()
        assert features.n_packed_dims() == 0
        assert len(cw_strides) == rank
        assert len(cw_dim) == rank

        strides = (
            f"(exo_Sm90_CUtensorMap_{rank}_strides)"
            + "{ {"
            + ", ".join(str(s) for s in cw_strides)
            + "} }"
        )
        dim = (
            f"(exo_Sm90_CUtensorMap_{rank}_dim)"
            + "{ {"
            + ", ".join(str(s) for s in cw_dim)
            + "} }"
        )
        return f"{sname}_encode({cw_dataptr}, {strides}, {dim})"

    def decode_array_offset(self, utils, window: CIR_Wrapper, n: int):
        return window.C_offsets[n]


def _validate_smem_box(
    smem_box_arg: Optional[Tuple[int]], default_smem_box: Tuple[int]
):
    if smem_box_arg is None:
        return default_smem_box
    # fmt: off
    assert all(isinstance(c, int) for c in smem_box_arg), "Non-integer smem_box given"
    # fmt: on
    minimal_arg = [c for c in smem_box_arg if c != 1]
    minimal_expected = [c for c in default_smem_box if c != 1]
    if minimal_arg != minimal_expected:
        raise ValueError(
            f"smem_box {smem_box_arg} isn't compatible with expected box "
            f"{default_smem_box} (we allow extra 1's but that's it)"
        )
    return tuple(smem_box_arg)


# str.format templates for CUtensorMap-related Exo window C definition
CUtensorMap_window_template = """\
typedef struct {sname} {{
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned C_offsets[{rank}];
}} {sname};
"""

CUtensorMap_strides_template = """\
typedef struct exo_Sm90_CUtensorMap_{rank}_strides {{
    // Stored in reverse-order as the raw CUtensorMap,
    // and in element count, not in bytes.
    // Leftmost stride is most-significant.
    unsigned C_strides[{rank}];
}} exo_Sm90_CUtensorMap_{rank}_strides;
"""

CUtensorMap_dim_template = """\
typedef struct exo_Sm90_CUtensorMap_{rank}_dim {{
    // Stored in the reverse-order as the raw CUtensorMap.
    // Leftmost dimension is the most-significant.
    unsigned C_dim[{rank}];
}} exo_Sm90_CUtensorMap_{rank}_dim;
"""

CUtensorMap_encode_template = """\
static inline CUtensorMap {sname}_encode(
        // Window dataptr, strides
        const void* globalAddress, exo_Sm90_CUtensorMap_{rank}_strides gmem_stride,
        // Tensor size
        exo_Sm90_CUtensorMap_{rank}_dim gmem_dim)
{{
    assert(gmem_stride.C_strides[{rank} - 1] == 1);

    CUtensorMap tensorMap;
    const CUtensorMapSwizzle swizzle = {cu_swizzle};

    cuuint64_t globalDim[{rank}];
    cuuint64_t allGlobalStrides[{rank}];  // allGlobalStrides[0] unused by CUDA
    cuuint32_t elementStrides[{rank}];

    // We translate from the Exo ordering (leftmost stride is most-significant)
    // to the CUDA ordering (leftmost stride is least-significant).
    for (uint32_t cu_idx = 0; cu_idx < {rank}; ++cu_idx) {{
        const uint32_t C_idx = {rank} - 1 - cu_idx;
        globalDim[cu_idx] = gmem_dim.C_dim[C_idx];
        allGlobalStrides[cu_idx] = ((cuuint64_t)gmem_stride.C_strides[C_idx]){stride_suffix};
        elementStrides[cu_idx] = 1;
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
    "ui8": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", ""),
    "ui16": ("CU_TENSOR_MAP_DATA_TYPE_UINT16", " * 2"),
    "ui32": ("CU_TENSOR_MAP_DATA_TYPE_UINT32", " * 4"),
    "i32": ("CU_TENSOR_MAP_DATA_TYPE_INT32", " * 4"),
    "u64": ("CU_TENSOR_MAP_DATA_TYPE_UINT64", " * 8"),
    "i64": ("CU_TENSOR_MAP_DATA_TYPE_INT64", " * 8"),
    "f16": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT16", " * 2"),
    "f32": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT32", " * 4"),
    "f64": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT64", " * 8"),
    "bf16": ("CU_TENSOR_MAP_DATA_TYPE_BFLOAT16", " * 2"),
    "u4": ("CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B", " / 2"),
}

_tma_elect_one_prefix = r"""// cute::elect_one_sync
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

_tma_get_rank_prefix = """constexpr auto rank = sizeof(window.C_offsets) / sizeof(window.C_offsets[0]);
    static_assert(rank >= 1 && rank <= 5);"""


def copy_tensor_to_smem_util(multicast: bool):
    cache_hint = 1152921504606846976  # copied from cutlass PTX

    # fmt: off
    # Note: indentation of code-in-strings here is dictated by output C++ requirements.
    expect_tx = f'asm("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" :: "r"(exo_tma_mbarrier), "r"(expect_tx));'

    def rank_case(rank: int):
        vector_fmt = "{" + ", ".join(f"%{r+2}" for r in range(rank)) + "}"
        ptx_fmt = f" [%0], [%1, {vector_fmt}], [%{rank+2}], %{rank+3}"
        if multicast:
            ptx_fmt += f", %{rank+4}"
        vector_args = [f'"r"(window.C_offsets[{rank - 1 - r}])' for r in range(0, rank)]
        vector_values = ", ".join(vector_args)
        if multicast:
            return f"""if constexpr (rank == {rank}) {{
            asm volatile(
                "cp.async.bulk.tensor.{rank}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                "{ptx_fmt};"
                :
                : "r"(exo_smemU32(dst)), "l"(&tensorMap), {vector_values},
                  "r"(exo_tma_mbarrier), "h"(cta_mask), "n"({cache_hint})
                : "memory");
        }}"""
        else:
            return f"""if constexpr (rank == {rank}) {{
            asm volatile(
            "cp.async.bulk.tensor.{rank}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            "{ptx_fmt};"
            :
            : "r"(exo_smemU32(dst)), "l"(&tensorMap), {vector_values},
              "r"(exo_tma_mbarrier), "n"({cache_hint})
            : "memory");
        }}"""

    if multicast:
        return f"""template <typename WindowOffsets>
EXO_CUDA_INLINE void
exo_Sm90_tma_to_smem_multicast(
        void* dst, const CUtensorMap& tensorMap, WindowOffsets window,
        uint32_t exo_tma_mbarrier, uint32_t expect_tx, uint16_t cta_mask)
{{
    {_tma_get_rank_prefix}
    {_tma_elect_one_prefix}
    if (pred) {{
        {expect_tx}
        {rank_case(1)}
        {rank_case(2)}
        {rank_case(3)}
        {rank_case(4)}
        {rank_case(5)}
    }}
}}"""
    else:
        return f"""template <typename WindowOffsets>
EXO_CUDA_INLINE void
exo_Sm90_tma_to_smem(
        void* dst, const CUtensorMap& tensorMap, WindowOffsets window,
        uint32_t exo_tma_mbarrier, uint32_t expect_tx)
{{
    {_tma_get_rank_prefix}
    {_tma_elect_one_prefix}
    if (pred) {{
        {expect_tx}
        {rank_case(1)}
        {rank_case(2)}
        {rank_case(3)}
        {rank_case(4)}
        {rank_case(5)}
    }}
}}"""
    # fmt: on


def copy_tensor_to_gmem_util(is_reduce: bool):
    # fmt: off
    def rank_case(rank: int):
        vector_fmt = "{" + ", ".join(f"%{r+1}" for r in range(rank)) + "}"
        ptx_fmt = f" [%0, {vector_fmt}], [%{rank+1}]"
        vector_args = [f'"r"(window.C_offsets[{rank - 1 - r}])' for r in range(0, rank)]
        vector_values = ", ".join(vector_args)
        return f"""if constexpr (rank == {rank}) {{
            asm volatile(
                "cp.{reduce_dot}async.bulk.tensor.{rank}d.global.shared::cta.{add_dot}tile.bulk_group"
                "{ptx_fmt};"
                :
                : "l"(&tensorMap),
                  {vector_values},
                  "r"(exo_smemU32(src))
                : "memory");
        }}"""
    _reduce = "_reduce" if is_reduce else ""
    reduce_dot = "reduce." if is_reduce else ""
    add_dot = "add." if is_reduce else ""

    return f"""template <typename WindowOffsets>
EXO_CUDA_INLINE void
exo_Sm90_tma_to_gmem{_reduce}(const CUtensorMap& tensorMap, WindowOffsets window, const void* src)
{{
    {_tma_get_rank_prefix}
    {_tma_elect_one_prefix}
    if (pred) {{
        {rank_case(1)}
        {rank_case(2)}
        {rank_case(3)}
        {rank_case(4)}
        {rank_case(5)}
    }}
}}"""

    # fmt: on


class copy_tensor_to_smem_impl(InstrInfo):
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
        self.access_info["dst"].mem = Sm90_get_mma_smem(swizzle)
        self.access_info["dst"].out_of_order = True
        self.access_info["src"].mem = Sm90_tensorMap(swizzle, *smem_box)
        self.access_info["src"].out_of_order = True
        self.access_info["src"].allow_out_of_bounds = True  # GMEM special case
        self.instr_tl = tma_to_smem_async_instr
        self.coll_unit = cuda_warp
        self.cu_utils.append(copy_tensor_to_smem_util(False))
        self.barrier_type = CudaMbarrier
        self.smem_box = smem_box
        self.swizzle = swizzle
        self.element_bits = element_bits

    def codegen(self, args: InstrArgs):
        box = self.smem_box
        lines = [f"exo_CudaUtil::exo_Sm90_tma_to_smem("]
        if self.swizzle:
            smem_data = args.dst.index(for_wgmma=True)
        else:
            smem_data = args.dst.index()
        CUtensorMap = args.src.get_separate_dataptr()
        src_struct = args.src.get_window()
        lines.append(f"  &{smem_data},")
        lines.append(f"  {CUtensorMap},")
        lines.append(f"  {src_struct},")
        lines.append(f"  {args.exo_barrier},")
        lines.append(f"  {prod(box) * self.element_bits // 8}")
        lines.append(");")
        return lines


class copy_tensor_to_gmem_impl(InstrInfo):
    def instance_impl(self, smem_box, swizzled, element_bits, is_reduce):
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
        self.access_info["src"].mem = Sm90_get_mma_smem(swizzle)
        self.access_info["src"].out_of_order = True
        self.access_info["dst"].mem = Sm90_tensorMap(swizzle, *smem_box)
        self.access_info["dst"].out_of_order = True
        self.access_info["dst"].allow_out_of_bounds = True  # GMEM special case
        if is_reduce:
            self.access_info["dst"].atomicity = AtomicityInfo([tma_to_gmem_async_qual])
        self.instr_tl = tma_to_gmem_async_instr
        self.coll_unit = cuda_warp
        self.cu_utils.append(copy_tensor_to_gmem_util(is_reduce))
        self.smem_box = smem_box
        self.element_bits = element_bits
        self.swizzle = swizzle
        self.is_reduce = is_reduce

    def codegen(self, args: InstrArgs):
        box = self.smem_box
        _reduce = "_reduce" if self.is_reduce else ""
        lines = [f"exo_CudaUtil::exo_Sm90_tma_to_gmem{_reduce}("]
        if self.swizzle:
            smem_data = args.src.index(for_wgmma=True)
        else:
            smem_data = args.src.index()
        CUtensorMap = args.dst.get_separate_dataptr()
        dst_struct = args.dst.get_window()
        lines.append(f"  {CUtensorMap},")
        lines.append(f"  {dst_struct},")
        lines.append(f"  &{smem_data}")
        lines.append(");")
        return lines


@instr
class Sm90_copy_tensor_to_smem_linear_2f32(copy_tensor_to_smem_impl):
    def behavior(
        size0: size, size1: size, dst: [f32][size0, size1], src: [f32][size0, size1]
    ):
        # We need to assert that the dst is densely packed.
        assert stride(dst, 1) == 1
        assert stride(dst, 0) == size1
        # src must be densely packed in last dimension
        assert stride(src, 1) == 1

        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]

    def instance(self, size0, size1, *, smem_box: Optional[Tuple[int]] = None):
        smem_box = _validate_smem_box(smem_box, (size0, size1))
        self.instance_impl(smem_box, False, 32)


__all__.append("Sm90_copy_tensor_to_smem_linear_2f32")


@instr
class Sm90_copy_tensor_to_smem_swizzled_2f32(copy_tensor_to_smem_impl):
    def behavior(
        size0: size,
        size1: size,
        dst: [f32][size0 / 8, 8, size1],
        src: [f32][size0, size1],
    ):
        assert size0 % 8 == 0
        assert size0 >= 8
        # We need to assert that the dst is densely packed.
        assert stride(dst, 2) == 1
        assert stride(dst, 1) == size1
        assert stride(dst, 0) == size1 * 8
        # src must be densely packed in last dimension
        assert stride(src, 1) == 1

        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0 / 8, i0 % 8, i1] = src[i0, i1]

    def instance(self, size0, size1, *, smem_box: Optional[Tuple[int]] = None):
        smem_box = _validate_smem_box(smem_box, (size0, size1))
        self.instance_impl(smem_box, True, 32)


__all__.append("Sm90_copy_tensor_to_smem_swizzled_2f32")


@instr
class Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(InstrInfo):
    smem_box: Tuple[int]
    swizzle: int
    element_bits: int
    coop_stride: int

    def behavior(
        ncta: size,
        size0: size,
        size1: size,
        dst: [f32][ncta, size0 / 8, 8, size1],
        src: [f32][size0, size1],
    ):
        assert size0 % 8 == 0
        assert size0 >= 8
        # We need to assert that the dst is densely packed.
        assert stride(dst, 3) == 1
        assert stride(dst, 2) == size1
        assert stride(dst, 1) == size1 * 8
        # src must be densely packed in last dimension
        assert stride(src, 1) == 1

        for cta in seq(0, ncta):
            for i0 in seq(0, size0):
                for i1 in seq(0, size1):
                    dst[cta, i0 / 8, i0 % 8, i1] = src[i0, i1]

    def instance(
        self, size0, size1, ncta, *, cta_stride, smem_box: Optional[Tuple[int]] = None
    ):
        assert size0 % (8 * ncta) == 0
        # The (size0, size1) copy is implemented as
        # ncta-many (coop_stride, size1) copies.
        coop_stride = size0 // ncta
        smem_box = _validate_smem_box(smem_box, (coop_stride, size1))
        self.instance_impl(smem_box, True, 32, ncta, cta_stride, coop_stride)

    def instance_impl(
        self, smem_box, swizzled, element_bits, ncta, cta_stride, coop_stride
    ):
        element_bits = 32
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
            assert 0, "not implemented, non-swizzled SMEM for TMA multicast"
            swizzle = 0
        self.access_info["dst"].mem = Sm90_get_mma_smem(swizzle)
        self.access_info["dst"].out_of_order = True
        self.access_info["src"].mem = Sm90_tensorMap(swizzle, *smem_box)
        self.access_info["src"].out_of_order = True
        self.access_info["src"].allow_out_of_bounds = True  # GMEM special case
        self.instr_tl = tma_to_smem_async_instr
        self.coll_unit = ncta * cuda_warp_in_cluster_strided(cta_stride)
        self.cu_utils.append(copy_tensor_to_smem_util(True))
        self.barrier_type = CudaMbarrier
        self.smem_box = smem_box
        self.element_bits = element_bits
        self.coop_stride = coop_stride
        self.swizzle = swizzle
        self.access_info["dst"].distributed_coll_units = [cuda_cta_in_cluster]
        self.access_info["dst"].access_by_owner_only = False
        self.barrier_coll_units = [cuda_cta_in_cluster]

    def codegen(self, args: InstrArgs):
        coop_stride = self.coop_stride
        lines = [f"exo_CudaUtil::exo_Sm90_tma_to_smem_multicast("]
        cta_idx = args.exo_wrap_cir(f"(blockIdx.x / {args.cta_stride}) % {args.ncta}")
        if self.swizzle:
            # dst[ncta, size0 / 8, 8, size1]
            # [ncta] corresponds to a distributed dimension, not indexed here
            # so per-CTA we have dst[size0 / 8, 8, size1]
            # We want to offset on the size0 dimension by (cta_idx * coop_stride)
            # which we have to divide by 8 due to the split dim.
            assert self.coop_stride % 8 == 0
            smem_data = args.dst.index(cta_idx * (coop_stride // 8), for_wgmma=True)
        else:
            assert 0, "not implemented: non-swizzled SMEM for TMA multicast"
        CUtensorMap = args.src.get_separate_dataptr()
        # src[size0, size1]
        # Each CTA handles
        # src[cta_idx * coop_stride : cta_idx * coop_stride + coop_stride, :]
        # Note, if src is a window taken from a tensor of higher dimensionality,
        # the WindowFeatures infrastructure ensures the (cta_idx * ...) offset
        # is applied to the correct dimension.
        src_struct = args.src[cta_idx * coop_stride : (cta_idx + 1) * coop_stride]
        lines.append(f"  &{smem_data},")
        lines.append(f"  {CUtensorMap},")
        lines.append(f"  {src_struct},")
        lines.append(f"  {args.exo_barrier},")
        lines.append(f"  {args.ncta * prod(self.smem_box) * self.element_bits // 8},")
        lines.append(f"  {args.exo_cta_mask}")
        lines.append(");")
        return lines


__all__.append("Sm90_multicast_copy_tensor_to_smem_swizzled_2f32")


@instr
class Sm90_copy_tensor_to_gmem_linear_2f32(copy_tensor_to_gmem_impl):
    def behavior(
        size0: size,
        size1: size,
        dst: [f32][size0, size1],
        src: [f32][size0, size1],
    ):
        # We need to assert that the src is densely packed.
        assert stride(src, 0) == size1
        assert stride(src, 1) == 1
        # dst must be densely packed in the last dimension (CUtensorMap requirement)
        assert stride(dst, 1) == 1

        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]

    def instance(self, size0, size1, *, smem_box: Optional[Tuple[int]] = None):
        smem_box = _validate_smem_box(smem_box, (size0, size1))
        self.instance_impl(smem_box, False, 32, False)


__all__.append("Sm90_copy_tensor_to_gmem_linear_2f32")


@instr
class Sm90_copy_tensor_to_gmem_swizzled_2f32(copy_tensor_to_gmem_impl):
    def behavior(
        size0: size,
        size1: size,
        dst: [f32][size0, size1],
        src: [f32][size0 / 8, 8, size1],
    ):
        assert size0 % 8 == 0
        assert size0 >= 8
        # We need to assert that the SMEM src is densely packed.
        assert stride(src, 2) == 1
        assert stride(src, 1) == size1
        assert stride(src, 0) == size1 * 8
        # dst must be densely packed in last dimension
        assert stride(dst, 1) == 1

        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0 / 8, i0 % 8, i1]

    def instance(self, size0, size1, *, smem_box: Optional[Tuple[int]] = None):
        smem_box = _validate_smem_box(smem_box, (size0, size1))
        self.instance_impl(smem_box, True, 32, False)


__all__.append("Sm90_copy_tensor_to_gmem_swizzled_2f32")


@instr
class Sm90_reduce_tensor_to_gmem_linear_2f32(copy_tensor_to_gmem_impl):
    def behavior(
        size0: size,
        size1: size,
        dst: [f32][size0, size1],
        src: [f32][size0, size1],
    ):
        # We need to assert that the src is densely packed.
        assert stride(src, 0) == size1
        assert stride(src, 1) == 1
        # dst must be densely packed in the last dimension (CUtensorMap requirement)
        assert stride(dst, 1) == 1

        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] += src[i0, i1]

    def instance(self, size0, size1, *, smem_box: Optional[Tuple[int]] = None):
        smem_box = _validate_smem_box(smem_box, (size0, size1))
        self.instance_impl(smem_box, False, 32, True)


__all__.append("Sm90_reduce_tensor_to_gmem_linear_2f32")


@instr
class Sm90_reduce_tensor_to_gmem_swizzled_2f32(copy_tensor_to_gmem_impl):
    def behavior(
        size0: size,
        size1: size,
        dst: [f32][size0, size1],
        src: [f32][size0 / 8, 8, size1],
    ):
        assert size0 % 8 == 0
        assert size0 >= 8
        # We need to assert that the SMEM src is densely packed.
        assert stride(src, 2) == 1
        assert stride(src, 1) == size1
        assert stride(src, 0) == size1 * 8
        # dst must be densely packed in last dimension
        assert stride(dst, 1) == 1

        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] += src[i0 / 8, i0 % 8, i1]

    def instance(self, size0, size1, *, smem_box: Optional[Tuple[int]] = None):
        smem_box = _validate_smem_box(smem_box, (size0, size1))
        self.instance_impl(smem_box, True, 32, True)


__all__.append("Sm90_reduce_tensor_to_gmem_swizzled_2f32")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# wgmma: warpgroup matrix multiply-accumulate


# TODO support more than float
store_d_util = """template <bool ColumnMajor, int32_t RegIndex, typename Window, typename Reg>
EXO_CUDA_INLINE void exo_Sm90_store_d_reg(
        Window dst, Reg value,
        int32_t m_offset, int32_t M_end, int32_t N_end)
{
    const uint32_t tid = threadIdx.x % 128u;
    const int32_t r_base = int32_t((tid / 32) * 16 + (tid % 32) / 4);
    const int32_t c_base = int32_t(tid % 4) * 2;
    const int32_t r = m_offset + r_base + ((RegIndex % 4) / 2) * 8;
    const int32_t c = c_base + (RegIndex / 4) * 8 + (RegIndex % 2);
    auto dst_ptr = reinterpret_cast<Reg*>(
            &dst.data[c * dst.strides[!ColumnMajor] + r * dst.strides[ColumnMajor]]);
    if (int32_t(r) < M_end && int32_t(c) < N_end) {
        *dst_ptr = value;
    }
}
"""

matrix_descriptor_util = """\
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor_encode(uint32_t val)
{
    return (val & 0x3FFFF) >> 4;
}

template <typename PackedMatrix>
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor(
    PackedMatrix* ptr, uint32_t mn_stride_as_packed, uint32_t mn_offset = 0)
{
    static_assert(sizeof(PackedMatrix) > 8, "Write a new impl for non-swizzled stuff");
    uint64_t mn_stride = mn_stride_as_packed * sizeof(PackedMatrix);
    return exo_matrix_descriptor_encode(exo_smemU32(ptr) + (mn_offset / 8u) * mn_stride)
           | exo_matrix_descriptor_encode(16) << 16u
           | exo_matrix_descriptor_encode(mn_stride) << 32u
           | uint64_t(PackedMatrix::get_swizzle_bits()) << 62;
}
"""


@dataclass(slots=True)
class WgmmaHelper:
    # wgmma.mma_async.sync.aligned.m{M}n{N}k{get_K()}.{ptx_dtype}.{ptx_atype}.{ptx.btype}
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
            self.wgmma_ss_function_def(),
        ]

    def cu_utils_rs(self):
        return [
            store_d_util,
            matrix_descriptor_util,
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


class Sm90_RmemMatrixA:
    # TODO implement this

    qual_tl_dict = cuda_rmem_qual_tl_dict | {wgmma_async_instr: wgmma_async_rmem_a_qual}


@memwin_template
def Sm90_RmemMatrixD(M, N):
    helper = WgmmaHelper(M, N, "f32", None, None)

    @window_indexer(RmemIndexer)
    class Sm90_RmemMatrixD(CudaBasicDeviceVisible):
        @classmethod
        def global_(cls):
            return helper.rmem_d_struct_def()

        qual_tl_dict = cuda_rmem_qual_tl_dict | {
            wgmma_async_instr: [wgmma_async_rmem_d_qual, wgmma_zero_qual]
        }

        @classmethod
        def device_permission(cls, device, instr_tl):
            return cls.device_allocated_impl(device, instr_tl)

        @classmethod
        def native_unit(cls):
            return cuda_warpgroup

        @classmethod
        def alloc(cls, new_name, prim_type, shape, srcinfo):
            element_bits = scalar_bits(prim_type)
            shape = cls.as_const_shape(new_name, shape, srcinfo, min_dim=2)
            array_shape = shape[:-2]
            assert prim_type == "float"  # TODO
            sname = helper.rmem_d_struct_name()
            arrays = "".join(f"[{s}]" for s in array_shape)
            return f"{sname} {new_name}{arrays};"

        @classmethod
        def free(cls, new_name, prim_type, shape, srcinfo):
            return ""

        @classmethod
        def packed_tensor_shape(cls, typ):
            return (M, N)

    return Sm90_RmemMatrixD


__all__.append("Sm90_RmemMatrixA")
__all__.append("Sm90_RmemMatrixD")


class RmemIndexer(WindowIndexer):
    def index(self, utils, features: WindowFeatures):
        code = features.get_dataptr()
        for i in range(features.n_array_dims()):
            code = code[features.get_array_offset(i)]
        return self.pack_result(code, False)


class mma_async_impl(InstrInfo):
    __slots__ = ["helper"]

    def instance_impl(self, M, N, ptx_dtype, ptx_atype, ptx_btype):
        helper = WgmmaHelper(M, N, ptx_dtype, ptx_atype, ptx_btype)
        self.helper = helper
        self.instr_tl = wgmma_async_instr
        self.coll_unit = cuda_warpgroup
        self.cu_utils = helper.cu_utils_ss()
        self.access_info["a"].out_of_order = True
        self.access_info["b"].out_of_order = True
        self.access_info["d"].out_of_order = False
        self.access_info["d"].mem = Sm90_RmemMatrixD(M, N)

    def codegen(self, args):
        helper = self.helper
        fname = "exo_CudaUtil::" + helper.wgmma_ss_function_name()
        lines = []
        lines.append(f"{fname}(")
        for m in range(0, args.M, 64):
            ref = args.a.index(for_wgmma=True)
            strides = args.a.to_strides_as_packed()
            lines.append(
                f"  exo_CudaUtil::exo_matrix_descriptor(&{ref}, {strides[0]}, {m}),"
            )
        ref = args.b.index(for_wgmma=True)
        strides = args.b.to_strides_as_packed()
        lines.append(f"  exo_CudaUtil::exo_matrix_descriptor(&{ref}, {strides[0]}),")
        d = args.d.index()
        lines.append("  " + "".join(f"{d}.{rname}," for rname in helper.dreg_names()))
        lines.append(f"  {d}.scale_d);")
        lines.append(f"{d}.scale_d = 1;")
        return lines


# For a wgmma D-matrix (in RMEM), set the scale-d flag to 0, so
# the NEXT wgmma.mma.async instruction will zero-initialize D.
# This is modelled in Exo as a zero-clear, even though the effect
# does not actually happen unless a subsequent mma.async occurs.
# We use the wgmma_zero_instr instr-tl to model this.
@instr
class Sm90_zero_scale_d_f32:
    def behavior(M: size, N: size, d: [f32][M, N]):
        for m in seq(0, M):
            for n in seq(0, N):
                d[m, n] = 0

    def instance(self, M, N):
        self.instr_tl = wgmma_zero_instr
        self.coll_unit = cuda_warpgroup
        self.access_info["d"].mem = Sm90_RmemMatrixD(M, N)
        self.access_info["d"].out_of_order = False

    def codegen(self, args):
        return [f"{args.d.index()}.scale_d = 0;"]


__all__.append("Sm90_zero_scale_d_f32")


@instr
class Sm90_mma_async_tf32(mma_async_impl):
    def behavior(
        M: size,
        N: size,
        d: [f32][M, N],  # @ Sm90_RmemMatrixD
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


class Sm90_mma_store_d_impl(InstrInfo):
    __slots__ = ["helper", "col_major"]

    def instance_impl(self, helper, col_major):
        self.helper = helper
        self.col_major = 1 if col_major else 0
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warpgroup
        self.cu_utils = helper.cu_utils_ss()
        self.access_info["src"].mem = Sm90_RmemMatrixD(helper.M, helper.N)

    def codegen(self, args):
        lines = []
        dst = str(args.dst)
        src = args.src.index()
        lines.append("{")
        lines.append(f"  const auto exo_Sm90_dst = {dst};")
        lines.append(f"  const int32_t exo_Sm90_M_end = int32_t({args.M_end});")
        lines.append(f"  const int32_t exo_Sm90_N_end = int32_t({args.N_end});")
        for m in range(0, args.M, 64):
            for reg_index, reg_name in enumerate(self.helper.dreg_names(m=m)):
                lines.append(
                    f"  exo_CudaUtil::exo_Sm90_store_d_reg<{self.col_major}, {reg_index}>(exo_Sm90_dst, {src}.{reg_name}, {m}, exo_Sm90_M_end, exo_Sm90_N_end);"
                )
        lines.append("}")
        return lines


@instr
class Sm90_mma_store_d_col_major_tf32(Sm90_mma_store_d_impl):
    def behavior(
        M: size,
        N: size,
        M_end: index,
        N_end: index,
        dst: [f32][N, M] @ CudaDeviceVisibleLinear,
        src: [f32][M, N],  # Sm90_RmemMatrixD
    ):
        for m in seq(0, M):
            if m < M_end:
                for n in seq(0, N):
                    if n < N_end:
                        dst[n, m] = src[m, n]  # Transposed

    def instance(self, M, N):
        helper = WgmmaHelper(M, N, "f32", "tf32", "tf32")
        self.instance_impl(helper, True)


@instr
class Sm90_mma_store_d_row_major_tf32(Sm90_mma_store_d_impl):
    def behavior(
        M: size,
        N: size,
        M_end: index,
        N_end: index,
        dst: [f32][M, N] @ CudaDeviceVisibleLinear,
        src: [f32][M, N],  # Sm90_RmemMatrixD
    ):
        for m in seq(0, M):
            if m < M_end:
                for n in seq(0, N):
                    if n < N_end:
                        dst[m, n] = src[m, n]

    def instance(self, M, N):
        helper = WgmmaHelper(M, N, "f32", "tf32", "tf32")
        self.instance_impl(helper, False)


__all__.append("Sm90_mma_store_d_col_major_tf32")
__all__.append("Sm90_mma_store_d_row_major_tf32")
