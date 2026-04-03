from __future__ import annotations

from .Sm90_fwd import *
from .Sm90_smem import *

from exo.API import *
from exo.platforms.cuda import *

from .Sm90_internal_util import *


__all__ = []  # Will be appended to


class Sm90_tensorMap_base(SpecialWindow):
    pass


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# CUtensorMap
# This is a 128-byte blob created on the CPU and passed to CUDA device functions
# which use it to execute TMA (cp.async.bulk) instructions.
# In Exo, we model this as a SpecialWindow (similar to Memory)
# that is constructed as a window to CudaGmemLinear.
@memwin_template
def Sm90_tensorMap(swizzle, *smem_box):
    """Represents CUtensorMap (bounds-checked multidimensional GMEM view)

    The Exo object language syntax is

        window_name = gmem[...] @ Sm90_tensorMap(swizzle, *smem_box)

    where the dimensionality of the resulting window is equal to
    the length of smem_box. The extent of the N-th interval of
    window expressions of the tensorMap must match the N-th smem_box coordinate,
    with point expressions counting as size 1. e.g.,

        window_name[x, 100:200, 0:1, y:y+32]  # `x` is a point expression.

    is compatible with Sm90_tensorMap(swizzle, 1, 100, 1, 32)

    The SMEM window must be stored in CudaSmemLinear if swizzle == 0,
    otherwise Sm90_SmemSwizzled(swizzle).
    If swizzled, the right-most smem_box coordinate must be
    swizzle / sizeof(Element).

    EXCEPTION: multicast TMA changes the smem_box size requirements.
    See those instructions for specific documentation.

    """

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
    class CUtensorMap(Sm90_tensorMap_base):
        @classmethod
        def global_(cls):
            return ""

        @classmethod
        def device_permission(cls, device, instr_tl):
            return CudaBasicDeviceVisible.host_allocated_impl(
                device, instr_tl, pinned=False
            )

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
                f"taking window of {dim}d tensor not supported given {len(smem_box)}d SMEM box {smem_box}; "
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
        init = "{ {" + ", ".join("0" for i in range(self.n_dims)) + "} }"
        return f"({self.exo_struct_name()}) {init}"

    def encode_special_separate_dataptr(
        self, utils: UtilInjector, features: WindowFeatures
    ):
        """For CudaGmemLinear -> CUtensorMap conversion. Make CUtensorMap blob"""
        sname = self.exo_struct_name()
        rank = self.n_dims
        swizzle = self.mem.swizzle()
        box = self.mem.smem_box()
        if swizzle == 0:
            cu_swizzle = "CU_TENSOR_MAP_SWIZZLE_NONE"
        else:
            cu_swizzle = f"CU_TENSOR_MAP_SWIZZLE_{swizzle}B"
            expected_coord = swizzle * 8 // self.scalar_info.bits
            if box[-1] != expected_coord:
                raise ValueError(
                    f"smem_box={box}; expect last coordinate {expected_coord} "
                    f"= swizzle / sizeof({self.scalar_info}) with swizzle={swizzle}"
                )
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

        if rank != len(box):
            better_box = (1,) * max(0, rank - len(box)) + tuple(box)
            raise ValueError(
                f"{features.srcinfo()}: "
                f"{rank}d window constructed does not match {len(box)}d smem_box={box}. "
                f"Consider left-padding smem_box with 1s, and update tma instrs with "
                f"template parameter smem_box={better_box}."
            )

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
    "e4m3": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", ""),
    "e5m2": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", ""),
    "e8m0": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", ""),
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
