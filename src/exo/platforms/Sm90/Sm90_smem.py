from .Sm90_fwd import *
from .Sm90_internal_util import *

from exo.platforms.cuda import *
from exo.API import *

from .Sm90_internal_util import *


from ..kittens.tk_types import cuda_tk_type_suffix_table


__all__ = [
    "Sm90_SmemSwizzled",
    "Sm90_get_mma_smem",
    "Sm90_SmemSwizzled_from_smem_box",
    "Sm90_codegen_smem_descriptor",
    "Sm90_codegen_smem_descriptor_util",
]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Swizzled shared memory, as used by wgmma and TMA (cp.async.bulk).
# This is a "position dependent swizzle layout" in Cutlass terminology.
# What this means is we swizzle (xor) the raw pointer, not the array offsets.
# Each {swizzle}-byte section is re-arranged depending on its position in SMEM.
# So if we shift a tensor allocation's location in memory, the swizzle pattern
# for the Nth section (relative to the base address) won't be the same.
@memwin_template
def Sm90_SmemSwizzled(swizzle):
    if swizzle not in (32, 64, 128):
        raise ValueError(f"swizzle must be 32, 64, or 128 bytes, not {swizzle}")

    swizzle_bits = 1 if swizzle == 128 else 2 if swizzle == 64 else 3
    mask = 0x70 if swizzle == 128 else 0x30 if swizzle == 64 else 0x10

    @window_indexer(SwizzledIndexer)
    class SwizzledImpl(CudaSmemAtomicity16B):
        @classmethod
        def get_swizzle_bytes(cls):
            return swizzle

        @classmethod
        def global_(cls):
            return f"""
#ifdef __CUDACC__
template <typename T>
struct exo_Sm90_SW{swizzle} {{
    T data;

    static EXO_CUDA_INLINE exo_Sm90_SW{swizzle}<T>* swizzle_pointer(uintptr_t addr)
    {{
        // Adapted from ThunderKittens appendix which actually documents CUDA correctly.
        uint32_t shr = uint32_t(addr) >> 3;
        const uint32_t mask = {mask};
        addr = addr ^ (shr & mask);
        return reinterpret_cast<exo_Sm90_SW{swizzle}*>(addr);
    }}

    static __host__ __device__ constexpr uint64_t get_swizzle_bits()
    {{
        return {swizzle_bits};
    }}

    static __host__ __device__ constexpr int get_swizzle_bytes()
    {{
        return {swizzle};
    }}

    EXO_CUDA_INLINE const T& swizzle_get() const
    {{
        return swizzle_pointer(reinterpret_cast<uintptr_t>(&data))->data;
    }}

    EXO_CUDA_INLINE T& swizzle_get()
    {{
        return swizzle_pointer(reinterpret_cast<uintptr_t>(&data))->data;
    }}
}};

// Element type for Sm90_SmemSwizzled({swizzle}) allocations where the
// last 3 array extents are [TileOuterCols, TileRows, TileInnerCols]
// (left-padded with 1 for 0D, 1D, 2D allocations).
template <typename T, int TileOuterCols, int TileRows, int TileInnerCols>
struct exo_Sm90_SW{swizzle}_tiled: public exo_Sm90_SW{swizzle}<T>
{{
    // Reinterpret-cast to kittens::st_subtile view of this shared memory.
    //
    // For better or worse, kittens 2D tile is expressed as a 3D tile in Exo-GPU.
    // `this` is assumed to point-to the NON-SWIZZLED base address of a complete
    // 3D tile. This returns a subtile view whose
    //
    // * size/extents are given by the Subtile template parameters
    //
    // * base offset from the `this` tile is given by runtime int offsets
    //
    // Assume for this discussion that the tile is row-major. Then,
    //
    // * Let swizzle_elements = {swizzle} / sizeof(T)
    //
    // * The 3D Exo-GPU tile is of size [TileOuterCols, TileRows, TileInnerCols],
    //   where TileInnerCols == swizzle_elements. If not, then this tile
    //   is incompatible with ThunderKittens, TMA, and wgmma.
    //
    // * The value at coordinates (r, c) in the 2D [Rows, Cols] tile is stored at
    //   address swizzle_pointer(
    //          tile_base_addr
    //          + sizeof(T) * (c / TileInnerCols) * TileRows * TileInnerCols
    //          + sizeof(T) * (r) * TileInnerCols
    //          + sizeof(T) * (c % TileInnerCols))
    //   Basically, the column is simultaneously the fastest (%) and slowest (/) dimension
    //
    // TMA: If TileOuterCols != 1, then ThunderKittens is capable of doing a single TMA
    // copy to load/store the tile, but Exo-GPU may have to issue multiple.
    template <
        int SubtileOuterCols,
        int SubtileRows,
        int SubtileInnerCols,
        template <int, int, bool, int> class st_typed>
    EXO_CUDA_INLINE auto as_tk_subtile(int col_outer_offset, int row_offset, int col_inner_offset) const
    {{
        static_assert(
            TileInnerCols * sizeof(T) == {swizzle},
            "Exo-GPU instr didn't assert strides properly."
            " This is needed to match kittens swizzle automation"
        );
        static_assert(
            SubtileInnerCols == TileInnerCols || SubtileOuterCols == 1,
            "Exo-GPU instr used wrong SubtileInnerCols (window size last dimension)."
            " This is needed for Exo-GPU and kittens to agree on column tiling"
        );
        using st_t = st_typed<TileRows, TileOuterCols * TileInnerCols, true, 0>;
        st_t* p_tile = const_cast<st_t*>(reinterpret_cast<const st_t*>(this));
        // Note, subtile implicitly multiplies (r, c) by the subtile size.
        // We have to work around this!
        // Also, can't mention ::kittens here, because it may not be included.
        // Also also, explicit kittens swizzle is broken in this function.
        auto subtile = p_tile->template subtile<SubtileRows, SubtileOuterCols * SubtileInnerCols>(int2(0, 0));
        subtile.row_offset = static_cast<int>(row_offset);
        subtile.col_offset = static_cast<int>(col_outer_offset) * TileInnerCols + static_cast<int>(col_inner_offset);
        return subtile;
    }}
}};

#endif
"""

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
        def smem_config(cls, inputs: SmemConfigInputs) -> SmemConfig:
            shape = inputs.const_shape
            shape_dims = len(shape)
            # fmt: off
            # Left-pad 3D tile size by 1 as promised.
            TileOuterCols = 1 if shape_dims < 3 else shape[-3]
            TileRows      = 1 if shape_dims < 2 else shape[-2]
            TileInnerCols = 1 if shape_dims < 1 else shape[-1]
            ctype = f"exo_Sm90_SW{swizzle}_tiled<{inputs.ctype()}, {TileOuterCols}, {TileRows}, {TileInnerCols}>"
            return SmemConfig(
                f"{ctype} (&)[]",
                swizzle,  # Force 32, 64, or 128 byte alignment.
            )
            # fmt: on

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


def Sm90_SmemSwizzled_from_smem_box(scalar_info: ScalarInfo, smem_box: Tuple[int, ...]):
    scalar_info = ScalarInfo(scalar_info)
    swizzle = smem_box[-1] * scalar_info.bits // 8
    if swizzle not in (32, 64, 128):
        raise ValueError(
            f"Invalid smem_box {smem_box}; "
            f"last dimension must lead to swizzle of "
            f"32, 64, or 128; not {swizzle}"
        )
    return Sm90_SmemSwizzled(swizzle)


class SwizzledIndexer(WindowIndexer):
    __slots__ = []

    def index(
        self, utils, features: WindowFeatures, for_wgmma=False, as_tk_subtile=None
    ):
        ctype = self.ctype()
        mem = features.get_mem()
        swizzle = mem.get_swizzle()

        n_dims = features.n_array_dims()
        if as_tk_subtile:
            assert len(as_tk_subtile) == 3
            col_outer_offset, row_offset, col_inner_offset = 0, 0, 0
            SubtileOuterCols, SubtileRows, SubtileInnerCols = as_tk_subtile
            if n_dims >= 3:
                # fmt: off
                assert SubtileOuterCols == 1 or features.get_array_interval_size(n_dims - 3) is not None
                col_outer_offset = features.get_array_offset(n_dims - 3)
            if n_dims >= 2:
                # fmt: off
                assert features.get_array_interval_size(n_dims - 2) is not None
                row_offset = features.get_array_offset(n_dims - 2)
            if n_dims >= 1:
                # fmt: off
                assert features.get_array_interval_size(n_dims - 1) is not None
                col_inner_offset = features.get_array_offset(n_dims - 1)
            n_strided_dims = max(0, n_dims - 3)
        else:
            n_strided_dims = n_dims

        dataptr = features.get_dataptr()
        array_offset = 0
        for i in range(n_strided_dims):
            dim_offset = features.get_array_offset(i)
            dim_stride = features.get_array_stride_as_packed(i)
            array_offset += dim_offset * dim_stride

        assert self.element_bits() >= 8, "TODO implement float4 etc."

        if as_tk_subtile:
            suffix = cuda_tk_type_suffix_table[self.scalar_info]
            code = (
                f"{dataptr}[{array_offset}].template as_tk_subtile"
                f"<{SubtileOuterCols}, {SubtileRows}, {SubtileInnerCols}, ::kittens::st_{suffix}>"
                f"({col_outer_offset}, {row_offset}, {col_inner_offset})"
            )
        elif for_wgmma:
            # This passes a non-swizzled pointer to wgmma or TMA.
            code = f"{dataptr}[{array_offset}]"
        else:
            # This path swizzles the pointer.
            # Needed for access with normal scalar code.
            code = f"{dataptr}[{array_offset}].swizzle_get()"

        return self.pack_result(code, False)


def Sm90_codegen_smem_descriptor(
    swizzle: int, arg: InstrWindowArg, K_major: bool, k: int
):
    assert swizzle in (32, 64, 128)
    element_strides: List[CIR_Wrapper] = arg.to_strides_as_packed()

    # 2D tile is [stride dimension, leading dimension]
    # expecting strides [expected_middle_stride, 1]
    #
    # 3D tile is [leading dimension // (?), stride dimension, leading dimension % (?)]
    # expecting strides [_, expected_middle_stride, 1]
    # and where (?) is expected_middle_stride.

    if len(element_strides) == 3:
        outer_stride = element_strides[-3].exo_to_int("constant")
    else:
        outer_stride = 1
        assert len(element_strides) == 2
    middle_stride = element_strides[-2].exo_to_int("constant")
    inner_stride = element_strides[-1].exo_to_int("constant")

    scalar_info: ScalarInfo = arg.get_scalar_info()

    expected_middle_stride = swizzle * 8 // scalar_info.bits
    if middle_stride != expected_middle_stride or inner_stride != 1:
        # TODO test me
        raise ValueError(
            f"Expected element strides of 2D {scalar_info} tile to be "
            f"[{expected_middle_stride}, 1], not [{middle_stride}, {inner_stride}]"
        )
    indices = [0 for _ in element_strides]
    if K_major:
        indices[-1] = k
    else:
        indices[-2] = k
    ptr = arg.index_ptr(*indices, for_wgmma=True)
    return f"exo_CudaUtil::exo_Sm90_smem_descriptor({ptr}, {outer_stride}, {8 * middle_stride})"


Sm90_codegen_smem_descriptor_util = """
EXO_CUDA_INLINE uint64_t exo_Sm90_matrix_descriptor_encode(uint32_t val)
{
    return (val & 0x3FFFF) >> 4;
}

template <typename SwizzledElement>
EXO_CUDA_INLINE
uint64_t exo_Sm90_smem_descriptor(SwizzledElement* ptr, uint32_t leading_stride_elements, uint32_t stride_stride_elements)
{
    uint32_t element_size = sizeof(SwizzledElement);
    return (
        exo_Sm90_matrix_descriptor_encode(exo_smemU32(ptr))
      | exo_Sm90_matrix_descriptor_encode(leading_stride_elements * element_size) << 16u
      | exo_Sm90_matrix_descriptor_encode(stride_stride_elements * element_size) << 32u
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
      | uint64_t(1) << 46  // Bit-field 46-48, Fixed constant value of 0b001 on Blackwell (?!!?)
#endif
      | uint64_t(SwizzledElement::get_swizzle_bits()) << 62
  );
}
"""
