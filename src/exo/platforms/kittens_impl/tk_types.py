from ..cuda import *

from exo.API import (
    instr,
    WindowFeatures,
    WindowEncoder,
    WindowIndexer,
    WindowIndexerResult,
    UtilInjector,
    window_indexer,
)
from exo.scalars import ScalarInfo, e4m3, e5m2, e8m0, f16, bf16, f32
from exo.core.memory import memwin_template

# Don't import from Sm90 here to avoid circular imports


__all__ = []  # Will be appended to


# Translate Exo ScalarInfo to suffixes for ThunderKittens convenience typedefs.
# NOTE: we can in principle use, e.g. st<exo_f16, ...> for st_hf<...>
# However, this assumes exo_f16/exo_bf16 are the same as the ThunderKittens types.
# This is not guaranteed (although this possibility is not really tested),
# so we use the official st_fl, st_hf, st_bf names.
cuda_tk_type_suffix_table = {
    f32: "fl",
    f16: "hf",
    bf16: "bf",
    e4m3: "fp8e4m3",
    e5m2: "fp8e5m2",
    e8m0: "fp8e8m0",
}

cuda_tk_typename_table = {
    f32: "float",
    f16: "::kittens::half",
    bf16: "::kittens::bf16",
    e4m3: "::kittens::e4m3",
    e5m2: "::kittens::e5m2",
    e8m0: "::kittens::e8m0",
}


cuda_tk_tile_layout_names = ["all", "row", "col"]
cuda_tk_vec_layout_names = ["all", "align", "ortho", "naive"]


cuda_tk_valid_num_types_all_pairs = {
    (dst, src) for dst in cuda_tk_typename_table for src in cuda_tk_typename_table
}


__all__.append("cuda_tk_type_suffix_table")
__all__.append("cuda_tk_typename_table")
__all__.append("cuda_tk_tile_layout_names")
__all__.append("cuda_tk_vec_layout_names")
__all__.append("cuda_tk_valid_num_types_all_pairs")


class _CudaTkWarpIndexer(WindowIndexer):
    def index(self, utils: UtilInjector, features: WindowFeatures):
        utils.add_cu_include("kittens.cuh")
        c_expr = features.get_dataptr()
        # All non-packed indices resolve to multidimensional array indexing.
        for i in range(features.n_array_dims()):
            c_expr = c_expr[features.get_array_offset(i)]
        return self.pack_result(c_expr, False)


@memwin_template
def CudaTkWarpTile(r, c, layout="row"):
    """Wrapper for kittens::rt<?, r, c, layout> (Per-warp register tile)

    Use layout="all" to get a common base class for all tiles of a
    given size with unknown layout. However, such tiles cannot be allocated.

    layout must be in ("all", "row", "col")

    However, currently "col" layout never seems to be used.
    wgmma uses only tiles in "row" ThunderKittens layout.

    In general, the layouts for A and D operands are NOT
    the same for wgmma register tiles, so this shouldn't work.
    However, conveniently, they are the same for f16, which
    is the only type ThunderKittens supports both A and D for.
    (f32 is only supported for D; tf32 operand support is omitted).

    """
    assert r % 16 == 0, f"CudaTkWarpTile requires r={r} to be divisible by 16"
    assert c % 16 == 0, f"CudaTkWarpTile requires c={c} to be divisible by 16"
    assert layout in cuda_tk_tile_layout_names, layout

    if layout == "all":
        base = CudaBasicDeviceVisible
    else:
        base = CudaTkWarpTile(r, c, "all")

    @window_indexer(_CudaTkWarpIndexer)
    class Tile(base):
        @classmethod
        def alloc(cls, new_name, prim_type, shape, srcinfo):
            scalar_info = ScalarInfo(prim_type)
            try:
                suffix = cuda_tk_type_suffix_table[scalar_info]
            except KeyError:
                raise TypeError(
                    f"CudaTkWarpTile currently does not support {scalar_info}"
                )
            array_dims = "".join(f"[{n}]" for n in shape[:-2])
            # fmt: off
            return f"::kittens::rt_{suffix}<{r}, {c}, ::kittens::ducks::rt_layout::{layout}> {new_name}{array_dims};"

        @classmethod
        def free(cls, new_name, prim_type, shape, srcinfo):
            return ""

        @classmethod
        def can_read(cls):
            return False

        @classmethod
        def packed_tensor_shape(cls, scalar_info: ScalarInfo):
            return (r, c)

        @classmethod
        def device_permission(cls, device, instr_tl):
            return cls.device_allocated_impl(device, instr_tl)

        @classmethod
        def native_unit(cls):
            return cuda_warp

        qual_tl_dict = cuda_rmem_qual_tl_dict

        if layout == "row":
            row_vec = CudaTkWarpVec(c, "align")
            col_vec = CudaTkWarpVec(r, "ortho")
        if layout == "col":
            # Not really tested.
            row_vec = CudaTkWarpVec(c, "ortho")
            col_vec = CudaTkWarpVec(r, "align")

    return Tile


__all__.append("CudaTkWarpTile")


@memwin_template
def CudaTkWarpVec(length, layout):
    """Wrapper for kittens::rv<?, length, layout> (Per-warp register tile)

    Use layout="all" to get a common base class for all vectors of a
    given length with unknown layout. However, such vectors cannot be allocated.

    layout must be in ("all", "align", "ortho", "naive")
    but consider using CudaTkWarpTile(...).row_vec or col_vec for clarity.

    """
    # fmt: off
    assert length % 16 == 0, f"CudaTkWarpVec requires length={length} to be divisible by 16"
    # fmt: on
    assert layout in cuda_tk_vec_layout_names, layout

    if layout == "all":
        base = CudaBasicDeviceVisible
    else:
        base = CudaTkWarpVec(length, "all")

    @window_indexer(_CudaTkWarpIndexer)
    class Vec(base):
        @classmethod
        def alloc(cls, new_name, prim_type, shape, srcinfo):
            scalar_info = ScalarInfo(prim_type)
            try:
                suffix = cuda_tk_type_suffix_table[scalar_info]
            except KeyError:
                raise TypeError(
                    f"CudaTkWarpTile currently does not support {scalar_info}"
                )
            assert shape[-1] == length
            array_dims = "".join(f"[{n}]" for n in shape[:-1])
            # fmt: off
            assert layout != "all", "Cannot allocate vector with 'all' layout"
            return f"::kittens::rv_{suffix}<{length}, ::kittens::ducks::rv_layout::{layout}> {new_name}{array_dims};"

        @classmethod
        def free(cls, new_name, prim_type, shape, srcinfo):
            return ""

        @classmethod
        def can_read(cls):
            return False

        @classmethod
        def packed_tensor_shape(cls, scalar_info: ScalarInfo):
            return (length,)

        @classmethod
        def device_permission(cls, device, instr_tl):
            return cls.device_allocated_impl(device, instr_tl)

        @classmethod
        def native_unit(cls):
            return cuda_warp

        qual_tl_dict = cuda_rmem_qual_tl_dict

    return Vec


__all__.append("CudaTkWarpVec")


cuda_tk_gl2_window_util = """\
// Adapted code from ThunderKittens
// https://github.com/HazyResearch/ThunderKittens
//
// Convert Exo 2D window type to something that quacks like
// ::kittens::gl (GMEM handle), but we don't have TMA handles.
// We have to do this to bridge between Exo C and ThunderKittens C++20.
// Assumes the last dimension is tightly packed.
// Can be used with {0, 0, 0, 0} as the COORD idx.
template <typename _T, int Rows>
struct exo_tk_gl2_window
{
    using identifier = ::kittens::ducks::gl::identifier;

    using T     = ::kittens::base_types::packing<_T>::unpacked_type;
    using T2    = ::kittens::base_types::packing<_T>::packed_type;
    using dtype = T;

    T* raw_ptr;
    int cols_internal;

    template <typename Window>
    EXO_CUDA_INLINE
    exo_tk_gl2_window(Window window)
    {
        static_assert(sizeof(Window::strides) == 2 * sizeof(Window::strides[0]));
        raw_ptr = const_cast<T*>(reinterpret_cast<const T*>(window.data));
        cols_internal = static_cast<int>(window.strides[0]);
    }

    static constexpr int batch() { return 1; }
    static constexpr int depth() { return 1; }
    static constexpr int rows() { return Rows; }
    __device__ __host__ inline int cols() const { return cols_internal; }
    __device__ __host__ inline size_t numel() const { return size_t(rows()) * cols(); }

    EXO_CUDA_INLINE T& operator[](const ::kittens::coord<::kittens::ducks::default_type>& idx) const
    {
        return raw_ptr[idx.r * cols() + idx.c];
    }

    template<int axis> __device__ inline size_t shape() const {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis==0) { return size_t(batch()); }
        else if constexpr (axis==1) { return size_t(depth()); }
        else if constexpr (axis==2) { return size_t(rows()); }
        else if constexpr (axis==3) { return size_t(cols()); }
    }
    template<int axis> __device__ inline size_t stride() const {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if      constexpr (axis==0) { return (size_t)depth()*rows()*cols(); }
        else if constexpr (axis==1) { return (size_t)rows()*cols(); }
        else if constexpr (axis==2) { return (size_t)cols(); }
        else if constexpr (axis==3) { return 1; }
    }
};
"""


__all__.append("cuda_tk_gl2_window_util")


# Quick check class structure went as planned.
assert CudaTkWarpTile(16, 256) == CudaTkWarpTile(16, 256, "row")
assert issubclass(CudaTkWarpTile(16, 256, "row"), CudaTkWarpTile(16, 256, "all"))
assert not issubclass(CudaTkWarpTile(16, 256, "all"), CudaTkWarpTile(16, 256, "row"))
assert CudaTkWarpTile(16, 256, "row").row_vec == CudaTkWarpVec(256, "align")
assert CudaTkWarpTile(16, 256, "row").col_vec == CudaTkWarpVec(16, "ortho")
assert CudaTkWarpTile(16, 256, "col").row_vec == CudaTkWarpVec(256, "ortho")
assert CudaTkWarpTile(16, 256, "col").col_vec == CudaTkWarpVec(16, "align")
