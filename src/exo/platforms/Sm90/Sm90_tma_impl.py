from __future__ import annotations

from .Sm90_fwd import *
from .Sm90_smem import *
from .Sm90_tensorMap import *

from exo.API import *
from exo.platforms.cuda import *

from .Sm90_internal_util import *


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


def tma_to_smem_util(is_multicast: bool):
    cache_hint = 1152921504606846976  # copied from cutlass PTX

    # fmt: off
    # Note: indentation of code-in-strings here is dictated by output C++ requirements.
    expect_tx = f'asm("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" :: "r"(exo_tma_mbarrier), "r"(expect_tx));'

    def rank_case(rank: int):
        vector_fmt = "{" + ", ".join(f"%{r+2}" for r in range(rank)) + "}"
        ptx_fmt = f" [%0], [%1, {vector_fmt}], [%{rank+2}], %{rank+3}"
        if is_multicast:
            ptx_fmt += f", %{rank+4}"
        vector_args = [f'"r"(window.C_offsets[{rank - 1 - r}])' for r in range(0, rank)]
        vector_values = ", ".join(vector_args)
        if is_multicast:
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

    if is_multicast:
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


def tma_to_gmem_util(is_reduce: bool):
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


def make_basic_tma(n_dims: int, to_gmem: bool, is_multicast: bool, is_reduce: bool):
    assert not to_gmem or not is_multicast
    assert to_gmem or not is_reduce
    assert 1 <= n_dims <= 5, n_dims

    class Base(InstrInfo):
        __slots__ = ["ncta", "sizes", "smem_box", "cta_stride", "swizzle"]

        valid_num_types = ScalarInfo.same()

        @staticmethod
        def _validate_smem_box(
            smem_box_arg: Optional[Tuple[int]], default_smem_box: Tuple[int]
        ):
            if smem_box_arg is None:
                return default_smem_box
            # fmt: off
            assert all(isinstance(c, int) for c in smem_box_arg), f"Non-integer smem_box={smem_box} given"
            # fmt: on
            minimal_arg = [c for c in smem_box_arg if c != 1]
            minimal_expected = [c for c in default_smem_box if c != 1]
            if minimal_arg != minimal_expected:
                raise ValueError(
                    f"smem_box {smem_box_arg} isn't compatible with expected box "
                    f"{default_smem_box} (we allow extra 1's but that's it)"
                )
            return tuple(smem_box_arg)

        def instance_impl(
            self: InstrInfo,
            ncta: int,
            sizes: Tuple[int],
            smem_box: Tuple[int],
            cta_stride: int,
            swizzle: int,
        ):
            self.ncta = ncta
            self.sizes = sizes
            self.smem_box = smem_box
            self.cta_stride = cta_stride
            self.swizzle = swizzle

            assert ncta == 1 or is_multicast

            if sizes[0] % ncta != 0:
                raise ValueError(
                    f"Multicast TMA requires size0={sizes[0]} to be divisible by ncta={ncta}"
                )

            if to_gmem:
                gmem = self.access_info["dst"]
                smem = self.access_info["src"]
            else:
                gmem = self.access_info["src"]
                smem = self.access_info["dst"]

            # Shape of the destination SMEM window may be of lower dimension
            # than the SMEM box because point expressions may be placed on
            # dimensions where the SMEM box coordinate is 1.
            # e.g. dst[0:M, x, 0:N] with smem_box=(M, 1, N).
            rank = len(smem_box)
            assert rank > 0
            assert len(sizes) <= len(smem_box)

            smem.mem = Sm90_get_mma_smem(swizzle)
            smem.out_of_order = True

            gmem.mem = Sm90_tensorMap(swizzle, *smem_box)
            gmem.out_of_order = True
            gmem.allow_out_of_bounds = True  # GMEM special case

            if to_gmem:
                self.instr_tl = tma_to_gmem_async_instr
                self.coll_unit = cuda_warp
                self.cu_utils.append(tma_to_gmem_util(is_reduce))
                if is_reduce:
                    dst.atomicity = AtomicityInfo([tma_to_gmem_async_qual])
            else:
                self.instr_tl = tma_to_smem_async_instr
                if is_multicast:
                    self.coll_unit = ncta * cuda_warp_in_cluster_strided(cta_stride)
                    # We are writing to an SMEM window of size [ncta, size0, size1,...]
                    # and each [size0, size1, ...] shard is owned by 1 CTA.
                    self.barrier_coll_units = (cuda_cta_in_cluster,)
                    smem.distributed_coll_units = (cuda_cta_in_cluster,)
                    smem.access_by_owner_only = False
                else:
                    # Same as 1 * cuda_warp_in_cluster_strided(1),
                    # but just use cuda_warp here to reduce user confusion.
                    self.coll_unit = cuda_warp
                self.cu_utils.append(tma_to_smem_util(is_multicast))
                self.barrier_mechanism = CudaMbarrier

        def codegen(self: InstrInfo, args: InstrArgs):
            # fmt: off
            box = self.smem_box
            gmem: InstrWindowArg
            smem: InstrWindowArg
            if to_gmem:
                _reduce = "_reduce" if is_reduce else ""
                fname = f"exo_Sm90_tma_to_gmem{_reduce}"
                gmem = args.dst
                smem = args.src
            else:
                _multicast = "_multicast" if is_multicast else ""
                fname = f"exo_Sm90_tma_to_smem{_multicast}"
                gmem = args.src
                smem = args.dst

            # For multicasting, we divide the (size0, size1, ...)
            # tile into (size0 // ncta, size1, ...) sized tiles, one
            # assigned to each CTA to copy.
            coop_size0 = args.size0 // self.ncta
            if self.ncta > 1:
                coop_cta_idx = (args.exo_wrap_cir("blockIdx.x") / self.cta_stride) % self.ncta
                coop_dim0_offset = coop_size0 * coop_cta_idx
            else:
                assert self.ncta == 1
                coop_cta_idx = 0
                coop_dim0_offset = 0

            # The GMEM argument is passed as a window struct with a
            # "separate dataptr". The separate dataptr is a CUtensorMap
            # blob, and the window struct encodes just offset coordinates.
            gmem_tensorMap = gmem.get_separate_dataptr()
            gmem_offsets = gmem[coop_dim0_offset : coop_dim0_offset + coop_size0]

            if self.swizzle:
                smem_ptr = smem.index_ptr(coop_dim0_offset, for_wgmma=True)
            else:
                smem_ptr = smem.index_ptr(coop_dim0_offset, )

            if to_gmem:
                c_args = [gmem_tensorMap, gmem_offsets, smem_ptr]
            else:
                mbarrier = args.exo_barrier  # Magical
                tx = self.ncta * prod(box) * smem.get_scalar_info().bits // 8
                c_args = [smem_ptr, gmem_tensorMap, gmem_offsets, mbarrier, tx]
                if is_multicast:
                    # Also magical, poorly-documented argument that
                    # LoopIR_compiler inserts just for us.
                    cta_mask = args.exo_cta_mask
                    c_args.append(cta_mask)

            lines = [f"exo_CudaUtil::{fname}("]
            for i, c_arg in enumerate(c_args):
                prefix = "    " if i == 0 else "  , "
                lines.append(f"{prefix}{c_arg}")
            lines.append(");")
            return lines

    return Base
