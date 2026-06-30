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
                if is_reduce:
                    gmem.atomicity = AtomicityInfo([tma_to_gmem_async_qual])
            else:
                self.instr_tl = tma_to_smem_async_instr
                if is_multicast:
                    self.coll_unit = ncta * cuda_warp_in_cluster_strided(cta_stride)
                    # We are writing to an SMEM window of size [ncta, size0, size1,...]
                    # and each [size0, size1, ...] shard is owned by 1 CTA.
                    self.barrier_coll_units = (cuda_cta_in_cluster,)
                    smem.distributed_coll_units = (cuda_cta_in_cluster,)
                else:
                    # Same as 1 * cuda_warp_in_cluster_strided(1),
                    # but just use cuda_warp here to reduce user confusion.
                    self.coll_unit = cuda_warp
                self.barrier_mechanism = CudaMbarrier

        def codegen(self: InstrInfo, args: InstrArgs):
            # fmt: off
            box = self.smem_box
            rank = len(box)
            cache_hint = 1152921504606846976  # copied from cutlass PTX

            gmem: InstrWindowArg
            smem: InstrWindowArg
            if to_gmem:
                gmem = args.dst
                smem = args.src
            else:
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
                smem_ptr = smem.index_ptr(coop_dim0_offset)

            lines = [
                "{",
                f"  auto exo_tmaWindow = {gmem_offsets};",
            ]

            # Offset vector: reversed from C order (least-significant first in PTX).
            # C_offsets[0] is most-significant; TMA wants least-significant first.
            offsets = [f"exo_tmaWindow.C_offsets[{rank - 1 - i}]" for i in range(rank)]

            if to_gmem:
                if is_reduce:
                    ptx_instr = f"cp.reduce.async.bulk.tensor.{rank}d.global.shared::cta.add.tile.bulk_group"
                else:
                    ptx_instr = f"cp.async.bulk.tensor.{rank}d.global.shared::cta.tile.bulk_group"

                # PTX: [tensorMap, {offsets...}], [smem_src]
                ptx = InlinePtxGen(f"{ptx_instr} [#1#, #2#], #3#;", volatile=True, elect_one_sync=True)
                ptx.add_arg(f"&({gmem_tensorMap})", constraint="l", log_as="ptr_data", N=1)
                ptx.add_arg(offsets, constraint="r", log_as="bits", N=2)
                ptx.add_arg(smem_ptr, constraint="smem", log_as="bits", N=3)
                lines.extend(ptx.as_c_lines(tab="  "))
            else:
                mbarrier = args.exo_barrier  # Magical
                tx = self.ncta * prod(box) * smem.get_scalar_info().bits // 8

                # mbarrier.expect_tx informs the barrier how many bytes to expect.
                expect_ptx = InlinePtxGen(
                    "mbarrier.expect_tx.shared::cta.b64 #1#, #2#;",
                    volatile=False,
                    elect_one_sync=True,
                )
                expect_ptx.add_arg(str(mbarrier), constraint="r", log_as="bits", N=1, brackets=True)
                expect_ptx.add_arg(tx, constraint="n", log_as="bits", N=2)
                lines.extend(expect_ptx.as_c_lines(tab="  "))

                # TMA to SMEM async copy.
                if is_multicast:
                    ptx_instr = f"cp.async.bulk.tensor.{rank}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                    # PTX arg order: [dst], [tensorMap, {offsets}], [mbar], ctaMask, cacheHint
                    ptx_fmt = f"{ptx_instr} #1#, [#2#, #3#], #4#, #5#, #6#;"
                else:
                    ptx_instr = f"cp.async.bulk.tensor.{rank}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                    ptx_fmt = f"{ptx_instr} #1#, [#2#, #3#], #4#, #5#;"

                ptx = InlinePtxGen(ptx_fmt, volatile=True, elect_one_sync=True)
                ptx.add_arg(smem_ptr, constraint="smem", log_as="bits", N=1)
                ptx.add_arg(f"&({gmem_tensorMap})", constraint="l", log_as="ptr_data", N=2)
                ptx.add_arg(offsets, constraint="r", log_as="bits", N=3)
                ptx.add_arg(str(mbarrier), constraint="r", log_as="bits", N=4, brackets=True)
                if is_multicast:
                    # Magical argument: LoopIR_compiler inserts clusterDim for us.
                    # ctaMask (#5#) must come before cacheHint (#6#) per PTX spec.
                    clusterDim = args.exo_clusterDim
                    cta_mask = self.codegen_cta_mask(clusterDim, self.ncta, self.cta_stride)
                    ptx.add_arg(cta_mask, constraint="h", log_as="bits", N=5)
                    ptx.add_arg(cache_hint, constraint="n", log_as=None, N=6)
                else:
                    ptx.add_arg(cache_hint, constraint="n", log_as=None, N=5)
                lines.extend(ptx.as_c_lines(tab="  "))

            lines.append("}")
            return lines

        def codegen_cta_mask(self, clusterDim, cta_count, cta_pitch):
            shift_mask = clusterDim - 1

            # CUDA model fundamentally assumes power-of-2 CTA counts
            cta_count_log2 = cta_count.bit_length() - 1
            cta_pitch_log2 = cta_pitch.bit_length() - 1
            assert cta_count == 1 << cta_count_log2
            assert cta_pitch == 1 << cta_pitch_log2

            for bit_idx in range(cta_pitch_log2, cta_pitch_log2 + cta_count_log2):
                shift_mask &= ~(1 << bit_idx)

            base_num = 1
            for i in range(1, cta_count):
                base_num = base_num << cta_pitch | 1

            if shift_mask == 0:
                return f"uint16_t({hex(base_num)})"
            else:
                return f"uint16_t({hex(base_num)} << (blockIdx.x & {hex(shift_mask)}))"

    return Base
