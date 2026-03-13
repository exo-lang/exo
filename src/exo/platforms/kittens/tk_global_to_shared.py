from __future__ import annotations

from exo import *
from exo.scalars import ScalarInfo

from ..cuda import *
from .tk_types import *
from ..Sm90 import Sm90_SmemSwizzled, Sm90_SmemSwizzled_from_smem_box


@instr
class tk_load_sg(InstrInfo):
    """Wrapper around ThunderKittens SMEM-from-GMEM load, using 16-byte aligned copies.

    Generally, use size1=128 / sizeof(T) and dst @ Sm90_SmemSwizzled(128)

    """

    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1],  # Sm90_SmemSwizzled(swizzle)
        src: [R][size0, size1] @ CudaGmemLinear,
    ):
        assert stride(dst, 0) == size1  # SMEM tile densely packed
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]

    def instance(self: InstrInfo, size0, size1):
        self.cu_includes.append("cuda_runtime.h")
        self.cu_includes.append("kittens.cuh")
        self.cu_utils.append(cuda_tk_gl2_window_util)
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        scalar_info: ScalarInfo = self.access_info["dst"].scalar_info
        self.access_info["src"].allow_out_of_bounds = True
        self.access_info["dst"].mem = Sm90_SmemSwizzled_from_smem_box(
            scalar_info, (size0, size1)
        )

    def codegen(self: InstrInfo, args: InstrArgs):
        # fmt: off
        tk_t = cuda_tk_typename_table[args.src.get_scalar_info()]
        src_c = f"exo_CudaUtil::exo_tk_gl2_window<{tk_t}, {args.size0}>({args.src})"
        dst_c = args.dst.index(as_tk_subtile=(args.size0, args.size1))
        return [
            "{  // Place dst/src handles in named temporaries, because ThunderKittens is not const-correct.",
            f"  auto exo_tk_dst = {dst_c};",
            f"  auto exo_tk_src = {src_c};",
            f"  ::kittens::warp::load(exo_tk_dst, exo_tk_src, ::kittens::coord(0, 0, 0, 0));",
            "}",
        ]

    valid_num_types = ScalarInfo.same()


@instr
class tk_store_sg(InstrInfo):
    """Wrapper around ThunderKittens SMEM-to-GMEM store, using 16-byte aligned copies.

    Generally, use size1=128 / sizeof(T) and src @ Sm90_SmemSwizzled(128)

    """

    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1] @ CudaGmemLinear,
        src: [R][size0, size1],  # Sm90_SmemSwizzled(swizzle)
    ):
        assert stride(src, 0) == size1  # SMEM tile densely packed
        assert stride(src, 1) == 1
        assert stride(dst, 1) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]

    def instance(self: InstrInfo, size0, size1):
        self.cu_includes.append("cuda_runtime.h")
        self.cu_includes.append("kittens.cuh")
        self.cu_utils.append(cuda_tk_gl2_window_util)
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        scalar_info: ScalarInfo = self.access_info["src"].scalar_info
        self.access_info["src"].mem = Sm90_SmemSwizzled_from_smem_box(
            scalar_info, (size0, size1)
        )
        self.access_info["dst"].allow_out_of_bounds = True

    def codegen(self: InstrInfo, args: InstrArgs):
        # fmt: off
        tk_t = cuda_tk_typename_table[args.dst.get_scalar_info()]
        src_c = args.src.index(as_tk_subtile=(args.size0, args.size1))
        dst_c = f"exo_CudaUtil::exo_tk_gl2_window<{tk_t}, {args.size0}>({args.dst})"
        return [
            "{  // Place dst/src handles in named temporaries, because ThunderKittens is not const-correct.",
            f"  auto exo_tk_dst = {dst_c};",
            f"  auto exo_tk_src = {src_c};",
            f"  ::kittens::warp::store(exo_tk_dst, exo_tk_src, ::kittens::coord(0, 0, 0, 0));",
            "}",
        ]

    valid_num_types = ScalarInfo.same()
