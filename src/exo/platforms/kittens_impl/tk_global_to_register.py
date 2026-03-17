from __future__ import annotations

from exo import *
from exo.scalars import ScalarInfo

from ..cuda import *
from .tk_types import *


__all__ = ["cuda_tk_load_rg", "cuda_tk_store_rg"]


@instr
class cuda_tk_load_rg(InstrInfo):
    """Warp loads tile into RMEM from GMEM. Not async."""

    valid_num_types = cuda_tk_valid_num_types_all_pairs

    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1],  # CudaTkWarpTile(size0, size1, layout="row")
        src: [R][size0, size1] @ CudaGmemLinear,
    ):
        assert stride(src, 1) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]

    def instance(self: InstrInfo, size0: int, size1: int, *, layout: str = "row"):
        self.cu_includes.append("kittens.cuh")
        self.cu_utils.append(cuda_tk_gl2_window_util)
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        self.access_info["dst"].mem = CudaTkWarpTile(size0, size1, layout)

    def codegen(self: InstrInfo, args: InstrArgs):
        # fmt: off
        tk_t = cuda_tk_typename_table[args.src.get_scalar_info()]
        src_c = f"exo_CudaUtil::exo_tk_gl2_window<{tk_t}, {args.size0}>({args.src})"
        return [
            "{  // Place GMEM handle in named temporary, because ThunderKittens is not const-correct.",
            f"  auto exo_tk_src = {src_c};",
            f"  ::kittens::warp::load({args.dst.index()}, exo_tk_src, ::kittens::coord(0, 0, 0, 0));",
            "}",
        ]


@instr
class cuda_tk_store_rg(InstrInfo):
    """Warp stores tile from RMEM into GMEM. Not async."""

    valid_num_types = cuda_tk_valid_num_types_all_pairs

    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1] @ CudaGmemLinear,
        src: [R][size0, size1],  # CudaTkWarpTile(size0, size1, layout="row")
    ):
        assert stride(dst, 1) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]

    def instance(self: InstrInfo, size0: int, size1: int, *, layout: str = "row"):
        self.cu_includes.append("kittens.cuh")
        self.cu_utils.append(cuda_tk_gl2_window_util)
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        self.access_info["src"].mem = CudaTkWarpTile(size0, size1, layout)

    def codegen(self: InstrInfo, args: InstrArgs):
        # fmt: off
        tk_t = cuda_tk_typename_table[args.dst.get_scalar_info()]
        dst_c = f"exo_CudaUtil::exo_tk_gl2_window<{tk_t}, {args.size0}>({args.dst})"
        return [
            "{  // Place GMEM handle in named temporary, because ThunderKittens is not const-correct.",
            f"  auto exo_tk_dst = {dst_c};",
            f"  ::kittens::warp::store(exo_tk_dst, {args.src.index()}, ::kittens::coord(0, 0, 0, 0));",
            "}",
        ]
