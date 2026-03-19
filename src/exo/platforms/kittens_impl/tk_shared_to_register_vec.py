# fmt: off

from __future__ import annotations
from exo.API import instr, InstrInfo, InstrArgs

from exo.platforms.cuda import CudaSmemLinear, cuda_warp, cuda_in_order_instr

from .tk_types import *


__all__ = ["cuda_tk_load_vec_rs", "cuda_tk_store_vec_rs"]


@instr
class cuda_tk_load_vec_rs(InstrInfo):
    """Warp loads vector into RMEM from SMEM. Not async."""

    valid_num_types = cuda_tk_valid_num_types_all_pairs

    def behavior(
        length: size,
        dst: [R][length],  # @ CudaTkWarpVec(length, layout)
        src: [R][length] @ CudaSmemLinear,
    ):
        assert stride(src, 0) == 1
        for i in seq(0, length):
            dst[i] = src[i]

    def instance(self: InstrInfo, length: int, *, layout: str):
        self.cu_includes.append("kittens.cuh")
        self.cu_utils.append(cuda_tk_cast_sv_util)
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        self.access_info["dst"].mem = CudaTkWarpVec(length, layout)
        self.access_info["src"].mem = CudaSmemLinear

    def codegen(self: InstrInfo, args: InstrArgs):
        return [
            f"::kittens::warp::load(",
            f"  {args.dst.index()},",
            f"  exo_CudaUtil::exo_tk_cast_sv<{args.length}>({args.src.index_ptr()})",
            ");",
        ]


@instr
class cuda_tk_store_vec_rs(InstrInfo):
    """Warp stores vector from RMEM into SMEM. Not async."""

    valid_num_types = cuda_tk_valid_num_types_all_pairs

    def behavior(
        length: size,
        dst: [R][length] @ CudaSmemLinear,
        src: [R][length],  # @ CudaTkWarpVec(length, layout)
    ):
        assert stride(dst, 0) == 1
        for i in seq(0, length):
            dst[i] = src[i]

    def instance(self: InstrInfo, length: int, *, layout: str):
        self.cu_includes.append("kittens.cuh")
        self.cu_utils.append(cuda_tk_cast_sv_util)
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        self.access_info["dst"].mem = CudaSmemLinear
        self.access_info["src"].mem = CudaTkWarpVec(length, layout)

    def codegen(self: InstrInfo, args: InstrArgs):
        return [
            f"::kittens::warp::store(",
            f"  exo_CudaUtil::exo_tk_cast_sv<{args.length}>({args.dst.index_ptr()}),",
            f"  {args.src.index()}",
            ");",
        ]
