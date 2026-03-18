from __future__ import annotations

from exo.API import instr, InstrInfo, InstrArgs
from exo.scalars import ScalarInfo

from ..cuda import *
from .tk_types import *


class cuda_tk_basic_tile_op(InstrInfo):
    def instance(
        self: InstrInfo,
        size0: int,
        size1: int,
        *,
        layout: str = "row",
    ):
        self.cu_includes.append("kittens.cuh")
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order

        for access_info in self.access_info:
            access_info.mem = CudaTkWarpTile(size0, size1, layout)
            access_info.out_of_order = False


class cuda_tk_basic_unary_tile_op(cuda_tk_basic_tile_op):
    def codegen(
        self: InstrInfo,
        args: InstrArgs,
    ):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::{self.kittens_op_name}({dst_c}, {src_c});"]


class cuda_tk_basic_binary_tile_op(cuda_tk_basic_tile_op):
    def codegen(
        self: InstrInfo,
        args: InstrArgs,
    ):
        dst_c = args.dst.index()
        lhs_c = args.lhs.index()
        rhs_c = args.rhs.index()
        return [f"::kittens::{self.kittens_op_name}({dst_c}, {lhs_c}, {rhs_c});"]


@instr
class cuda_tk_tile_copy(cuda_tk_basic_unary_tile_op):
    valid_num_types = cuda_tk_valid_num_types_all_pairs
    kittens_op_name = "copy"

    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1],  # @ CudaTkWarpTile(size0, size1, layout="row")
        src: [R][size0, size1],  # @ CudaTkWarpTile(size0, size1, layout="row")v
    ):
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]


# TODO transpose
