from __future__ import annotations

from exo.API import instr, InstrInfo, InstrArgs
from exo.scalars import ScalarInfo

from ..cuda import *
from .tk_types import *


class basic_map_vec_op(InstrInfo):
    def instance(self: InstrInfo, length: int, *, layout: str):
        self.cu_includes.append("kittens.cuh")
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr

        for access_info in self.access_info.values():
            access_info.mem = CudaTkWarpVec(length, layout)
            access_info.out_of_order = False


class basic_0ary_vec_op(basic_map_vec_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c});"]


class basic_unary_vec_op(basic_map_vec_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c});"]


class basic_binary_3op_vec_op(basic_map_vec_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        lhs_c = args.lhs.index()
        rhs_c = args.rhs.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {lhs_c}, {rhs_c});"]


class basic_binary_lhs_vec_op(basic_map_vec_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {dst_c}, {src_c});"]


class basic_binary_rhs_vec_op(basic_map_vec_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c}, {dst_c});"]


class basic_binary_vec_scalar_op(InstrInfo):
    def instance(
        self: InstrInfo,
        length: int,
        *,
        layout: str,
    ):
        self.cu_includes.append("kittens.cuh")
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr

        dst = self.access_info["dst"]
        dst.mem = CudaTkWarpVec(length, layout)
        dst.out_of_order = False

        if "lhs" in self.access_info:
            lhs = self.access_info["lhs"]
            lhs.mem = CudaTkWarpVec(length, layout)
            lhs.out_of_order = False


class basic_binary_3op_vec_scalar_op(basic_binary_vec_scalar_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        lhs_c = args.lhs.index()
        rhs_c = args.rhs.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {lhs_c}, {rhs_c});"]


class basic_binary_lhs_vec_scalar_op(basic_binary_vec_scalar_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {dst_c}, {src_c});"]


class basic_copy_layout_vec_op(InstrInfo):
    def instance(
        self: InstrInfo,
        length: int,
        *,
        dst_layout: str,
        src_layout: str,
    ):
        self.cu_includes.append("kittens.cuh")
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        dst = self.access_info["dst"]
        src = self.access_info["src"]
        dst.mem = CudaTkWarpVec(length, dst_layout)
        dst.out_of_order = False
        src.mem = CudaTkWarpVec(length, src_layout)
        src.out_of_order = False

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c});"]
