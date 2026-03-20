from __future__ import annotations

from exo.API import instr, InstrInfo, InstrArgs
from exo.scalars import ScalarInfo

from ..cuda import *
from .tk_types import *


class basic_map_tile_op(InstrInfo):
    def instance(
        self: InstrInfo,
        rows: int,
        cols: int,
        *,
        layout: str = "row",
    ):
        self.cu_includes.append("kittens.cuh")
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr

        for access_info in self.access_info.values():
            access_info.mem = CudaTkWarpTile(rows, cols, layout)
            access_info.out_of_order = False


class basic_0ary_tile_op(basic_map_tile_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c});"]


class basic_make_causal_op(basic_map_tile_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        ctype = cuda_tk_typename_table[args.dst.get_scalar_info()]
        return [
            f"::kittens::warp::{self.kittens_op_name}({dst_c}, {dst_c},",
            f"    ::kittens::base_types::constants<{ctype}>::{self.kittens_constant_name}());",
        ]


class basic_unary_tile_op(basic_map_tile_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c});"]


class basic_binary_tile_op(basic_map_tile_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        lhs_c = args.lhs.index()
        rhs_c = args.rhs.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {lhs_c}, {rhs_c});"]


class basic_binary_lhs_tile_op(basic_map_tile_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {dst_c}, {src_c});"]


class basic_binary_rhs_tile_op(basic_map_tile_op):
    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c}, {dst_c});"]


class basic_row_reduce_op(InstrInfo):
    def instance(
        self: InstrInfo,
        rows: int,
        cols: int,
        *,
        layout: str = "row",
    ):
        self.cu_includes.append("kittens.cuh")
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        tile_mem = CudaTkWarpTile(rows, cols, layout)
        vec_mem = tile_mem.col_vec
        dst = self.access_info["dst"]
        src = self.access_info["src"]
        src.mem = tile_mem
        src.out_of_order = False
        dst.mem = vec_mem
        dst.out_of_order = False

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c}, {dst_c});"]


class basic_col_reduce_op(InstrInfo):
    def instance(
        self: InstrInfo,
        rows: int,
        cols: int,
        *,
        layout: str = "row",
    ):
        self.cu_includes.append("kittens.cuh")
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        tile_mem = CudaTkWarpTile(rows, cols, layout)
        vec_mem = tile_mem.row_vec
        dst = self.access_info["dst"]
        src = self.access_info["src"]
        src.mem = tile_mem
        src.out_of_order = False
        dst.mem = vec_mem
        dst.out_of_order = False

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c}, {dst_c});"]


class basic_broadcast_row_op(InstrInfo):
    def instance(
        self: InstrInfo,
        rows: int,
        cols: int,
        *,
        layout: str = "row",
    ):
        self.cu_includes.append("kittens.cuh")
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        tile_mem = CudaTkWarpTile(rows, cols, layout)
        vec_mem = tile_mem.col_vec
        dst = self.access_info["dst"]
        src = self.access_info["src"]
        src.mem = vec_mem
        src.out_of_order = False
        dst.mem = tile_mem
        dst.out_of_order = False

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        if "broadcast" in self.kittens_op_name:
            return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c});"]
        else:
            return [
                f"::kittens::warp::{self.kittens_op_name}({dst_c}, {dst_c}, {src_c});"
            ]


class basic_broadcast_col_op(InstrInfo):
    def instance(
        self: InstrInfo,
        rows: int,
        cols: int,
        *,
        layout: str = "row",
    ):
        self.cu_includes.append("kittens.cuh")
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order_instr
        tile_mem = CudaTkWarpTile(rows, cols, layout)
        vec_mem = tile_mem.row_vec
        dst = self.access_info["dst"]
        src = self.access_info["src"]
        src.mem = vec_mem
        src.out_of_order = False
        dst.mem = tile_mem
        dst.out_of_order = False

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        if "broadcast" in self.kittens_op_name:
            return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c});"]
        else:
            return [
                f"::kittens::warp::{self.kittens_op_name}({dst_c}, {dst_c}, {src_c});"
            ]
