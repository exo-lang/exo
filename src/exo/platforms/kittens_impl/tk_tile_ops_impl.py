from __future__ import annotations

from exo.API import instr, InstrInfo, InstrArgs
from exo.scalars import ScalarInfo

from ..cuda import *
from .tk_types import *


class basic_map_tile_op(InstrInfo):
    __slots__ = []

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
    __slots__ = []

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c});"]


class basic_make_causal_op(basic_map_tile_op):
    __slots__ = []

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        ctype = cuda_tk_typename_table[args.dst.get_scalar_info()]
        # fmt: off
        # ThunderKittens for whatever reason doesn't have a causal-with-offset op.
        # They just decomp to 16x16 tiles and use these pragma unroll loops wherever
        # they need to do this (which is always), so we emulate that here.
        # Note, 8-bit types don't use 16x16 tile size;
        # this would have to be changed in that case.
        return [
            f"#pragma unroll",
            f"for (int exo_causal_r = 0; exo_causal_r < {args.rows >> 4}; ++exo_causal_r) {{",
            f"  #pragma unroll",
            f"  for (int exo_causal_c = 0; exo_causal_c < {args.cols >> 4}; ++exo_causal_c) {{",
            f"    int exo_causal_delta = exo_causal_r * 16 - exo_causal_c * 16 + static_cast<int>({args.row_offset} - {args.col_offset});",
            f"    ::kittens::rt<{ctype}, 16, 16> exo_causal_subtile;",
            f"    exo_causal_subtile.tiles[0][0] = {dst_c}.tiles[exo_causal_r][exo_causal_c];",
            f"    const auto exo_causal_identity = ::kittens::base_types::constants<{ctype}>::{self.kittens_constant_name}();",
            f"    if (exo_causal_delta {self.cmp_op} 0) ::kittens::warp::{self.kittens_constant_name}(exo_causal_subtile);",
            f"    if (exo_causal_delta == 0) ::kittens::warp::{self.kittens_op_name}(exo_causal_subtile, exo_causal_subtile, exo_causal_identity);",
            f"    {dst_c}.tiles[exo_causal_r][exo_causal_c] = exo_causal_subtile.tiles[0][0];",
            f"  }}",
            f"}}",
        ]


class basic_unary_tile_op(basic_map_tile_op):
    __slots__ = []

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c});"]


class basic_binary_3op_tile_op(basic_map_tile_op):
    __slots__ = []

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        lhs_c = args.lhs.index()
        rhs_c = args.rhs.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {lhs_c}, {rhs_c});"]


class basic_binary_lhs_tile_op(basic_map_tile_op):
    __slots__ = []

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {dst_c}, {src_c});"]


class basic_binary_rhs_tile_op(basic_map_tile_op):
    __slots__ = []

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {src_c}, {dst_c});"]


class basic_binary_tile_scalar_op(InstrInfo):
    __slots__ = []

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

        dst = self.access_info["dst"]
        dst.mem = CudaTkWarpTile(rows, cols, layout)
        dst.out_of_order = False

        if "lhs" in self.access_info:
            lhs = self.access_info["lhs"]
            lhs.mem = CudaTkWarpTile(rows, cols, layout)
            lhs.out_of_order = False


class basic_binary_3op_tile_scalar_op(basic_binary_tile_scalar_op):
    __slots__ = []

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        lhs_c = args.lhs.index()
        rhs_c = args.rhs.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {lhs_c}, {rhs_c});"]


class basic_binary_lhs_tile_scalar_op(basic_binary_tile_scalar_op):
    __slots__ = []

    def codegen(self: InstrInfo, args: InstrArgs):
        dst_c = args.dst.index()
        src_c = args.src.index()
        return [f"::kittens::warp::{self.kittens_op_name}({dst_c}, {dst_c}, {src_c});"]


class basic_row_reduce_op(InstrInfo):
    __slots__ = []

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
    __slots__ = []

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
    __slots__ = []

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
    __slots__ = []

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
