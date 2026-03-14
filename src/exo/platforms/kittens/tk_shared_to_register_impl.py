from dataclasses import dataclass

from typing import Type

from exo import *
from exo.API import Memory, Procedure
from exo.scalars import ScalarInfo

from ..cuda_fwd import (
    CudaBasicDeviceVisible,
    CudaSmemAtomicity16B,
    cuda_warp,
    cuda_in_order_instr,
)
from .tk_types import *
from ..Sm90 import Sm90_SmemSwizzled, Sm90_SmemSwizzled_from_smem_box


def make_tk_load_rs_base(inner_cols):
    class tk_load_rs_impl(InstrInfo):
        valid_num_types = cuda_tk_valid_num_types_all_pairs

        def instance(self: InstrInfo, rows: int, outer_cols: int):
            self.cu_includes.append("kittens.cuh")
            self.coll_unit = cuda_warp
            self.instr_tl = cuda_in_order_instr
            self.access_info["dst"].mem = CudaTkWarpTile(
                rows, outer_cols * inner_cols, "row"
            )
            self.access_info["src"].mem = Sm90_SmemSwizzled_from_smem_box(
                self.access_info["src"].scalar_info,
                (outer_cols, rows, inner_cols),
            )

        def codegen(self: InstrInfo, args: InstrArgs):
            subtile_size = (args.outer_cols, args.rows, inner_cols)
            subtile_c = args.src.index(as_tk_subtile=subtile_size)
            return [
                "{  // Place SMEM handle in named temporary, because ThunderKittens is not const-correct.",
                f"  auto exo_tk_subtile = {subtile_c};",
                f"  ::kittens::warp::load({args.dst.index()}, exo_tk_subtile);",
                "}",
            ]

    return tk_load_rs_impl


def make_tk_store_rs_base(inner_cols):
    class tk_store_rs_impl(InstrInfo):
        valid_num_types = cuda_tk_valid_num_types_all_pairs

        def instance(self: InstrInfo, rows: int, outer_cols: int):
            self.cu_includes.append("kittens.cuh")
            self.coll_unit = cuda_warp
            self.instr_tl = cuda_in_order_instr
            self.access_info["src"].mem = CudaTkWarpTile(
                rows, outer_cols * inner_cols, "row"
            )
            self.access_info["dst"].mem = Sm90_SmemSwizzled_from_smem_box(
                self.access_info["dst"].scalar_info,
                (outer_cols, rows, inner_cols),
            )

        def codegen(self: InstrInfo, args: InstrArgs):
            subtile_size = (args.outer_cols, args.rows, inner_cols)
            subtile_c = args.dst.index(as_tk_subtile=subtile_size)
            return [
                "{  // Place SMEM handle in named temporary, because ThunderKittens is not const-correct.",
                f"  auto exo_tk_subtile = {subtile_c};",
                f"  ::kittens::warp::store(exo_tk_subtile, {args.src.index()});",
                "}",
            ]

    return tk_store_rs_impl


@dataclass(slots=True)
class CudaTkRsInstrAdvice:
    instr: Procedure
    rmem: Type[CudaBasicDeviceVisible]
    smem: Type[CudaSmemAtomicity16B]
    swizzle_elements: int

    # Just fyi for the generated instr; can be ignored.
    outer_cols: int
    rows: int
    inner_cols: int  # Same as swizzle_elements


def get_tk_rs_instr_advice_impl(size0, size1, dst, src, swizzle, instr_dict, is_store):
    assert size0 % 16 == 0
    assert size1 % 16 == 0

    dst = ScalarInfo(dst)
    src = ScalarInfo(src)

    assert swizzle in (32, 64, 128)
    inner_cols = 8 * swizzle // (dst.bits if is_store else src.bits)
    outer_cols = size1 // inner_cols
    rows = size0

    # fmt: off
    assert size1 % inner_cols == 0, f"size1={size1} needs to be divisible by {inner_cols} for swizzle={swizzle}"
    # fmt: on

    instr_template = instr_dict[inner_cols]
    instr = instr_template(rows=rows, outer_cols=outer_cols, dst=dst, src=src)
    return CudaTkRsInstrAdvice(
        instr,
        CudaTkWarpTile(size0, size1, "row"),
        Sm90_SmemSwizzled(swizzle),
        inner_cols,
        outer_cols,
        rows,
        inner_cols,
    )
