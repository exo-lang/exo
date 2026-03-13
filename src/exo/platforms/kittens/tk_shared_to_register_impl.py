from exo import *
from exo.scalars import ScalarInfo

from ..cuda import *
from .tk_types import *
from ..Sm90 import Sm90_SmemSwizzled, Sm90_SmemSwizzled_from_smem_box


def make_tk_load_rs_base(inner_cols):
    class tk_load_rs_impl(InstrInfo):
        valid_num_types = cuda_tk_valid_num_types_all_pairs

        def instance(self: InstrInfo, rows: int, outer_cols: int):
            self.cu_includes.append("cuda_runtime.h")
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
            self.cu_includes.append("cuda_runtime.h")
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


def get_tk_load_rs_instr_impl():
    pass


def get_tk_store_rs_instr_impl():
    pass
