from .Sm90_fwd import *
from .Sm90_smem import *

from exo.API import *
from exo.platforms.cuda import *
from exo.platforms.cuda_tk import CudaTkWarpTile, cuda_tk_typename_table
from exo.scalars import f16, f32

from .Sm90_internal_util import *


@memwin_template
def Sm90_TkRmemTileA(K: int):
    class Tile(CudaTkWarpTile(16, K, "row")):
        qual_tl_dict = cuda_rmem_qual_tl_dict | {
            wgmma_async_instr: wgmma_async_rmem_a_qual
        }

    return Tile


@memwin_template
def Sm90_TkRmemTileD(N: int):
    class Tile(CudaTkWarpTile(16, N, "row")):
        qual_tl_dict = cuda_rmem_qual_tl_dict | {
            wgmma_async_instr: wgmma_async_rmem_d_qual
        }

    return Tile


def make_basic_mma(a_mode, b_mode, m64):
    assert a_mode in ("row", "col", "rmem")
    assert b_mode in ("row", "col")
    assert isinstance(m64, bool)

    if a_mode == "col":
        assert m64, "trans_a requires M=64"

    # f16/f16/f16 supports everything
    _valid_num_types = {(f16, f16, f16)}

    # D=f32/AB/AB support, with rules:
    #   * "Transpose" (MN-major) requires 16-bit AB type
    #     Exo-GPU further requires 128-bit swizzle (enforced later).
    #   * ThunderKittens doesn't allow tf32 register operands.
    for AB in cuda_tk_typename_table:
        if a_mode == "col" or b_mode == "row":
            if AB.bits != 16:
                continue
        if AB.bits > 16 and a_mode == "rmem":
            continue
        _valid_num_types.add((f32, AB, AB))

    class WgmmaBase(InstrInfo):
        # All of these can be looked up when needed, but it's more clear
        # to cache them in this InstrInfo object once.
        __slots__ = ["a_type", "b_type", "d_type", "M", "N", "K", "swizzle"]
        a_type: ScalarInfo
        b_type: ScalarInfo
        d_type: ScalarInfo
        M: int
        N: int
        K: int
        swizzle: int

        valid_num_types = _valid_num_types

        def instance_impl(self: InstrInfo, M, N, K, swizzle):
            if a_mode == "col" or b_mode == "row":
                # fmt: off
                assert swizzle == 128, "Sorry, unimplemented, transpose for swizzle != 128"

            a_access = self.access_info["A"]
            b_access = self.access_info["B"]
            d_access = self.access_info["D"]
            a_type: ScalarInfo = a_access.scalar_info
            b_type: ScalarInfo = b_access.scalar_info
            d_type: ScalarInfo = d_access.scalar_info
            self.a_type = a_type
            self.b_type = b_type
            self.d_type = d_type
            self.M = M
            self.N = N
            self.K = K
            self.swizzle = swizzle
            self.coll_unit = cuda_warpgroup
            self.instr_tl = wgmma_async_instr

            # Each underlying wgmma.mma_async instr
            # processes 32 bytes in the K dimension.
            K_divisor = 32 * 8 // a_type.bits

            # fmt: off
            assert swizzle in (32, 64, 128), f"swizzle={swizzle} invalid for wgmma"
            assert N % 8 == 0 and N >= 8 and N <= 256, f"N={N} invalid for wgmma"
            assert K % K_divisor == 0, f"K={K} invalid for wgmma with a_type={a_type}"
            # fmt: on

            d_access.distributed_coll_units = (cuda_warp,)
            d_access.mem = Sm90_TkRmemTileD(N)
            d_access.out_of_order = False

            if a_mode == "rmem":
                a_access.distributed_coll_units = (cuda_warp,)
                a_access.mem = Sm90_TkRmemTileA(K)
            else:
                a_access.mem = Sm90_SmemSwizzled(swizzle)
            a_access.out_of_order = True

            b_access.out_of_order = True
            b_access.mem = Sm90_SmemSwizzled(swizzle)

        # m64=True omits M64 instance template parameter
        if m64:
            if b_mode == "col":

                def instance(self, N, K, *, swizzle=128):
                    self.instance_impl(64, N, K, swizzle)

            else:
                # MN-major B requires N divisible by 64 (Exo-GPU restriction)
                def instance(self, N64, K, *, swizzle=128):
                    self.instance_impl(64, 64 * N64, K, swizzle)

        # m64=False includes M64 instance template parameter
        else:
            if b_mode == "col":

                def instance(self, M64, N, K, *, swizzle=128):
                    self.instance_impl(64 * M64, N, K, swizzle)

            else:

                def instance(self, M64, N64, K, *, swizzle=128):
                    self.instance_impl(64 * M64, 64 * N64, K, swizzle)

        def codegen(self: InstrInfo, args):
            return []  # TODO

    return WgmmaBase
