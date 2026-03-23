from .Sm90_fwd import *
from .Sm90_smem import *

from exo.API import *
from exo.platforms.cuda import *
from exo.platforms.cuda_tk import CudaTkWarpTile, cuda_tk_typename_table
from exo.scalars import e4m3, e5m2, f16, bf16, f32

from .Sm90_internal_util import *


@memwin_template
def Sm90_TkRmemTileA(K: int):
    """Specialized (16 x K) ThunderKittens warp tile. wgmma.mma_async A operand"""

    class Tile(CudaTkWarpTile(16, K, "row")):
        qual_tl_dict = cuda_rmem_qual_tl_dict | {
            wgmma_async_instr: wgmma_async_rmem_a_qual
        }

    return Tile


@memwin_template
def Sm90_TkRmemTileD(N: int):
    """Specialized (16 x N) ThunderKittens warp tile. wgmma.mma_async D operand"""

    class Tile(CudaTkWarpTile(16, N, "row")):
        qual_tl_dict = cuda_rmem_qual_tl_dict | {
            wgmma_async_instr: [wgmma_async_rmem_d_qual, wgmma_zero_qual],
        }

        @classmethod
        def alloc(cls, new_name, prim_type, shape, srcinfo):
            scalar_info = ScalarInfo(prim_type)
            try:
                tk_typename = cuda_tk_typename_table[scalar_info]
            except KeyError:
                raise TypeError(
                    f"CudaTkWarpTile currently does not support {scalar_info}"
                )
            array_dims = "".join(f"[{n}]" for n in shape[:-2])
            # fmt: off
            # Unfortunately, we have to rely on the wgmma instr to add the needed cu_util.
            return f"exo_CudaUtil::exo_Sm90_TkRmemTileD<{tk_typename}, {N}> {new_name}{array_dims};"

    return Tile


Sm90_TkRmemTileD_util = """
template <typename T, int N>
struct exo_Sm90_TkRmemTileD: public ::kittens::rt<T, 16, N, ::kittens::ducks::rt_layout::row>
{
    // Set to 0 to trigger a zero-clear on the next wgmma instr.
    // Each wgmma instr resets this to 1.
    int scale_d = 1;
};
"""


def make_basic_mma(a_mode, b_mode):
    assert a_mode in ("row", "col", "rmem")
    assert b_mode in ("row", "col")

    _valid_num_types = set()

    # Valid D/A/B types, restrictions (with culprit identified)
    #   * CUDA: f16 D cannot be used with bf16 or tf32 A/B.
    #   * CUDA: "Transpose" (MN-major) requires 16-bit AB type.
    #   * CUDA: Cannot mix different A/B types unless both are fp8.
    #   * Exo-GPU: further requires 128-bit swizzle for transpose (enforced later).
    #   * ThunderKittens: doesn't allow tf32 register operands.
    #   * ThunderKittens: no support for integer types at all.
    for D in (f16, f32):
        for A in (e4m3, e5m2, f16, bf16, f32):
            for B in (e4m3, e5m2, f16, bf16, f32):
                if a_mode == "col" or b_mode == "row":
                    if A.bits != 16 or B.bits != 16:
                        continue  # Unsupported A/B for transpose
                if D == f16 and (A == bf16 or A == f32):
                    continue  # Unsupported A/B for D == f16
                if A.bits != 8 or B.bits != 8:
                    if A != B:
                        continue  # Mixed A/B rule
                if A.bits == 32 and a_mode == "rmem":
                    continue  # ThunderKittens no tf32 A in registers
                _valid_num_types.add((D, A, B))

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
            self.cu_includes.append("kittens.cuh")
            self.cu_utils.append(Sm90_TkRmemTileD_util)
            self.cu_utils.append(Sm90_codegen_smem_descriptor_util)

            # Each underlying wgmma.mma_async instr
            # processes 32 bytes in the K dimension.
            K_native = 32 * 8 // a_type.bits

            # fmt: off
            assert swizzle in (32, 64, 128), f"swizzle={swizzle} invalid for wgmma"
            assert N % 8 == 0 and N >= 8 and N <= 256, f"N={N} invalid for wgmma"
            assert K % K_native == 0, f"K={K} invalid for wgmma with a_type={a_type}"
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

        if b_mode == "col":

            def instance(self, N, K, *, swizzle=128):
                self.instance_impl(64, N, K, swizzle)

        else:
            # MN-major B requires N divisible by 64 (Exo-GPU restriction)
            def instance(self, N64, K, *, swizzle=128):
                self.instance_impl(64, 64 * N64, K, swizzle)

        def codegen(self: InstrInfo, args):
            # fmt: off
            K_native = 32 * 8 // self.a_type.bits
            K_iters = self.K // K_native

            instr_name = f"wgmma.mma_async.sync.aligned.m{self.M}n{self.N}k{K_native}.{self.d_type}"
            for typ in (self.a_type, self.b_type):
                if typ == f32:
                    instr_name += ".tf32"
                else:
                    instr_name += f".{typ}"

            # Figure out trailing args first (the ones after D, A, B, scale-d).
            # imm-scale-a, imm-scale-b, im-trans-a, im-trans-b
            # with last two only relevant for SMEM 16-bit arguments.
            trailing_args = "1, 1"
            if a_mode != "rmem" and self.a_type.bits == 16:
                if a_mode == "col":
                    trailing_args += ", 1"
                else:
                    assert a_mode == "row"
                    trailing_args += ", 0"  # K-major = A row major
            if self.b_type.bits == 16:
                if b_mode == "col":
                    trailing_args += ", 0"  # K-major = B col major
                else:
                    assert b_mode == "row"
                    trailing_args += ", 1"

            instr_fmt = f"""{{
    .reg .pred p;
    setp.ne.b32 p, #1#, 0;
    {instr_name} #0#,
    p, {trailing_args};
}}"""

            lines = []

            for k in range(K_iters):
                lines.append("{")
                if K_iters != 1:
                    lines.append(f"  // K = {k * K_native} out of {self.K}")
                d_arg = args.D.index()
                ptx = InlinePtxGen(instr_fmt, volatile=True)

                # p=scale_d arg to wgmma
                ptx.add_arg(d_arg + ".scale_d", constraint="r", log_as=None, N=1)

                # Add vector of registers (D argument)
                if self.d_type == f32:
                    d_vec = []
                    for tile_idx in range(0, self.N // 16):
                        tile = f"{d_arg}.tiles[0][{tile_idx}]"
                        d_vec.append(tile + ".data[0].x")
                        d_vec.append(tile + ".data[0].y")
                        d_vec.append(tile + ".data[1].x")
                        d_vec.append(tile + ".data[1].y")
                        d_vec.append(tile + ".data[2].x")
                        d_vec.append(tile + ".data[2].y")
                        d_vec.append(tile + ".data[3].x")
                        d_vec.append(tile + ".data[3].y")
                    ptx.add_arg(d_vec, constraint="+f", log_as=None)
                else:
                    d_vec = []
                    assert self.d_type == f16
                    for tile_idx in range(0, self.N // 16):
                        tile = f"{d_arg}.tiles[0][{tile_idx}]"
                        d_vec.append(f"*reinterpret_cast<uint32_t*>(&{tile}.data[0])")
                        d_vec.append(f"*reinterpret_cast<uint32_t*>(&{tile}.data[1])")
                        d_vec.append(f"*reinterpret_cast<uint32_t*>(&{tile}.data[2])")
                        d_vec.append(f"*reinterpret_cast<uint32_t*>(&{tile}.data[3])")
                    ptx.add_arg(d_vec, constraint="+r", log_as=None)

                # Add A argument
                if a_mode == "rmem":
                    a_vec = []
                    a_arg = args.A.index()
                    tile = f"{a_arg}.tiles[0][{k}]"
                    a_vec.append(f"*reinterpret_cast<const uint32_t*>(&{tile}.data[0])")
                    a_vec.append(f"*reinterpret_cast<const uint32_t*>(&{tile}.data[1])")
                    a_vec.append(f"*reinterpret_cast<const uint32_t*>(&{tile}.data[2])")
                    a_vec.append(f"*reinterpret_cast<const uint32_t*>(&{tile}.data[3])")
                    ptx.add_arg(a_vec, constraint="r", log_as=None)
                else:
                    A_K_major = a_mode == "row"
                    desc_A = Sm90_codegen_smem_descriptor(self.swizzle, args.A, A_K_major, k * K_native)
                    lines.append(f"  const uint64_t exo_descA = {desc_A};")
                    ptx.add_arg("exo_descA", constraint="l", log_as=None)

                # Add B argument
                B_K_major = b_mode == "col"
                desc_B = Sm90_codegen_smem_descriptor(self.swizzle, args.B, B_K_major, k * K_native)
                lines.append(f"  const uint64_t exo_descB = {desc_B};")
                ptx.add_arg("exo_descB", constraint="l", log_as=None)

                # Write out the code for this K iteration.
                # Implicitly reset scale_d flag after wgmma.
                lines.extend(ptx.as_c_lines(tab="  "))
                lines.append(f"  {d_arg}.scale_d = 1;")
                lines.append("}")
            # End for k in range(K_iters)
            return lines

    return WgmmaBase
