# Memory, instructions, instr-tl, and sync-tl specific to CUDA sm_80 (Ampere/A100)
# All names exported by this module contain Sm80_
from __future__ import annotations

import math

# Currently we import from the exo.spork directory,
# which users shouldn't import directly.
from ..spork.timelines import (
    Sm80_cp_async,
    Sm80_cp_async_instr,
    Sm80_generic,
    cuda_rmem_qual_tl_dict,
    cuda_in_order_ram_qual,
)

__all__ = [
    "Sm80_cp_async",
    "Sm80_cp_async_instr",
    "Sm80_generic",
]


# We use these but don't put them in __all__
from .cuda import InlinePtxGen
from ..API import (
    instr,
    memwin_template,
    WindowIndexer,
    window_indexer,
    WindowIndexerResult,
    InstrArgs,
    InstrInfo,
    AtomicityInfo,
)
from ..spork.cuda_memory import *
from ..spork.timelines import cuda_in_order, cuda_in_order_instr
from ..spork.coll_algebra import cuda_warp

from exo.scalars import ScalarInfo, f32, f16, bf16, i32

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# cp.async instruction
# 1 CUDA thread copies 4, 8, or 16 bytes asynchronously.
# In Exo, we model this with instr_tl=Sm80_cp_async_instr.


class Sm80_cp_async_base(InstrInfo):
    __slots__ = ["n_bytes"]
    n_bytes: int

    def instance_impl(self, *size_tuple):
        scalar_info: ScalarInfo = self.access_info["dst"].scalar_info
        n_bytes = scalar_info.bits * math.prod(size_tuple) // 8
        if n_bytes not in (4, 8, 16):
            typ = f"{scalar_info.shorthand}{list(size_tuple)}"
            raise ValueError(
                f"cp.async copies 4, 8, or 16 bytes, not {n_bytes} ({typ})"
            )
        self.instr_tl = Sm80_cp_async_instr
        self.n_bytes = n_bytes
        self.access_info["dst"].out_of_order = True
        self.access_info["src"].out_of_order = True

    def codegen(self, args):
        cg_ca = "cg" if self.n_bytes == 16 else "ca"
        ptx = InlinePtxGen(f"cp.async.{cg_ca}.shared.global #0#;", volatile=True)
        ptx.add_arg(str(args.dst.index_ptr()), constraint="smem", log_as="bits")
        ptx.add_arg(str(args.src.index_ptr()), constraint="generic", log_as="bits")
        ptx.add_arg(self.n_bytes, constraint="n", log_as="bits")
        return ptx.as_c_lines(py_format=False)

    valid_num_types = ScalarInfo.same()


@instr
class Sm80_cp_async_1d(Sm80_cp_async_base):
    def behavior(
        size0: size,
        dst: [R][size0] @ CudaSmemAtomicity16B,
        src: [R][size0] @ CudaGmemAtomicity16B,
    ):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i0 in seq(0, size0):
            dst[i0] = src[i0]

    def instance(self, size0):
        self.instance_impl(size0)


# TODO test
@instr
class Sm80_cp_async_2d(Sm80_cp_async_base):
    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1] @ CudaSmemAtomicity16B,
        src: [R][size0, size1] @ CudaSmemAtomicity16B,
    ):
        assert stride(dst, 0) == size1
        assert stride(src, 0) == size1
        assert stride(dst, 1) == 1
        assert stride(src, 1) == 1
        for i0 in seq(0, size0):
            for i1 in seq(0, size1):
                dst[i0, i1] = src[i0, i1]

    def instance(self, size0, size1):
        self.instance_impl(size0, size1)


# Legacy
@instr
class Sm80_cp_async_f32(Sm80_cp_async_base):
    def behavior(
        size: size,
        dst: [f32][size] @ CudaSmemAtomicity16B,
        src: [f32][size] @ CudaGmemAtomicity16B,
    ):
        assert stride(dst, 0) == 1
        assert stride(src, 0) == 1
        for i in seq(0, size):
            dst[i] = src[i]

    def instance(self, size):
        self.instance_impl(size)


__all__.append("Sm80_cp_async_base")
__all__.append("Sm80_cp_async_1d")
__all__.append("Sm80_cp_async_2d")
__all__.append("Sm80_cp_async_f32")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# ldmatrix, 4 matrix version
#
# The CUDA description for this instruction is incredibly misleading because
# they use "row" and "column" without defining the majorness of the matrix.
# I use "L-row" to mean what the PTX docs call a "row".
# A single L-row is 16 aligned bytes of data.
# So if the matrix is column-major (B usually is), then an L-row is a column!!!
# A "matrix" (here, L-matrix) is 8 L-rows.
#
# ldmatrix loads four L-matrices from SMEM into RMEM.
# Each L-matrix lives inside one register of each thread in a warp.


class Sm80_ldmatrix_base(InstrInfo):
    def instance(self: InstrInfo, nmat0: int, nmat1: int):
        if nmat0 * nmat1 != 4:
            raise ValueError(f"Need nmat0={nmat0} * nmat1={nmat1} == 4")
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        self.access_info["dst"].distributed_coll_units = [4 * cuda_thread, cuda_thread]

    def codegen(self, args):
        ptx = InlinePtxGen(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 #0#;", volatile=True
        )
        registers = [
            args.dst.index(i // args.nmat1, i % args.nmat1, ptx_data=True)
            for i in range(4)
        ]
        matrix_index = args.exo_wrap_cir(f"threadIdx.x") % 32 / 8
        matrix0_index = matrix_index / args.nmat1
        matrix1_index = matrix_index % args.nmat1
        l_row_index = args.exo_wrap_cir("threadIdx.x % 8")
        smem_expr = args.src.index_ptr(
            8 * matrix0_index + l_row_index, 8 * matrix1_index
        )
        ptx.add_arg(registers, constraint="=r", log_as=None)
        ptx.add_arg(smem_expr, constraint="smem", log_as="bits")
        return ptx.as_c_lines()


@instr
class Sm80_ldmatrix_16bit(Sm80_ldmatrix_base):
    valid_num_types = {(f16, f16), (bf16, bf16)}

    # fmt: off
    def behavior(
            nmat0: size, nmat1: size,  # Substitute constants so nmat0 * nmat1 == 4
            dst: [R][
                8,          # 8 L-rows, distributed by 4*cuda_thread
                4,          # 4 registers per L-row, distributed by cuda_thread
                nmat0,      # Number of L-matrices in the outer dimension (M or N, usually)
                nmat1,      # Number of L-matrices in the inner dimenision (K, usually)
                2]          # 2 f16 values per register
                @ CudaRmemPacked32,
            src: [R][8 * nmat0, 8 * nmat1] @ CudaSmemAtomicity16B):
        # Iterate over L-rows (ldmatrix PTX docs assumes row major)
        # Distributed by register index and threads (thread pitch 4)
        for oR in seq(0, nmat0):
            for oT in seq(0, 8):
                # "columns"
                # Distributed by register index, threads (thread pitch 1), bit pack
                for iR in seq(0, nmat1):
                    for iT in seq(0, 4):
                        for iB in seq(0, 2):
                            dst[oT, iT, oR, iR, iB] = src[8 * oR + oT, 8 * iR + 2 * iT + iB]
                            # NOTE: the 2 * iT is what makes it impossible currently
                            # to make this generic between 16-bit and 32-bit types.


@instr
class Sm80_ldmatrix_32bit(Sm80_ldmatrix_base):
    valid_num_types = {(f32, f32), (i32, i32)}

    # fmt: off
    def behavior(
            nmat0: size, nmat1: size,  # Substitute constants so nmat0 * nmat1 == 4
            dst: [R][
                8,          # 8 L-rows, distributed by 4*cuda_thread
                4,          # 4 registers per L-row, distributed by cuda_thread
                nmat0,      # Number of L-matrices in the outer dimension (M or N, usually)
                nmat1,      # Number of L-matrices in the inner dimenision (K, usually)
                1]          # 1 f32 values per register
                @ CudaRmemPacked32,
            src: [R][8 * nmat0, 8 * nmat1] @ CudaSmemAtomicity16B):
        # Iterate over L-rows (ldmatrix PTX docs assumes row major)
        # Distributed by register index and threads (thread pitch 4)
        for oR in seq(0, nmat0):
            for oT in seq(0, 8):
                # "columns"
                # Distributed by register index, threads (thread pitch 1), bit pack
                for iR in seq(0, nmat1):
                    for iT in seq(0, 4):
                        for iB in seq(0, 1):
                            dst[oT, iT, oR, iR, iB] = src[8 * oR + oT, 8 * iR + iT + iB]


Sm80_ldmatrix_f32 = Sm80_ldmatrix_32bit.partial(dst=f32, src=f32)
Sm80_ldmatrix_f16 = Sm80_ldmatrix_16bit.partial(dst=f16, src=f16)
Sm80_ldmatrix_bf16 = Sm80_ldmatrix_16bit.partial(dst=bf16, src=bf16)
Sm80_ldmatrix_i32 = Sm80_ldmatrix_32bit.partial(dst=i32, src=i32)


__all__.append("Sm80_ldmatrix_16bit")
__all__.append("Sm80_ldmatrix_32bit")
__all__.append("Sm80_ldmatrix_f32")
__all__.append("Sm80_ldmatrix_f16")
__all__.append("Sm80_ldmatrix_bf16")
__all__.append("Sm80_ldmatrix_i32")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# mma.sync.m16n8k? instructions, wrapped as Sm80_mma_m16n8
# These take A and B operands in CudaRmemPacked32.
# The C/D operand lives in a special opaque Sm80_RmemMatrixD(16, 8) type.
# In principle it could also be CudaRmemPacked32, but this
# made instruction unification too hard.


class Sm80_RmemMatrixIndexer(WindowIndexer):
    def index(self, utils, features):
        data = features.get_dataptr()
        for i in range(features.n_array_dims()):
            data = data[features.get_array_offset(i)]
        return self.pack_result(data, False)


@window_indexer(Sm80_RmemMatrixIndexer)
class Sm80_BasicRmemMatrix(CudaBasicDeviceVisible):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        scalar_info: ScalarInfo = ScalarInfo(prim_type)
        tile_shape = cls.mma_packed_tensor_shape

        if scalar_info == f16:
            regcount = tile_shape[0] * tile_shape[1] // 64
            regtype = "int32_t"
        elif scalar_info == f32:
            regcount = tile_shape[0] * tile_shape[1] // 32
            regtype = "float"
        elif scalar_info == i32:
            regcount = tile_shape[0] * tile_shape[1] // 32
            regtype = "int32_t"
        else:
            raise TypeError(f"Sm80_BasicRmemMatrix doesn't support {scalar_info}")

        # Last array dimension corresponds to encoded matrix tile.
        # Leading dimensions correspond to the Exo user's array dimensions.
        leading = "".join(f"[{c}]" for c in shape[:-2])
        return f"{regtype} {new_name}{leading}[{regcount}];"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def device_permission(cls, device, instr_tl):
        return cls.device_allocated_impl(device, instr_tl)

    qual_tl_dict = cuda_rmem_qual_tl_dict

    @classmethod
    def native_unit(cls):
        return cuda_warp

    @classmethod
    def packed_tensor_shape(cls, _):
        return cls.mma_packed_tensor_shape


@memwin_template
def Sm80_RmemMatrixD(M: int, N: int):
    class Sm80_RmemMatrixD(Sm80_BasicRmemMatrix):
        """Matrix tile for sm_80+ warp MMA accumulator (C, D) operands"""

        mma_packed_tensor_shape = (M, N)

    return Sm80_RmemMatrixD


# Note, no other shapes are currently supported.
# This is a bit of a holdover from Exo-GPU development.
Sm80_RmemMatrixD_m16n8 = Sm80_RmemMatrixD(16, 8)


__all__ += ["Sm80_RmemMatrixD", "Sm80_RmemMatrixD_m16n8"]


AB_ptx_names = {
    f32: "f32",
    f16: "f16",
    bf16: "bf16",
    i32: "s32",
}


CD_ptx_names = {
    f32: "f32",
    f16: "f16",
    bf16: "bf16",
    i32: "s32",
}


@instr
class Sm80_mma_m16n8(InstrInfo):
    valid_num_types = {
        (f32, f32, f32),
        (f32, bf16, bf16),
        (f32, f16, f16),
        (f16, f16, f16),
        (i32, i32, i32),
    }

    def behavior(
        K_pack: size,
        # D: opaque tile of 16 x 8
        D: [R][16, 8] @ Sm80_RmemMatrixD_m16n8,
        # A: [4 threads in 32, 1 thread in 4, 2 registers, bit packing]
        A: [R][8, 4, 2, K_pack] @ CudaRmemPacked32,
        # B: [4 threads in 32, 1 thread in 4, bit packing]
        B: [R][8, 4, K_pack] @ CudaRmemPacked32,
    ):
        for m_reg in seq(0, 2):
            for m_thread in seq(0, 8):
                for n_thread in seq(0, 8):
                    for k_thread in seq(0, 4):
                        for k_pack in seq(0, K_pack):
                            D[m_reg * 8 + m_thread, n_thread] += (
                                A[m_thread, k_thread, m_reg, k_pack]
                                * B[n_thread, k_thread, k_pack]
                            )

    def instance(self: InstrInfo, K_pack: int):
        Dtype: ScalarInfo = self.access_info["D"].scalar_info
        Atype: ScalarInfo = self.access_info["A"].scalar_info
        Btype: ScalarInfo = self.access_info["B"].scalar_info
        assert Atype == Btype

        if Atype.bits * K_pack != 32:
            raise ValueError(
                f"A={Atype} requires K_pack=32 // {Atype.bits}, not {K_pack}"
            )

        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        distributed_coll_units = [4 * cuda_thread, cuda_thread]
        self.access_info["A"].distributed_coll_units = distributed_coll_units
        self.access_info["B"].distributed_coll_units = distributed_coll_units

    def codegen(self: InstrInfo, args: InstrArgs):
        Dtype = args.D.get_scalar_info()
        Atype = args.A.get_scalar_info()
        Btype = args.B.get_scalar_info()
        K = 4 * args.K_pack

        Dt = CD_ptx_names[Dtype]
        At = AB_ptx_names[Atype]
        Bt = AB_ptx_names[Btype]
        fmt = f"mma.sync.aligned.m16n8k{K}.row.col.{Dt}.{At}.{Bt}.{Dt} #0#;"
        ptx = InlinePtxGen(fmt, volatile=False)

        CD_nreg = args.D.get_scalar_info().bits // 8
        CD_data = args.D.index()
        CD_args = [f"{CD_data}[{n}]" for n in range(CD_nreg)]
        CD_constraint = "f" if Dtype == f32 else "r"

        # D vector of registers, passed as "f" if f32 else "r"
        ptx.add_arg(CD_args, constraint="=" + CD_constraint, log_as=None)
        # A vector of registers, always force passed as int32_t
        ptx.add_arg(
            [
                f"*reinterpret_cast<const int32_t*>({args.A.index_ptr(0, ptx_data=True)})",
                f"*reinterpret_cast<const int32_t*>({args.A.index_ptr(1, ptx_data=True)})",
            ],
            constraint="r",
            log_as=None,
        )
        # B vector of registers, always force passed as int32_t
        ptx.add_arg(
            [
                f"*reinterpret_cast<const int32_t*>({args.B.index_ptr(ptx_data=True)})",
            ],
            constraint="r",
            log_as=None,
        )
        # C vector of registers, passed as "f" if f32 else "r"
        ptx.add_arg(CD_args, constraint=CD_constraint, log_as=None)
        # Note, CUDA requires tf32 to be treated as int32_t bits.

        return ptx.as_c_lines(py_format=False)


__all__.append("Sm80_mma_m16n8")
Sm80_mma_m16n8k4_f32_tf32 = Sm80_mma_m16n8(K_pack=1, D=f32, A=f32, B=f32)
__all__.append("Sm80_mma_m16n8k4_f32_tf32")
Sm80_mma_m16n8k8_f32_bf16 = Sm80_mma_m16n8(K_pack=2, D=f32, A=bf16, B=bf16)
__all__.append("Sm80_mma_m16n8k8_f32_bf16")
Sm80_mma_m16n8k8_f32_f16 = Sm80_mma_m16n8(K_pack=2, D=f32, A=f16, B=f16)
__all__.append("Sm80_mma_m16n8k8_f32_f16")
Sm80_mma_m16n8k8_f16_f16 = Sm80_mma_m16n8(K_pack=2, D=f16, A=f16, B=f16)
__all__.append("Sm80_mma_m16n8k8_f16_f16")
Sm80_mma_m16n8k4_s32_s32 = Sm80_mma_m16n8(K_pack=1, D=i32, A=i32, B=i32)
__all__.append("Sm80_mma_m16n8k4_s32_s32")


# The Sm80_RmemMatrixD_m16n8 is an opaque per-warp tile.
# Convert to/from per-thread registers or 0 with these instrs.
# For scheduling, you can use stage_mem and use these
# to replace the generated load loops.


@instr
class Sm80_mma_m16n8_zero(InstrInfo):
    valid_num_types = ScalarInfo.same()

    def behavior(D: [R][16, 8] @ Sm80_RmemMatrixD_m16n8):
        for m in seq(0, 16):
            for n in seq(0, 8):
                D[m, n] = 0

    def instance(self):
        self.coll_unit = cuda_warp
        self.instr_tl = cuda_in_order

    def codegen(self, args):
        regcount = args.D.get_scalar_info().bits // 8
        D_data = args.D.index()
        return [f"{D_data}[{n}] = 0;" for n in range(regcount)]


__all__.append("Sm80_mma_m16n8_zero")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# LEGACY MMA EVERYTHING SHOULD BE REMOVED
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


# TODO remove
@memwin_template
def Sm80_RmemMatrixA(M: int, K: int):
    class Sm80_RmemMatrixA(Sm80_BasicRmemMatrix):
        """Matrix tile for sm_80+ warp MMA A operand"""

        mma_packed_tensor_shape = (M, K)

    return Sm80_RmemMatrixA


# TODO remove
@memwin_template
def Sm80_RmemMatrixB(N: int, K: int):
    class Sm80_RmemMatrixB(Sm80_BasicRmemMatrix):
        """Matrix tile for sm_80+ warp MMA B operand"""

        # TODO consider N/K ordering confusion (swap here)
        mma_packed_tensor_shape = (K, N)

    return Sm80_RmemMatrixB


__all__ += ["Sm80_RmemMatrixA", "Sm80_RmemMatrixB"]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Instructions for sm_80 MMA
# Unlike later tensor cores, these are NOT async instructions.
# In exo terminology, these operate with instr_tl=cuda_in_order_instr


# TODO remove
class Sm80_mma_load_base(InstrInfo):
    def instance_impl(self, K: int, matrix_name: str):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        if K != 4 and K != 8:
            raise ValueError("Require K=4 or K=8")
        self.K = K
        if matrix_name == "A":
            self.access_info["rmem"].mem = Sm80_RmemMatrixA(16, K)
        else:
            assert matrix_name == "B" or matrix_name == "A"
            self.access_info["rmem"].mem = Sm80_RmemMatrixB(8, K)

    def codegen_impl(self, matrix_name: str, args: InstrArgs, *, k_major: bool):
        # fmt: off
        preamble = [
          "{",
          "  const unsigned exo_lane = threadIdx.x % 32;",
          "  const unsigned exo_mn = exo_lane / 4;",
          "  const unsigned exo_k = exo_lane % 4;",
        ]
        regs = str(args.rmem.index())
        src = args.src
        if k_major:
            def index(mn, k):
                return src.index(mn, k)
        else:
            def index(mn, k):
                return src.index(k, mn)
        rhs_list = []
        assert matrix_name == "B" or matrix_name == "A"
        rhs_list.append(index("exo_mn + 0", "exo_k + 0"))
        if matrix_name == "A":
            rhs_list.append(index("exo_mn + 8", "exo_k + 0"))
        if self.K == 8:
            rhs_list.append(index("exo_mn + 0", "exo_k + 4"))
            if matrix_name == "A":
                rhs_list.append(index("exo_mn + 8", "exo_k + 4"))
        body = [f"  {regs}[{i}] = {rhs};" for i, rhs in enumerate(rhs_list)]
        return preamble + body + ["}"]
        # fmt: on


# TODO remove
@instr
class Sm80_mma_load_a_row_major_tf32(Sm80_mma_load_base):
    K: int

    def behavior(
        K: size,
        rmem: [f32][16, K],
        src: [f32][16, K] @ CudaDeviceVisibleAtomicity16B,
    ):
        for m in seq(0, 16):
            for k in seq(0, K):
                rmem[m, k] = src[m, k]

    def instance(self, K):
        return self.instance_impl(K, "A")

    def codegen(self, args: InstrArgs):
        return self.codegen_impl("A", args, k_major=True)


__all__.append("Sm80_mma_load_a_row_major_tf32")


# TODO remove
@instr
class Sm80_mma_load_a_col_major_tf32(Sm80_mma_load_base):
    K: int

    def behavior(
        K: size,
        rmem: [f32][16, K],
        src: [f32][K, 16] @ CudaDeviceVisibleAtomicity16B,
    ):
        for m in seq(0, 16):
            for k in seq(0, K):
                rmem[m, k] = src[k, m]

    def instance(self, K):
        return self.instance_impl(K, "A")

    def codegen(self, args: InstrArgs):
        return self.codegen_impl("A", args, k_major=False)


__all__.append("Sm80_mma_load_a_col_major_tf32")


# TODO remove
@instr
class Sm80_mma_load_b_row_major_tf32(Sm80_mma_load_base):
    K: int

    def behavior(
        K: size,
        rmem: [f32][K, 8],
        src: [f32][K, 8] @ CudaDeviceVisibleAtomicity16B,
    ):
        for k in seq(0, K):
            for n in seq(0, 8):
                rmem[k, n] = src[k, n]

    def instance(self, K):
        self.instance_impl(K, "B")

    def codegen(self, args: InstrArgs):
        return self.codegen_impl("B", args, k_major=False)


__all__.append("Sm80_mma_load_b_row_major_tf32")


# TODO remove
@instr
class Sm80_mma_load_b_col_major_tf32(Sm80_mma_load_base):
    K: int

    def behavior(
        K: size,
        rmem: [f32][K, 8],
        src: [f32][8, K] @ CudaDeviceVisibleAtomicity16B,
    ):
        for k in seq(0, K):
            for n in seq(0, 8):
                rmem[k, n] = src[n, k]

    def instance(self, K):
        self.instance_impl(K, "B")

    def codegen(self, args: InstrArgs):
        return self.codegen_impl("B", args, k_major=True)


__all__.append("Sm80_mma_load_b_col_major_tf32")


# TODO remove
@instr
class Sm80_mma_tf32:
    def behavior(
        K: size,
        D: [f32][16, 8],
        A: [f32][16, K],
        B: [f32][K, 8],
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                for k in seq(0, K):
                    D[m, n] += A[m, k] * B[k, n]

    def instance(self, K):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        if K != 4 and K != 8:
            raise ValueError("Require K=4 or K=8")
        ptx_instr = f"mma.sync.aligned.m16n8k{K}.row.col.f32.tf32.tf32.f32"
        ptx = InlinePtxGen(f"{ptx_instr} #0#;", volatile=False)
        D_nreg = 4
        A_nreg = K // 2
        B_nreg = K // 4
        ptx.add_arg(
            [f"{{D_data}}[{i}]" for i in range(D_nreg)], log_as=None, constraint="=f"
        )
        ptx.add_arg(
            [f"__float_as_uint({{A_data}}[{i}])" for i in range(A_nreg)],
            log_as=None,
            constraint="r",
        )
        ptx.add_arg(
            [f"__float_as_uint({{B_data}}[{i}])" for i in range(B_nreg)],
            log_as=None,
            constraint="r",
        )
        ptx.add_arg(
            [f"{{D_data}}[{i}]" for i in range(D_nreg)], log_as=None, constraint="f"
        )
        self.instr_format = ptx.as_c_lines(py_format=True)
        self.access_info["D"].mem = Sm80_RmemMatrixD(16, 8)
        self.access_info["A"].mem = Sm80_RmemMatrixA(16, K)
        self.access_info["B"].mem = Sm80_RmemMatrixB(8, K)


__all__.append("Sm80_mma_tf32")


# TODO remove
def _codegen_Sm80_d_tf32(args: InstrArgs, fmt: str, *, row_major: bool):
    # fmt: off
    preamble = [
      "{",
      "  const unsigned exo_lane = threadIdx.x % 32;",
      "  const unsigned exo_m = exo_lane / 4;",
      "  const unsigned exo_n = (exo_lane % 4) * 2;",
    ]

    if row_major:
        def index(m, n):
            return dst.index(m, n)
    else:
        def index(m, n):
            return dst.index(n, m)

    regs = str(args.rmem.index())
    dst = args.dst
    lhs_list = [
        index("exo_m + 0", "exo_n + 0"),
        index("exo_m + 0", "exo_n + 1"),
        index("exo_m + 8", "exo_n + 0"),
        index("exo_m + 8", "exo_n + 1"),
    ]
    body = [fmt.format(i=i, regs=regs, lhs=lhs) for i, lhs in enumerate(lhs_list)]
    return preamble + body + ["}"]
    # fmt: on


# TODO remove
@instr
class Sm80_mma_store_d_row_major_tf32:
    def behavior(
        dst: [f32][16, 8] @ CudaDeviceVisibleAtomicity16B,
        rmem: [f32][16, 8] @ Sm80_RmemMatrixD(16, 8),
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                dst[m, n] = rmem[m, n]

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp

    def codegen(self, args: InstrArgs):
        return _codegen_Sm80_d_tf32(args, "  {lhs} = {regs}[{i}];", row_major=True)


__all__.append("Sm80_mma_store_d_row_major_tf32")


# TODO remove
@instr
class Sm80_mma_reduce_d_row_major_tf32:
    def behavior(
        dst: [f32][16, 8] @ CudaDeviceVisibleAtomicity16B,
        rmem: [f32][16, 8] @ Sm80_RmemMatrixD(16, 8),
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                dst[m, n] += rmem[m, n]

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp

    def codegen(self, args: InstrArgs):
        return _codegen_Sm80_d_tf32(args, "  {lhs} += {regs}[{i}];", row_major=True)


__all__.append("Sm80_mma_reduce_d_row_major_tf32")


# TODO remove
@instr
class Sm80_mma_atomic_reduce_d_row_major_tf32:
    def behavior(
        dst: [f32][16, 8] @ CudaDeviceVisibleAtomicity16B,
        rmem: [f32][16, 8] @ Sm80_RmemMatrixD(16, 8),
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                dst[m, n] += rmem[m, n]

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        self.access_info["dst"].atomicity = AtomicityInfo([cuda_in_order_ram_qual])

    def codegen(self, args: InstrArgs):
        return _codegen_Sm80_d_tf32(
            args,
            "  atomicAdd(&{lhs}, {regs}[{i}]);",
            row_major=True,
        )


__all__.append("Sm80_mma_atomic_reduce_d_row_major_tf32")


# TODO remove
@instr
class Sm80_mma_store_d_col_major_tf32:
    def behavior(
        dst: [f32][8, 16] @ CudaDeviceVisibleAtomicity16B,
        rmem: [f32][16, 8] @ Sm80_RmemMatrixD(16, 8),
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                dst[n, m] = rmem[m, n]

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp

    def codegen(self, args: InstrArgs):
        return _codegen_Sm80_d_tf32(args, "  {lhs} = {regs}[{i}];", row_major=False)


__all__.append("Sm80_mma_store_d_col_major_tf32")


# TODO remove
@instr
class Sm80_mma_reduce_d_col_major_tf32:
    def behavior(
        dst: [f32][8, 16] @ CudaDeviceVisibleAtomicity16B,
        rmem: [f32][16, 8] @ Sm80_RmemMatrixD(16, 8),
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                dst[n, m] += rmem[m, n]

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp

    def codegen(self, args: InstrArgs):
        return _codegen_Sm80_d_tf32(args, "  {lhs} += {regs}[{i}];", row_major=False)


__all__.append("Sm80_mma_reduce_d_col_major_tf32")


# TODO remove
@instr
class Sm80_mma_atomic_reduce_d_col_major_tf32:
    def behavior(
        dst: [f32][8, 16] @ CudaDeviceVisibleAtomicity16B,
        rmem: [f32][16, 8] @ Sm80_RmemMatrixD(16, 8),
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                dst[n, m] += rmem[m, n]

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        self.access_info["dst"].atomicity = AtomicityInfo([cuda_in_order_ram_qual])

    def codegen(self, args: InstrArgs):
        return _codegen_Sm80_d_tf32(
            args,
            "  atomicAdd(&{lhs}, {regs}[{i}]);",
            row_major=False,
        )


__all__.append("Sm80_mma_atomic_reduce_d_col_major_tf32")


# TODO remove
@instr
class Sm80_mma_zero_d_tf32:
    def behavior(rmem: [f32][16, 8] @ Sm80_RmemMatrixD(16, 8)):
        for m in seq(0, 16):
            for n in seq(0, 8):
                rmem[m, n] = 0

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp

    def codegen(self, args):
        regs = str(args.rmem.index())
        return [
            f"{regs}[0] = 0;",
            f"{regs}[1] = 0;",
            f"{regs}[2] = 0;",
            f"{regs}[3] = 0;",
        ]


__all__.append("Sm80_mma_zero_d_tf32")
