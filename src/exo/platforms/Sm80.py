# Memory, instructions, instr-tl, and sync-tl specific to CUDA sm_80 (Ampere/A100)
# All names exported by this module contain Sm80_
from __future__ import annotations

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
    InstrInfo,
    AtomicityInfo,
)
from ..spork.cuda_memory import *
from ..spork.timelines import cuda_in_order, cuda_in_order_instr
from ..spork.coll_algebra import cuda_warp

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# cp.async instruction
# 1 CUDA thread copies 4, 8, or 16 bytes asynchronously.
# In Exo, we model this with instr_tl=Sm80_cp_async_instr.


class cp_async_impl(InstrInfo):
    n_bytes: int

    def instance_impl(self, n_bytes):
        if n_bytes not in (4, 8, 16):
            raise ValueError(f"cp.async copies 4, 8, or 16 bytes, not {n_bytes}")
        self.instr_tl = Sm80_cp_async_instr
        self.n_bytes = n_bytes


@instr
class Sm80_cp_async_f32(cp_async_impl):
    def behavior(
        size: size,
        smem: [f32][size] @ CudaSmemAtomicity16B,
        gmem: [f32][size] @ CudaGmemLinear,
    ):
        assert stride(smem, 0) == 1
        for i in seq(0, size):
            smem[i] = gmem[i]

    def instance(self, size):
        self.instance_impl(4 * size)
        self.access_info["smem"].out_of_order = True
        self.access_info["gmem"].out_of_order = True

    def codegen(self, args):
        cg_ca = "cg" if self.n_bytes == 16 else "ca"
        ptx = InlinePtxGen(f"cp.async.{cg_ca}.shared.global #0#;", volatile=True)
        ptx.add_arg(str(args.smem.index_ptr()), constraint="smem", log_as="bits")
        ptx.add_arg(str(args.gmem.index_ptr()), constraint="generic", log_as="bits")
        ptx.add_arg(self.n_bytes, constraint="n", log_as="bits")
        return ptx.as_c_lines(py_format=False)


__all__.append("Sm80_cp_async_f32")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Matrix tiles for sm_80 MMA instructions
# 32 threads' registers collectively store a matrix tile for a problem size
# m16n8k4 or m16n8k8


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
        tile_shape = cls.mma_packed_tensor_shape
        assert prim_type == "float"  # TODO
        regcount = tile_shape[0] * tile_shape[1] // 32

        # Last array dimension corresponds to uint32_t-encoded matrix tile
        # Leading dimensions correspond to the Exo user's array dimensions.
        leading = "".join(f"[{c}]" for c in shape[:-2])
        return f"unsigned {new_name}{leading}[{regcount}];"

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
def Sm80_RmemMatrixA(M: int, K: int):
    class Sm80_RmemMatrixA(Sm80_BasicRmemMatrix):
        """Matrix tile for sm_80+ warp MMA A operand"""

        mma_packed_tensor_shape = (M, K)

    return Sm80_RmemMatrixA


@memwin_template
def Sm80_RmemMatrixB(N: int, K: int):
    class Sm80_RmemMatrixB(Sm80_BasicRmemMatrix):
        """Matrix tile for sm_80+ warp MMA B operand"""

        # TODO consider N/K ordering confusion (swap here)
        mma_packed_tensor_shape = (K, N)

    return Sm80_RmemMatrixB


@memwin_template
def Sm80_RmemMatrixD(M: int, N: int):
    class Sm80_RmemMatrixD(Sm80_BasicRmemMatrix):
        """Matrix tile for sm_80+ warp MMA accumulator (C, D) operands"""

        mma_packed_tensor_shape = (M, N)

    return Sm80_RmemMatrixD


__all__ += ["Sm80_RmemMatrixA", "Sm80_RmemMatrixB", "Sm80_RmemMatrixD"]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Instructions for sm_80 MMA
# Unlike later tensor cores, these are NOT async instructions.
# In exo terminology, these operate with instr_tl=cuda_in_order_instr


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
        body = [f"  {regs}[{i}] = __float_as_uint({rhs});" for i, rhs in enumerate(rhs_list)]
        return preamble + body + ["}"]
        # fmt: on


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
            [f"{{D_data}}[{i}]" for i in range(D_nreg)], log_as=None, constraint="=r"
        )
        ptx.add_arg(
            [f"{{A_data}}[{i}]" for i in range(A_nreg)], log_as=None, constraint="r"
        )
        ptx.add_arg(
            [f"{{B_data}}[{i}]" for i in range(B_nreg)], log_as=None, constraint="r"
        )
        ptx.add_arg(
            [f"{{D_data}}[{i}]" for i in range(D_nreg)], log_as=None, constraint="r"
        )
        self.instr_format = ptx.as_c_lines(py_format=True)
        self.access_info["D"].mem = Sm80_RmemMatrixD(16, 8)
        self.access_info["A"].mem = Sm80_RmemMatrixA(16, K)
        self.access_info["B"].mem = Sm80_RmemMatrixB(8, K)


__all__.append("Sm80_mma_tf32")


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
        return _codegen_Sm80_d_tf32(
            args, "  {lhs} = __uint_as_float({regs}[{i}]);", row_major=True
        )


__all__.append("Sm80_mma_store_d_row_major_tf32")


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
        return _codegen_Sm80_d_tf32(
            args, "  {lhs} += __uint_as_float({regs}[{i}]);", row_major=True
        )


__all__.append("Sm80_mma_reduce_d_row_major_tf32")


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
            "  atomicAdd(&{lhs}, __uint_as_float({regs}[{i}]));",
            row_major=True,
        )


__all__.append("Sm80_mma_atomic_reduce_d_row_major_tf32")


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
        return _codegen_Sm80_d_tf32(
            args, "  {lhs} = __uint_as_float({regs}[{i}]);", row_major=False
        )


__all__.append("Sm80_mma_store_d_col_major_tf32")


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
        return _codegen_Sm80_d_tf32(
            args, "  {lhs} += __uint_as_float({regs}[{i}]);", row_major=False
        )


__all__.append("Sm80_mma_reduce_d_col_major_tf32")


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
            "  atomicAdd(&{lhs}, __uint_as_float({regs}[{i}]));",
            row_major=False,
        )


__all__.append("Sm80_mma_atomic_reduce_d_col_major_tf32")


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
