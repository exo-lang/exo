# Memory, instructions, instr-tl, and sync-tl specific to CUDA sm_80 (Ampere/A100)
# All names exported by this module contain Sm80_
from __future__ import annotations

# Currently we import from the exo.spork directory,
# which users shouldn't import directly.
from ..spork.timelines import (
    Sm80_cp_async,
    Sm80_cp_async_instr,
    Sm80_generic,
    cuda_sync_rmem_usage,
    cuda_ram_usage,
)

__all__ = [
    "Sm80_cp_async",
    "Sm80_cp_async_instr",
    "Sm80_generic",
    "cuda_sync_rmem_usage",
    "cuda_ram_usage",
]


# We use these but don't put them in __all__
from .cuda import InlinePtxGen
from ..API import (
    instr,
    memwin_template,
    WindowIndexer,
    window_indexer,
    WindowIndexerResult,
)
from ..spork.cuda_memory import *
from ..spork.timelines import cuda_in_order, cuda_in_order_instr
from ..spork.coll_algebra import cuda_warp

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# cp.async instruction
# 1 CUDA thread copies 4, 8, or 16 bytes asynchronously.
# In Exo, we model this with instr_tl=Sm80_cp_async_instr.


class cp_async_impl:
    def instance_impl(self, n_bytes):
        if n_bytes not in (4, 8, 16):
            raise ValueError(f"cp.async copies 4, 8, or 16 bytes, not {n_bytes}")
        ptx = InlinePtxGen("cp.async.cg.shared.global #0#;", volatile=True)
        ptx.add_arg("&{smem_data}", constraint="smem", log_as="bits")
        ptx.add_arg("&{gmem_data}", constraint="generic", log_as="bits")
        ptx.add_arg(n_bytes, constraint="n", log_as="bits")
        self.instr_format = ptx.as_c_lines(py_format=True)
        self.instr_tl = Sm80_cp_async_instr


@instr
class Sm80_cp_async_f32(cp_async_impl):
    def behavior(
        size: size,
        smem: [f32][size] @ CudaSmemLinear,
        gmem: [f32][size] @ CudaGmemLinear,
    ):
        assert stride(smem, 0) == 1
        for i in seq(0, size):
            smem[i] = gmem[i]

    def instance(self, size):
        self.instance_impl(4 * size)
        self.access_info["smem"].out_of_order = True
        self.access_info["gmem"].out_of_order = True


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
    def instr_tl_permission(cls, instr_tl, is_instr):
        return cls.device_allocated_impl(instr_tl, is_instr)

    @classmethod
    def native_unit(cls):
        return cuda_warp

    @classmethod
    def default_usage_tl(cls, instr_tl):
        return timelines.cuda_sync_rmem_usage

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


@instr
class Sm80_mma_load_a_tf32:
    K: int

    def behavior(
        K: size,
        rmem: [f32][16, K],
        src: [f32][16, K] @ CudaDeviceVisibleLinear,
    ):
        for m in seq(0, 16):
            for k in seq(0, K):
                rmem[m, k] = src[m, k]

    def instance(self, K):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        self.cu_utils = [Sm80_mma_load_util]
        if K != 4 and K != 8:
            raise ValueError("Require K=4 or K=8")
        self.K = K
        self.access_info["rmem"].mem = Sm80_RmemMatrixA(16, K)

    def codegen(self, args: InstrArgs):
        return [
            f"exo_CudaUtil::Sm80_mma_load_a_k{self.K}({args.rmem.index()}, {args.src});"
        ]


__all__.append("Sm80_mma_load_a_tf32")


@instr
class Sm80_mma_load_b_tf32:
    K: int

    def behavior(
        K: size,
        rmem: [f32][K, 8],
        src: [f32][K, 8] @ CudaDeviceVisibleLinear,
    ):
        for k in seq(0, K):
            for n in seq(0, 8):
                rmem[k, n] = src[k, n]

    def instance(self, K):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        self.cu_utils = [Sm80_mma_load_util]
        if K != 4 and K != 8:
            raise ValueError("Require K=4 or K=8")
        self.K = K
        self.access_info["rmem"].mem = Sm80_RmemMatrixB(8, K)

    def codegen(self, args: InstrArgs):
        return [
            f"exo_CudaUtil::Sm80_mma_load_b_k{self.K}({args.rmem.index()}, {args.src});"
        ]


__all__.append("Sm80_mma_load_b_tf32")


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


@instr
class Sm80_mma_store_d_tf32:
    def behavior(
        dst: [f32][16, 8] @ CudaDeviceVisibleLinear,
        rmem: [f32][16, 8] @ Sm80_RmemMatrixD(16, 8),
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                dst[m, n] = rmem[m, n]

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        self.cu_utils = [Sm80_mma_store_util]

    def codegen(self, args: InstrArgs):
        return [f"exo_CudaUtil::Sm80_mma_store_d({args.dst}, {args.rmem.index()});"]


__all__.append("Sm80_mma_store_d_tf32")


@instr
class Sm80_mma_zero_d_tf32:
    def behavior(rmem: [f32][16, 8] @ Sm80_RmemMatrixD(16, 8)):
        for m in seq(0, 16):
            for n in seq(0, 8):
                rmem[m, n] = 0

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warp
        self.cu_utils = [Sm80_mma_zero_util]
        self.instr_format = ["exo_CudaUtil::Sm80_mma_zero_d({rmem_data});"]


__all__.append("Sm80_mma_zero_d_tf32")


Sm80_mma_load_util = r"""
EXO_CUDA_INLINE void Sm80_mma_load_a_k8(unsigned rmem[4], struct exo_win_2f32c src)
{
  const unsigned row_stride = src.strides[0];
  const unsigned col_stride = src.strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &src.data[warp_lane / 4u * row_stride + warp_lane % 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
  rmem[1] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride]);
  rmem[2] = __float_as_uint(gmem_thread_baseaddr[4 * col_stride]);
  rmem[3] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride + 4 * col_stride]);
}

EXO_CUDA_INLINE void Sm80_mma_load_a_k4(unsigned rmem[2], struct exo_win_2f32c src)
{
  const unsigned row_stride = src.strides[0];
  const unsigned col_stride = src.strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &src.data[warp_lane / 4u * row_stride + warp_lane % 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
  rmem[1] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride]);
}

EXO_CUDA_INLINE void Sm80_mma_load_b_k8(unsigned rmem[2], struct exo_win_2f32c src)
{
  const unsigned row_stride = src.strides[0];
  const unsigned col_stride = src.strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &src.data[warp_lane % 4u * row_stride + warp_lane / 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
  rmem[1] = __float_as_uint(gmem_thread_baseaddr[4 * row_stride]);
}

EXO_CUDA_INLINE void Sm80_mma_load_b_k4(unsigned rmem[1], struct exo_win_2f32c src)
{
  const unsigned row_stride = src.strides[0];
  const unsigned col_stride = src.strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &src.data[warp_lane % 4u * row_stride + warp_lane / 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
}
"""

Sm80_mma_store_util = r"""
EXO_CUDA_INLINE void Sm80_mma_store_d(struct exo_win_2f32 dst, const unsigned rmem[4])
{
  const unsigned row_stride = dst.strides[0];
  const unsigned col_stride = dst.strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  float* gmem_thread_baseaddr = &dst.data[(warp_lane / 4u) * row_stride + (warp_lane % 4u) * 2u * col_stride];
  gmem_thread_baseaddr[0] = __uint_as_float(rmem[0]);
  gmem_thread_baseaddr[col_stride] = __uint_as_float(rmem[1]);
  gmem_thread_baseaddr[8 * row_stride] = __uint_as_float(rmem[2]);
  gmem_thread_baseaddr[8 * row_stride + col_stride] = __uint_as_float(rmem[3]);
}
"""

Sm80_mma_zero_util = r"""
EXO_CUDA_INLINE void Sm80_mma_zero_d(unsigned rmem[4])
{
  rmem[0] = 0;
  rmem[1] = 0;
  rmem[2] = 0;
  rmem[3] = 0;
}
"""
