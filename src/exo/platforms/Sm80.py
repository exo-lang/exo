# Memory, instructions, and actor kinds specific to CUDA sm_80 (Ampere/A100)
# All names exported by this module contain Sm80_
from __future__ import annotations

# Currently we import from the exo.spork directory,
# which users shouldn't import directly.
from ..spork.actor_kinds import (
    Sm80_cp_async,
    Sm80_generic,
    sig_Sm80_cp_async,
)

__all__ = ["Sm80_cp_async", "Sm80_generic", "sig_Sm80_cp_async"]


# We use these but don't put them in __all__
from .cuda import InlinePtxGen
from ..API import instr
from ..spork.cuda_memory import *
from ..spork.actor_kinds import sig_cuda_classic, cuda_classic
from ..spork.coll_algebra import cuda_warp

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# cp.async instruction
# 1 CUDA thread copies 4, 8, or 16 bytes asynchronously.
# In Exo, we model this with actor_kind=Sm80_cp_async.


class cp_async_impl:
    def instance_impl(self, n_bytes):
        if n_bytes not in (4, 8, 16):
            raise ValueError(f"cp.async copies 4, 8, or 16 bytes, not {n_bytes}")
        ptx = InlinePtxGen("cp.async.cg.shared.global #0#;", volatile=True)
        ptx.add_arg("&{smem_data}", constraint="smem", log_as="bits")
        ptx.add_arg("&{gmem_data}", constraint="generic", log_as="bits")
        ptx.add_arg(n_bytes, constraint="n", log_as="bits")
        self.instr_format = ptx.as_c_lines(py_format=True)
        self.actor_kind = Sm80_cp_async
        self.access_info["smem"].actor_signature = sig_Sm80_cp_async
        self.access_info["gmem"].actor_signature = sig_Sm80_cp_async


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


__all__.append("Sm80_cp_async_f32")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Matrix tiles for sm_80 MMA instructions
# 32 threads' registers collectively store a matrix tile for a problem size
# m16n8k4 or m16n8k8


class Sm80_BasicRmemMatrix(CudaBasicDeviceVisible):
    @classmethod
    def window_definition(cls, ctx):
        if ctx.n_dims() != 2:
            raise MemGenError(
                f"{ctx.srcinfo()}: Only support windows to a single tile (n_dims 2)"
            )
        return ctx.generate_default("Sm80_RmemMatrix", "unsigned")

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) < 2:
            raise MemGenError(f"{srcinfo}: Require at least 2D tile for Sm80 MMA tile")
        array_shape = shape[:-2]
        tile_shape = shape[-2:]

        try:
            int_tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
            if int_tile_shape not in cls.expected_tile_shapes:
                raise ValueError("WRONG")
        except Exception:
            raise MemGenError(
                f"{srcinfo}: last 2 dims (tile_shape) {tile_shape} must match "
                f"one of {cls.expected_tile_shapes}"
            )

        assert prim_type == "float"  # TODO
        regcount = int_tile_shape[0] * int_tile_shape[1] // 32

        # Last array dimension corresponds to uint32_t-encoded matrix tile
        # Leading dimensions correspond to the Exo user's array dimensions.
        leading = "".join(f"[{c}]" for c in array_shape)
        return f"unsigned {new_name}{leading}[{regcount}];"

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        if basetyp.is_win():
            return f"*{baseptr}.data"
        assert len(strides) >= 2
        # assert strides[-2] == str(cls.tile_shape[1])
        assert strides[-1] == "1"
        leading = "".join(f"[{c}]" for c in indices[:-2])
        return f"{baseptr}{leading}"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def actor_kind_permission(cls, actor_kind, is_instr):
        return cls.device_allocated_impl(actor_kind, is_instr)

    @classmethod
    def native_unit(cls):
        return cuda_warp


class Sm80_RmemMatrixA(Sm80_BasicRmemMatrix):
    """Matrix tile for sm_80+ warp MMA A operand"""

    expected_tile_shapes = [(16, 4), (16, 8)]


class Sm80_RmemMatrixB(Sm80_BasicRmemMatrix):
    """Matrix tile for sm_80+ warp MMA B operand"""

    expected_tile_shapes = [(4, 8), (8, 8)]


class Sm80_RmemMatrixD(Sm80_BasicRmemMatrix):
    """Matrix tile for sm_80+ warp MMA accumulator (C, D) operands"""

    expected_tile_shapes = [(16, 8)]


__all__ += ["Sm80_RmemMatrixA", "Sm80_RmemMatrixB", "Sm80_RmemMatrixD"]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Instructions for sm_80 MMA
# Unlike later tensor cores, these are NOT async instructions.
# In exo terminology, these operate with actor_kind=cuda_classic


class mma_instr_impl:
    def instance_common(self):
        for v in self.access_info.values():
            v.actor_signature = sig_cuda_classic
        self.actor_kind = cuda_classic
        self.coll_unit = cuda_warp
        self.cu_includes = ["cuda/std/array"]
        self.cu_utils = [Sm80_mma_load_store_util]


@instr
class Sm80_mma_load_a_tf32(mma_instr_impl):
    def behavior(
        K: size,
        rmem: [f32][16, K] @ Sm80_RmemMatrixA,
        smem: [f32][16, K] @ CudaSmemLinear,
    ):
        for m in seq(0, 16):
            for k in seq(0, K):
                rmem[m, k] = smem[m, k]

    def instance(self, K):
        self.instance_common()
        if K != 4 and K != 8:
            raise ValueError("Require K=4 or K=8")
        self.instr_format = [
            (
                "exo_CudaUtil::Sm80_mma_load_a_k"
                + str(K)
                + "({rmem_data}, &{smem_data}, {smem_layout});"
            )
        ]


__all__.append("Sm80_mma_load_a_tf32")


@instr
class Sm80_mma_load_b_tf32(mma_instr_impl):
    def behavior(
        K: size,
        rmem: [f32][K, 8] @ Sm80_RmemMatrixB,
        smem: [f32][K, 8] @ CudaSmemLinear,
    ):
        for k in seq(0, K):
            for n in seq(0, 8):
                rmem[k, n] = smem[k, n]

    def instance(self, K):
        self.instance_common()
        if K != 4 and K != 8:
            raise ValueError("Require K=4 or K=8")
        self.instr_format = [
            (
                "exo_CudaUtil::Sm80_mma_load_b_k"
                + str(K)
                + "({rmem_data}, &{smem_data}, {smem_layout});"
            )
        ]


__all__.append("Sm80_mma_load_b_tf32")


@instr
class Sm80_mma_tf32(mma_instr_impl):
    def behavior(
        K: size,
        D: [f32][16, 8] @ Sm80_RmemMatrixD,
        A: [f32][16, K] @ Sm80_RmemMatrixA,
        B: [f32][K, 8] @ Sm80_RmemMatrixB,
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                for k in seq(0, K):
                    D[m, n] += A[m, k] * B[k, n]

    def instance(self, K):
        self.instance_common()
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


__all__.append("Sm80_mma_tf32")


@instr
class Sm80_mma_store_d_tf32(mma_instr_impl):
    def behavior(
        gmem: [f32][16, 8] @ CudaDeviceVisibleLinear,
        rmem: [f32][16, 8] @ Sm80_RmemMatrixD,
    ):
        for m in seq(0, 16):
            for n in seq(0, 8):
                gmem[m, n] = rmem[m, n]

    def instance(self):
        self.instance_common()
        self.instr_format = [
            (
                "exo_CudaUtil::Sm80_mma_store_d(&{gmem_data}, {rmem_data}, {gmem_layout});"
            )
        ]


__all__.append("Sm80_mma_store_d_tf32")


@instr
class Sm80_mma_zero_d_tf32(mma_instr_impl):
    def behavior(rmem: [f32][16, 8] @ Sm80_RmemMatrixD):
        for m in seq(0, 16):
            for n in seq(0, 8):
                rmem[m, n] = 0

    def instance(self):
        self.instance_common()
        self.instr_format = ["exo_CudaUtil::Sm80_mma_zero_d({rmem_data});"]


__all__.append("Sm80_mma_zero_d_tf32")


Sm80_mma_load_store_util = r"""
EXO_CUDA_INLINE void Sm80_mma_load_a_k8(unsigned rmem[4], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &gmem[warp_lane / 4u * row_stride + warp_lane % 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
  rmem[1] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride]);
  rmem[2] = __float_as_uint(gmem_thread_baseaddr[4 * col_stride]);
  rmem[3] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride + 4 * col_stride]);
}

EXO_CUDA_INLINE void Sm80_mma_load_a_k4(unsigned rmem[2], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &gmem[warp_lane / 4u * row_stride + warp_lane % 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
  rmem[1] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride]);
}

EXO_CUDA_INLINE void Sm80_mma_load_b_k8(unsigned rmem[2], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &gmem[warp_lane % 4u * row_stride + warp_lane / 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
  rmem[1] = __float_as_uint(gmem_thread_baseaddr[4 * row_stride]);
}

EXO_CUDA_INLINE void Sm80_mma_load_b_k4(unsigned rmem[1], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &gmem[warp_lane % 4u * row_stride + warp_lane / 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
}

EXO_CUDA_INLINE void Sm80_mma_store_d(float* gmem, const unsigned rmem[4], cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  float* gmem_thread_baseaddr = &gmem[(warp_lane / 4u) * row_stride + (warp_lane % 4u) * 2u * col_stride];
  gmem_thread_baseaddr[0] = __uint_as_float(rmem[0]);
  gmem_thread_baseaddr[col_stride] = __uint_as_float(rmem[1]);
  gmem_thread_baseaddr[8 * row_stride] = __uint_as_float(rmem[2]);
  gmem_thread_baseaddr[8 * row_stride + col_stride] = __uint_as_float(rmem[3]);
}

EXO_CUDA_INLINE void Sm80_mma_zero_d(unsigned rmem[4])
{
  rmem[0] = 0;
  rmem[1] = 0;
  rmem[2] = 0;
  rmem[3] = 0;
}
"""
