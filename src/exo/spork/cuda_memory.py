from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Type
from math import prod

from ..core.LoopIR import scalar_bits
from ..core.prelude import SrcInfo, ScalarInfo
from ..core.memory import (
    ScalarInfo,
    Memory,
    MemGenError,
    DRAM,
    BarrierMechanism,
    BarrierMechanismTraits,
    MemIncludeC,
    MemGlobalC,
    FreePoolTag,
    cuda_smem_free_pool_tag,
    full_scope_free_pool_tag,
    WindowFeatures,
    window_indexer,
    WindowIndexer,
    WindowIndexerResult,
    UtilInjector,
)
from . import timelines
from .coll_algebra import (
    CollUnit,
    cuda_thread,
    cuda_warp,
    cuda_warpgroup,
    cuda_cta_in_cluster,
)


class CudaBasicDeviceVisible(Memory):
    """All Memory types allocatable in CUDA device code must inherit from this.
    The LoopIR compiler expects this subclassing.

    Converse is not true -- this class represents only that the
    memory is device visible, not allocable. Subclasses implement:

    * device_permission, using one of the _impl functions
    * qual_tl_dict (usually timelines.cuda_ram_qual_tl_dict
      or timelines.cuda_rmem_qual_tl_dict)
    * native_unit, if allocable on the CUDA device.

    NB SpecialWindow, BarrierMechanism are not Memory.

    """

    @classmethod
    def sync_exempt(cls) -> bool:
        return False

    @classmethod
    @abstractmethod
    def device_permission(cls, device, instr_tl):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def native_unit(cls) -> CollUnit:
        raise NotImplementedError()

    @classmethod
    def device_allocated_impl(cls, device, instr_tl):
        """Only allocated and used on the CUDA device"""
        if device == timelines.cuda_basic_device:
            return "rwc"
        else:
            return ""

    @classmethod
    def host_allocated_impl(cls, device, instr_tl, pinned):
        """Allocated on the CPU and accessed on the CUDA device"""
        if instr_tl == timelines.cpu_cuda_stream_instr:
            return "rwc"
        elif device == timelines.cpu_basic_device:
            return "rwc" if pinned else "c"
        elif device == timelines.cuda_basic_device:
            return "rw"
        else:
            return ""

    @classmethod
    def grid_constant_impl(cls, device, instr_tl):
        if device == timelines.cpu_basic_device:
            return "rwc"
        elif device == timelines.cuda_basic_device:
            return "r"
        else:
            return ""


@dataclass(slots=True)
class SmemConfig:
    """Subclasses of CudaBasicSmem (CUDA shared memory) must not implement
    alloc and free directly. Instead, return SmemConfig in smem_config()
    and the compiler will generate the alloc/free for you.

    reftype: C++ REFERENCE type for the SMEM allocation.
        e.g. "float (&) [128]"

    alignment: minimum byte alignment (power of 2)
        If the allocation size is divisible by a power of 2
        up to 128, then the implementation implicitly aligns
        at least to that power-of-2 (opportunistic alignment)
    """

    reftype: str
    alignment: int = 1


SmemConfig.opportunistic_alignment = 128


@dataclass(slots=True)
class SmemConfigInputs:
    scalar_info: ScalarInfo
    const_shape: List[int]  # Tensor shape as list of ints
    srcinfo: SrcInfo  # Include this in error messages
    mem: Type[Memory]

    def make_reftype(self, ctype=None, shape=None):
        """Helper for initializing SmemConfig.reftype

        By default we generate either a scalar reference, or a reference
        to an array of size = product of shape dimensions."""
        ctype = ctype or self.scalar_info.ctype
        if shape is None and self.const_shape:
            shape = [prod(self.const_shape)]
        if not shape:
            return f"{ctype}&"
        else:
            return f"{ctype} (&) [{']['.join(str(c) for c in shape)}]"

    def ctype(self):
        return self.scalar_info.ctype

    def element_bits(self):
        return self.scalar_info.bits


class CudaBasicSmem(CudaBasicDeviceVisible):
    """Mandatory base class for all SMEM-resident memory types, which require
    compiler support to be lowered correctly. alloc/free are not implemented,
    instead implement smem_config().

    All allocations can only be lowered if their shape is a constant."""

    @classmethod
    def device_permission(cls, device, instr_tl):
        return cls.device_allocated_impl(device, instr_tl)

    @classmethod
    def native_unit(cls):
        return cuda_cta_in_cluster

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        """Use smem_config instead. cuda_backend.py will handle generating
        the allocation for you.

        If you must do your own checks, you may implement alloc(...), but
        it must return an empty string (which will be ignored by the compiler).

        """
        return ""

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def free_pool_tag(cls):
        return cuda_smem_free_pool_tag

    @classmethod
    def is_cuda_smem(cls):
        return True

    @classmethod
    @abstractmethod
    def smem_config(cls, inputs: SmemConfigInputs) -> SmemConfig:
        """Substitute for alloc/free. Return SmemConfig."""
        raise NotImplementedError()

    qual_tl_dict = timelines.cuda_ram_qual_tl_dict


class CudaDeviceVisibleAtomicity16B(CudaBasicDeviceVisible):
    """Any memory in swizzled C array order visible to the CUDA device,
    where the swizzle pattern keeps aligned groups of 16 bytes in the
    unswizzled layout contiguous in the swizzled layout."""

    @classmethod
    def can_read(cls):
        return True

    @classmethod
    def write(cls, s, lhs, rhs):
        return f"{lhs} = {rhs};"

    @classmethod
    def reduce(cls, s, lhs, rhs):
        return f"{lhs} += {rhs};"


class CudaDeviceVisibleLinear(CudaDeviceVisibleAtomicity16B):
    """Any memory in C array order visible to CUDA device"""

    pass


# TODO grid constants require special compiler support. Consider additional
# abstraction if we support other similar API concepts, e.g. Vulkan push constants.
class CudaGridConstant(CudaDeviceVisibleLinear, DRAM):
    """CUDA Grid constant; usable as both cuda device memory and CPU DRAM.

    Scalar or fixed-size array allocated and writeable on the CPU;
    copied to the CUDA device as a parameter to the kernel launch.
    Cannot be modified on the device.

    """

    @classmethod
    def sync_exempt(cls) -> bool:
        return True

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # Allocated "on the stack"
        for extent in shape:
            try:
                int(extent)
            except ValueError as e:
                raise MemGenError(
                    f"CudaGridConstant requires constant shapes. Saw: {shape}"
                ) from e

        if len(shape) == 0:
            return f"{prim_type} {new_name};"
        else:
            return f'{prim_type} {new_name}[{" * ".join(shape)}];'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def device_permission(cls, device, instr_tl):
        return cls.grid_constant_impl(device, instr_tl)

    qual_tl_dict = timelines.cuda_ram_qual_tl_dict


gmem_code = """
#ifndef exo_cudaMallocAsync
#ifndef __cplusplus
static
#endif
inline void* exo_cudaMallocAsync_default(size_t size, cudaStream_t exo_cudaStream,
                                         const char* file __attribute__((unused)),
                                         int line __attribute__((unused)) )
{
    void* out;
    cudaMallocAsync(&out, size, exo_cudaStream);
    if (exo_excut_log_file_enabled()) {
        exo_excut_begin_log_action("cudaMallocAsync");
        exo_excut_log_ptr_arg(out);
        exo_excut_log_ptr_arg((void*)(size));
        exo_excut_log_ptr_arg(exo_cudaStream);
        exo_excut_end_log_action("cpu", 0, 0, file, line);
    }
    return out;
}
#define exo_cudaMallocAsync(size, stream) exo_cudaMallocAsync_default(size, stream, __FILE__, __LINE__)
#endif

#ifndef exo_cudaFreeAsync
#ifndef __cplusplus
static
#endif
inline void exo_cudaFreeAsync_default(void* ptr, cudaStream_t exo_cudaStream,
                                      const char* file __attribute__((unused)),
                                      int line __attribute__((unused)) )
{
    cudaFreeAsync(ptr, exo_cudaStream);
    if (exo_excut_log_file_enabled()) {
        exo_excut_begin_log_action("cudaFreeAsync");
        exo_excut_log_ptr_arg(ptr);
        exo_excut_log_ptr_arg(exo_cudaStream);
        exo_excut_end_log_action("cpu", 0, 0, file, line);
    }
}
#define exo_cudaFreeAsync(ptr, stream) exo_cudaFreeAsync_default(ptr, stream, __FILE__, __LINE__)
#endif
"""


class CudaGmemAtomicity16B(CudaDeviceVisibleAtomicity16B):
    """Any shared memory with CudaDeviceVisibleAtomicity16B requirements met.

    Abstract base class, not allocable.

    """

    pass


class CudaGmemLinear(CudaDeviceVisibleLinear, CudaGmemAtomicity16B):
    """Global memory in C array order

    Consider CudaDeviceVisibleLinear when you do not truly need this
    to be global memory.

    """

    @classmethod
    def global_(cls):
        return MemGlobalC("CudaGmemLinear", gmem_code)

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            raise MemGenError("Cannot allocate scalar CudaGmemLinear")
        return (
            f"{prim_type} *{new_name} = "
            f"({prim_type}*) exo_cudaMallocAsync({' * '.join(shape)} * sizeof(*{new_name}), exo_cudaStream);"
        )

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            raise MemGenError("Cannot allocate scalar CudaGmemLinear")
        return f"exo_cudaFreeAsync({new_name}, exo_cudaStream);"

    @classmethod
    def device_permission(cls, device, instr_tl):
        return cls.host_allocated_impl(device, instr_tl, pinned=False)

    qual_tl_dict = timelines.cuda_ram_qual_tl_dict


class CudaSmemAtomicity16B(CudaDeviceVisibleAtomicity16B, CudaBasicSmem):
    """Any shared memory with CudaDeviceVisibleAtomicity16B requirements met.

    Abstract base class, not allocable.

    """

    pass


class CudaSmemLinear(CudaDeviceVisibleLinear, CudaSmemAtomicity16B):
    """Shared memory in C array order"""

    @classmethod
    def smem_config(cls, inputs: SmemConfigInputs) -> SmemConfig:
        return SmemConfig(inputs.make_reftype())

    qual_tl_dict = timelines.cuda_ram_qual_tl_dict


class CudaRmem(CudaDeviceVisibleLinear):
    """Per-thread registers"""

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            return f"{prim_type} {new_name};"

        const_shape = cls.as_const_shape(new_name, shape, srcinfo)

        return f'{prim_type} {new_name}[{" * ".join(shape)}];'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def device_permission(cls, device, instr_tl):
        return cls.device_allocated_impl(device, instr_tl)

    @classmethod
    def native_unit(cls):
        return cuda_thread

    qual_tl_dict = timelines.cuda_rmem_qual_tl_dict


global_CudaRmemPacked32 = MemGlobalC(
    "exo_CudaRmemPacked32",
    """
#ifdef __CUDACC__

template <typename PtxType, typename Scalar, typename PackedStruct>
struct exo_CudaRmemPacked32
{
    static_assert(sizeof(PtxType) == sizeof(PackedStruct));
    PtxType ptx_data;

    template <typename Index>
    __device__ auto operator[] (Index i) const -> Scalar
    {
        if constexpr (sizeof(Scalar) == 4)
            return ptx_data;
        else if (i == 0)
            return reinterpret_cast<const PackedStruct*>(&ptx_data)->x;
        else
            return reinterpret_cast<const PackedStruct*>(&ptx_data)->y;
    }
};

using exo_CudaRmemPacked32_f32 = exo_CudaRmemPacked32<float, float, float>;
using exo_CudaRmemPacked32_i32 = exo_CudaRmemPacked32<int32_t, int32_t, int32_t>;
using exo_CudaRmemPacked32_f16 = exo_CudaRmemPacked32<int32_t, __half, __half2>;
using exo_CudaRmemPacked32_bf16 = exo_CudaRmemPacked32<int32_t, __nv_bfloat16, __nv_bfloat162>;

#endif
""",
)


class CudaRmemPacked32_Indexer(WindowIndexer):
    def index(self, utils: UtilInjector, features: WindowFeatures, *, ptx_data=False):
        expr = features.get_dataptr()
        # All non-packed indices resolve to multidimensional array indexing.
        for i in range(features.n_array_dims()):
            expr = expr[features.get_array_offset(i)]
        # If the caller requested the raw ptx_data, give the 32-bit word directly.
        if ptx_data:
            expr = expr.ptx_data
        # Otherwise, we defer to the overloaded operator[]
        else:
            expr = expr[features.get_packed_offset(0)]
        return self.pack_result(expr, False)


@window_indexer(CudaRmemPacked32_Indexer)
class CudaRmemPacked32(CudaBasicDeviceVisible):
    """Per-thread registers, with scalar data packed as 32 bit words.

    The rightmost dimension must be 4 / sizeof(ElementType).
    This includes the degenerate case of float or int32_t
    (where only one scalar is "packed" per word) for consistency.

    Currently scalar code (non-instr) may only read, not write.

    """

    allowed_types = {"f32", "i32", "f16", "bf16"}

    @classmethod
    def global_(cls):
        return global_CudaRmemPacked32

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        assert shape
        const_shape = cls.as_const_shape(new_name, shape, srcinfo)
        scalar_info = ScalarInfo(prim_type)
        gpuir_name = scalar_info.shorthand
        if gpuir_name not in cls.allowed_types:
            raise TypeError(f"CudaRmemPacked32 doesn't support {gpuir_name}")
        # Don't generate array dimensions for final (bit-packed) dimension.
        dims = [f"[{n}]" for n in const_shape[:-1]]
        return f'exo_CudaRmemPacked32_{gpuir_name} {new_name}{"".join(dims)};'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def can_read(cls):
        return True

    @classmethod
    def packed_tensor_shape(cls, scalar_info: ScalarInfo):
        return (32 // scalar_info.bits,)

    @classmethod
    def device_permission(cls, device, instr_tl):
        return cls.device_allocated_impl(device, instr_tl)

    @classmethod
    def native_unit(cls):
        return cuda_thread

    qual_tl_dict = timelines.cuda_rmem_qual_tl_dict


# TODO implement this.
class CudaEvent(BarrierMechanism):
    @classmethod
    def traits(cls) -> BarrierMechanismTraits:
        return BarrierMechanismTraits(
            requires_guarding=True, requires_arrive_first=True
        )

    @classmethod
    def sync_exempt(cls) -> bool:
        return True


class CudaDeviceBarrier(BarrierMechanism):
    @classmethod
    def device_permission(cls, device, instr_tl):
        return "rwc" if device == timelines.cuda_basic_device else ""

    qual_tl_dict = timelines.cuda_ram_qual_tl_dict


class CudaMbarrier(CudaDeviceBarrier):
    @classmethod
    def traits(cls) -> BarrierMechanismTraits:
        return BarrierMechanismTraits(
            negative_await_N=True,
            uniform_await_N=True,
            different_arrive_await_threads=True,
            requires_guarding=True,
            requires_arrive_first=False,
            supports_guards=True,
            supports_arrive_multicast=True,
        )

    @classmethod
    def sync_exempt(cls) -> bool:
        return False

    @classmethod
    def free_pool_tag(cls):
        return full_scope_free_pool_tag

    @classmethod
    def is_cuda_smem(cls):
        return True

    qual_tl_dict = timelines.cuda_ram_qual_tl_dict

    # Bespoke functions (not really externalizable) for mbarrier, which
    # is the only barrier type subject to synchronization checking.
    # We give the qual_tl used to model the access associated with
    # an arrive/await with the given Sync_tl parameter.
    @classmethod
    def arrive_qual_tl(cls, L1: timelines.Sync_tl):
        if L1.get_full_timeline_set_bits() & timelines.Sm80_cp_async_qual.as_bit():
            return timelines.Sm80_cp_async_qual
        return timelines.cuda_in_order_ram_qual

    @classmethod
    def await_qual_tl(cls, L2: timelines.Sync_tl):
        return timelines.cuda_in_order_ram_qual


class CudaCommitGroup(CudaDeviceBarrier):
    @classmethod
    def traits(cls) -> BarrierMechanismTraits:
        return BarrierMechanismTraits(non_negative_await_N=True)

    @classmethod
    def sync_exempt(cls) -> bool:
        return True


class CudaClusterSync(CudaDeviceBarrier):
    @classmethod
    def traits(cls) -> BarrierMechanismTraits:
        return BarrierMechanismTraits(
            requires_guarding=True, requires_arrive_first=True
        )

    @classmethod
    def sync_exempt(cls) -> bool:
        return True
