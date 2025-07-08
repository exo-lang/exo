from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Type
from math import prod

from ..core.LoopIR import scalar_bits
from ..core.prelude import SrcInfo, ScalarInfo
from ..core.memory import (
    Memory,
    MemGenError,
    DRAM,
    BarrierType,
    BarrierTypeTraits,
    MemIncludeC,
    MemGlobalC,
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
    """All memory types allocatable in CUDA device code must inherit from this.
    The LoopIR compiler expects this subclassing, and expects
    the native_unit() function to be implemented if allocable on the device.

    Converse is not true -- this class represents only that the
    memory is device visible, not allocable. Subclasses should
    implement instr_tl_permission in terms of one of the impl
    functions based on the correct behavior.
    """

    @classmethod
    @abstractmethod
    def instr_tl_permission(cls, instr_tl, is_instr):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def native_unit(cls) -> CollUnit:
        raise NotImplementedError()

    @classmethod
    def device_allocated_impl(cls, instr_tl, is_instr):
        """Only allocated and used on the CUDA device"""
        if instr_tl == timelines.cuda_in_order_instr:
            return "rwc"
        elif instr_tl in timelines.cuda_async_instr_tl:
            return "rwc" if is_instr else "c"
        else:
            return ""

    @classmethod
    def host_allocated_impl(cls, instr_tl, is_instr, pinned):
        """Allocated on the CPU and accessed on the CUDA device"""
        if instr_tl == timelines.cpu_in_order_instr:
            return "rwc" if pinned else "c"
        elif instr_tl == timelines.cuda_in_order_instr:
            return "rw"
        elif instr_tl in timelines.cuda_async_instr_tl:
            return "rw" if is_instr else ""
        else:
            return ""

    @classmethod
    def grid_constant_impl(cls, instr_tl, is_instr):
        if instr_tl == timelines.cpu_in_order_instr:
            return "rwc"
        elif instr_tl == timelines.cuda_in_order_instr:
            return "r"
        elif instr_tl in timelines.cuda_async_instr_tl:
            return "r" if is_instr else ""
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
        Will be implicitly increased if the scalar type needs it.
    """

    reftype: str
    alignment: int = 1


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

    def require_shape_divisibility(self, divisors):
        """Shape divisibility check

        Require that the rightmost len(divisors)-many dimensions are
        divisible by the respective divisor values

        """
        shape = self.const_shape
        if len(shape) < len(divisors):
            raise MemGenError(
                f"{self.srcinfo}: {self.mem.name()} tensor shape "
                f"must be at least {len(divisors)}-dimensional. "
                f"Got {shape} (after removing distributed dimensions)"
            )
        for i in range(1, len(divisors) + 1):
            if shape[-i] % divisors[-i] != 0:
                raise MemGenError(
                    f"{self.srcinfo}: {self.mem.name()} tensor shape "
                    f"must be a multiple of {divisors}. "
                    f"Got {shape} (after removing distributed dimensions)"
                )

    def require_shape_tile(self, tile):
        """Shape tiling check

        Require that the rightmost len(divisors)-many dimensions are
        exactly matching the respective tile values

        """
        shape = self.const_shape
        if len(shape) < len(tile):
            raise MemGenError(
                f"{self.srcinfo}: {self.mem.name()} tensor shape "
                f"must be at least {len(tile)}-dimensional. "
                f"Got {shape} (after removing distributed dimensions)"
            )
        for i in range(1, len(tile) + 1):
            if shape[-i] != tile[-i] != 0:
                raise MemGenError(
                    f"{self.srcinfo}: {self.mem.name()} tensor shape "
                    f"rightmost dims must match {tile}. "
                    f"Got {shape} (after removing distributed dimensions)"
                )


class CudaBasicSmem(CudaBasicDeviceVisible):
    """Mandatory base class for all SMEM-resident memory types, which require
    compiler support to be lowered correctly. alloc/free are not implemented,
    instead implement smem_config().

    All allocations can only be lowered if their shape is a constant."""

    @classmethod
    def instr_tl_permission(cls, instr_tl, is_instr):
        return cls.device_allocated_impl(instr_tl, is_instr)

    @classmethod
    def native_unit(cls):
        return cuda_cta_in_cluster

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        """Use smem_config instead. cuda_backend.py will handle generating
        the allocation for you.

        If you must do your own checks, you may implement alloc(...), but
        it must return an empty string (which will be ignored by the compiler).
        Consider SmemConfigInputs.require_shape_divisibility.
        """
        return ""

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    @abstractmethod
    def smem_config(cls, inputs: SmemConfigInputs) -> SmemConfig:
        """Substitute for alloc/free. Return SmemConfig."""
        raise NotImplementedError()

    @classmethod
    def default_usage_tl(cls, instr_tl):
        return timelines.cuda_ram_usage


class CudaDeviceVisibleLinear(CudaBasicDeviceVisible):
    """Any memory in C array order visible to CUDA device"""

    @classmethod
    def can_read(cls):
        return True

    @classmethod
    def write(cls, s, lhs, rhs):
        return f"{lhs} = {rhs};"

    @classmethod
    def reduce(cls, s, lhs, rhs):
        return f"{lhs} += {rhs};"

    @classmethod
    def default_usage_tl(cls, instr_tl):
        return timelines.cuda_ram_usage


# TODO grid constants require special compiler support. Consider additional
# abstraction if we support other similar API concepts, e.g. Vulkan push constants.
class CudaGridConstant(CudaDeviceVisibleLinear, DRAM):
    """CUDA Grid constant; usable as both cuda device memory and CPU DRAM.

    Scalar or fixed-size array allocated and writeable on the CPU;
    copied to the CUDA device as a parameter to the kernel launch.
    Cannot be modified on the device.

    """

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
    def instr_tl_permission(cls, instr_tl, is_instr):
        return cls.grid_constant_impl(instr_tl, is_instr)

    @classmethod
    def default_usage_tl(cls, instr_tl):
        if instr_tl == cpu_in_order_instr:
            return timelines.cpu_usage
        else:
            return timelines.cuda_sync_rmem_usage


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
        exo_excut_log_ptr_arg((void*)(size));
        exo_excut_log_ptr_arg(out);
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
        exo_excut_end_log_action("cpu", 0, 0, file, line);
    }
}
#define exo_cudaFreeAsync(ptr, stream) exo_cudaFreeAsync_default(ptr, stream, __FILE__, __LINE__)
#endif
"""


class CudaGmemLinear(CudaDeviceVisibleLinear):
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
    def instr_tl_permission(cls, instr_tl, is_instr):
        return cls.host_allocated_impl(instr_tl, is_instr, pinned=False)

    @classmethod
    def default_usage_tl(cls, instr_tl):
        return timelines.cuda_ram_usage


class CudaSmemLinear(CudaDeviceVisibleLinear, CudaBasicSmem):
    """Shared memory in C array order"""

    @classmethod
    def smem_config(cls, inputs: SmemConfigInputs) -> SmemConfig:
        return SmemConfig(inputs.make_reftype())


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
    def instr_tl_permission(cls, instr_tl, is_instr):
        return cls.device_allocated_impl(instr_tl, is_instr)

    @classmethod
    def native_unit(cls):
        return cuda_thread

    @classmethod
    def default_usage_tl(cls, instr_tl):
        return timelines.cuda_sync_rmem_usage


class CudaEvent(BarrierType):
    @classmethod
    def traits(cls) -> BarrierTypeTraits:
        return BarrierTypeTraits(requires_pairing=True, requires_arrive_first=True)

    @classmethod
    def default_usage_tl(cls, instr_tl):
        return timelines.cpu_usage


class CudaDeviceBarrier(BarrierType):
    @classmethod
    def instr_tl_permission(cls, instr_tl, is_instr):
        if (
            instr_tl == timelines.cuda_in_order_instr
            or instr_tl in timelines.cuda_async_instr_tl
        ):
            return "rwc"
        else:
            return ""

    @classmethod
    def default_usage_tl(cls, instr_tl):
        return timelines.cuda_ram_usage


class CudaMbarrier(CudaDeviceBarrier):
    @classmethod
    def traits(cls) -> BarrierTypeTraits:
        return BarrierTypeTraits(
            negative_await_N=True,
            uniform_await_N=True,
            supports_back_array=True,
            requires_pairing=True,
            requires_arrive_first=False,
            supports_arrive_multicast=True,
        )


class CudaCommitGroup(CudaDeviceBarrier):
    @classmethod
    def traits(cls) -> BarrierTypeTraits:
        return BarrierTypeTraits(non_negative_await_N=True)


class CudaClusterSync(CudaDeviceBarrier):
    @classmethod
    def traits(cls) -> BarrierTypeTraits:
        return BarrierTypeTraits()
