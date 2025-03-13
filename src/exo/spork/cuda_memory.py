from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Type

from ..core.LoopIR import ctype_bits
from ..core.prelude import SrcInfo
from ..core.memory import Memory, MemGenError
from . import actor_kinds
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
    implement actor_kind_permission in terms of one of the impl
    functions based on the correct behavior.
    """

    @classmethod
    @abstractmethod
    def actor_kind_permission(cls, actor_kind, is_instr):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def native_unit(cls) -> CollUnit:
        raise NotImplementedError()

    @classmethod
    def device_allocated_impl(cls, actor_kind, is_instr):
        """Only allocated and used on the CUDA device"""
        if actor_kind == actor_kinds.cuda_classic:
            return "rwa"
        elif actor_kind in actor_kinds.cuda_async_actor_kinds:
            return "rwa" if is_instr else "a"
        else:
            return ""

    @classmethod
    def host_allocated_impl(cls, actor_kind, is_instr, pinned):
        """Allocated on the CPU and accessed on the CUDA device"""
        if actor_kind == actor_kinds.cpu:
            return "rwa" if pinned else "a"
        elif actor_kind == actor_kinds.cuda_classic:
            return "rw"
        elif actor_kind in actor_kinds.cuda_async_actor_kinds:
            return "rw" if is_instr else ""
        else:
            return ""

    @classmethod
    def grid_constant_impl(cls, actor_kind, is_instr, pinned):
        if actor_kind == actor_kinds.cpu:
            return "rwa"
        elif actor_kind == actor_kinds.cuda_classic:
            return "r"
        elif actor_kind in actor_kinds.cuda_async_actor_kinds:
            return "r" if is_instr else ""
        else:
            return ""


@dataclass
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


@dataclass
class SmemConfigInputs:
    ctype: str  # C type name e.g. "float", "int32_t"
    const_shape: List[int]  # Tensor shape as list of ints
    srcinfo: SrcInfo  # Include this in error messages
    mem: Type[Memory]

    def make_reftype(self, ctype=None, shape=None):
        """Helper for initializing SmemConfig.reftype

        By default we generate either a scalar reference, or a reference
        to an array of size = product of shape dimensions."""
        ctype = ctype or self.ctype
        if shape is None and self.const_shape:
            prod = 1
            for n in self.const_shape:
                prod *= n
            shape = [prod]
        if not shape:
            return f"{ctype}&"
        else:
            return f"{ctype} (&) [{']['.join(str(c) for c in shape)}]"

    def element_bits(self):
        return ctype_bits(self.ctype)

    def require_shape_divisibility(self, divisors):
        shape = self.const_shape
        if len(shape) < len(divisors):
            raise ValueError(
                f"{self.srcinfo}: {self.mem.name()} tensor shape "
                f"must be at least {len(divisors)}-dimensional. "
                "Got {shape} (after removing distributed dimensions)"
            )
        for i in range(1, len(divisors) + 1):
            if shape[-i] % divisors[-i] != 0:
                raise ValueError(
                    f"{self.srcinfo}: {self.mem.name()} tensor shape "
                    f"must be a multiple of {divisors}. "
                    "Got {shape} (after removing distributed dimensions)"
                )


class CudaBasicSmem(CudaBasicDeviceVisible):
    """Mandatory base class for all SMEM-resident memory types, which require
    compiler support to be lowered correctly. alloc/free are not implemented,
    instead implement smem_config().

    All allocations can only be lowered if their shape is a constant."""

    @classmethod
    def actor_kind_permission(cls, actor_kind, is_instr):
        return cls.device_allocated_impl(actor_kind, is_instr)

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


class CudaDeviceVisibleLinear(CudaBasicDeviceVisible):
    @classmethod
    def can_read(cls):
        return True

    @classmethod
    def write(cls, s, lhs, rhs):
        return f"{lhs} = {rhs};"

    @classmethod
    def reduce(cls, s, lhs, rhs):
        return f"{lhs} += {rhs};"


class CudaGmemLinear(CudaDeviceVisibleLinear):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        raise MemGenError("TODO implement CudaGmemLinear.alloc")

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        raise MemGenError("TODO implement CudaGmemLinear.free")

    @classmethod
    def actor_kind_permission(cls, actor_kind, is_instr):
        return cls.host_allocated_impl(actor_kind, is_instr, pinned=False)


class CudaSmemLinear(CudaDeviceVisibleLinear, CudaBasicSmem):
    @classmethod
    def smem_config(cls, inputs: SmemConfigInputs) -> SmemConfig:
        return SmemConfig(inputs.make_reftype())


class CudaRmem(CudaDeviceVisibleLinear):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            return f"{prim_type} {new_name};"

        for extent in shape:
            try:
                int(extent)
            except ValueError as e:
                raise MemGenError(
                    f"CudaRmem requires constant shapes. Saw: {extent}"
                ) from e

        return f'{prim_type} {new_name}[{" * ".join(shape)}];'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def actor_kind_permission(cls, actor_kind, is_instr):
        return cls.device_allocated_impl(actor_kind, is_instr)

    @classmethod
    def native_unit(cls):
        return cuda_thread


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
            raise MemGenError("Require at least 2D tile for Sm80 MMA tile")
        array_shape = shape[:-2]
        tile_shape = shape[-2:]
        assert prim_type == "float"  # TODO
        regcount = cls.tile_shape[0] * cls.tile_shape[1] // 32

        assert len(cls.tile_shape) == 2
        for i, c in enumerate(tile_shape):
            try:
                if int(c) != int(cls.tile_shape[i]):
                    raise ValueError("WRONG")
            except Exception:
                raise MemGenError(
                    f"Expected last 2 dimensions of size "
                    f"{cls.tile_shape}, not {tile_shape}"
                )

        # Last array dimension corresponds to uint32_t-encoded matrix tile
        # Leading dimensions correspond to the Exo user's array dimensions.
        leading = "".join(f"[{c}]" for c in array_shape)
        return f"unsigned {new_name}{leading}[{regcount}];"

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        if basetyp.is_win():
            return f"*{baseptr}.data"
        assert len(strides) >= 2
        assert strides[-2] == str(cls.tile_shape[1])
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
    tile_shape = (16, 8)


class Sm80_RmemMatrixB(Sm80_BasicRmemMatrix):
    tile_shape = (8, 8)


class Sm80_RmemMatrixD(Sm80_BasicRmemMatrix):
    tile_shape = (16, 8)
