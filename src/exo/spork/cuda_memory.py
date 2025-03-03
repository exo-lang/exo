from ..core.memory import Memory, MemGenError


class CudaBasicDeviceVisible(Memory):
    pass


class CudaBasicSmem(Memory):
    pass


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


class CudaSmemLinear(CudaDeviceVisibleLinear, CudaBasicSmem):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # Should check shape is constexpr
        # TODO Shared memory should use extern allocation.
        if len(shape) == 0:
            return f"__shared__ {prim_type} {new_name};"
        return f"__shared__ {prim_type} {new_name}[{' * '.join(shape)}];"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""


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


class Sm70_BasicRmemMatrix(CudaBasicDeviceVisible):
    @classmethod
    def matrix_alloc_impl(cls, new_name, prim_type, shape, srcinfo, regcount):
        if len(shape) < 2:
            raise MemGenError("Require at least 2D tile for Sm70 MMA tile")
        array_shape = shape[:-2]
        tile_shape = shape[-2:]
        assert prim_type == "float"  # TODO
        if len(array_shape) == 0:
            return f"unsigned {new_name}[{regcount}]"
        return f"unsigned {new_name}[' * '.join(array_shape)][{regcount}]"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""
