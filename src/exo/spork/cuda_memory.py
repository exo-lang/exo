from ..core.memory import Memory, MemGenError


class CudaBasicDeviceVisible(Memory):
    pass


class CudaBasicSmem(CudaBasicDeviceVisible):
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


class Sm80_RmemMatrixA(Sm80_BasicRmemMatrix):
    tile_shape = (16, 8)


class Sm80_RmemMatrixB(Sm80_BasicRmemMatrix):
    tile_shape = (8, 8)


class Sm80_RmemMatrixD(Sm80_BasicRmemMatrix):
    tile_shape = (16, 8)
