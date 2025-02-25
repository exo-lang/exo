from ..core.memory import Memory


class CudaSmemLinear(Memory):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # Should check shape is constexpr
        if len(shape) == 0:
            return f"__shared__ {prim_type} {new_name};"
        return f"__shared__ {prim_type} {new_name}[{' * '.join(shape)}];"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def can_read(cls):
        return True

    @classmethod
    def write(cls, s, lhs, rhs):
        return f"{lhs} = {rhs};"

    @classmethod
    def reduce(cls, s, lhs, rhs):
        return f"{lhs} += {rhs};"
