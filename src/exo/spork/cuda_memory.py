from ..core.memory import Memory

# Unchecked and not really working placeholders for now.
# These generate the correct code in specific circumstances but otherwise will break.


class CudaRegisters(Memory):
    # XXX this only works due to a compiler backdoor for now.
    # We strip all indexing information.
    # This should be replaced by the "distributed memory" idea later.
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        return f"{prim_type} {new_name};"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def can_read(cls):
        return False  # Handled with backdoor

    @classmethod
    def write(cls, s, lhs, rhs):
        return f"{cls.strip_indexing(lhs)} = {rhs};"

    @classmethod
    def reduce(cls, s, lhs, rhs):
        return f"{cls.strip_indexing(lhs)} += {rhs};"

    @staticmethod
    def strip_indexing(lhs):
        # Ridiculous hack
        return lhs.split("[")[0]


class CudaShared(Memory):
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
