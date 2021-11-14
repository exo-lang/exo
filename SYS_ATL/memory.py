# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Helper Functions

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory Objects

"""
#   A memory consists of
#     - a global C-code block
#     - C-code macros (as Python functions) to:
#           * allocate from the memory
#           * free an allocation
#           * read from the memory (optional)
#           * write to the memory (optional)
#           * reduce to the memory (optional)
"""
from abc import ABC, abstractmethod

"""
--- Alloc specifications ---
    - new_name is a string with the variable name to be allocated.
    - prim_type is a string with the c-type of elements of the new
      buffer: e.g. 'float', 'double', 'bfloat16', 'int32', etc.
    - shape is a python tuple containing positive integers and/or
      strings.  The strings represent size-variable names, which
      will be in-scope for the C-code.  Size-variables have type 'int'

    NOTE: If `shape` is a length-0 tuple, then `new_name` should have
          C-type `prim_type` and be stack-allocated if appropriate,
          whereas if `shape` is not a length-0 tuple,
          then `new_name` should have C-type `prim_type *`.

    NOTE Further: if a given memory can only support pointers to it,
                  then attempting to allocate a variable with shape ()
                  should trigger an error rather than do something weird.
                  The user is then responsible for converting the code
                  to use a buffer of type prim_type[1], which is
                  equivalent to a scalar, but will be processed as
                  a pointer-indirected buffer at the C-level.

    Memory Ordering: The memory may choose an ordering of the buffer, but
                     should prefer a "row-major" layout in which the last
                     dimension of shape is iterated most frequently
                     and the first dimension of shape is iterated least
                     frequently.

    Error Handling:
        If the memory is unable to successfully generate an
        allocation, free, read, write, or reduction, then the
        macro invoked should raise a MemGenError with a useful
        message.  This error will be reported to the user.
"""


class MemGenError(Exception):
    pass


class Memory(ABC):
    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    def global_(cls):
        """
        C code
        """
        return ''

    @classmethod
    @abstractmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        """
        python gemmini_extended_compute_preloaded
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        raise NotImplementedError()

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        offset = ' + '.join(f'({i}) * ({s})' for i, s in zip(indices, strides))
        if basetyp.is_win():
            baseptr = f'{baseptr}.data'
        return f'{baseptr}[{offset}]'

    @classmethod
    def can_read(cls):
        raise False

    @classmethod
    def write(cls, s, lhs, rhs):
        raise MemGenError(f"{s.srcinfo}: cannot write to buffer "
                          f"'{s.name}' in memory '{cls.name()}'")

    @classmethod
    def reduce(cls, s, lhs, rhs):
        raise MemGenError(f"{s.srcinfo}: cannot reduce to buffer "
                          f"'{s.name}' in memory '{cls.name()}'")


# ----------- DRAM on LINUX ----------------

class DRAM(Memory):
    @classmethod
    def global_(cls):
        return (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
        )

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"

        return (f"{prim_type} *{new_name} = "
                f"malloc({' * '.join(shape)} * sizeof(*{new_name}));")

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""
        return f"free({new_name});"

    @classmethod
    def can_read(cls):
        return True

    @classmethod
    def write(cls, s, lhs, rhs):
        return f'{lhs} = {rhs};'

    @classmethod
    def reduce(cls, s, lhs, rhs):
        return f"{lhs} += {rhs};"
