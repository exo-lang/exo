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
    def name(self):
        return type(self).__name__

    @property
    @abstractmethod
    def global_(self):
        """
        C code
        """
        raise NotImplementedError()

    @abstractmethod
    def alloc(self, new_name, prim_type, shape, srcinfo):
        """
        python gemmini_extended_compute_preloaded
        """
        raise NotImplementedError()

    @abstractmethod
    def free(self, new_name, prim_type, shape, srcinfo):
        raise NotImplementedError()

    def window(self, basetyp, baseptr, idx_expr, indices, strides, srcinfo):
        if basetyp.is_win():
            baseptr = f'{baseptr}.data'
        return f'{baseptr} + {idx_expr}'

    @property
    @abstractmethod
    def can_read(self):
        raise False

    def write(self, s, lhs, rhs):
        raise MemGenError(f"{s.srcinfo}: cannot write to buffer "
                          f"'{s.name}' in memory '{self.name()}'")

    def reduce(self, s, lhs, rhs):
        raise MemGenError(f"{s.srcinfo}: cannot reduce to buffer "
                          f"'{s.name}' in memory '{self.name()}'")


# ----------- DRAM on LINUX ----------------

class DRAMBase(Memory):
    @property
    def global_(self):
        return "#include <stdio.h>\n" + "#include <stdlib.h>\n"

    def alloc(self, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"
        else:
            size_str = shape[0]
            for s in shape[1:]:
                size_str = f"{s} * {size_str}"
            return (f"{prim_type} *{new_name} = "
                    f"({prim_type}*) malloc ({size_str} * sizeof({prim_type}));")

    def free(self, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""
        else:
            return f"free({new_name});"

    @property
    def can_read(self):
        return True

    def write(self, s, lhs, rhs):
        return f'{lhs} = {rhs};'

    def reduce(self, s, lhs, rhs):
        return f"{lhs} += {rhs};"


DRAM = DRAMBase()
