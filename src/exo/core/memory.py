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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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


def generate_offset(indices, strides, vector_size=1):
    assert isinstance(vector_size, int), "generalize this if needed"
    assert vector_size >= 1

    def index_expr(i, s):
        if s == "0" or i == "0":
            return ""

        if s == "1":
            return i
        if i == "1":
            return s

        if len(s) == 1:
            return f"({i}) * {s}"
        else:
            return f"({i}) * ({s})"

    exprs = [e for i, s in zip(indices, strides) if (e := index_expr(i, s)) != ""]

    expr = " + ".join(exprs) if len(exprs) > 0 else "0"
    if vector_size != 1 and expr != "0":
        expr = f"({expr}) / {vector_size}"

    return expr


class WindowStructCtx(object):
    __slots__ = [
        "_ctype",
        "_type_shorthand",
        "_n_dims",
        "_is_const",
        "_srcinfo",
        "_struct_name",
        "_guard_macro",
    ]

    def __init__(self, ctype, type_shorthand, n_dims, is_const, srcinfo):
        """For internal use of LoopIR compiler"""
        self._ctype = ctype
        self._type_shorthand = type_shorthand
        self._n_dims = n_dims
        self._is_const = is_const
        self._srcinfo = srcinfo

        self._struct_name = None
        self._guard_macro = None

    def generate_default(self, mem_win_name, data_ctype=None):
        sname = self.struct_name(mem_win_name)
        if data_ctype is None:
            data_ctype = self._ctype
        # Spacing difference gives byte-for-byte compatibility with Exo 1.
        struct_cptr = "const " * self._is_const + data_ctype + " *"
        dataptr_ctype = "const " * self._is_const + data_ctype + "*"

        sdef = (
            f"struct {sname}{{\n"
            f"    {struct_cptr} const data;\n"
            f"    const int_fast32_t strides[{self._n_dims}];\n"
            f"}};"
        )
        return dataptr_ctype, sdef

    def struct_name(self, mem_win_name: str) -> str:
        assert isinstance(mem_win_name, str), "use str (avoid silent mistakes)"
        assert mem_win_name
        const_suffix = "c" if self._is_const else ""
        base_sname = f"exo_win_{self._n_dims}{self._type_shorthand}{const_suffix}"
        mem_suffix = "" if mem_win_name == "DRAM" else "_" + mem_win_name
        sname = base_sname + mem_suffix

        assert self._struct_name is None or self.struct_name == sname
        self._struct_name = sname
        self._guard_macro = base_sname.upper() + mem_suffix  # case-sensitive

        return sname

    def n_dims(self) -> int:
        return self._n_dims

    def is_const(self) -> bool:
        return self._is_const

    def ctype(self) -> str:
        """return C name for scalar type tensor is made of e.g. float, uint16_t"""
        return self._ctype

    def type_shorthand(self) -> str:
        """e.g. f32, u16"""
        return self._type_shorthand

    def srcinfo(self):
        """Convert to str and include in error messages"""
        return self._srcinfo


class MemWin(ABC):
    """Common base class of allocable Memory and non-allocable SpecialWindow"""

    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    def global_(cls):
        """
        C code
        """
        return ""

    @classmethod
    @abstractmethod
    def window_definition(cls, ctx: WindowStructCtx):
        """
        C code defining struct.
        Get the required parameters from the WindowStructCtx.
        Return (dataptr : str, window_struct : str)

        dataptr: C type for a raw pointer (e.g. __m256d*, float*)

        window_struct: C code defining a struct named ctx.struct_name()

        The compiler will include a header guard for you.
        """
        raise NotImplementedError()

    @classmethod
    def separate_dataptr(cls):
        """separate_dataptr: return False for the usual case.

        If True, the window is passed to functions as separate arguments
        (dataptr, window_struct) rather than a combined window struct;
        the window struct only contains layout information in this case,
        and you must define this custom layout (see window(...))
        """
        return False

    @classmethod
    def window(cls, basetyp, in_expr, indices, strides, srcinfo) -> str:
        """
        Return one of the following:

        Base case:      data : str
        Custom layout:  (dataptr : str, layout : str)

        Where dataptr and layout are both C strings used to initialize
        the window struct. (A default layout is provided in non-custom cases).
        We implicitly take dataptr = &data in the base case.

        If you wish to implement can_read/write/reduce, you should not use
        a custom layout. Furthermore, currently custom layouts don't support
        reducing the dimensionality of a window (can be changed later).

        basetyp: LoopIR.Tensor instance

        in_expr: C expression of the following type:

          basetyp.is_win() = false: dense tensor type (as generated by alloc)
            Won't occur if implementing a SpecialWindow

          basetyp.is_win() = True: window type
            str if no separate_dataptr, else (dataptr : str, layout : str)

        indices: C expressions of indices (offsets per dimension)
          e.g. [1:10, 42:46] -> ["1", "42"] (we don't provide the slice sizes)

        strides: C expressions of per-dim strides, in units of scalars. (*)
          If basetyp.is_win() and you define a custom layout, don't use this,
          (*) consider passing vector_size to generate_offset.

        srcinfo: include this when throwing an exception.
        """
        return cls.default_window(1, basetyp, in_expr, indices, strides, srcinfo)

    @classmethod
    def default_window(cls, vector_size, basetyp, in_expr, indices, strides, srcinfo):
        """Helper for simple window(...) implementations. Don't override this"""
        offset = generate_offset(indices, strides, vector_size)
        dataptr = f"{in_expr}.data" if basetyp.is_win() else in_expr
        return f"{dataptr}[{offset}]"

    @classmethod
    @abstractmethod
    def can_read(cls):
        raise NotImplementedError()

    @classmethod
    def write(cls, s, lhs, rhs):
        raise MemGenError(
            f"{s.srcinfo}: cannot write to buffer "
            f"'{s.name}' in memory '{cls.name()}'"
        )

    @classmethod
    def reduce(cls, s, lhs, rhs):
        raise MemGenError(
            f"{s.srcinfo}: cannot reduce to buffer "
            f"'{s.name}' in memory '{cls.name()}'"
        )


class Memory(MemWin):
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
    def window_definition(cls, ctx: WindowStructCtx):
        """This is not correct for non-scalar cases but we provide this
        for backwards compatibility with Exo 1 ... programs worked OK
        if they never materialized the faulty default window struct"""
        return ctx.generate_default("DRAM")


class WindowFromDenseCtx(object):
    __slots__ = [
        "_basetyp",
        "_baseptr",
        "_shape_impl",
        "_stride_impl",
        "_is_const",
        "_ctype",
        "_type_shorthand",
        "_srcinfo",
    ]

    def __init__(
        self,
        basetyp,
        baseptr,
        shape_impl,
        stride_impl,
        is_const,
        ctype,
        type_shorthand,
        srcinfo,
    ):
        """For internal use of LoopIR compiler"""
        self._basetyp = basetyp
        self._baseptr = baseptr
        self._shape_impl = shape_impl
        self._stride_impl = stride_impl
        self._is_const = is_const
        self._ctype = ctype
        self._type_shorthand = type_shorthand
        self._srcinfo = srcinfo

    def basetyp(self) -> LoopIR.Tensor:
        """return LoopIR.Tensor type of input tensor"""
        return self._basetyp

    def baseptr(self):
        """return C name of allocated tensor"""
        return self._baseptr

    def shape_strs(self) -> List[str]:
        return self._shape_impl()

    def stride_strs(self) -> List[str]:
        return self._stride_impl()

    def is_const(self) -> bool:
        return self._is_const

    def ctype(self) -> str:
        """return C name for scalar type tensor is made of e.g. float, uint16_t"""
        return self._ctype

    def type_shorthand(self) -> str:
        """e.g. f32, u16"""
        return self._type_shorthand

    def srcinfo(self):
        """Convert to str and include in error messages"""
        return self._srcinfo


class SpecialWindow(MemWin):
    @classmethod
    @abstractmethod
    def memory_type(cls) -> type:
        """Return memory type expected as input to window statement"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def window_from_dense(cls, ctx: WindowFromDenseCtx):
        """Callback for generating C code initializing a window to an entire tensor.

        You may assume the input tensor is the correct memory type and is dense
        (not is_win()).

        If separate_dataptr(), return (dataptr : str, layout : str) of
        C expressions that can initialize the two respective window variables.
        Otherwise, return a single C expression that can be used
        to initialize a struct of the window type.
        """
        raise NotImplementedError()

    # Remember to implement everything in base class MemWin as well


# ----------- DRAM on LINUX ----------------


class DRAM(Memory):
    @classmethod
    def global_(cls):
        return "#include <stdio.h>\n#include <stdlib.h>\n"

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"

        return (
            f"{prim_type} *{new_name} = "
            f"({prim_type}*) malloc({' * '.join(shape)} * sizeof(*{new_name}));"
        )

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
        return f"{lhs} = {rhs};"

    @classmethod
    def reduce(cls, s, lhs, rhs):
        return f"{lhs} += {rhs};"


# ----------- Static Memory "Register Allocation"  ----------------


class StaticMemory(Memory):
    """
    A free list implementation of register allocation for static
    memory.
    """

    is_chunk_allocated = []

    @classmethod
    def init_state(cls, size):
        cls.is_chunk_allocated = [False] * size

    @classmethod
    def find_free_chunk(cls):
        try:
            idx = cls.is_chunk_allocated.index(False)
        except ValueError as e:
            raise MemGenError(
                f"Cannot allocate more than {len(cls.is_chunk_allocated)} chunks at a time."
            ) from e
        return idx

    @classmethod
    def mark(cls, idx):
        cls.is_chunk_allocated[idx] = True

    @classmethod
    def unmark(cls, idx):
        cls.is_chunk_allocated[idx] = False
