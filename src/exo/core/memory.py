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
from dataclasses import dataclass
from typing import Optional, Set, Type
from ..spork.timelines import cpu_in_order_instr, cpu_usage, Instr_tl, Usage_tl

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


_memwin_template_names = {}
_memwin_template_cache = {}


class MemGenError(Exception):
    pass


@dataclass
class MemIncludeC:
    """Give back MemIncludeC(...) in global_() to require a header file"""

    header_name: str
    _used_by: Set[Type[MemWin]]

    def __init__(self, header_name: str):
        self.header_name = header_name
        self._used_by = set()

    def used_by_strs(self) -> List[str]:
        return sorted(u.name() for u in self._used_by)


@dataclass
class MemGlobalC:
    """Give back MemGlobalC(...) in global_() to require some source code.

    The MemGlobalC contains a C identifier `name`, and C code `code`.
    The code is injected into either the header file, or the C/CUDA
    source file, wrapped with a `name`-based header guard.
    The code goes into the header file only if it's required by some
    MemWin type that's part of a public proc's interface.

    For two MemGlobalC instances a and b, Exo requires that
    (a.name == b.name) iff (a.code == b.code).

    Any code in the self.depends_on list will get injected before self.code.

    """

    name: str
    code: str
    depends_on: Tuple[MemGlobalC | MemIncludeC]
    _used_by: Set[Type[MemWin]]

    def __init__(
        self, name: str, code: str, depends_on: List[MemGlobalC | MemIncludeC] = None
    ):
        assert all(c == "_" or c.isalnum() for c in name)
        self.name = name
        self.code = code
        if depends_on:
            assert all(isinstance(c, (MemIncludeC, MemGlobalC)) for c in depends_on)
            self.depends_on = tuple(depends_on)
        else:
            self.depends_on = ()
        self._used_by = set()

    def used_by_strs(self) -> List[str]:
        return sorted(u.name() for u in self._used_by)

    def header_guard_name(self) -> str:
        return "EXO_MEM_GLOBAL_" + self.name


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
        "_separate_dataptr",
        "_srcinfo",
        "_struct_name",
        "_guard_macro",
    ]

    def __init__(
        self, ctype, type_shorthand, n_dims, is_const, separate_dataptr, srcinfo
    ):
        """For internal use of LoopIR compiler"""
        self._ctype = ctype
        self._type_shorthand = type_shorthand
        self._n_dims = n_dims
        self._is_const = is_const
        self._separate_dataptr = separate_dataptr
        self._srcinfo = srcinfo

        self._struct_name = None
        self._guard_macro = None

    def generate_default(self, memwin_name, data_ctype=None, mangle_parameters=None):
        sname = self.struct_name(memwin_name, mangle_parameters)
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

    def struct_name(self, memwin_name: str, mangle_parameters=None) -> str:
        """Must be called at least once (and consistently) to name the struct."""
        assert isinstance(memwin_name, str), "use str (avoid silent mistakes)"
        assert memwin_name

        if mangle_parameters:
            for p in mangle_parameters:
                assert isinstance(p, int), "Only support mangled names for ints"
                if p >= 0:
                    memwin_name += f"_{p}"
                else:
                    memwin_name += f"_n{-p}"

        # As promised in MemWin.separate_dataptr, if True, disable const suffix
        const_suffix = "c" if self._is_const and not self._separate_dataptr else ""
        base_sname = f"exo_win_{self._n_dims}{self._type_shorthand}{const_suffix}"
        mem_suffix = "" if memwin_name == "DRAM" else "_" + memwin_name
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
        return _memwin_template_names.get(cls) or cls.__name__

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

        In this case, the layout-only window struct is the same for both
        const and non-const windows.
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

    @classmethod
    def instr_tl_permission(cls, instr_tl: Instr_tl, is_instr):
        """For a given instr_tl, return a string of permission letters.

        r: read
        w: write
        c: create (allocate Memory or create SpecialWindow)

        The syntax similar to e.g. "rw" for opening a file
        NB for now, we expect 'r' if 'w' is given (i.e. no write-only mems)

        is_instr is True iff the access is via calling an instr, and not
        a scalar Read/Assign/Reduce statement. If is_instr is False,
        whether a Read/Assign/Reduce will actually compile additionally
        depends on the can_read/write/reduce member functions."""
        if instr_tl == cpu_in_order_instr:
            return "rwc"
        else:
            return ""

    @classmethod
    def default_usage_tl(cls, instr_tl: Instr_tl):
        assert (
            instr_tl == cpu_in_order_instr
        ), f"{cls} needs to implement default_usage_tl(instr_tl={instr_tl})"
        return cpu_usage

    @classmethod
    def as_const_shape(cls, new_name, shape, srcinfo, *, min_dim=0, max_dim=None):
        if len(shape) < min_dim:
            raise MemGenError(
                f"{srcinfo}: {new_name} @ {cls.name()} requires at least {min_dim} dimensions"
            )

        if max_dim is not None and len(shape) > max_dim:
            raise MemGenError(
                f"{srcinfo}: {new_name} @ {cls.name()} requires at most {max_dim} dimensions"
            )

        def to_int(extent):
            try:
                return int(extent)
            except ValueError as e:
                raise MemGenError(
                    f"{srcinfo}: {new_name} @ {cls.name()} requires constant shape. Saw: {extent}"
                ) from e

        return tuple(to_int(extent) for extent in shape)


class AllocableMemWin(MemWin):
    pass


class Memory(AllocableMemWin):
    """Memory type for backing non-barrier allocations"""

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


@dataclass(slots=True)
class BarrierTypeTraits:
    """Bare minimum vocab to describe valid barrier usage patterns

    Describes differences between cudaEvent_t, commit_group, mbarrier
    to parts of the codebase that are not deeply connected to
    CUDA. Re-think if we externalize.

    Instr_tl and CollTiling are to be handled by barrier lowering code.
    """

    # N = 1 always for an Arrive as of 2025-06-27
    # N = 0 for Await if neither of the following:
    non_negative_await_N: bool = False  # Require N >= 0 if true
    negative_await_N: bool = False  # Require N < 0 if true

    # Each Await stmt for same queue barrier array must use the same N
    uniform_await_N: bool = False

    # Forbid back queue barrier array (-name)
    supports_back_array: bool = False

    requires_pairing: bool = False
    requires_arrive_first: bool = False

    # Allow : in Arrive trailing queue barrier expr
    supports_arrive_multicast: bool = False


class BarrierType(AllocableMemWin):
    """MemWin type for backing barrier allocations

    The user parameterizes barrier allocations with `@ type`, in a manner
    similar to memory type annotations. The type must be one of the
    subclasses of BarrierType already defined by the Exo stdlib.

    Unlike memory, we currently do not have an interface for externalizing
    barrier types. Much of the logic is by-cases within the Exo->C compiler.
    So don't subclass this yourself (unless you plan to modify the compiler).

    """

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # prim_type and shape make no sense here
        return f"// Scope of named barrier {new_name}"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window_definition(cls, ctx):
        assert False, "Internal Exo error: window of barrier?"

    @classmethod
    def traits(cls) -> BarrierTypeTraits:
        raise NotImplementedError()


class SpecialWindowFromMemoryCtx(object):
    # TODO since we only give access to runtime window struct,
    # it's currently not possible to compile-time assert stride info.
    __slots__ = [
        "_src_data",
        "_src_layout",
        "_dst_dataptr_ctype",
        "_dst_struct_name",
        "_tensor_type",
        "_shape_strs",
        "_is_const",
        "_ctype",
        "_type_shorthand",
        "_srcinfo",
    ]

    def __init__(
        self,
        src_data,
        src_layout,
        dst_dataptr_ctype,
        dst_struct_name,
        tensor_type,
        shape_strs,
        is_const,
        ctype,
        type_shorthand,
        srcinfo,
    ):
        """For internal use of LoopIR compiler"""
        self._src_data = src_data
        self._src_layout = src_layout
        self._dst_dataptr_ctype = dst_dataptr_ctype
        self._dst_struct_name = dst_struct_name
        self._tensor_type = tensor_type
        self._shape_strs = shape_strs
        self._is_const = is_const
        self._ctype = ctype
        self._type_shorthand = type_shorthand
        self._srcinfo = srcinfo

    def src_data(self):
        """C initializer for source window data pointer

        Passed through from Memory.window of the source memory type"""
        return self._src_data

    def src_layout(self):
        """Untyped C initializer for source window layout (e.g. strides)

        Passed through (or default strides) from Memory.window of
        the source memory type"""
        return self._src_layout

    def dst_dataptr_ctype(self):
        """C type name of SpecialWindow data pointer (you defined this)"""
        return self._dst_dataptr_ctype

    def dst_struct_name(self):
        """C struct name of SpecialWindow window struct (you defined this)"""
        return self._dst_struct_name

    def tensor_type(self) -> LoopIR.Tensor:
        """return LoopIR.Tensor type of input tensor"""
        assert isinstance(self._tensor_type, LoopIR.Tensor)
        return self._tensor_type

    def shape_strs(self):
        """C strings defining dimension sizes of window"""
        return self._shape_strs

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
    def source_memory_type(cls) -> type:
        """Return memory type expected as input to window statement"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_memory(cls, ctx: SpecialWindowFromMemoryCtx):
        """Callback for generating C code initializing a special window
        from a window to a tensor of the source memory type.

        If separate_dataptr(), return (dataptr : str, layout : str) of
        C expressions that can initialize the two respective window variables.
        Otherwise, return a single C expression that can be used
        to initialize a struct of the window type.
        """
        raise NotImplementedError()

    # Remember to implement everything in base class MemWin as well


# ----------- TEMPLATE SYSTEM -------------


def memwin_template(class_factory):
    """Wrapper for creating MemWin types parameterized on a tuple of args.

    The name of the generated class will look like a function call
    e.g. MyMemoryName(64, 128) [akin to MyMemoryName<64, 128> in C++].
    Cached: identically parameterized MemWins will be identical Python types.

    The parameter tuple is injected to the class as memwin_template_parameters

    Usage:

    @memwin_template
    def MyMemoryName(*parameters):
        class MemoryImpl(Memory):  # class name is ignored
            ...implement memory normally
        return MemoryImpl
    """

    def class_factory_wrapper(*parameters, **kwargs):
        assert not kwargs, "No support for keyword template parameters"
        cache_key = (id(class_factory), parameters)
        cls = _memwin_template_cache.get(cache_key)
        if not cls:
            cls = class_factory(*parameters)
            cls_name = f"{class_factory.__name__}{parameters}"
            _memwin_template_cache[cache_key] = cls
            _memwin_template_names[cls] = cls_name
            cls.memwin_template_parameters = parameters
        return cls

    return class_factory_wrapper


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
