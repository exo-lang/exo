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
from .prelude import ScalarInfo
from .c_window import (
    WindowEncoder,
    WindowEncoderArgs,
    WindowIndexer,
    WindowIndexerArgs,
    FallbackWindowEncoder,
    FallbackWindowIndexer,
)

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
_memwin_template_base_names = {}
_memwin_template_cache = {}


class MemGenError(Exception):
    pass


@dataclass(slots=True)
class MemIncludeC:
    """Add MemIncludeC to MemGlobalC.depends_on to require a header file"""

    header_name: str


@dataclass(slots=True, init=False)
class MemGlobalC:
    """Give back MemGlobalC(...) in global_() to require some source code.

    The MemGlobalC contains a C identifier `name`, and C code `code`.
    The code is injected into either the header file, or the C/CUDA
    source file, wrapped with a `name`-based header guard.
    The code goes into the header file only if it's required by some
    MemWin type that's part of a public proc's interface.

    For two MemGlobalC instances a and b with a.code, b.code both not empty,
    Exo requires that (a.name == b.name) iff (a.code == b.code)

    Any code in the self.depends_on list will get injected before self.code.

    """

    name: str
    code: str
    depends_on: Tuple[MemGlobalC | MemIncludeC]

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


def generate_offset(indices, strides):
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
    return expr


class MemWin(ABC):
    """Common base class of allocable Memory and non-allocable SpecialWindow"""

    # Injected by @window_encoder, @window_indexer
    # The fallbacks are only used if the user never specifes a custom encoder/indexer
    _exo_window_encoder_type: Optional[Type[WindowEncoder]] = FallbackWindowEncoder
    _exo_window_encoder_origin_memwin: Optional[Type[MemWin]] = None
    _exo_window_indexer_type: Optional[Type[WindowIndexer]] = FallbackWindowIndexer
    _exo_window_indexer_origin_memwin: Optional[Type[MemWin]] = None

    # Injected by @memwin_template
    memwin_template_parameters: tuple = ()

    @classmethod
    def name(cls):
        return _memwin_template_names.get(cls) or cls.__name__

    @classmethod
    def base_name(cls):
        """Name without template parameters"""
        return _memwin_template_base_names.get(cls) or cls.__name__

    @classmethod
    def mangled_name(cls, base_name=None):
        """Unique C identifier for (MemWin, template parameters) combination"""
        fragments = [base_name or cls.base_name()]
        mangle_parameters = cls.memwin_template_parameters
        for p in mangle_parameters:
            assert isinstance(
                p, int
            ), f"{cls.name()}: only support mangled names for ints, not {p}"
            fragments.append(f"{p}" if p >= 0 else "n{-p}")
        return "_".join(fragments)

    @classmethod
    def global_(cls) -> str | MemGlobalC:
        """
        C code
        """
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        offset = generate_offset(indices, strides)

        if basetyp.is_win():
            baseptr = f"{baseptr}.data"

        return f"{baseptr}[{offset}]"

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
    def packed_tensor_shape(cls, scalar_info) -> List[int]:
        return ()

    # TODO remove?
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

    @classmethod
    def has_window_encoder(cls):
        """Do not override"""
        return cls._exo_window_encoder_type is not None

    @classmethod
    def make_window_encoder(cls, scalar_info, n_dims, const):
        """Do not override"""
        origin_memwin = cls._exo_window_encoder_origin_memwin
        args = WindowEncoderArgs(
            cls,
            ScalarInfo(scalar_info),
            n_dims,
            const,
            "" if not origin_memwin else origin_memwin.base_name(),
        )
        return cls._exo_window_encoder_type(args)

    @classmethod
    def window_struct_name(cls, scalar_info, n_dims, const):
        """Do not override"""
        return cls.make_window_encoder(
            ScalarInfo(scalar_info), n_dims, const
        ).exo_struct_name()

    @classmethod
    def has_window_indexer(cls):
        """Do not override"""
        return cls._exo_window_indexer_type is not None

    @classmethod
    def make_window_indexer(cls, scalar_info, n_dims, const):
        """Do not override"""
        sname = None
        scalar_info = ScalarInfo(scalar_info)
        if cls.has_window_encoder():
            sname = cls.window_struct_name(scalar_info.shorthand, n_dims, const)
        args = WindowIndexerArgs(scalar_info, n_dims, const, sname)
        return cls._exo_window_indexer_type(args)

    @classmethod
    def wrapped_smem_type(cls):
        """Do not override; used in the compiler internally"""
        return cls


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
    def traits(cls) -> BarrierTypeTraits:
        raise NotImplementedError()


class SpecialWindow(MemWin):
    _exo_window_encoder_type: Optional[Type[WindowEncoder]] = None
    _exo_window_indexer_type: Optional[Type[WindowIndexer]] = None

    @classmethod
    @abstractmethod
    def source_memory_type(cls) -> type:
        """Return memory type expected as input to window statement"""
        raise NotImplementedError()

    # Remember to implement everything in base class MemWin as well


# ----------- TEMPLATE SYSTEM -------------


def memwin_template(class_factory, *, hide_parameters=False):
    """Wrapper for creating MemWin types parameterized on a tuple of args.

    The name of the generated class will look like a function call
    e.g. MyMemoryName(64, 128) [akin to MyMemoryName<64, 128> in C++].
    Cached: identically parameterized MemWins will be identical Python types.

    The parameter tuple is injected to the class as memwin_template_parameters

    Usage:

    @memwin_template
    def MyMemoryName(*parameters):
        @window_encoder(...)  # optional
        @window_indexer(...)  # optional
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
            assert cls, f"forgot return from {class_factory}?"
            cls_name = f"{class_factory.__name__}{parameters}"
            _memwin_template_cache[cache_key] = cls
            _memwin_template_names[cls] = cls_name
            _memwin_template_base_names[cls] = class_factory.__name__
            if not hide_parameters:
                cls.memwin_template_parameters = parameters
        return cls

    return class_factory_wrapper


# ----------- WINDOW ENCODER, INDEXER DECORATORS -------------


def window_encoder(encoder_cls):
    """MemWin class decorator, add WindowEncoder"""
    assert issubclass(encoder_cls, WindowEncoder) or encoder_cls is None
    assert not issubclass(
        encoder_cls, FallbackWindowEncoder
    ), "you are not allowed to use this explicitly"

    def add_encoder(mem_cls):
        assert issubclass(mem_cls, MemWin)
        mem_cls._exo_window_encoder_type = encoder_cls
        mem_cls._exo_window_encoder_origin_memwin = mem_cls
        if mem_cls._exo_window_indexer_type is FallbackWindowIndexer:
            mem_cls._exo_window_indexer_type = None
        return mem_cls

    return add_encoder


def window_indexer(indexer_cls):
    """MemWin class decorator, add WindowIndexer"""
    assert issubclass(indexer_cls, WindowIndexer) or indexer_cls is None
    assert not issubclass(
        indexer_cls, FallbackWindowIndexer
    ), "you are not allowed to use this explicitly"

    def add_indexer(mem_cls):
        assert issubclass(mem_cls, MemWin)
        mem_cls._exo_window_indexer_type = indexer_cls
        mem_cls._exo_window_indexer_origin_memwin = mem_cls
        if mem_cls._exo_window_encoder_type is FallbackWindowEncoder:
            mem_cls._exo_window_encoder_type = None
        return mem_cls

    return add_indexer


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
