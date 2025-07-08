"""Codegen utilites for windows"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Type

from .cir import CIR, CIR_Wrapper
from . import LoopIR
from .prelude import SrcInfo


class UtilInjector:
    __slots__ = []

    def add_c_util(self, code):
        """Add snippet of C code at global scope to appear before your code"""
        raise NotImplementedError()

    def add_c_include(self, header_name):
        """Add header file to generated C code"""
        raise NotImplementedError()

    def add_cu_util(self, code):
        """Add CUDA utility to appear before your code"""
        raise NotImplementedError()

    def add_cu_include(self, header_name):
        """Add header file to generated CUDA code"""
        raise NotImplementedError()


@dataclass(slots=True, init=False)
class WindowFeatures:
    """Container holding C syntax describing "features" of a window

    Important tricky implementation notes:

    1. The WindowFeatures more-or-less encodes a window *expression*,
    which contain both points and intervals (x[pt] vs x[lo:hi]).
    The "strides" (should they exist) correspond to both point and
    intervals dimensions, so a classic-style Exo window struct will
    have to filter out strides corresponding to points, e.g.
    encode(x[a, b:c, d]) will use get_*_stride(0) and get_*_stride(2).

    2. Exo-GPU leverages the "Sym to cname" conversion (Compiler.env)
    to sometimes give different C names to the same Exo variable
    in C and CUDA code (e.g. foo vs exo_deviceArgs.foo). Thus the use
    of CIR_Wrapper is required even for "trivial" expressions; this
    delays to Sym-to-str lookup so we know which name to use.

    """

    _mem: Type["MemWin"]
    _scalars_per_packed_tensor: int
    _varname: CIR_Wrapper
    _dataptr: CIR_Wrapper
    _array_strides_as_packed: Tuple[CIR_Wrapper]  # empty if strides not supported
    _array_offsets: Tuple[CIR_Wrapper]
    _array_interval_sizes: Tuple[Optional[CIR_Wrapper]]
    _packed_offsets: Tuple[CIR_Wrapper]
    _packed_interval_sizes: Tuple[Optional[CIR_Wrapper]]
    _encoder: Optional["WindowEncoder"]
    _indexer: Optional["WindowIndexer"]

    # For use by FallbackWindowEncoder, FallbackWindowIndexer only
    _legacy_basetyp: object  # LoopIR type

    _srcinfo: SrcInfo

    def copy(self):
        new = WindowFeatures()
        for attr in self.__slots__:
            setattr(new, attr, getattr(self, attr))
        return new

    def get_memwin(self) -> Type["MemWin"]:
        return self._mem

    def get_memwin_name(self) -> str:
        """Name of the MemWin type of the source tensor/window"""
        return self._mem.name()

    def get_raw_name(self) -> str:
        return str(self._varname)

    def get_dataptr(self) -> CIR_Wrapper:
        """C syntax for data pointer of the window"""
        return self._dataptr

    def n_array_dims(self) -> int:
        """Number of array dimensions of the source tensor/window

        If you are encoding the window and support reducing the dimensionality,
        then the number of array dimensions of the encoded window will be this
        minus the number of None array interval sizes.
        """
        dims = len(self._array_offsets)
        assert len(self._array_interval_sizes) == dims, "LoopIR internal error"
        return dims

    def get_array_stride_as_packed(self, i) -> CIR_Wrapper:
        """Get the stride of the i-th array dimension in units of packed tensors"""
        strides = self._array_strides_as_packed
        if not strides:
            raise ValueError(f"{self._memwin_name} does not support explicit strides")
        assert len(self._array_offsets) == len(strides), "LoopIR internal error"
        return strides[i]

    def get_array_stride_as_scalars(self, i) -> CIR_Wrapper:
        """Get the stride of the i-th array dimension in units of C scalars"""
        return self.get_array_stride_as_packed(i) * self._scalars_per_packed_tensor

    def get_array_offset(self, i) -> CIR_Wrapper:
        """Offset with respect to the dataptr on the i-th array dimension.

        Essentially, window[0, 0, ..., 0] refers to the data at
        dataptr[get_array_offset(0), get_array_offset(1),
                ..., get_array_offset(n_array_dims() - 1),
                get_packed_offset(0), ..., get_packed_offset(n_packed_dims() - 1)]

        for some reasonable definition of indexing the dataptr.

        See also get_packed_offset (only needed for WindowIndexer).
        """
        return self._array_offsets[i]

    def get_array_interval_size(self, i) -> Optional[CIR_Wrapper]:
        """Get the size of the window on the i-th array dimension.

        With Off[n] being get_array_offset(n) for the input_window (usually 0),

        If the n-th index in the window expression is an interval lo:hi, then
        get_array_offset(n) =        Off[n] + lo
        get_array_interval_size(n) = hi - lo

        If the n-th index in the window expression is a point pt, then
        get_array_offset(n) =        Off[n] + pt
        get_array_interval_size(n) = None

        For WindowEncoder, the latter case (point) will only occur if
        WindowEncoder.supports_dim_change()

        """
        return self._array_interval_sizes[i]

    def n_packed_dims(self) -> int:
        """Number of packed dimensions of the source tensor/window.

        This is always the number of packed dimensions specified by
        the source MemWin type.

        """
        dims = len(self._packed_offsets)
        assert len(self._packed_interval_sizes) == dims, "LoopIR internal error"
        return dims

    def get_packed_offset(self, i) -> CIR_Wrapper:
        """Offset with respect to the dataptr on the i-th packed dimension.

        Only WindowIndexer uses this.

        This is always 0 for WindowEncoder, and it may assume that the packed
        dimensions have complete intervals.
        """
        return self._packed_offsets[i]

    def get_packed_interval_size(self, i) -> Optional[CIR_Wrapper]:
        """Get the size of the window on the i-th packed dimension.

        WindowEncoder ignores this.

        For WindowIndexer, with A being n_array_dims(),

        If the (A + n)-th index in the window expression is an interval lo:hi,
        get_packed_offset(n) =          lo
        get_packed_interval_size(n) =   hi - lo

        If the (A + n)-th index of the window expression is a point pt,
        get_packed_offset(n) =         pt
        get_packed_interval_size(n) =  None
        """

    def get_encoder(self) -> "WindowEncoder":
        """Get WindowEncoder for this window struct, if the MemWin supports it"""
        encoder = self._encoder
        if encoder is None:
            raise ValueError(f"{self._memwin_name} does not support WindowEncoder")
        assert isinstance(encoder, WindowEncoder)
        return encoder

    def get_indexer(self) -> "WindowIndexer":
        """Get WindowIndexer for this window, if the MemWin supports it"""
        indexer = self._indexer
        if indexer is None:
            raise ValueError(f"{self._memwin_name} does not support WindowIndexer")
        assert isinstance(indexer, WindowIndexer)
        return indexer

    def separate_dataptr(self) -> bool:
        if self._encoder is None:
            return False
        else:
            return self._encoder.separate_dataptr()

    def srcinfo(self) -> SrcInfo:
        return self._srcinfo

    def new_window(
        self,
        idxs: List[int | CIR_Wrapper],
        interval_sizes: List[Optional[CIR_Wrapper]],
        srcinfo: SrcInfo,
    ):
        """For the compiler. Create a new WindowFeatures holding the features for
        a window made by applying a window expression [idx0, idx1, ...] to self.

        idxN is a point pt when (idxs[N] = pt, interval_sizes[N] = None)

        idxN is an interval lo:hi when (idxs[N] = lo, interval_sizes[N] = hi - lo)

        idxs may be shorter than the dimensionality of the window
        (implies trailing 0:hi which have no effect).

        """
        assert isinstance(srcinfo, SrcInfo)
        assert len(idxs) == len(interval_sizes), "Internal error"
        new = self.copy()
        new_offsets = list(new._array_offsets) + list(new._packed_offsets)
        new_interval_sizes = list(new._array_interval_sizes) + list(
            new._packed_interval_sizes
        )
        assert len(new_offsets) == len(new_interval_sizes), "Internal error"
        i = -1
        for idx, interval in zip(idxs, interval_sizes):
            # Find the next interval to update
            # Skip points
            while True:
                i += 1
                if i > len(new_interval_sizes):
                    raise ValueError("Too many indices given to window code generator")
                if new_interval_sizes[i] is not None:
                    break
            # Update here
            new_offsets[i] += idx
            new_interval_sizes[i] = interval

        n_array_dims = new.n_array_dims()
        new._array_offsets = tuple(new_offsets[:n_array_dims])
        new._array_interval_sizes = tuple(new_interval_sizes[:n_array_dims])
        new._packed_offsets = tuple(new_offsets[n_array_dims:])
        new._packed_interval_sizes = tuple(new_interval_sizes[n_array_dims:])
        new._srcinfo = srcinfo
        return new


@dataclass(slots=True)
class WindowEncoderArgs:
    mem: type
    type_shorthand: str  # Exo name for scalar type, e.g. f64, i32
    n_dims: int  # Number of dimensions
    const: bool
    base_memwin_name: str


@dataclass(slots=True)
class WindowEncoder:
    # Filled by __init__ (if you don't override it)
    mem: type
    type_shorthand: str
    ctype: str
    n_dims: int
    const: bool
    _exo_base_memwin_name: str

    def __init__(self, args: WindowEncoderArgs):
        assert isinstance(args.type_shorthand, str)
        assert isinstance(args.n_dims, int)
        self.mem = args.mem
        self.type_shorthand = args.type_shorthand
        self.ctype = LoopIR.loopir_from_uast_type_table[
            type(LoopIR.uast_prim_types[args.type_shorthand])
        ].ctype()
        self.n_dims = args.n_dims
        self.const = args.const
        self._exo_base_memwin_name = args.base_memwin_name

    def exo_struct_name(self) -> str:
        """Dictates the C name you use for the window struct. DON'T override this"""

        if self._exo_base_memwin_name:
            mem_suffix = "_" + self.mem.mangled_name(self._exo_base_memwin_name)
        else:
            # Special case when using FallbackWindowEncoder
            mem_suffix = ""
        if not self.separate_dataptr() and self.const:
            # const_suffix suppressed if separate dataptr enabled as promised
            # in separate_dataptr()
            const_suffix = "c"
        else:
            const_suffix = ""

        return f"exo_win_{self.n_dims}{self.type_shorthand}{const_suffix}{mem_suffix}"

    def dataptr_ctype(self) -> str:
        if self.const:
            return f"const {self.ctype}*"
        else:
            return f"{self.ctype}*"

    def separate_dataptr(self) -> bool:
        """Override this to return True to opt-in to separate dataptr mode.

        In this mode, window structs are given as a pair of C objects
        {exo_struct_name()} {varname};
        {dataptr_ctype()} exo_data_{varname};

        If the dataptr is separate, the struct name will be the same
        for the const and non-const versions of a certain window.

        This is primarily for CUtensorMap.

        """
        return False

    def define_struct(self, depends_on: list) -> str:
        """Give window struct definition as C string.

        The struct must contain the dataptr as a member named .data,
        unless a separate dataptr type is defined.

        Optionally append MemIncludeC and MemGlobalC objects to depends_on
        to have them injected (along with include guards) at some point
        before this struct's definition in the generated C/CUDA code.

        NOTE: you are intentionally not given a UtilInjector here.
        Defer adding C/CUDA utilities until actually generating code
        for encoding a struct, or in WindowIndexer too if needed.

        """
        raise NotImplementedError()

    def supports_dim_change(self) -> bool:
        """Override to True to indicate the encoder allows creating a window
        from another window of greater dimension.

        """
        return False

    def encode_window(
        self, utils: UtilInjector, features: WindowFeatures
    ) -> str | CIR_Wrapper:
        """Return C expression (str or CIR_Wrapper) giving window struct

        e.g. (struct struct_name) { foo, bar };

        For SpecialWindow, this is used to convert from an existing
        SpecialWindow. See encode_special_window.

        """
        raise NotImplementedError()

    def encode_separate_dataptr(
        self, utils: UtilInjector, features: WindowFeatures
    ) -> str | CIR_Wrapper:
        """Return C expression (str or CIR_Wrapper) giving the dataptr.

        This is only used when a separate_dataptr() is true.
        For SpecialWindow, this is used to convert from an existing
        SpecialWindow. See encode_special_separate_window.

        """
        raise NotImplementedError()

    def encode_special_window(
        self, utils: UtilInjector, features: WindowFeatures
    ) -> str | CIR_Wrapper:
        """Used for converting ordinary Memory to SpecialWindow.

        Same usage as encode_window()

        """
        raise NotImplementedError()

    def encode_special_separate_dataptr(
        self, utils: UtilInjector, features: WindowFeatures
    ) -> str | CIR_Wrapper:
        """Used for converting ordinary Memory to SpecialWindow.

        Same usage as encode_separate_dataptr()

        """
        raise NotImplementedError()

    def decode_array_offset(
        self, utils: UtilInjector, window: CIR_Wrapper, n: int
    ) -> int | CIR_Wrapper:
        """Return the i-th array offset with respect to the dataptr.

        If you adjusted the dataptr to already point to the [0, 0, ...0]-th
        element, then this should always return 0.

        """
        raise NotImplementedError()

    def decode_array_stride_as_packed(
        self, utils: UtilInjector, window: CIR_Wrapper, n: int
    ) -> int | CIR_Wrapper:
        """Optional: return the stride on the i-th array dimension.

        This is given in units of number of packed tensors.
        If the MemWin type defines 0 packed dimensions (default), then this
        is equivalent to a stride given in count of scalars.

        If the MemWin defines a WindowEncoder and this function is not defined,
        then stride queries aren't supported for that MemWin type.
        """
        raise NotImplementedError()

    # NOTE: by design, Exo window structs cannot encode intervals,
    # strides, or offsets into packed dimensions.


@dataclass(slots=True)
class WindowIndexerArgs:
    type_shorthand: str  # Exo name for scalar type, e.g. f64, i32
    n_dims: int  # Number of dimensions
    const: bool
    exo_struct_name: Optional[str]


@dataclass(slots=True)
class WindowIndexerResult:
    code: str
    is_ptr: bool

    def __init__(self, code, is_ptr):
        self.code = str(code)
        self.is_ptr = bool(is_ptr)


@dataclass(slots=True)
class WindowIndexer:
    # Filled by __init__ (if you don't override it)
    type_shorthand: str
    ctype: str
    n_dims: int
    const: bool
    _exo_struct_name: Optional[str]

    def __init__(self, args: WindowIndexerArgs):
        self.type_shorthand = args.type_shorthand
        self.ctype = LoopIR.loopir_from_uast_type_table[
            type(LoopIR.uast_prim_types[args.type_shorthand])
        ].ctype()
        self.n_dims = args.n_dims
        self.const = args.const
        self._exo_struct_name = args.exo_struct_name

    def exo_struct_name(self):
        """Give the name of the window struct as defined by the WindowEncoder"""
        assert self._exo_struct_name, "seems to be no WindowEncoder"
        return self._exo_struct_name

    def index(
        self, utils: UtilInjector, features: WindowFeatures
    ) -> WindowIndexerResult:
        """Return WindowIndexerResult(code, is_ptr)

        code must be a str or CIR_Wrapper giving a C expression
        delivering the window window/scalar implied by the given features.
        (The result is a scalar if all indices are points, not intervals).

        This "delivery" isn't necessarily in struct form; how the requested
        data gets delivered is up to the MemWin type author to specify.

        is_ptr determines whether the result resolves to C type
        `ctype*` (true) or `ctype` (false).  This is only relevant if
        you support Exo's built-in scalar reads or writes
        (i.e. without using a custom instr).

        """

        raise NotImplementedError()


_default_struct_template = """\
struct {sname} {{
    {const_keyword}{ctype} * const data;
    const int_fast32_t strides[{n_dims}];
}};"""


class FallbackWindowEncoder(WindowEncoder):
    __slots__ = []

    def define_struct(self, depends_on: list) -> str:
        sname = self.exo_struct_name()
        const_keyword = "const " if self.const else ""
        return _default_struct_template.format(
            sname=sname,
            ctype=self.ctype,
            n_dims=self.n_dims,
            const_keyword=const_keyword,
        )

    def supports_dim_change(self) -> bool:
        return True

    def encode_window(self, utils: UtilInjector, features: WindowFeatures) -> str:
        assert (
            features.n_packed_dims() == 0
        ), "Implement your own @window_encoder if you have packed dimensions"
        sname = self.exo_struct_name()
        mem = features.get_memwin()
        n_dims = features.n_array_dims()

        indices_strs = [str(features.get_array_offset(i)) for i in range(n_dims)]
        strides_strs = [
            str(features.get_array_stride_as_packed(i)) for i in range(n_dims)
        ]
        dataptr = mem.window(
            features._legacy_basetyp,
            features.get_raw_name(),
            indices_strs,
            strides_strs,
            features.srcinfo(),
        )

        # Remove strides corresponding to point expressions.
        filtered_strides_strs = [
            strides_strs[i]
            for i in range(n_dims)
            if features.get_array_interval_size(i) is not None
        ]
        strides = "{" + ", ".join(filtered_strides_strs) + "}"

        return f"(struct {sname}) {{ &{dataptr}, {strides} }}"

    def decode_array_offset(
        self, utils: UtilInjector, window: CIR_Wrapper, n: int
    ) -> int:
        return 0

    def decode_array_stride_as_packed(
        self, utils: UtilInjector, window: CIR_Wrapper, n: int
    ) -> CIR_Wrapper:
        return window.strides[n]


class FallbackWindowIndexer(WindowIndexer):
    def index(
        self, utils: UtilInjector, features: WindowFeatures
    ) -> WindowIndexerResult:
        assert (
            features.n_packed_dims() == 0
        ), "Implement your own @window_indexer if you have packed dimensions"
        mem = features.get_memwin()
        n_dims = features.n_array_dims()

        indices_strs = [str(features.get_array_offset(i)) for i in range(n_dims)]
        strides_strs = [
            str(features.get_array_stride_as_packed(i)) for i in range(n_dims)
        ]
        code = mem.window(
            features._legacy_basetyp,
            features.get_raw_name(),
            indices_strs,
            strides_strs,
            features.srcinfo(),
        )

        return WindowIndexerResult(code, False)
