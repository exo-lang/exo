"""Codegen utilites for windows"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple

from .cir import CIR, CIR_Wrapper
from .LoopIR import uast_prim_types, loopir_from_uast_type_table


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


@dataclass(slots=True)
class WindowFeatures:
    """Container holding C syntax describing "features" of a window"""

    _memwin_name: str
    _scalars_per_packed_tensor: int
    _dataptr: CIR_Wrapper
    _array_strides_as_packed: Tuple[CIR_Wrapper]  # empty if strides not supported
    _array_offsets: Tuple[CIR_Wrapper]
    _array_interval_sizes: Tuple[Optional[CIR_Wrapper]]
    _packed_offsets: Tuple[CIR_Wrapper]
    _packed_interval_sizes: Tuple[Optional[CIR_Wrapper]]
    _encoder: Optional["WindowEncoder"]
    _indexer: Optional["WindowIndexer"]

    def get_memwin_name(self) -> str:
        """Name of the MemWin type of the source tensor/window"""
        return self._memwin_name

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
        assert len(self._array_strides_as_packed) == dims, "LoopIR internal error"
        assert len(self._array_interval_sizes) == dims, "LoopIR internal error"

    def get_array_stride_as_packed(self, i) -> CIR_Wrapper:
        """Get the stride of the i-th array dimension in units of packed tensors"""
        strides = self._array_strides_as_packed
        if not strides:
            raise ValueError(f"{self._memwin_name} does not support explicit strides")
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

        For WindowIndexer, see WindowIndexerSupport.
        """
        return self._array_interval_sizes[i]

    def n_packed_dims(self) -> int:
        """Number of packed dimensions of the source tensor/window.

        This is always the number of packed dimensions specified by
        the source MemWin type.

        """
        dims = len(self._packed_offsets)
        assert len(self._packed_interval_sizes) == dims, "LoopIR internal error"

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

        For WindowIndexer, see WindowIndexerSupport, and,
        with A being n_array_dims(),

        If the (A + n)-th index in the window expression is an interval lo:hi,
        get_packed_offset(n) =          lo
        get_packed_interval_size(n) =   hi - lo
        where in fact we guarantee lo = 0 and
        hi = the n-th coordinate of the packed tensor size

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


@dataclass(slots=True)
class WindowEncoderArgs:
    type_shorthand: str  # Exo name for scalar type, e.g. f64, i32
    n_dims: int  # Number of dimensions
    const: bool


@dataclass(slots=True)
class WindowEncoder:
    # Filled by __init__ (if you don't override it)
    type_shorthand: str
    ctype: str
    n_dims: int
    const: bool

    # Injected implicitly
    _exo_base_memwin_name: str
    _exo_memwin_template_parameters: Tuple[int]

    def __init__(self, args: WindowEncoderArgs):
        self.type_shorthand = args.type_shorthand
        self.ctype = loopir_from_uast_type_table[
            uast_prim_types[args.type_shorthand]
        ].ctype()
        self.n_dims = args.n_dims
        self.const = args.const

    def exo_struct_ctype(self) -> str:
        """Dictates the C name you use for the window struct. DON'T override this"""
        memwin_name = self._exo_base_minwin_name
        mangle_parameters = self._exo_memwin_template_parameters

        assert isinstance(memwin_name, str)
        if mangle_parameters:
            for p in mangle_parameters:
                assert isinstance(p, int), "Only support mangled names for ints"
                if p >= 0:
                    memwin_name += f"_{p}"
                else:
                    memwin_name += f"_n{-p}"
        if memwin_name == "DRAM":
            mem_suffix = ""
        else:
            mem_suffix = "_" + memwin_name
        if not self.separate_dataptr_ctype() and self.const:
            # const_suffix suppressed if separate dataptr enabled as promised
            # in separate_dataptr_ctype()
            const_suffix = "c"
        else:
            const_suffix = ""

        return f"exo_win_{self.n_dims}_{self.type_shorthand}{const_suffix}{mem_suffix}"

    def separate_dataptr_ctype(self) -> Optional[str]:
        """Override this to return a str to opt-in to separate dataptr mode.

        In this mode, window structs are given as a pair of C objects
        {exo_struct_ctype()} {varname};
        {separate_dataptr_ctype()} exo_data_{varname};

        If the dataptr is separate, the struct name will be the same
        for the const and non-const versions of a certain window.

        This is primarily for CUtensorMap.

        """
        return None

    def define_struct(self, depends_on: list) -> str:
        """Give window struct definition as C string.

        The struct must contain the dataptr as a member named .data,
        unless a separate dataptr type is defined.

        Optionally append MemIncludeC and MemGlobalC objects to depends_on
        to have them injected (along with include guards) at some point
        before this struct's definition in the generated C/CUDA code.

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

        """
        raise NotImplementedError()

    def decode_array_offset(
        self, utils: UtilInjector, window: CIR_Wrapper, n: int
    ) -> CIR_Wrapper:
        """Return the i-th array offset with respect to the dataptr.

        If you adjusted the dataptr to already point to the [0, 0, ...0]-th
        element, then this should always return 0.

        """
        raise NotImplementedError()

    def decode_array_stride_as_packed(
        self, utils: UtilInjector, window: CIR_Wrapper, n: int
    ) -> CIR_Wrapper:
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


class WindowIndexerSupport(Enum):
    points_only = auto()
    intervals_only = auto()
    points_before_intervals = auto()


@dataclass(slots=True)
class WindowIndexerArgs:
    type_shorthand: str  # Exo name for scalar type, e.g. f64, i32
    n_dims: int  # Number of dimensions
    const: bool
    exo_struct_ctype: Optional[str]


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
    _exo_struct_ctype: Optional[str]

    def __init__(self, args: WindowIndexerArgs):
        self.type_shorthand = args.type_shorthand
        self.ctype = loopir_from_uast_type_table[
            uast_prim_types[args.type_shorthand]
        ].ctype()
        self.n_dims = args.n_dims
        self.const = args.const
        self._exo_struct_ctype = args.exo_struct_ctype

    def exo_struct_ctype(self):
        """Give the name of the window struct as defined by the WindowEncoder"""
        assert self._exo_struct_ctype
        return self._exo_struct_ctype

    def array_dim_support(self) -> WindowIndexerSupport:
        """The WindowIndexer only supports encoding window expressions
        where the indices corresponding to array dimensions consist of:

        * points only, if WindowIndexerSupport.points_only
        * intervals only, if WindowIndexerSupport.intervals_only
        * any number of points followed by any number of intervals,
          if WindowIndexerSupport.points_before_intervals

        """
        return WindowIndexerSupport.points_only

    def packed_dim_support(self) -> WindowIndexerSupport:
        """The WindowIndexer only supports encoding window expressions
        where the indices corresponding to packed dimensions consist of:

        * points only, if WindowIndexerSupport.points_only
        * intervals only, if WindowIndexerSupport.intervals_only
        * any number of points followed by any number of intervals,
          if WindowIndexerSupport.points_before_intervals

        """
        return WindowIndexerSupport.intervals_only

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
