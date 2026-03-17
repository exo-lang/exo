from enum import Enum, auto

# Can't import LoopIR or prelude here.
# David Zhao Akeley 2026-01-09: This is a really unfortunate inversion
# of the normal dependency order, where API imports internals.


class ProcedureBase:
    pass


class ExoType(Enum):
    # CUDA 8-bit float types E{exponent bits}M{mantissa bits}
    E4M3 = auto()
    E5M2 = auto()
    E8M0 = auto()

    # 16-bit float types
    # David Zhao Akeley 2026-03-17: F16 is _Float16 in C, __half in CUDA
    # BF16 is not supported in C but could be in the future (e.g. AVX512 BF16)
    BF16 = auto()
    F16 = auto()

    # Typical float types
    F32 = auto()
    F64 = auto()

    # Numeric (data) integer types
    UI8 = auto()
    I8 = auto()
    UI16 = auto()
    I32 = auto()

    # Number of unknown precision
    R = auto()

    # Control value types
    Index = auto()
    Bool = auto()
    Size = auto()
    Int = auto()
    Stride = auto()

    def is_indexable(self):
        return self in [ExoType.Index, ExoType.Size, ExoType.Int, ExoType.Stride]

    def is_numeric(self):
        return self in self.numerics_set

    def is_bool(self):
        return self == ExoType.Bool


# Implementation note: will be updated by ScalarInfo.extclass for fixed-width types
ExoType.numerics_set = {ExoType.R}


def loopir_type_to_exotype(typ: "LoopIR.type") -> ExoType:
    try:
        from .core.prelude import ScalarInfo

        return ScalarInfo(typ).exotype
    except KeyError:
        from .core.LoopIR import LoopIR

        mapping = {
            LoopIR.Num: ExoType.R,
            LoopIR.Index: ExoType.Index,
            LoopIR.Bool: ExoType.Bool,
            LoopIR.Size: ExoType.Size,
            LoopIR.Int: ExoType.Int,
            LoopIR.Stride: ExoType.Stride,
        }
        return mapping[type(typ)]
