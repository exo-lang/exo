from enum import Enum, auto

# Can't import LoopIR or prelude here.
# David Zhao Akeley 2026-01-09: This is a really unfortunate inversion
# of the normal dependency order, where API imports internals.


class ProcedureBase:
    pass


class ExoType(Enum):
    CU_f16 = auto()
    CU_bf16 = auto()
    F16 = auto()
    F32 = auto()
    F64 = auto()
    UI8 = auto()
    I8 = auto()
    UI16 = auto()
    I32 = auto()
    R = auto()
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


# Will be updated by ScalarInfo.extclass for fixed-width types
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
