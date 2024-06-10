from enum import Enum, auto
from .LoopIR import LoopIR, T


class ProcedureBase:
    pass


class ExoType(Enum):
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
        return self in [
            ExoType.F16,
            ExoType.F32,
            ExoType.F64,
            ExoType.I8,
            ExoType.UI8,
            ExoType.UI16,
            ExoType.I32,
            ExoType.R,
        ]

    def is_bool(self):
        return self == ExoType.Bool


def loopir_type_to_exotype(typ: T) -> ExoType:
    mapping = {
        LoopIR.F16: ExoType.F16,
        LoopIR.F32: ExoType.F32,
        LoopIR.F64: ExoType.F64,
        LoopIR.UINT8: ExoType.UI8,
        LoopIR.INT8: ExoType.I8,
        LoopIR.UINT16: ExoType.UI16,
        LoopIR.INT32: ExoType.I32,
        LoopIR.Num: ExoType.R,
        LoopIR.Index: ExoType.Index,
        LoopIR.Bool: ExoType.Bool,
        LoopIR.Size: ExoType.Size,
        LoopIR.Int: ExoType.Int,
        LoopIR.Stride: ExoType.Stride,
    }
    for key, val in mapping.items():
        if isinstance(typ, key):
            return val
    assert False, f"Type {typ} not found"
