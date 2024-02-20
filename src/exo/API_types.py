from enum import Enum, auto


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

    def is_indexable(self):
        return self in [ExoType.Index, ExoType.Size, ExoType.Int]

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
