from enum import Enum, auto


class ProcedureBase:
    pass


class ExoType(Enum):
    F32 = auto()
    F64 = auto()
    I8 = auto()
    I32 = auto()
    R = auto()
    Index = auto()
    Bool = auto()
    Size = auto()

    def is_indexable(self):
        return self in [ExoType.Index, ExoType.Size]

    def is_numeric(self):
        return self in [ExoType.F32, ExoType.F64, ExoType.I8, ExoType.I32, ExoType.R]

    def is_bool(self):
        return self == ExoType.Bool
