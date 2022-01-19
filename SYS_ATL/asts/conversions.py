def uast_to_type(typ):
    from . import LoopIR
    from . import UAST

    return {
        UAST.Size():   LoopIR.Size(),
        UAST.Bool():   LoopIR.Bool(),
        UAST.Index():  LoopIR.Index(),
        UAST.Stride(): LoopIR.Stride(),
        UAST.F32():    LoopIR.F32(),
        UAST.F64():    LoopIR.F64(),
        UAST.INT8():   LoopIR.INT8(),
        UAST.INT32():  LoopIR.INT32(),
    }[typ]
