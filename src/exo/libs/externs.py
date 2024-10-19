from exo.core.extern import Extern, _EErr


class _Sin(Extern):
    def __init__(self):
        super().__init__("sin")

    def typecheck(self, args):
        if len(args) != 1:
            raise _EErr(f"expected 1 argument, got {len(args)}")

        atyp = args[0].type
        if not atyp.is_real_scalar():
            raise _EErr(
                f"expected argument 1 to be a real scalar value, but "
                f"got type {atyp}"
            )
        return atyp

    def globl(self, prim_type):
        return "#include <math.h>"

    #    def interpret(self, args):
    #        return math.sin(args[0])

    def compile(self, args, prim_type):
        return f"sin(({prim_type}){args[0]})"


sin = _Sin()


class _Relu(Extern):
    def __init__(self):
        super().__init__("relu")

    def typecheck(self, args):
        if len(args) != 1:
            raise _EErr(f"expected 1 argument, got {len(args)}")

        atyp = args[0].type
        if not atyp.is_real_scalar():
            raise _EErr(
                f"expected argument 1 to be a real scalar value, but "
                f"got type {atyp}"
            )
        return atyp

    def globl(self, prim_type):
        s = (
            f"{prim_type} _relu_{prim_type}({prim_type} x) " + "{\n"
            "    if (x > 0.0) return x;\n"
            "    else return 0.0;\n"
            "}\n"
        )
        return s

    #    def interpret(self, args):
    #        if args[0] > 0:
    #            return args[0]
    #        else:
    #            return 0

    def compile(self, args, prim_type):
        return f"_relu_{prim_type}(({prim_type}){args[0]})"


relu = _Relu()


class _Select(Extern):
    def __init__(self):
        super().__init__("select")

    def typecheck(self, args):
        if len(args) != 4:
            raise _EErr(f"expected 4 arguments, got {len(args)}")

        for i in range(len(args)):
            atyp = args[i].type
            if not atyp.is_real_scalar():
                raise _EErr(
                    f"expected argument {i+1} to be a real scalar value, but "
                    f"got type {atyp}"
                )
        return atyp

    def globl(self, prim_type):
        s = (
            f"{prim_type} _select_{prim_type}({prim_type} x,{prim_type} v,{prim_type} y,{prim_type} z)"
            + " {\n"
            "    if (x < v) return y;\n"
            "    else return z;\n"
            "}\n"
        )
        return s

    #    def interpret(self, args):
    #        x = args[0]
    #        v = args[1]
    #        y = args[2]
    #        z = args[3]
    #        if x < v:
    #            return y
    #        else:
    #            return z

    def compile(self, args, prim_type):
        return f"_select_{prim_type}(({prim_type}){args[0]}, ({prim_type}){args[1]}, ({prim_type}){args[2]}, ({prim_type}){args[3]})"


select = _Select()


class _Expf(Extern):
    def __init__(self):
        super().__init__("expf")

    def typecheck(self, args):
        if len(args) != 1:
            raise _EErr(f"expected 1 argument, got {len(args)}")

        atyp = args[0].type
        if not atyp.is_real_scalar():
            raise _EErr(
                f"expected argument 1 to be a real scalar value, but "
                f"got type {atyp}"
            )
        return atyp

    def globl(self, prim_type):
        return "#include <math.h>"

    #    def interpret(self, args):
    #        return math.expf(args[0])

    def compile(self, args, prim_type):
        return f"expf(({prim_type})({args[0]}))"


expf = _Expf()


class _FmaxF(Extern):
    def __init__(self):
        super().__init__("fmaxf")

    def typecheck(self, args):
        if len(args) != 2:
            raise _EErr(f"expected 2 argument, got {len(args)}")

        for i in range(len(args)):
            atyp = args[i].type
            if not atyp.is_real_scalar():
                raise _EErr(
                    f"expected argument {i+1} to be a real scalar value, but "
                    f"got type {atyp}"
                )
        return atyp

    def globl(self, prim_type):
        return "#include <math.h>"

    #    def interpret(self, args):
    #        return math.fmaxf(args[0], args[1])

    def compile(self, args, prim_type):
        return f"fmaxf(({prim_type})({args[0]}), ({prim_type})({args[1]}))"


fmaxf = _FmaxF()


class _Sigmoid(Extern):
    def __init__(self):
        super().__init__("sigmoid")

    def typecheck(self, args):
        if len(args) != 1:
            raise _EErr(f"expected 1 argument, got {len(args)}")

        atyp = args[0].type
        if not atyp.is_real_scalar():
            raise _EErr(
                f"expected argument 1 to be a real scalar value, but "
                f"got type {atyp}"
            )
        return atyp

    def globl(self, prim_type):
        return f"""
#include <math.h>
{prim_type} sigmoid({prim_type} x) {{
    return 1 / (1 + exp(-x));
}}
"""

    #    def interpret(self, args):
    #        return math.sigmoid(args[0])

    def compile(self, args, prim_type):
        return f"sigmoid(({prim_type})({args[0]}))"


sigmoid = _Sigmoid()


class _Sqrt(Extern):
    def __init__(self):
        super().__init__("sqrt")

    def typecheck(self, args):
        if len(args) != 1:
            raise _EErr(f"expected 1 argument, got {len(args)}")

        atyp = args[0].type
        if not atyp.is_real_scalar():
            raise _EErr(
                f"expected argument 1 to be a real scalar value, but "
                f"got type {atyp}"
            )
        return atyp

    def globl(self, prim_type):
        return "#include <math.h>"

    #    def interpret(self, args):
    #        return math.sqrt(args[0])

    def compile(self, args, prim_type):
        return f"sqrt(({prim_type})({args[0]}))"


sqrt = _Sqrt()
