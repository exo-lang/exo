# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# BuiltIn superclass


class BuiltIn_Typecheck_Error(Exception):
    def __init__(self, msg):
        self._builtin_err_msg = str(msg)

    def __str__(self):
        return self._builtin_err_msg


_BErr = BuiltIn_Typecheck_Error


class BuiltIn:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def globl(self):
        raise NotImplementedError()

    def typecheck(self, args):
        raise NotImplementedError()

    def compile(self, args):
        raise NotImplementedError()


class _Sin(BuiltIn):
    def __init__(self):
        super().__init__("sin")

    def typecheck(self, args):
        if len(args) != 1:
            raise _BErr(f"expected 1 argument, got {len(args)}")

        atyp = args[0].type
        if not atyp.is_real_scalar():
            raise _BErr(
                f"expected argument 1 to be a real scalar value, but "
                f"got type {atyp}"
            )
        return atyp

    def globl(self):
        return "#include <math.h>"

    def compile(self, args):
        return f"sin((double)*{args[0]})"


sin = _Sin()


class _Relu(BuiltIn):
    def __init__(self):
        super().__init__("relu")

    def typecheck(self, args):
        if len(args) != 1:
            raise _BErr(f"expected 1 argument, got {len(args)}")

        atyp = args[0].type
        if not atyp.is_real_scalar():
            raise _BErr(
                f"expected argument 1 to be a real scalar value, but "
                f"got type {atyp}"
            )
        return atyp

    def globl(self):
        s = (
            "double _relu_(double x) {\n"
            "    if (x > 0.0) return x;\n"
            "    else return 0.0;\n"
            "}\n"
        )
        return s

    def compile(self, args):
        return f"_relu_((double)*{args[0]})"


relu = _Relu()


class _Select(BuiltIn):
    def __init__(self):
        super().__init__("select")

    def typecheck(self, args):
        if len(args) != 4:
            raise _BErr(f"expected 4 arguments, got {len(args)}")

        atyp = args[0].type
        if not atyp.is_real_scalar():
            raise _BErr(
                f"expected argument 1 to be a real scalar value, but "
                f"got type {atyp}"
            )

        atyp = args[1].type
        if not atyp.is_real_scalar():
            raise _BErr(
                f"expected argument 2 to be a real scalar value, but "
                f"got type {atyp}"
            )

        atyp = args[2].type
        if not atyp.is_real_scalar():
            raise _BErr(
                f"expected argument 3 to be a real scalar value, but "
                f"got type {atyp}"
            )

        atyp = args[3].type
        if not atyp.is_real_scalar():
            raise _BErr(
                f"expected argument 4 to be a real scalar value, but "
                f"got type {atyp}"
            )
        return atyp

    def globl(self):
        s = (
            "double _select_(double x, double v, double y, double z) {\n"
            "    if (x < v) return y;\n"
            "    else return z;\n"
            "}\n"
        )
        return s

    def compile(self, args):
        return f"_select_((double)*{args[0]}, (double)*{args[1]}, (double)*{args[2]}, (double)*{args[3]})"


select = _Select()
