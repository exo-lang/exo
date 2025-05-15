from ..core.extern import Extern, _EErr
import numpy as np

from ..rewrite.constraint_solver import Constraint, DisjointConstraint, Expression
from ..core.prelude import Sym


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

    def interpret(self, args):
        return np.sin(args[0])

    def transpile(self, args):
        return f"Math.sin({args[0]})"

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

    def interpret(self, args):
        if args[0] > 0:
            return args[0]
        else:
            return 0

    def transpile(self, args):
        return f"(({args[0]}>0)?{args[0]}:0)"

    def compile(self, args, prim_type):
        return f"_relu_{prim_type}(({prim_type}){args[0]})"

    def express_in_constraints(
        self, args: tuple[Expression, ...], out_sym: Sym
    ) -> DisjointConstraint:
        result_expr = Expression.from_sym(out_sym)
        return (
            Constraint(args[0], True)
            .lift_to_disjoint_constraint()
            .intersect(
                Constraint(
                    args[0].add(result_expr.negate()), False
                ).lift_to_disjoint_constraint()
            )
            .union(
                Constraint(args[0].negate(), True)
                .lift_to_disjoint_constraint()
                .intersect(Constraint(result_expr, False).lift_to_disjoint_constraint())
            )
        )


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

    def interpret(self, args):
        x = args[0]
        v = args[1]
        y = args[2]
        z = args[3]
        if x < v:
            return y
        else:
            return z

    def transpile(self, args):
        return f"(({args[0]}<{args[1]})?{args[2]}:{args[3]})"

    def compile(self, args, prim_type):
        return f"_select_{prim_type}(({prim_type}){args[0]}, ({prim_type}){args[1]}, ({prim_type}){args[2]}, ({prim_type}){args[3]})"

    def express_in_constraints(
        self, args: tuple[Expression, ...], out_sym: Sym
    ) -> DisjointConstraint:
        result_expr = Expression.from_sym(out_sym)
        return (
            Constraint(
                args[1].add(args[0].add(Expression.from_constant(1)).negate()), True
            )
            .lift_to_disjoint_constraint()
            .intersect(
                Constraint(
                    args[2].add(result_expr.negate()), False
                ).lift_to_disjoint_constraint()
            )
            .union(
                Constraint(args[0].add(args[1].negate()), True)
                .lift_to_disjoint_constraint()
                .intersect(
                    Constraint(
                        args[3].add(result_expr.negate()), False
                    ).lift_to_disjoint_constraint()
                )
            )
        )


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

    def interpret(self, args):
        return np.exp(args[0])

    def transpile(self, args):
        return f"Math.exp({args[0]})"

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

    def interpret(self, args):
        return np.nanmax([args[0], args[1]])

    def transpile(self, args):
        return f"(({args[0]}<{args[1]}&&{args[1]}=={args[1]})?{args[1]}:{args[0]})"

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

    def interpret(self, args):
        return 1 / (1 + np.exp(-args[0]))

    def transpile(self, args):
        return f"1/(1+Math.exp(-{args[0]}))"

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

    def interpret(self, args):
        return np.sqrt(args[0])

    def transpile(self, args):
        return f"Math.sqrt({args[0]})"

    def compile(self, args, prim_type):
        return f"sqrt(({prim_type})({args[0]}))"


sqrt = _Sqrt()


class _IntMin(Extern):
    def __init__(self):
        super().__init__("intmin")

    def typecheck(self, args):
        if len(args) != 2:
            raise _EErr(f"expected 2 arguments, got {len(args)}")

        for i in range(len(args)):
            atyp = args[i].type
            if not atyp.is_indexable() and not atyp.is_real_scalar():
                raise _EErr(
                    f"expected argument {i+1} to be a real scalar value or "
                    f"control flow value, but got type {atyp}"
                )
        return atyp

    def globl(self, prim_type):
        s = (
            f"{prim_type} _intmin_{prim_type}({prim_type} x,{prim_type} v)" + " {\n"
            "    if (x < v) return x;\n"
            "    else return v;\n"
            "}\n"
        )
        return s

    def interpret(self, args):
        x = args[0]
        v = args[1]
        if x < v:
            return x
        else:
            return v

    def transpile(self, args):
        return f"(({args[0]}<{args[1]})?{args[0]}:{args[1]})"

    def compile(self, args, prim_type):
        return f"_intmin_{prim_type}(({prim_type}){args[0]}, ({prim_type}){args[1]})"

    def express_in_constraints(
        self, args: tuple[Expression, ...], out_sym: Sym
    ) -> DisjointConstraint:
        result_expr = Expression.from_sym(out_sym)
        return (
            Constraint(
                args[1].add(args[0].add(Expression.from_constant(1)).negate()), True
            )
            .lift_to_disjoint_constraint()
            .intersect(
                Constraint(
                    args[0].add(result_expr.negate()), False
                ).lift_to_disjoint_constraint()
            )
            .union(
                Constraint(args[0].add(args[1].negate()), True)
                .lift_to_disjoint_constraint()
                .intersect(
                    Constraint(
                        args[1].add(result_expr.negate()), False
                    ).lift_to_disjoint_constraint()
                )
            )
        )


intmin = _IntMin
