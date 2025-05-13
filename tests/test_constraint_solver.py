from __future__ import annotations
from exo.core.prelude import Sym

from exo.rewrite.constraint_solver import ConstraintMaker, DisjointConstraint
from exo.core.LoopIR import T
from exo import proc
from exo.rewrite.chexo import TypeVisitor


def stringify_proc_constraint(p, invert=False):
    p_type = TypeVisitor()
    p_type.visit(p._loopir_proc)
    constraint = ConstraintMaker(p_type.type_map).make_constraint(
        p._loopir_proc.preds[0]
    )
    return (constraint.invert() if invert else constraint).pretty_print()


def solve_proc_assertion(p):
    p_type = TypeVisitor()
    p_type.visit(p._loopir_proc)
    cm = ConstraintMaker(p_type.type_map)
    constraint = cm.make_constraint(p._loopir_proc.preds[0])
    return "\n".join(
        sorted(
            [
                f"{str(sym)} = {val}"
                for sym, val in cm.solve_constraint(
                    constraint, bound=100, search_limit=10, seed=13
                ).var_assignments.items()
            ]
        )
    )


def test_make_constraint(golden):
    @proc
    def foo(a: size, b: size, c: size):
        assert ((a * 4 + b > c) or (a <= 3)) and (b < 5)
        pass

    assert golden == stringify_proc_constraint(foo)


def test_solve(golden):
    @proc
    def foo(a: size, b: size, c: size):
        assert ((a * 4 + b > c) or (a <= 3)) and (b < 5)
        pass

    assert golden == solve_proc_assertion(foo)


def test_divmod(golden):
    @proc
    def foo(a: size, b: size, c: size):
        assert ((a * 4 + b > c) or (a <= 3)) and (b < 5) and (a % 4 == 3)
        pass

    assert golden == stringify_proc_constraint(foo)


def test_divmod_solve(golden):
    @proc
    def foo(a: size, b: size, c: size):
        assert ((a * 4 + b > c) or (a <= 3)) and (b < 5) and (a % 4 == 3)
        pass

    assert golden == solve_proc_assertion(foo)


def test_large_slack(golden):
    @proc
    def foo(a: size):
        assert a <= 1000000
        pass

    assert golden == solve_proc_assertion(foo)


def test_disjunction(golden):
    @proc
    def foo(a: size, b: size):
        assert (a <= 3 or b <= 4) and (a + b < 4)
        pass

    assert golden == stringify_proc_constraint(foo)


def test_inversion(golden):
    @proc
    def foo(a: size, b: size):
        assert (a <= 3 or b <= 4) and (a + b == 4)
        pass

    assert golden == stringify_proc_constraint(foo, True)
