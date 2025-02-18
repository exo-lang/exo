from dataclasses import dataclass, field
from typing import Union, Optional

from exo.core.prelude import Sym
from ..core.LoopIR import LoopIR, T
import numpy as np


@dataclass
class ConstraintTerm:
    coefficient: int
    syms: tuple[Sym]

    def negate(self) -> "ConstraintTerm":
        return ConstraintTerm(-self.coefficient, self.syms)

    def multiply(self, other) -> "ConstraintTerm":
        return ConstraintTerm(
            self.coefficient * other.coefficient, self.syms + other.syms
        )

    def apply_assignments(
        self, assignments: dict[Sym, int]
    ) -> Optional[tuple[int, Optional[Sym]]]:
        target_sym = None
        acc = self.coefficient
        for sym in self.syms:
            if sym in assignments:
                acc *= assignments[sym]
            else:
                if target_sym is None:
                    target_sym = sym
                else:
                    return None
        return (acc, target_sym)

    def collect_nonlinear_syms(self) -> frozenset[Sym]:
        occurrences = set()
        result = set()
        for sym in self.syms:
            if sym in occurrences:
                result.add(sym)
            else:
                occurrences.add(sym)
        return frozenset(result)


@dataclass
class LinearConstraint:
    coefficients: dict[Sym, int]
    offset: int


@dataclass
class Constraint:
    terms: tuple[ConstraintTerm]

    def apply_assignments(
        self, assignments: dict[Sym, int]
    ) -> Optional[LinearConstraint]:
        coefficients = {}
        offset = 0
        for term in self.terms:
            assign_result = term.apply_assignments(assignments)
            if assign_result is None:
                return None
            else:
                coefficient, sym = assign_result
                if sym is None:
                    offset += coefficient
                else:
                    if sym not in coefficients:
                        coefficients[sym] = 0
                    coefficients[sym] += coefficient
        return LinearConstraint(coefficients, offset)

    def collect_syms(self) -> frozenset[Sym]:
        return frozenset(sym for term in self.terms for sym in term.syms)

    def collect_nonlinear_syms(self) -> frozenset[Sym]:
        return frozenset().union(
            *[term.collect_nonlinear_syms() for term in self.terms]
        )

    def pretty_print(self) -> str:
        return (
            " + ".join(
                [
                    f"{' * '.join([str(term.coefficient)] + [str(sym) for sym in term.syms])}"
                    for term in self.terms
                ]
            )
            + " == 0"
        )


class ConstraintMaker:
    def __init__(self, type_map: dict[Sym, LoopIR.type]):
        self.unconstrained_var_subs: dict[Sym, tuple[ConstraintTerm]] = {}
        self.extra_constraints: list[Constraint] = []
        self.stride_dummies: dict[tuple[Sym, int], Sym] = {}
        for sym, sym_type in type_map.items():
            if isinstance(sym_type, T.Size, T.Stride):
                # positive constraint
                self.extra_constraints.append(
                    Constraint(
                        (
                            ConstraintTerm(1, (sym,)),
                            ConstraintTerm(-1, ()),
                            ConstraintTerm(-1, (Sym("slack"),)),
                        )
                    )
                )
            elif isinstance(sym_type, T.Int, T.Num):
                # unsigned variables are represented as a - b, where a and b are nonnegative
                a, b = Sym("a"), Sym("b")
                self.unconstrained_var_subs[sym] = (
                    ConstraintTerm(1, (a,)),
                    ConstraintTerm(-1, (b,)),
                )
            elif isinstance(sym_type, T.Bool):
                # constrained to [0, 1]
                self.extra_constraints.append(
                    Constraint(
                        (
                            ConstraintTerm(1, (sym,)),
                            ConstraintTerm(-1, ()),
                            ConstraintTerm(1, (Sym("slack"))),
                        )
                    )
                )

    def make_constraint_terms(self, expr: LoopIR.expr) -> tuple[ConstraintTerm]:
        # expect that expr is int type
        if isinstance(expr, LoopIR.Read):
            assert (
                len(expr.idx) == 0
            ), "indexing not supported in assertions (yet, todo)"
            if expr.name in self.unconstrained_var_subs:
                return self.unconstrained_var_subs[expr.name]
            else:
                return (ConstraintTerm(1, (expr.name,)),)
        elif isinstance(expr, LoopIR.Const):
            return (ConstraintTerm(expr.val, ()),)
        elif isinstance(expr, LoopIR.USub):
            return tuple(term.negate() for term in self.make_constraint_terms(expr.arg))
        elif isinstance(expr, LoopIR.BinOp):
            # TODO: support mod and div using extra variables
            lhs_terms = self.make_constraint_terms(expr.lhs)
            rhs_terms = self.make_constraint_terms(expr.rhs)
            if expr.op == "+":
                return lhs_terms + rhs_terms
            elif expr.op == "-":
                return lhs_terms + tuple(term.negate() for term in rhs_terms)
            elif expr.op == "*":
                return tuple(
                    lhs_term.multiply(rhs_term)
                    for lhs_term in lhs_terms
                    for rhs_term in rhs_terms
                )
            elif expr.op in ["/", "%"]:
                div, rem = Sym("div"), Sym("rem")
                self.extra_constraints.append(
                    Constraint(
                        lhs_terms
                        + (ConstraintTerm(1, (rem,)))
                        + tuple(
                            rhs_term.multiply(ConstraintTerm(1, (div,)))
                            for rhs_term in rhs_terms
                        )
                    )
                )
                self.extra_constraints.append(
                    Constraint(
                        (
                            ConstraintTerm(-1, (rem,)),
                            ConstraintTerm(-1, (Sym("slack"),)),
                        )
                        + rhs_terms
                    )
                )
                return (ConstraintTerm(1, (rem if expr.op == "%" else div,)),)
            else:
                assert False, f"unsupported op in assertion: {expr.op}"
        elif isinstance(expr, LoopIR.StrideExpr):
            if (expr.name, expr.dim) not in self.stride_dummies:
                new_sym = Sym("stride")
                self.stride_dummies[(expr.name, expr.dim)] = new_sym
                self.nonneg_vars.add(new_sym)
            dummy = self.stride_dummies[(expr.name, expr.dim)]
            return (ConstraintTerm(1, (dummy,)),)
        else:
            assert False, f"unsupported expr"

    def make_constraints(
        self,
        expr: LoopIR.expr,
    ) -> tuple[Constraint]:
        # expect that expr is bool type
        if isinstance(expr, LoopIR.BinOp):
            if expr.op == "and":
                return self.make_constraints(expr.lhs) + self.make_constraints(expr.rhs)
            elif expr.op == "or":
                # disjunction multiplies all constraints
                lhs_constraints, rhs_constraints = self.make_constraints(
                    expr.lhs
                ), self.make_constraints(expr.rhs)
                return tuple(
                    Constraint(
                        tuple(
                            lhs_term.multiply(rhs_term)
                            for lhs_term in lhs_constraint.terms
                            for rhs_term in rhs_constraint.terms
                        )
                    )
                    for lhs_constraint in lhs_constraints
                    for rhs_constraint in rhs_constraints
                )
            else:
                return (
                    self.make_constraint_from_inequality(expr.lhs, expr.rhs, expr.op),
                )
        elif isinstance(expr, LoopIR.Read):
            assert len(expr.idx) == 0, "cannot index into boolean"
            return (
                Constraint((ConstraintTerm(1, (expr.name,)), ConstraintTerm(-1, ()))),
            )
        elif isinstance(expr, LoopIR.Const):
            if expr.val:
                return Constraint(())
            else:
                return Constraint((ConstraintTerm(1, ())))
        else:
            assert False, "only boolean expected"

    def make_constraint_from_inequality(
        self, lhs: LoopIR.expr, rhs: LoopIR.expr, op: str
    ) -> Constraint:
        lhs_terms = self.make_constraint_terms(lhs)
        rhs_terms = self.make_constraint_terms(rhs)
        main_terms = rhs_terms + tuple(term.negate() for term in lhs_terms)
        if op == "<":
            slack_terms = (
                ConstraintTerm(-1, (Sym("slack"),)),
                ConstraintTerm(-1, ()),
            )
        elif op == ">":
            slack_terms = (
                ConstraintTerm(1, (Sym("slack"),)),
                ConstraintTerm(1, ()),
            )
        elif op == "<=":
            slack_terms = (ConstraintTerm(-1, (Sym("slack"),)),)
        elif op == ">=":
            slack_terms = (ConstraintTerm(1, (Sym("slack"),)),)
        elif op == "==":
            slack_terms = ()
        else:
            assert False, "boolean ops expected"
        return Constraint(main_terms + slack_terms)

    def solve_constraints(
        self,
        constraints: tuple[Constraint],
        *,
        search_limit: int,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed=seed)
        all_constraints = constraints + tuple(self.extra_constraints)
        assignments = {}

        def solve_recursive():
            linear_constraints: list[LinearConstraint] = []
            linear_constraint_syms: set[Sym] = set()
            nonlinear_syms: set[Sym] = set()
            for constraint in all_constraints:
                assign_result = constraint.apply_assignments(assignments)
                if assign_result is not None:
                    linear_constraints.append(assign_result)
                    linear_constraint_syms |= {
                        sym for sym in assign_result.coefficients.keys()
                    }

                nonlinear_syms |= constraint.collect_nonlinear_syms()
            sym_ordering = {sym: i for i, sym in enumerate(linear_constraint_syms)}
            matrix_Ab = np.zeros(
                (len(linear_constraints), len(linear_constraint_syms) + 1),
                dtype=np.int32,
            )
            for row, linear_constraint in enumerate(linear_constraints):
                for sym, coefficient in linear_constraint.coefficients:
                    matrix_Ab[row, sym_ordering[sym]] = coefficient
                matrix_Ab[row, len(linear_constraint_syms)] = linear_constraint.offset

        for _ in range(search_limit):
            if solve_recursive():
                return assignments
            else:
                assignments = {}

        return None
