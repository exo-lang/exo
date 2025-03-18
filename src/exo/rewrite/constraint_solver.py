from dataclasses import dataclass, field
from typing import Literal, Union, Optional

from ..core.configs import Config
from ..core.prelude import Sym
from ..core.LoopIR import LoopIR, T
import numpy as np
from scipy.optimize import linprog
from hsnf import smith_normal_form
import textwrap


@dataclass
class ConstraintTerm:
    coefficient: int
    syms: tuple[Sym, ...]

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
    has_slack: bool


@dataclass
class Constraint:
    terms: tuple[ConstraintTerm, ...]
    has_slack: bool

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
        return LinearConstraint(coefficients, offset, self.has_slack)

    def collect_syms(self) -> frozenset[Sym]:
        return frozenset(sym for term in self.terms for sym in term.syms)

    def collect_nonlinear_syms(self) -> frozenset[Sym]:
        return frozenset().union(
            *[term.collect_nonlinear_syms() for term in self.terms]
        )

    def lift_to_disjoint_constraint(self) -> "DisjointConstraint":
        return DisjointConstraint((ConstraintClause((self,)),))

    def invert(self) -> "DisjointConstraint":
        if self.has_slack:
            return Constraint(
                tuple(term.negate() for term in self.terms) + (ConstraintTerm(-1, ()),),
                True,
            ).lift_to_disjoint_constraint()
        else:
            return DisjointConstraint(
                (
                    ConstraintClause(
                        (Constraint(self.terms + (ConstraintTerm(-1, ()),), True),)
                    ),
                    ConstraintClause(
                        (
                            Constraint(
                                tuple(term.negate() for term in self.terms)
                                + (ConstraintTerm(-1, ()),),
                                True,
                            ),
                        )
                    ),
                )
            )

    def pretty_print(self) -> str:
        return (
            " + ".join(
                [
                    f"{' * '.join([str(term.coefficient)] + [str(sym) for sym in term.syms])}"
                    for term in self.terms
                ]
            )
            + f" {'>=' if self.has_slack else '=='} 0"
        )


@dataclass
class ConstraintClause:
    constraints: tuple[Constraint, ...]

    def invert(self) -> "DisjointConstraint":
        acc = FALSE_CONSTRAINT
        for constraint in self.constraints:
            acc = acc.union(constraint.invert())
        return acc

    def pretty_print(self) -> str:
        lines = [
            "intersect(",
            *list(
                textwrap.indent(constraint.pretty_print(), "\t") + ","
                for constraint in self.constraints
            ),
            ")",
        ]
        return "\n".join(lines)


@dataclass
class DisjointConstraint:
    clauses: tuple[ConstraintClause, ...]

    def intersect(self, other: "DisjointConstraint"):
        return DisjointConstraint(
            tuple(
                ConstraintClause(lhs_clause.constraints + rhs_clause.constraints)
                for lhs_clause in self.clauses
                for rhs_clause in other.clauses
            )
        )

    def union(self, other: "DisjointConstraint"):
        return DisjointConstraint(self.clauses + other.clauses)

    def invert(self) -> "DisjointConstraint":
        acc = TRUE_CONSTRAINT
        for clause in self.clauses:
            acc = acc.intersect(clause.invert())
        return acc

    def pretty_print(self) -> str:
        lines = [
            "union(",
            *list(
                textwrap.indent(clause.pretty_print(), "\t") + ","
                for clause in self.clauses
            ),
            ")",
        ]
        return "\n".join(lines)


TRUE_CONSTRAINT = DisjointConstraint((ConstraintClause(()),))
FALSE_CONSTRAINT = DisjointConstraint(())


@dataclass
class Expression:
    terms: tuple[ConstraintTerm, ...]

    def apply_assignments(self, assignments: dict[Sym, int]) -> Optional[int]:
        result = 0
        for term in self.terms:
            assign_result = term.apply_assignments(assignments)
            if assign_result is None:
                return None
            else:
                coeff, target = assign_result
                if target is None:
                    result += coeff
                else:
                    return None
        return result


@dataclass
class Solution:
    ctxt: dict[tuple[Config, str], int]
    var_assignments: dict[Sym, int]


class ConstraintMaker:
    def __init__(self, type_map: dict[Sym, LoopIR.type]):
        self.var_subs: dict[Sym, Expression] = {}
        self.ctxt: dict[tuple[Config, str], Expression] = {}
        self.extra_constraints: list[Constraint] = []
        self.stride_dummies: dict[tuple[Sym, int], Sym] = {}
        for sym, sym_type in type_map.items():
            var_sub_result = self.make_var_sub(sym.name(), sym_type)
            if var_sub_result is not None:
                self.var_subs[sym] = var_sub_result

    def make_var_sub(self, name: str, var_type: LoopIR.type) -> Optional[Expression]:
        if isinstance(var_type, (T.Size, T.Stride)):
            # positive variable
            return Expression(
                (ConstraintTerm(1, (Sym(f"{name}_m1"),)), ConstraintTerm(1, ()))
            )
        elif isinstance(var_type, (T.Int, T.Index)):
            # unsigned variables are represented as a - b, where a and b are nonnegative
            a, b = Sym(f"{name}_a"), Sym(f"{name}_b")
            return Expression((ConstraintTerm(1, (a,)), ConstraintTerm(-1, (b,))))
        elif isinstance(var_type, T.Bool):
            # constrained to [0, 1]
            sym = Sym(name)
            self.extra_constraints.append(
                Constraint(
                    (
                        ConstraintTerm(-1, (sym,)),
                        ConstraintTerm(1, ()),
                    ),
                    True,
                )
            )
            return Expression((ConstraintTerm(1, (sym,)),))
        else:
            return None

    def make_constraint_terms(
        self, expr: Union[LoopIR.expr, Sym]
    ) -> tuple[ConstraintTerm, ...]:
        # expect that expr is int type
        if isinstance(expr, Sym):
            return self.var_subs[expr].terms
        elif isinstance(expr, LoopIR.Read):
            assert (
                len(expr.idx) == 0
            ), "indexing not supported in assertions (yet, todo)"
            return self.var_subs[expr.name].terms
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
                        tuple(lhs_term.negate() for lhs_term in lhs_terms)
                        + (ConstraintTerm(1, (rem,)),)
                        + tuple(
                            rhs_term.multiply(ConstraintTerm(1, (div,)))
                            for rhs_term in rhs_terms
                        ),
                        False,
                    )
                )
                self.extra_constraints.append(
                    Constraint(
                        (
                            ConstraintTerm(-1, (rem,)),
                            ConstraintTerm(-1, ()),
                        )
                        + rhs_terms,
                        True,
                    )
                )
                return (ConstraintTerm(1, (rem if expr.op == "%" else div,)),)
            else:
                assert False, f"unsupported op in assertion: {expr.op}"
        elif isinstance(expr, LoopIR.StrideExpr):
            if (expr.name, expr.dim) not in self.stride_dummies:
                new_sym = Sym("stride")
                self.stride_dummies[(expr.name, expr.dim)] = new_sym
            dummy = self.stride_dummies[(expr.name, expr.dim)]
            return (ConstraintTerm(1, (dummy,)),)
        elif isinstance(expr, LoopIR.ReadConfig):
            if (expr.config, expr.field) not in self.ctxt:
                field_type = expr.config.lookup_type(expr.field)
                var_sub_result = self.make_var_sub(
                    f"{expr.config.name()}_{expr.field}", field_type
                )
                assert (
                    var_sub_result is not None
                ), "constraints can only occur on control variables"
                self.ctxt[(expr.config, expr.field)] = var_sub_result
            return self.ctxt[(expr.config, expr.field)].terms
        else:
            assert False, f"unsupported expr"

    def make_constraint(
        self,
        expr: LoopIR.expr,
    ) -> DisjointConstraint:
        # expect that expr is bool type
        if isinstance(expr, LoopIR.BinOp):
            if expr.op == "and":
                lhs_constraints, rhs_constraints = self.make_constraint(
                    expr.lhs
                ), self.make_constraint(expr.rhs)
                return lhs_constraints.intersect(rhs_constraints)
            elif expr.op == "or":
                lhs_constraints, rhs_constraints = self.make_constraint(
                    expr.lhs
                ), self.make_constraint(expr.rhs)
                return lhs_constraints.union(rhs_constraints)
            else:
                return self.make_constraint_from_inequality(
                    expr.lhs, expr.rhs, expr.op
                ).lift_to_disjoint_constraint()
        elif isinstance(expr, LoopIR.Read):
            assert len(expr.idx) == 0, "cannot index into boolean"
            return Constraint(
                (
                    ConstraintTerm(1, (expr.name,)),
                    ConstraintTerm(-1, ()),
                ),
                True,
            ).lift_to_disjoint_constraint()
        elif isinstance(expr, LoopIR.Const):
            return TRUE_CONSTRAINT if expr.val else FALSE_CONSTRAINT
        else:
            assert False, "only boolean expected"

    def make_constraint_from_inequality(
        self, lhs: Union[LoopIR.expr, Sym], rhs: Union[LoopIR.expr, Sym], op: str
    ) -> Constraint:
        lhs_terms = self.make_constraint_terms(lhs)
        rhs_terms = self.make_constraint_terms(rhs)
        has_slack = True
        if op == "<":
            terms = (
                rhs_terms
                + tuple(term.negate() for term in lhs_terms)
                + (ConstraintTerm(-1, ()),)
            )
        elif op == ">":
            terms = (
                lhs_terms
                + tuple(term.negate() for term in rhs_terms)
                + (ConstraintTerm(-1, ()),)
            )
        elif op == "<=":
            terms = rhs_terms + tuple(term.negate() for term in lhs_terms)
        elif op == ">=":
            terms = lhs_terms + tuple(term.negate() for term in rhs_terms)
        elif op == "==":
            has_slack = False
            terms = rhs_terms + tuple(term.negate() for term in lhs_terms)
        else:
            assert False, "boolean ops expected"
        return Constraint(terms, has_slack)

    def solve_constraint(
        self,
        disjoint_constraint: DisjointConstraint,
        *,
        bound: int,
        search_limit: int,
        seed: Optional[int] = None,
    ) -> Optional[Solution]:
        if seed is not None:
            np.random.seed(seed=seed)
        if len(disjoint_constraint.clauses) == 0:
            return None
        chosen_clause = np.random.choice(list(disjoint_constraint.clauses))
        assert isinstance(chosen_clause, ConstraintClause)
        all_constraints = chosen_clause.constraints + tuple(self.extra_constraints)
        assignments = {}
        sym_universe = set()
        for constraint in all_constraints:
            sym_universe |= constraint.collect_syms()

        def solve_helper():
            while len(assignments) < len(sym_universe):
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
                nonlinear_syms -= assignments.keys()
                priority_syms = nonlinear_syms & linear_constraint_syms
                if len(priority_syms) == 0 and len(nonlinear_syms) != 0:
                    chosen_sym = np.random.choice(
                        sorted(list(nonlinear_syms), key=lambda sym: sym._id)
                    )
                    assignments[chosen_sym] = np.random.randint(0, bound)
                    continue
                sym_ordering = {
                    sym: i
                    for i, sym in enumerate(
                        sorted(
                            list(linear_constraint_syms),
                            key=lambda sym: sym._id,
                        )
                    )
                }
                n = len(linear_constraints)
                m_nonslack = len(linear_constraint_syms)
                matrix_A = np.zeros(
                    (n, m_nonslack),
                    dtype=np.int32,
                )
                m = m_nonslack
                vec_b = np.zeros(n, dtype=np.int32)
                for row, linear_constraint in enumerate(linear_constraints):
                    for sym, coefficient in linear_constraint.coefficients.items():
                        matrix_A[row, sym_ordering[sym]] = coefficient
                    if linear_constraint.has_slack:
                        slack_col = np.zeros((n, 1), dtype=np.int32)
                        slack_col[row, 0] = -1
                        matrix_A = np.hstack((matrix_A, slack_col))
                        m += 1
                    vec_b[row] = -linear_constraint.offset
                matrix_B, matrix_U, matrix_V = smith_normal_form(matrix_A)
                vec_d = matrix_U @ vec_b
                k = min(n, m)
                vec_f = np.zeros(m)
                for i in range(min(n, m)):
                    if matrix_B[i, i] == 0:
                        k = i
                        break
                    if vec_d[i] % matrix_B[i, i] != 0:
                        return False
                    vec_f += vec_d[i] / matrix_B[i, i] * matrix_V[:, i]
                if m == k:
                    solution = vec_f
                    if not np.all(vec_f >= 0):
                        return False
                else:
                    matrix_C = matrix_V[:, k:]
                    upper_bound_matrix = np.concatenate(
                        (matrix_C[:m_nonslack, :], -matrix_C), axis=0
                    )
                    upper_bound_offset = np.concatenate(
                        (np.ones(m_nonslack) * bound - vec_f[:m_nonslack], vec_f),
                        axis=0,
                    )
                    lp = linprog(
                        np.zeros(m - k),
                        A_ub=upper_bound_matrix,
                        b_ub=upper_bound_offset,
                        bounds=(None, None),
                    )
                    if not lp.success:
                        return False
                    cur_y = lp.x
                    har_iter = 50
                    last_int_y = None
                    for _ in range(har_iter):
                        direction = np.random.normal(size=m - k)
                        direction = direction / np.linalg.norm(direction)
                        lower_bounds = -matrix_C @ cur_y - vec_f
                        upper_bounds = lower_bounds + bound
                        upper_bounds[m_nonslack:] = -np.nan
                        coefficients = matrix_C @ direction
                        lower_bounds = lower_bounds[coefficients != 0]
                        upper_bounds = upper_bounds[coefficients != 0]
                        coefficients = coefficients[coefficients != 0]
                        max_lambda = np.nanmin(
                            np.where(coefficients < 0, lower_bounds, upper_bounds)
                            / coefficients
                        )
                        min_lambda = np.nanmax(
                            np.where(coefficients >= 0, lower_bounds, upper_bounds)
                            / coefficients
                        )
                        new_y = cur_y + direction * (
                            np.random.rand() * (max_lambda - min_lambda) + min_lambda
                        )
                        new_int_y = np.round(new_y)
                        cur_y = new_y
                        if np.all(upper_bound_matrix @ new_int_y <= upper_bound_offset):
                            last_int_y = new_int_y
                    if last_int_y is not None:
                        solution = matrix_C @ last_int_y + vec_f
                    else:
                        return False

                chosen_sym = None
                if len(priority_syms) != 0:
                    chosen_sym = np.random.choice(
                        sorted(list(priority_syms), key=lambda sym: sym._id)
                    )
                elif len(linear_constraint_syms) != 0:
                    chosen_sym = np.random.choice(
                        sorted(list(linear_constraint_syms), key=lambda sym: sym._id)
                    )
                if chosen_sym is None:
                    free_syms = (
                        sym_universe
                        - linear_constraint_syms
                        - assignments.keys()
                        - nonlinear_syms
                    )
                    chosen_sym = np.random.choice(
                        sorted(list(free_syms), key=lambda sym: sym._id)
                    )
                    assignments[chosen_sym] = np.random.randint(0, bound)
                else:
                    assignments[chosen_sym] = int(solution[sym_ordering[chosen_sym]])
            return True

        for _ in range(search_limit):
            if solve_helper():
                var_assignments = {}
                for sym, sub in self.var_subs.items():
                    result = sub.apply_assignments(assignments)
                    if result is not None:
                        var_assignments[sym] = result
                ctxt = {}
                for (config, field), sub in self.ctxt.items():
                    result = sub.apply_assignments(assignments)
                    if result is not None:
                        ctxt[(config, field)] = result
                return Solution(ctxt, var_assignments)
            else:
                assignments = {}

        return None
