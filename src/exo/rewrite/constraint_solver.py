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

    def substitute(self, assignments: dict[Sym, int]) -> "ConstraintTerm":
        new_syms = []
        new_coefficient = self.coefficient
        for sym in self.syms:
            if sym in assignments:
                new_coefficient *= assignments[sym]
            else:
                new_syms.append(sym)
        return ConstraintTerm(new_coefficient, tuple(new_syms))

    def collect_nonlinear_syms(self) -> frozenset[Sym]:
        occurrences = set()
        result = set()
        for sym in self.syms:
            if sym in occurrences:
                result.add(sym)
            else:
                occurrences.add(sym)
        return frozenset(result)

    def pretty_print(self) -> str:
        return (
            f"{' * '.join([str(self.coefficient)] + [str(sym) for sym in self.syms])}"
        )

    def rename_syms(self, lookup: dict[Sym, Sym]) -> "ConstraintTerm":
        return ConstraintTerm(
            self.coefficient,
            tuple(lookup[sym] if sym in lookup else sym for sym in self.syms),
        )


@dataclass
class LinearConstraint:
    coefficients: dict[Sym, int]
    offset: int
    has_slack: bool


@dataclass
class Expression:
    terms: tuple[ConstraintTerm, ...]

    @staticmethod
    def from_constant(const: int) -> "Expression":
        return Expression((ConstraintTerm(const, ()),))

    @staticmethod
    def from_sym(sym: Sym) -> "Expression":
        return Expression((ConstraintTerm(1, (sym,)),))

    def negate(self) -> "Expression":
        return Expression(tuple(term.negate() for term in self.terms))

    def add(self, other: "Expression") -> "Expression":
        return Expression((*self.terms, *other.terms))

    def multiply(self, other: "Expression") -> "Expression":
        return Expression(
            tuple(
                term1.multiply(term2) for term1 in self.terms for term2 in other.terms
            )
        )

    def substitute(self, assignments: dict[Sym, int]) -> "Expression":
        coefficients: dict[tuple[Sym, ...], int] = {}
        for term in self.terms:
            sub_term = term.substitute(assignments)
            if sub_term.syms not in coefficients:
                coefficients[sub_term.syms] = 0
            coefficients[sub_term.syms] += sub_term.coefficient
        return Expression(
            tuple(
                ConstraintTerm(coefficient, syms)
                for syms, coefficient in coefficients.items()
            )
        )

    def get_trivial_result(self) -> Optional[int]:
        if len(self.terms) == 0:
            return 0
        elif len(self.terms) == 1 and len(self.terms[0].syms) == 0:
            return self.terms[0].coefficient
        return None

    def collect_syms(self) -> frozenset[Sym]:
        return frozenset(sym for term in self.terms for sym in term.syms)

    def collect_nonlinear_syms(self) -> frozenset[Sym]:
        return frozenset().union(
            *[term.collect_nonlinear_syms() for term in self.terms]
        )

    def pretty_print(self):
        return " + ".join([term.pretty_print() for term in self.terms])

    def rename_syms(self, lookup: dict[Sym, Sym]) -> "Expression":
        return Expression(tuple(term.rename_syms(lookup) for term in self.terms))


@dataclass
class Constraint:
    lhs: Expression
    has_slack: bool

    def linearize(self, assignments: dict[Sym, int]) -> Optional[LinearConstraint]:
        new_lhs = self.lhs.substitute(assignments)
        offset = 0
        coefficients = {}
        for term in new_lhs.terms:
            if len(term.syms) == 0:
                offset += term.coefficient
            elif len(term.syms) == 1:
                coefficients[term.syms[0]] = term.coefficient
            else:
                return None
        return LinearConstraint(coefficients, offset, self.has_slack)

    def collect_syms(self) -> frozenset[Sym]:
        return self.lhs.collect_syms()

    def collect_nonlinear_syms(self) -> frozenset[Sym]:
        return self.lhs.collect_nonlinear_syms()

    def lift_to_disjoint_constraint(self) -> "DisjointConstraint":
        return DisjointConstraint((ConstraintClause((self,)),))

    def invert(self) -> "DisjointConstraint":
        if self.has_slack:
            return Constraint(
                self.lhs.negate().add(Expression.from_constant(-1)),
                True,
            ).lift_to_disjoint_constraint()
        else:
            return DisjointConstraint(
                (
                    ConstraintClause(
                        (
                            Constraint(
                                self.lhs.add(Expression.from_constant(-1)),
                                True,
                            ),
                        )
                    ),
                    ConstraintClause(
                        (
                            Constraint(
                                self.lhs.negate().add(Expression.from_constant(-1)),
                                True,
                            ),
                        )
                    ),
                )
            )

    def pretty_print(self) -> str:
        return f"{self.lhs.pretty_print()} {'>=' if self.has_slack else '=='} 0"

    def substitute(self, assignments: dict[Sym, int]) -> "Constraint":
        return Constraint(self.lhs.substitute(assignments), self.has_slack)

    def get_trivial_result(self) -> Optional[bool]:
        lhs_result = self.lhs.get_trivial_result()
        if lhs_result is not None:
            return (lhs_result >= 0 and self.has_slack) or lhs_result == 0
        return None

    def rename_syms(self, lookup: dict[Sym, Sym]) -> "Constraint":
        return Constraint(self.lhs.rename_syms(lookup), self.has_slack)


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

    def collect_syms(self) -> frozenset[Sym]:
        return frozenset().union(
            *(constraint.collect_syms() for constraint in self.constraints)
        )

    def substitute(self, assignments: dict[Sym, int]) -> "ConstraintClause":
        new_constraints = []
        for constraint in self.constraints:
            new_constraint = constraint.substitute(assignments)
            trivial_result = new_constraint.get_trivial_result()
            if trivial_result is None:
                new_constraints.append(new_constraint)
            elif not trivial_result:
                return ConstraintClause((new_constraint,))
        return ConstraintClause(tuple(new_constraints))

    def get_trivial_result(self) -> Optional[bool]:
        if len(self.constraints) == 0:
            return True
        elif len(self.constraints) == 1:
            return self.constraints[0].get_trivial_result()
        return None

    def rename_syms(self, lookup: dict[Sym, Sym]) -> "ConstraintClause":
        return ConstraintClause(
            tuple(constraint.rename_syms(lookup) for constraint in self.constraints)
        )


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

    def collect_syms(self) -> frozenset[Sym]:
        return frozenset().union(*(clause.collect_syms() for clause in self.clauses))

    def substitute(self, assignments: dict[Sym, int]) -> "DisjointConstraint":
        new_clauses = []
        for clause in self.clauses:
            new_clause = clause.substitute(assignments)
            trivial_result = new_clause.get_trivial_result()
            if trivial_result is None:
                new_clauses.append(new_clause)
            elif trivial_result:
                return DisjointConstraint((new_clause,))
        return DisjointConstraint(tuple(new_clauses))

    def get_trivial_result(self) -> Optional[bool]:
        if len(self.clauses) == 0:
            return False
        elif len(self.clauses) == 1:
            return self.clauses[0].get_trivial_result()
        return None

    def rename_syms(self, lookup: dict[Sym, Sym]) -> "DisjointConstraint":
        return DisjointConstraint(
            tuple(clause.rename_syms(lookup) for clause in self.clauses)
        )


TRUE_CONSTRAINT = DisjointConstraint((ConstraintClause(()),))
FALSE_CONSTRAINT = DisjointConstraint(())


@dataclass
class Solution:
    ctxt: dict[tuple[Config, str], int]
    var_assignments: dict[Sym, int]
    substitutions: dict[Sym, int]

    def merge_solutions(self, other: "Solution", other_renaming: dict[Sym, Sym]):
        return Solution(
            self.ctxt,
            self.var_assignments,
            {
                other_renaming[key] if key in other_renaming else key: value
                for key, value in other.substitutions.items()
            }
            | self.substitutions,
        )


class ConstraintMaker:
    def __init__(self, type_map: dict[Sym, LoopIR.type]):
        self.var_subs: dict[Sym, Expression] = {}
        self.ctxt: dict[tuple[Config, str], Expression] = {}
        self.extra_constraints: list[Constraint] = []
        self.hidden_vars: set[Sym] = set()
        self.stride_dummies: dict[tuple[Sym, int], Sym] = {}
        for sym, sym_type in type_map.items():
            var_sub_result = self.make_var_sub(sym.name(), sym_type)
            if var_sub_result is not None:
                self.var_subs[sym] = var_sub_result

    def get_var_sub(self, var_sym: Sym) -> Expression:
        return self.var_subs[var_sym]

    def make_var_sub(self, name: str, var_type: LoopIR.type) -> Optional[Expression]:
        if isinstance(var_type, (T.Size, T.Stride)):
            # positive variable
            return Expression.from_sym(Sym(f"{name}_m1")).add(
                Expression.from_constant(1)
            )
        elif isinstance(var_type, (T.Int, T.Index)):
            # unsigned variables are represented as a - b, where a and b are nonnegative
            a, b = Sym(f"{name}_a"), Sym(f"{name}_b")
            return Expression.from_sym(a).add(Expression.from_sym(b).negate())
        elif isinstance(var_type, T.Bool):
            # constrained to [0, 1]
            sym = Sym(name)
            self.extra_constraints.append(
                Constraint(
                    Expression.from_sym(sym).negate().add(Expression.from_constant(1)),
                    True,
                )
            )
            return Expression.from_sym(sym)
        else:
            return None

    def make_expression(self, expr: Union[LoopIR.expr, Sym]) -> Expression:
        # expect that expr is int type
        if isinstance(expr, Sym):
            return self.var_subs[expr]
        elif isinstance(expr, LoopIR.Read):
            assert (
                len(expr.idx) == 0
            ), "indexing not supported in assertions (yet, todo)"
            return self.var_subs[expr.name]
        elif isinstance(expr, LoopIR.Const):
            return Expression.from_constant(expr.val)
        elif isinstance(expr, LoopIR.USub):
            return self.make_expression(expr.arg).negate()
        elif isinstance(expr, LoopIR.BinOp):
            # TODO: support mod and div using extra variables
            lhs = self.make_expression(expr.lhs)
            rhs = self.make_expression(expr.rhs)
            if expr.op == "+":
                return lhs.add(rhs)
            elif expr.op == "-":
                return lhs.add(rhs.negate())
            elif expr.op == "*":
                return lhs.multiply(rhs)
            elif expr.op in ["/", "%"]:
                div, rem = Sym("div"), Sym("rem")
                self.hidden_vars.update((div, rem))
                self.extra_constraints.append(
                    Constraint(
                        lhs.negate()
                        .add(Expression.from_sym(rem))
                        .add(rhs.multiply(Expression.from_sym(div))),
                        False,
                    )
                )
                self.extra_constraints.append(
                    Constraint(
                        Expression.from_sym(rem)
                        .add(Expression.from_constant(1))
                        .negate()
                        .add(rhs),
                        True,
                    )
                )
                return Expression.from_sym(rem if expr.op == "%" else div)
            else:
                assert False, f"unsupported op in assertion: {expr.op}"
        elif isinstance(expr, LoopIR.StrideExpr):
            if (expr.name, expr.dim) not in self.stride_dummies:
                new_sym = Sym("stride")
                self.stride_dummies[(expr.name, expr.dim)] = new_sym
            dummy = self.stride_dummies[(expr.name, expr.dim)]
            return Expression.from_sym(dummy)
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
            return self.ctxt[(expr.config, expr.field)]
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
            elif expr.op == "==" and isinstance(expr.lhs.type, LoopIR.Bool):
                lhs_constraints, rhs_constraints = self.make_constraint(
                    expr.lhs
                ), self.make_constraint(expr.rhs)
                return (
                    lhs_constraints.invert().intersect(rhs_constraints.invert())
                ).union(lhs_constraints.intersect(rhs_constraints))
            else:
                return self.make_constraint_from_inequality(
                    expr.lhs, expr.rhs, expr.op
                ).lift_to_disjoint_constraint()
        elif isinstance(expr, LoopIR.Read):
            assert len(expr.idx) == 0, "cannot index into boolean"
            return Constraint(
                Expression.from_sym(expr.name).add(Expression.from_constant(-1)),
                True,
            ).lift_to_disjoint_constraint()
        elif isinstance(expr, LoopIR.Const):
            return TRUE_CONSTRAINT if expr.val else FALSE_CONSTRAINT

        elif isinstance(expr, LoopIR.ReadConfig):
            if (expr.config, expr.field) not in self.ctxt:
                field_type = expr.config.lookup_type(expr.field)
                assert isinstance(field_type, LoopIR.Bool)
                var_sub_result = self.make_var_sub(
                    f"{expr.config.name()}_{expr.field}", field_type
                )
                assert (
                    var_sub_result is not None
                ), "constraints can only occur on control variables"
                self.ctxt[(expr.config, expr.field)] = var_sub_result
            return Constraint(
                self.ctxt[(expr.config, expr.field)].add(Expression.from_constant(-1)),
                False,
            ).lift_to_disjoint_constraint()
        else:
            assert False, "only boolean expected"

    def make_constraint_from_inequality(
        self, lhs: Union[LoopIR.expr, Sym], rhs: Union[LoopIR.expr, Sym], op: str
    ) -> Constraint:
        lhs_expr = self.make_expression(lhs)
        rhs_expr = self.make_expression(rhs)
        if op == "<":
            return Constraint(
                rhs_expr.add(lhs_expr.negate()).add(Expression.from_constant(-1)), True
            )
        elif op == ">":
            return Constraint(
                lhs_expr.add(rhs_expr.negate()).add(Expression.from_constant(-1)), True
            )
        elif op == "<=":
            return Constraint(rhs_expr.add(lhs_expr.negate()), True)
        elif op == ">=":
            return Constraint(lhs_expr.add(rhs_expr.negate()), True)
        elif op == "==":
            return Constraint(lhs_expr.add(rhs_expr.negate()), False)
        else:
            assert False, "boolean ops expected"

    def _make_solution_from_assignments(self, assignments: dict[Sym, int]) -> Solution:
        var_assignments = {}
        for sym, sub in self.var_subs.items():
            result = sub.substitute(assignments).get_trivial_result()
            if result is not None:
                var_assignments[sym] = result
        ctxt = {}
        for (config, field), sub in self.ctxt.items():
            result = sub.substitute(assignments).get_trivial_result()
            if result is not None:
                ctxt[(config, field)] = result
        return Solution(ctxt, var_assignments, assignments)

    def _solve_for_assignments(
        self, all_constraints: tuple[Constraint, ...], bound: int
    ) -> Union[Literal["failed", "infeasible"], dict[Sym, int]]:
        sym_universe = set()
        for constraint in all_constraints:
            sym_universe |= constraint.collect_syms()
        assignments = {}
        while len(assignments) < len(sym_universe):
            linear_constraints: list[LinearConstraint] = []
            linear_constraint_syms: set[Sym] = set()
            nonlinear_syms: set[Sym] = set()
            for constraint in all_constraints:
                linear_result = constraint.linearize(assignments)
                if linear_result is not None:
                    linear_constraints.append(linear_result)
                    linear_constraint_syms |= {
                        sym for sym in linear_result.coefficients.keys()
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
                    return "infeasible" if len(assignments) == 0 else "failed"
                vec_f += vec_d[i] / matrix_B[i, i] * matrix_V[:, i]
            if m == k:
                solution = vec_f
                if not np.all(vec_f >= 0):
                    return "infeasible" if len(assignments) == 0 else "failed"
            else:
                matrix_C = matrix_V[:, k:]
                upper_bound_matrix = np.concatenate(
                    (matrix_C[:m_nonslack, :], -matrix_C), axis=0
                )
                upper_bound_offset = np.concatenate(
                    (np.ones(m_nonslack) * bound - vec_f[:m_nonslack], vec_f),
                    axis=0,
                )
                radius_row = np.zeros((1, m - k + 1))
                radius_row[0, -1] = -1
                upper_bound_matrix_with_radius = np.concatenate(
                    (
                        np.concatenate(
                            (
                                upper_bound_matrix,
                                np.linalg.norm(upper_bound_matrix, axis=1)[
                                    :, np.newaxis
                                ],
                            ),
                            axis=1,
                        ),
                        radius_row,
                    ),
                    axis=0,
                )
                upper_bound_offset_with_radius = np.concatenate(
                    (upper_bound_offset, np.array([0])), axis=0
                )
                objective = np.zeros(m - k + 1)
                objective[-1] = -1
                lp = linprog(
                    objective,
                    A_ub=upper_bound_matrix_with_radius,
                    b_ub=upper_bound_offset_with_radius,
                    bounds=(None, None),
                    method="highs",
                )
                if not lp.success:
                    return "infeasible" if len(assignments) == 0 else "failed"
                cur_y = lp.x[: m - k]
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
                    return "infeasible" if len(assignments) == 0 else "failed"

            if len(nonlinear_syms) == 0:
                for sym in linear_constraint_syms:
                    assignments[sym] = int(solution[sym_ordering[sym]])
                for sym in sym_universe - assignments.keys():
                    assignments[sym] = np.random.randint(0, bound)
            else:
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
        return assignments

    def solve_constraint(
        self,
        disjoint_constraint: DisjointConstraint,
        *,
        partial_solution: Optional[Solution] = None,
        bound: int,
        search_limit: int,
        seed: Optional[int] = None,
    ) -> Optional[Solution]:
        if seed is not None:
            np.random.seed(seed=seed)
        if partial_solution is not None:
            disjoint_constraint = disjoint_constraint.substitute(
                partial_solution.substitutions
            )

        clauses = list(disjoint_constraint.clauses)
        for _ in range(search_limit):
            if len(clauses) == 0:
                return None
            chosen_clause = np.random.choice(clauses)
            assert isinstance(chosen_clause, ConstraintClause)
            all_constraints = chosen_clause.constraints + tuple(self.extra_constraints)
            assignment_result = self._solve_for_assignments(all_constraints, bound)
            if assignment_result == "failed":
                continue
            elif assignment_result == "infeasible":
                clauses = list(clause for clause in clauses if clause != chosen_clause)
            else:
                return self._make_solution_from_assignments(
                    ({} if partial_solution is None else partial_solution.substitutions)
                    | assignment_result
                )
        return None

    def rename_sym_set(
        self, syms: frozenset[Sym], free_vars: frozenset[Sym]
    ) -> tuple[dict[Sym, Sym], dict[Sym, Sym]]:
        var_renaming = {}
        sym_renaming = {sym: Sym(sym.name()) for sym in self.hidden_vars & syms}
        for var in free_vars:
            var_sub = self.var_subs[var]
            var_sub_syms = var_sub.collect_syms()
            if len(var_sub_syms & syms) != 0:
                sym_renaming |= {sym: Sym(sym.name()) for sym in var_sub_syms}
                renamed_var = Sym(var.name())
                var_renaming[var] = renamed_var
                self.var_subs[renamed_var] = var_sub.rename_syms(sym_renaming)
        self.extra_constraints.extend(
            tuple(
                extra_constraint.rename_syms(sym_renaming)
                for extra_constraint in self.extra_constraints
                if len(extra_constraint.collect_syms() & sym_renaming.keys()) != 0
            )
        )
        return (
            sym_renaming,
            var_renaming,
        )
