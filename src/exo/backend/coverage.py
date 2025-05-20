from dataclasses import dataclass, field
from itertools import groupby
from typing import Generator, Iterable, Optional, Union
import numpy as np

from ..rewrite.constraint_solver import (
    Constraint,
    ConstraintMaker,
    ConstraintTerm,
    DisjointConstraint,
    TRUE_CONSTRAINT,
    Expression,
    Solution,
)
from ..core.prelude import Sym
from ..core.internal_cursors import Node, NodePath


@dataclass
class CoverageProgress:
    covered_cases: int
    total_cases: int

    def merge(self, other: "CoverageProgress") -> "CoverageProgress":
        return CoverageProgress(
            self.covered_cases + other.covered_cases,
            self.total_cases + other.total_cases,
        )

    def is_finished(self) -> bool:
        return self.covered_cases == self.total_cases


@dataclass
class CoverageSolverState:
    current_constraint: DisjointConstraint
    is_base_constraint: bool
    current_solution: Solution
    cm: ConstraintMaker
    free_vars: frozenset[Sym]
    bound: int
    search_limit: int

    def update_solution(
        self, new_constraint: DisjointConstraint, new_solution: Solution
    ):
        return CoverageSolverState(
            new_constraint,
            False,
            new_solution,
            self.cm,
            self.free_vars,
            self.bound,
            self.search_limit,
        )


@dataclass(order=True, frozen=True)
class IndexedFiller:
    index: int
    placefiller: str


@dataclass
class CoverageSkeletonBranch:
    true_child: "CoverageSkeletonNode"
    false_child: "CoverageSkeletonNode"

    def write_coverage_declarations(self, decls: dict[Sym, str]):
        self.true_child.write_coverage_declarations(decls)
        self.false_child.write_coverage_declarations(decls)

    def get_indexed_fillers(self) -> Generator[IndexedFiller, None, None]:
        yield from self.true_child.get_indexed_fillers()
        yield from self.false_child.get_indexed_fillers()

    def get_coverage_syms(self) -> frozenset[Sym]:
        return (
            self.true_child.get_coverage_syms() | self.false_child.get_coverage_syms()
        )

    def update_coverage(self, coverage_result: dict[str, Union[bool, memoryview]]):
        self.true_child.update_coverage(coverage_result)
        self.false_child.update_coverage(coverage_result)

    def get_coverage_progress(self) -> CoverageProgress:
        return self.true_child.get_coverage_progress().merge(
            self.false_child.get_coverage_progress()
        )

    def solve_coverage(self, state: CoverageSolverState) -> CoverageSolverState:
        uncovered_path = None
        if self.true_child.visited and not self.false_child.visited:
            uncovered_path = False
        elif self.false_child.visited and not self.true_child.visited:
            uncovered_path = True

        if uncovered_path is not None:
            path_constraint = (
                self.true_child.get_complete_constraint()
                if uncovered_path
                else self.false_child.get_complete_constraint()
            )
            sym_renaming, _ = state.cm.rename_sym_set(
                path_constraint.collect_syms(),
                state.free_vars,
            )
            new_constraint = state.current_constraint.intersect(
                path_constraint.rename_syms(sym_renaming)
            )
            new_solution = state.cm.solve_constraint(
                new_constraint, bound=state.bound, search_limit=state.search_limit
            )
            if new_solution is None and state.is_base_constraint:
                (self.true_child if uncovered_path else self.false_child).mark_visited()
            if new_solution is not None:
                if uncovered_path:
                    self.true_child.visited = True
                else:
                    self.false_child.visited = True
                return state.update_solution(new_constraint, new_solution)
        elif self.true_child.visited and self.false_child.visited:
            return self.false_child.solve_coverage(
                self.true_child.solve_coverage(state)
            )
        return state


@dataclass
class CoverageSkeletonNode:
    coverage_sym: Optional[Sym]
    parent_edge: Optional[tuple["CoverageSkeletonNode", DisjointConstraint]]
    indexed_fillers: tuple[IndexedFiller, ...]
    branches: list[CoverageSkeletonBranch] = field(default_factory=lambda: [])
    visited: bool = False  # mutable

    def write_coverage_declarations(self, decls: dict[Sym, str]):
        if self.coverage_sym is not None:
            decls[self.coverage_sym] = "false"
        for branch in self.branches:
            branch.write_coverage_declarations(decls)

    def get_indexed_fillers(self) -> Generator[IndexedFiller, None, None]:
        for indexed_filler in self.indexed_fillers:
            yield indexed_filler
        for branch in self.branches:
            yield from branch.get_indexed_fillers()

    def get_coverage_syms(self) -> frozenset[Sym]:
        return frozenset(
            (self.coverage_sym,) if self.coverage_sym is not None else ()
        ).union(*tuple(branch.get_coverage_syms() for branch in self.branches))

    def update_coverage(self, coverage_result: dict[str, Union[bool, memoryview]]):
        if self.coverage_sym is None:
            self.visited = True
        else:
            covered = coverage_result[repr(self.coverage_sym)]
            assert isinstance(covered, bool)
            self.visited |= covered
        for branch in self.branches:
            branch.update_coverage(coverage_result)

    def get_complete_constraint(self) -> DisjointConstraint:
        current_edge = self.parent_edge
        result = TRUE_CONSTRAINT
        while current_edge is not None:
            result = result.intersect(current_edge[1])
            current_edge = current_edge[0].parent_edge
        return result

    def get_coverage_progress(self) -> CoverageProgress:
        result = CoverageProgress(1 if self.visited else 0, 1)
        for branch in self.branches:
            result = result.merge(branch.get_coverage_progress())
        return result

    def solve_coverage(self, state: CoverageSolverState) -> CoverageSolverState:
        current_state = state
        for branch in self.branches:
            current_state = branch.solve_coverage(current_state)
        return current_state

    def mark_visited(self):
        self.visited = True
        for branch in self.branches:
            branch.true_child.mark_visited()
            branch.false_child.mark_visited()


@dataclass
class MemoryAccess:
    coverage_sym: Sym
    node: CoverageSkeletonNode
    index: tuple[Expression, ...]
    indexed_fillers: tuple[IndexedFiller, ...]

    def get_indexed_fillers(self) -> Generator[IndexedFiller, None, None]:
        for indexed_filler in self.indexed_fillers:
            yield indexed_filler

    def make_renamed_constraint_and_indices(
        self, state: CoverageSolverState
    ) -> tuple[DisjointConstraint, tuple[Expression, ...], dict[Sym, Sym]]:
        path_constraint = self.node.get_complete_constraint()
        sym_renaming, var_renaming = state.cm.rename_sym_set(
            path_constraint.collect_syms().union(
                *(index_expr.collect_syms() for index_expr in self.index)
            ),
            state.free_vars,
        )
        return (
            path_constraint.rename_syms(sym_renaming),
            tuple(index_expr.rename_syms(sym_renaming) for index_expr in self.index),
            var_renaming,
        )


@dataclass
class MemoryAccessPair:
    access1: MemoryAccess
    access2: MemoryAccess
    visited_aliasing: bool = False  # mutable
    visited_nonaliasing: bool = False  # mutable

    def get_indexed_fillers(self) -> Generator[IndexedFiller, None, None]:
        yield from self.access1.get_indexed_fillers()
        yield from self.access2.get_indexed_fillers()

    def get_coverage_syms(self) -> frozenset[Sym]:
        return frozenset((self.access1.coverage_sym, self.access2.coverage_sym))

    def update_coverage(self, coverage_result: dict[str, Union[bool, memoryview]]):
        access1_view = coverage_result[repr(self.access1.coverage_sym)]
        access2_view = coverage_result[repr(self.access2.coverage_sym)]
        if isinstance(access1_view, memoryview):
            assert isinstance(access2_view, memoryview)
            access1_arr = np.asarray(access1_view)
            access2_arr = np.asarray(access2_view)
            self.visited_aliasing |= np.any(access1_arr & access2_arr)
            self.visited_nonaliasing |= np.any(access1_arr & ~access2_arr) and np.any(
                ~access1_arr & access2_arr
            )
        else:
            assert isinstance(access2_view, bool)
            aliased = access1_view and access2_view
            self.visited_aliasing |= aliased  # nonaliasing not possible without tensor

    def get_coverage_progress(self) -> CoverageProgress:
        return CoverageProgress(
            (1 if self.visited_aliasing else 0)
            + (1 if self.visited_nonaliasing else 0),
            2,
        )

    def solve_coverage(self, state: CoverageSolverState) -> CoverageSolverState:
        uncovered_path = None
        if self.visited_aliasing and not self.visited_nonaliasing:
            uncovered_path = False
        elif self.visited_nonaliasing and not self.visited_aliasing:
            uncovered_path = True

        if uncovered_path is not None:
            (
                access1_path_constraint,
                access1_indices,
                _,
            ) = self.access1.make_renamed_constraint_and_indices(state)
            (
                access2_path_constraint,
                access2_indices,
                _,
            ) = self.access2.make_renamed_constraint_and_indices(state)
            path_constraints = access1_path_constraint.intersect(
                access2_path_constraint
            )
            alias_constraint = TRUE_CONSTRAINT
            for index1, index2 in zip(access1_indices, access2_indices):
                alias_constraint = alias_constraint.intersect(
                    Constraint(
                        index1.negate().add(index2),
                        False,
                    ).lift_to_disjoint_constraint()
                )
            if not uncovered_path:
                alias_constraint = alias_constraint.invert()
            new_constraint = state.current_constraint.intersect(
                path_constraints
            ).intersect(alias_constraint)
            new_solution = state.cm.solve_constraint(
                new_constraint, bound=state.bound, search_limit=state.search_limit
            )
            if (
                new_solution is None and state.is_base_constraint
            ) or new_solution is not None:
                if uncovered_path:
                    self.visited_aliasing = True
                else:
                    self.visited_nonaliasing = True
            if new_solution is not None:
                return state.update_solution(new_constraint, new_solution)
        elif not self.visited_nonaliasing and not self.visited_aliasing:
            (
                access1_path_constraint,
                access1_indices,
                _,
            ) = self.access1.make_renamed_constraint_and_indices(state)
            (
                access2_path_constraint,
                access2_indices,
                _,
            ) = self.access2.make_renamed_constraint_and_indices(state)
            path_constraints = access1_path_constraint.intersect(
                access2_path_constraint
            )
            new_constraint = state.current_constraint.intersect(path_constraints)
            new_solution = state.cm.solve_constraint(
                new_constraint, bound=state.bound, search_limit=state.search_limit
            )
            if new_solution is None:
                self.visited_aliasing = True
                self.visited_nonaliasing = True
            else:
                return state.update_solution(new_constraint, new_solution)

        return state


@dataclass
class FailureCondition:
    coverage_sym: Sym
    fail_cond: DisjointConstraint
    node: CoverageSkeletonNode
    indexed_fillers: tuple[IndexedFiller, ...]
    visited_failure: bool = False  # mutable

    def get_indexed_fillers(self) -> Generator[IndexedFiller, None, None]:
        for indexed_filler in self.indexed_fillers:
            yield indexed_filler

    def get_coverage_syms(self) -> frozenset[Sym]:
        return frozenset((self.coverage_sym,))

    def update_coverage(self, coverage_result: dict[str, Union[bool, memoryview]]):
        failed = coverage_result[repr(self.coverage_sym)]
        assert isinstance(failed, bool)
        self.visited_failure |= failed

    def get_coverage_progress(self) -> CoverageProgress:
        return CoverageProgress(
            1 if self.visited_failure else 0,
            1,
        )

    def solve_coverage(self, state: CoverageSolverState) -> CoverageSolverState:
        if not self.visited_failure:
            path_constraint = self.node.get_complete_constraint().intersect(
                self.fail_cond
            )
            sym_renaming, _ = state.cm.rename_sym_set(
                path_constraint.collect_syms(),
                state.free_vars,
            )
            new_constraint = state.current_constraint.intersect(
                path_constraint.rename_syms(sym_renaming)
            )
            new_solution = state.cm.solve_constraint(
                new_constraint, bound=state.bound, search_limit=state.search_limit
            )
            if (
                new_solution is None and state.is_base_constraint
            ) or new_solution is not None:
                self.visited_failure = True
            if new_solution is not None:
                return state.update_solution(new_constraint, new_solution)
        return state


@dataclass
class SymbolicPoint:
    index: Expression

    def collect_syms(self) -> frozenset[Sym]:
        return self.index.collect_syms()

    def rename_syms(self, lookup: dict[Sym, Sym]) -> "SymbolicPoint":
        return SymbolicPoint(self.index.rename_syms(lookup))


@dataclass
class SymbolicSlice:
    lower_bound: Expression
    upper_bound: Expression

    def collect_syms(self) -> frozenset[Sym]:
        return self.lower_bound.collect_syms() | self.upper_bound.collect_syms()

    def rename_syms(self, lookup: dict[Sym, Sym]) -> "SymbolicSlice":
        return SymbolicSlice(
            self.lower_bound.rename_syms(lookup), self.upper_bound.rename_syms(lookup)
        )


SymbolicWindowIndex = Union[SymbolicPoint, SymbolicSlice]


# for stage_mem
@dataclass
class StagingOverlap:
    overlap_sym: Sym
    disjoint_sym: Sym
    staged_window: tuple[SymbolicWindowIndex, ...]
    access_window: tuple[SymbolicWindowIndex, ...]
    node: CoverageSkeletonNode
    access_cursor: NodePath
    indexed_fillers: tuple[IndexedFiller, ...]
    has_overlap: bool = False
    has_disjoint_access: bool = False
    visited_overlap: bool = False
    visited_disjoint: bool = False

    def get_indexed_fillers(self) -> Generator[IndexedFiller, None, None]:
        for indexed_filler in self.indexed_fillers:
            yield indexed_filler

    def get_coverage_syms(self) -> frozenset[Sym]:
        return frozenset((self.overlap_sym, self.disjoint_sym))

    def update_coverage(self, coverage_result: dict[str, Union[bool, memoryview]]):
        overlap = coverage_result[repr(self.overlap_sym)]
        disjoint = coverage_result[repr(self.disjoint_sym)]
        assert isinstance(overlap, bool) and isinstance(disjoint, bool)
        self.has_overlap |= overlap
        self.visited_overlap |= overlap
        self.has_disjoint_access |= disjoint
        self.visited_disjoint |= disjoint

    def get_coverage_progress(self) -> CoverageProgress:
        return CoverageProgress(
            (1 if self.visited_overlap else 0) + (1 if self.visited_disjoint else 0),
            2,
        )

    def solve_coverage(self, state: CoverageSolverState) -> CoverageSolverState:
        uncovered_overlap = None
        if self.visited_overlap and not self.visited_disjoint:
            uncovered_overlap = False
        elif self.visited_disjoint and not self.visited_overlap:
            uncovered_overlap = True

        if uncovered_overlap is not None:
            path_constraint = self.node.get_complete_constraint()
            sym_renaming, _ = state.cm.rename_sym_set(
                path_constraint.collect_syms().union(
                    *(
                        (access_window_idx.collect_syms())
                        for access_window_idx in self.access_window
                    ),
                    *(
                        staged_window_idx.collect_syms()
                        for staged_window_idx in self.staged_window
                    ),
                ),
                state.free_vars,
            )
            path_constraint = path_constraint.rename_syms(sym_renaming)
            overlap_constraint = TRUE_CONSTRAINT
            for access_idx, staged_idx in zip(
                map(lambda idx: idx.rename_syms(sym_renaming), self.access_window),
                map(lambda idx: idx.rename_syms(sym_renaming), self.staged_window),
            ):
                if isinstance(staged_idx, SymbolicPoint) and isinstance(
                    access_idx, SymbolicPoint
                ):
                    index_overlap_constraint = Constraint(
                        access_idx.index.negate().add(staged_idx.index),
                        False,
                    ).lift_to_disjoint_constraint()
                elif isinstance(staged_idx, SymbolicSlice) and isinstance(
                    access_idx, SymbolicPoint
                ):
                    index_overlap_constraint = (
                        Constraint(
                            staged_idx.lower_bound.negate().add(access_idx.index),
                            True,
                        )
                        .lift_to_disjoint_constraint()
                        .intersect(
                            Constraint(
                                access_idx.index.negate()
                                .add(staged_idx.upper_bound)
                                .add(Expression.from_constant(-1)),
                                True,
                            ).lift_to_disjoint_constraint()
                        )
                    )
                elif isinstance(staged_idx, SymbolicPoint) and isinstance(
                    access_idx, SymbolicSlice
                ):
                    index_overlap_constraint = (
                        Constraint(
                            access_idx.lower_bound.negate().add(staged_idx.index),
                            True,
                        )
                        .lift_to_disjoint_constraint()
                        .intersect(
                            Constraint(
                                staged_idx.index.negate()
                                .add(access_idx.upper_bound)
                                .add(Expression.from_constant(-1)),
                                True,
                            ).lift_to_disjoint_constraint()
                        )
                    )
                elif isinstance(staged_idx, SymbolicSlice) and isinstance(
                    access_idx, SymbolicSlice
                ):
                    index_overlap_constraint = (
                        Constraint(
                            staged_idx.lower_bound.negate()
                            .add(access_idx.upper_bound)
                            .add(Expression.from_constant(-1)),
                            True,
                        )
                        .lift_to_disjoint_constraint()
                        .intersect(
                            Constraint(
                                access_idx.lower_bound.negate()
                                .add(staged_idx.upper_bound)
                                .add(Expression.from_constant(-1)),
                                True,
                            ).lift_to_disjoint_constraint()
                        )
                    )
                else:
                    assert False
                overlap_constraint = overlap_constraint.intersect(
                    index_overlap_constraint
                )
            if not uncovered_overlap:
                overlap_constraint = overlap_constraint.invert()
            new_constraint = state.current_constraint.intersect(
                path_constraint
            ).intersect(overlap_constraint)
            new_solution = state.cm.solve_constraint(
                new_constraint, bound=state.bound, search_limit=state.search_limit
            )
            if (
                new_solution is None and state.is_base_constraint
            ) or new_solution is not None:
                if uncovered_overlap:
                    self.visited_overlap = True
                else:
                    self.visited_disjoint = True
            if new_solution is not None:
                return state.update_solution(new_constraint, new_solution)
        return state


@dataclass
class ParallelAccess:
    node: CoverageSkeletonNode
    index: tuple[Expression, ...]
    indexed_fillers: tuple[IndexedFiller, ...]

    def get_indexed_fillers(self) -> Generator[IndexedFiller, None, None]:
        for indexed_filler in self.indexed_fillers:
            yield indexed_filler

    def make_renamed_constraint_and_indices(
        self, state: CoverageSolverState
    ) -> tuple[DisjointConstraint, tuple[Expression, ...], dict[Sym, Sym]]:
        path_constraint = self.node.get_complete_constraint()
        sym_renaming, var_renaming = state.cm.rename_sym_set(
            path_constraint.collect_syms().union(
                *(index_expr.collect_syms() for index_expr in self.index)
            ),
            state.free_vars,
        )
        return (
            path_constraint.rename_syms(sym_renaming),
            tuple(index_expr.rename_syms(sym_renaming) for index_expr in self.index),
            var_renaming,
        )


@dataclass
class ParallelAccessPair:
    coverage_sym: Sym
    iter_sym: Sym
    access1: ParallelAccess
    access2: ParallelAccess
    indexed_fillers: tuple[IndexedFiller, ...]
    has_aliased: bool = False
    visited_aliasing: bool = False  # mutable

    def get_indexed_fillers(self) -> Generator[IndexedFiller, None, None]:
        yield from self.access1.get_indexed_fillers()
        yield from self.access2.get_indexed_fillers()
        for indexed_filler in self.indexed_fillers:
            yield indexed_filler

    def get_coverage_syms(self) -> frozenset[Sym]:
        return frozenset((self.coverage_sym,))

    def update_coverage(self, coverage_result: dict[str, Union[bool, memoryview]]):
        aliased = coverage_result[repr(self.coverage_sym)]
        assert isinstance(aliased, bool)
        self.has_aliased |= aliased
        self.visited_aliasing |= aliased

    def get_coverage_progress(self) -> CoverageProgress:
        return CoverageProgress(
            1 if self.visited_aliasing else 0,
            1,
        )

    def solve_coverage(self, state: CoverageSolverState) -> CoverageSolverState:
        if not self.visited_aliasing:
            (
                access1_path_constraint,
                access1_indices,
                access1_var_renaming,
            ) = self.access1.make_renamed_constraint_and_indices(state)
            (
                access2_path_constraint,
                access2_indices,
                access2_var_renaming,
            ) = self.access2.make_renamed_constraint_and_indices(state)
            path_constraints = access1_path_constraint.intersect(
                access2_path_constraint
            )
            alias_constraint = TRUE_CONSTRAINT
            for index1, index2 in zip(access1_indices, access2_indices):
                alias_constraint = alias_constraint.intersect(
                    Constraint(
                        index1.negate().add(index2),
                        False,
                    ).lift_to_disjoint_constraint()
                )

            different_iteration_constraint = TRUE_CONSTRAINT
            if (
                self.iter_sym in access1_var_renaming
                and self.iter_sym in access2_var_renaming
            ):
                different_iteration_constraint = Constraint(
                    Expression.from_sym(access1_var_renaming[self.iter_sym])
                    .negate()
                    .add(Expression.from_sym(access2_var_renaming[self.iter_sym])),
                    False,
                ).invert()

            new_constraint = (
                state.current_constraint.intersect(path_constraints)
                .intersect(alias_constraint)
                .intersect(different_iteration_constraint)
            )
            new_solution = state.cm.solve_constraint(
                new_constraint, bound=state.bound, search_limit=state.search_limit
            )
            if (
                new_solution is None and state.is_base_constraint
            ) or new_solution is not None:
                self.visited_aliasing = True
            if new_solution is not None:
                return state.update_solution(new_constraint, new_solution)
        return state


@dataclass
class CoverageSkeleton:
    roots: tuple[CoverageSkeletonNode, ...]
    aliasable_accesses: tuple[MemoryAccessPair, ...]
    failure_conditions: tuple[FailureCondition, ...]
    staging_overlaps: tuple[StagingOverlap, ...]
    parallel_accesses: tuple[ParallelAccessPair, ...]
    free_vars: frozenset[Sym]

    def merge(self, other: "CoverageSkeleton") -> "CoverageSkeleton":
        return CoverageSkeleton(
            self.roots + other.roots,
            self.aliasable_accesses + other.aliasable_accesses,
            self.failure_conditions + other.failure_conditions,
            self.staging_overlaps + other.staging_overlaps,
            self.parallel_accesses + other.parallel_accesses,
            self.free_vars | other.free_vars,
        )

    def get_indexed_fillers(self) -> Generator[IndexedFiller, None, None]:
        for root in self.roots:
            yield from root.get_indexed_fillers()
        for aliasable_access in self.aliasable_accesses:
            yield from aliasable_access.get_indexed_fillers()
        for failure_condition in self.failure_conditions:
            yield from failure_condition.get_indexed_fillers()
        for staging_overlap in self.staging_overlaps:
            yield from staging_overlap.get_indexed_fillers()
        for parallel_access in self.parallel_accesses:
            yield from parallel_access.get_indexed_fillers()

    def get_coverage_syms(self) -> frozenset[Sym]:
        return frozenset().union(
            *tuple(root_node.get_coverage_syms() for root_node in self.roots),
            *tuple(
                aliasable_access.get_coverage_syms()
                for aliasable_access in self.aliasable_accesses
            ),
            *tuple(
                failure_condition.get_coverage_syms()
                for failure_condition in self.failure_conditions
            ),
            *tuple(
                staging_overlap.get_coverage_syms()
                for staging_overlap in self.staging_overlaps
            ),
            *tuple(
                parallel_access.get_coverage_syms()
                for parallel_access in self.parallel_accesses
            ),
        )

    def update_coverage(self, coverage_result: dict[str, Union[bool, memoryview]]):
        for root_node in self.roots:
            root_node.update_coverage(coverage_result)
        for aliasable_access in self.aliasable_accesses:
            aliasable_access.update_coverage(coverage_result)
        for failure_condition in self.failure_conditions:
            failure_condition.update_coverage(coverage_result)
        for staging_overlap in self.staging_overlaps:
            staging_overlap.update_coverage(coverage_result)
        for parallel_access in self.parallel_accesses:
            parallel_access.update_coverage(coverage_result)

    def get_coverage_progress(self) -> CoverageProgress:
        result = CoverageProgress(0, 0)
        for root_node in self.roots:
            result = root_node.get_coverage_progress()
        for aliasable_access in self.aliasable_accesses:
            result = result.merge(aliasable_access.get_coverage_progress())
        for failure_condition in self.failure_conditions:
            result = result.merge(failure_condition.get_coverage_progress())
        for staging_overlap in self.staging_overlaps:
            result = result.merge(staging_overlap.get_coverage_progress())
        for parallel_access in self.parallel_accesses:
            result = result.merge(parallel_access.get_coverage_progress())
        return result

    def solve_constraint_with_coverage(
        self,
        cm: ConstraintMaker,
        base_constraint: DisjointConstraint,
        *,
        bound: int,
        search_limit: int,
    ) -> Optional[Solution]:
        base_solution = cm.solve_constraint(
            base_constraint, bound=bound, search_limit=search_limit
        )
        if base_solution is None:
            return None
        state = CoverageSolverState(
            base_constraint,
            True,
            base_solution,
            cm,
            self.free_vars,
            bound,
            search_limit,
        )
        for parallel_access in self.parallel_accesses:
            state = parallel_access.solve_coverage(state)
        for staging_overlap in self.staging_overlaps:
            state = staging_overlap.solve_coverage(state)
        for failure_condition in self.failure_conditions:
            state = failure_condition.solve_coverage(state)
        for aliasable_access in self.aliasable_accesses:
            state = aliasable_access.solve_coverage(state)
        for root_node in self.roots:
            state = root_node.solve_coverage(state)
        return state.current_solution
