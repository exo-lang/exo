from functools import reduce
from itertools import chain
from string import Template
from typing import Any, Callable, Generator, Iterable, Optional, Union

from ..core.configs import Config

from ..core.prelude import Sym
from ..core.LoopIR import LoopIR, T
from .coverage import (
    CoverageSkeleton,
    CoverageSkeletonNode,
    CoverageSkeletonBranch,
    FailureCondition,
    IndexedFiller,
    MemoryAccess,
    MemoryAccessPair,
    ParallelAccess,
    ParallelAccessPair,
    StagingBoundCheck,
    SymbolicPoint,
    SymbolicSlice,
    StagingOverlap,
    SymbolicWindowIndex,
)
from ..core.internal_cursors import Block, Cursor, Node, NodePath
from ..rewrite.constraint_solver import (
    TRUE_CONSTRAINT,
    Constraint,
    ConstraintMaker,
    DisjointConstraint,
    Expression,
)
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BaseType:
    loopir_type: Any
    dtype: type
    javascript_array_type: str


base_types = (
    BaseType(T.F16, np.float16, "Float16Array"),
    BaseType(T.F32, np.float32, "Float32Array"),
    BaseType(T.F64, np.float64, "Float64Array"),
    BaseType(T.INT8, np.int8, "Int8Array"),
    BaseType(T.UINT8, np.uint8, "Uint8Array"),
    BaseType(T.UINT16, np.uint16, "Uint16Array"),
    BaseType(T.INT32, np.int32, "Int32Array"),
    BaseType(T.Num, np.float64, "Float64Array"),
)


def lookup_loopir_type(loopir_type: Any) -> BaseType:
    return next(
        base_type
        for base_type in base_types
        if isinstance(loopir_type, base_type.loopir_type)
    )


def lookup_dtype(dtype: type) -> BaseType:
    return next(base_type for base_type in base_types if base_type.dtype == dtype)


@dataclass
class Constant:
    name: str


@dataclass
class Reference:
    name: str
    is_config: bool


@dataclass
class Point:
    index: str


@dataclass
class Slice:
    lower_bound: str
    upper_bound: str


WindowIndex = Union[Point, Slice]


@dataclass
class Dimension:
    size: str
    stride: str
    window_idx: WindowIndex


@dataclass
class Tensor:
    name: Sym
    dims: tuple[Dimension, ...]


ExoValue = Union[Constant, Reference, Tensor]


CONTEXT_OBJECT_NAME = "ctxt"
INITIAL_DYNAMIC_SIZE = 16


@dataclass
class SymbolicTensor:
    name: Sym
    dims: tuple[SymbolicWindowIndex, ...]
    resize_placeholder: Optional[int]


class AliasingTracker:
    def __init__(self, parent_state: "CoverageState"):
        self.writes: dict[Sym, list[MemoryAccess]] = {}
        self.reads: dict[Sym, list[MemoryAccess]] = {}
        self.parent_state = parent_state

    def access_tensor(
        self,
        js_tensor: Tensor,
        js_idxs: tuple[str, ...],
        symbolic_idxs: tuple[Expression],
        access_placeholder: int,
        _access_cursor: Node,
        cov_node: CoverageSkeletonNode,
        is_write: bool,
    ):
        js_access = "+".join(
            f"Math.imul({js_idx},{js_dim.stride})"
            for js_idx, js_dim in zip(js_idxs, js_tensor.dims)
        )
        access_sym = Sym("access")
        mark_stmt = f"{repr(access_sym)}[{js_access}]=1;"
        base_size = f"Math.imul({js_tensor.dims[0].size},{js_tensor.dims[0].stride})"
        resize_placeholder = self.parent_state.symbolic_tensors[
            js_tensor.name
        ].resize_placeholder
        if resize_placeholder is None:
            decl_stmt = f"let {repr(access_sym)}=new Uint8Array({base_size});"
            fillers = (
                IndexedFiller(access_placeholder, mark_stmt),
                IndexedFiller(self.parent_state.cov_placeholder, decl_stmt),
            )
        else:
            temp_sym = Sym("temp")
            decl_stmt = (
                f"let {repr(access_sym)}=new Uint8Array({INITIAL_DYNAMIC_SIZE});"
            )
            resize_stmt = f"if({base_size}>{repr(access_sym)}.length){{let {repr(temp_sym)}=new Uint8Array(Math.max(2*{repr(access_sym)}.length,{base_size}));for(let i=0;i<{repr(access_sym)}.length;i++){repr(temp_sym)}[i]={repr(access_sym)}[i];{repr(access_sym)}={repr(temp_sym)}}};"
            fillers = (
                IndexedFiller(access_placeholder, mark_stmt),
                IndexedFiller(self.parent_state.cov_placeholder, decl_stmt),
                IndexedFiller(resize_placeholder, resize_stmt),
            )

        dest = self.writes if is_write else self.reads
        if js_tensor.name not in dest:
            dest[js_tensor.name] = []
        dest[js_tensor.name].append(
            MemoryAccess(access_sym, cov_node, symbolic_idxs, fillers)
        )

    def access_scalar(
        self,
        name: Sym,
        access_placeholder: int,
        _access_cursor: Node,
        cov_node: CoverageSkeletonNode,
        is_write: bool,
    ):
        access_sym = Sym("access")
        mark_stmt = f"{repr(access_sym)}=true;"
        decl_stmt = f"let {repr(access_sym)}=false;"

        dest = self.writes if is_write else self.reads
        if name not in dest:
            dest[name] = []
        dest[name].append(
            MemoryAccess(
                access_sym,
                cov_node,
                (),
                (
                    IndexedFiller(access_placeholder, mark_stmt),
                    IndexedFiller(self.parent_state.cov_placeholder, decl_stmt),
                ),
            )
        )

    def make_aliasing_accesses(self) -> tuple[MemoryAccessPair, ...]:
        aliasable_accesses: list[MemoryAccessPair] = []
        for sym, write_indices in self.writes.items():
            read_indices = self.reads[sym] if sym in self.reads else []
            for i, index1 in enumerate(write_indices):
                for index2 in chain(write_indices[i + 1 :], read_indices):
                    aliasable_accesses.append(MemoryAccessPair(index1, index2))
        return tuple(aliasable_accesses)


class FailureTracker:
    def __init__(self, scope: Block, parent_state: "CoverageState"):
        self.scope = scope
        self.in_scope = False
        self.call_depth = 0
        self.failure_conditions: list[FailureCondition] = []
        self.parent_state = parent_state

    def enter_stmt(self, stmt_cursor: Node):
        if stmt_cursor in self.scope:
            self.in_scope = True

    def exit_stmt(self, stmt_cursor):
        if stmt_cursor in self.scope:
            self.in_scope = False

    def enter_proc_body(self):
        self.call_depth += 1

    def exit_proc_body(self):
        self.call_depth -= 1

    def add_assertion(
        self, asserted_cond: DisjointConstraint, js_cond: str, placeholder: int
    ):
        if self.in_scope and self.call_depth == 1:
            fail_sym = Sym("fail")
            self.failure_conditions.append(
                FailureCondition(
                    fail_sym,
                    asserted_cond.invert(),
                    self.parent_state.current_node,
                    (
                        IndexedFiller(
                            self.parent_state.cov_placeholder,
                            f"let {repr(fail_sym)}=false;",
                        ),
                        IndexedFiller(
                            placeholder, f"if(!({js_cond})){repr(fail_sym)}=true;"
                        ),
                    ),
                )
            )

    def make_failures(self) -> tuple[FailureCondition, ...]:
        return tuple(self.failure_conditions)


@dataclass
class SymbolicWindow:
    name: Sym
    index: tuple[SymbolicWindowIndex, ...]


@dataclass
class StagedWindowExpr:
    indices: tuple[Union[tuple[LoopIR.expr, LoopIR.expr], LoopIR.expr], ...]


@dataclass
class StageMemArgs:
    buffer_sym: Sym
    staged_window_expr: StagedWindowExpr
    scope: Block


class StageMemTracker:
    def __init__(self, args: StageMemArgs, parent_state: "CoverageState"):
        self.scope: Block = args.scope
        self.buffer_sym = args.buffer_sym
        self.staged_window_expr = args.staged_window_expr
        self.staged_window: Optional[tuple[SymbolicTensor, Tensor]] = None
        self.enabled: bool = False
        self.overlaps: list[StagingOverlap] = []
        self.bound_checks: list[StagingBoundCheck] = []
        self.parent_state: "CoverageState" = parent_state

    def enter_stmt(self, stmt_node: Node):
        if stmt_node in self.scope:
            self.enabled = True

            if self.staged_window is None:
                staged_win_sym = Sym("win")
                js_parent = self.parent_state.parent_transpiler._lookup_sym(
                    self.buffer_sym
                )
                js_staged = self.parent_state.parent_transpiler._transpile_window(
                    staged_win_sym,
                    self.buffer_sym,
                    Cursor.create(self.staged_window_expr)._child_block("indices"),
                )
                assert isinstance(js_parent, Tensor)
                symbolic_parent = self.parent_state.symbolic_tensors[self.buffer_sym]
                symbolic_staged = self.parent_state.symbolic_tensors[staged_win_sym]
                stage_placeholder = (
                    self.parent_state.parent_transpiler._make_placeholder()
                )
                for (
                    symbolic_parent_dim,
                    symbolic_staged_dim,
                    js_parent_dim,
                    js_staged_dim,
                ) in zip(
                    symbolic_parent.dims,
                    symbolic_staged.dims,
                    (dim.window_idx for dim in js_parent.dims),
                    (dim.window_idx for dim in js_staged.dims),
                ):
                    if isinstance(symbolic_parent_dim, SymbolicSlice):
                        assert isinstance(js_parent_dim, Slice)
                        out_of_bounds_sym = Sym("oob")
                        js_out_of_bounds_cond = (
                            "&&".join(
                                (
                                    f"({js_staged_dim.index}<{js_parent_dim.upper_bound})",
                                    f"({js_parent_dim.lower_bound}<={js_staged_dim.index})",
                                )
                            )
                            if isinstance(js_staged_dim, Point)
                            else "&&".join(
                                (
                                    f"({js_parent_dim.lower_bound}<={js_staged_dim.lower_bound})",
                                    f"({js_staged_dim.upper_bound})<{js_parent_dim.upper_bound})",
                                )
                            )
                        )
                        self.bound_checks.append(
                            StagingBoundCheck(
                                out_of_bounds_sym,
                                symbolic_staged_dim,
                                symbolic_parent_dim,
                                self.parent_state.current_node,
                                (
                                    IndexedFiller(
                                        self.parent_state.cov_placeholder,
                                        f"let {repr(out_of_bounds_sym)}=false;",
                                    ),
                                    IndexedFiller(
                                        stage_placeholder,
                                        f"if({js_out_of_bounds_cond}){{let {repr(out_of_bounds_sym)}=true;}}",
                                    ),
                                ),
                            )
                        )
                self.staged_window = (
                    symbolic_staged,
                    js_staged,
                )

    def exit_stmt(self, stmt_cursor: Node):
        if stmt_cursor in self.scope:
            self.enabled = False

    def access_tensor(
        self,
        js_tensor: Tensor,
        js_idxs: tuple[str, ...],
        symbolic_idxs: tuple[Expression],
        access_placeholder: int,
        access_cursor: Node,
        cov_node: CoverageSkeletonNode,
        _is_write: bool,
    ):
        if (
            self.staged_window is not None
            and self.enabled
            and self.staged_window[1].name == js_tensor.name
        ):
            symbolic_staged_window, js_staged_window = self.staged_window
            overlap_sym = Sym("overlap")
            disjoint_sym = Sym("disjoint")
            access_window = tuple(SymbolicPoint(idx) for idx in symbolic_idxs)
            js_overlap_cond = "&&".join(
                f"(({js_staged_slice.lower_bound})<=({js_idx}))&&(({js_staged_slice.upper_bound})>({js_idx}))"
                for js_idx, js_staged_slice in zip(
                    js_idxs,
                    (
                        dim.window_idx
                        for dim in js_staged_window.dims
                        if isinstance(dim.window_idx, Slice)
                    ),
                )
            )
            self.overlaps.append(
                StagingOverlap(
                    overlap_sym,
                    disjoint_sym,
                    symbolic_staged_window.dims,
                    access_window,
                    cov_node,
                    access_cursor.get_path(),
                    (
                        IndexedFiller(
                            access_placeholder,
                            f"if({js_overlap_cond}){{{repr(overlap_sym)}=true}}else{{{repr(disjoint_sym)}=true}}",
                        ),
                        IndexedFiller(
                            self.parent_state.cov_placeholder,
                            f"let {repr(overlap_sym)}=false;let {repr(disjoint_sym)}=false;",
                        ),
                    ),
                )
            )

    def make_staging_overlaps(self) -> tuple[StagingOverlap, ...]:
        return tuple(self.overlaps)

    def make_staging_bound_checks(self) -> tuple[StagingBoundCheck, ...]:
        return tuple(self.bound_checks)


@dataclass
class ParallelScope:
    iter_sym: Sym
    loop_entrance_placeholder: int
    writes: dict[Sym, list[ParallelAccess]] = field(default_factory=lambda: {})
    reads: dict[Sym, list[ParallelAccess]] = field(default_factory=lambda: {})
    access_set_syms: dict[Sym, Sym] = field(default_factory=lambda: {})


class ParallelAccessTracker:
    def __init__(self, parent_state: "CoverageState"):
        self.parallel_scopes: list[ParallelScope] = []
        self.coverage_sym: Sym = Sym("par")
        self.pairs: list[ParallelAccessPair] = []
        self.parent_state: "CoverageState" = parent_state

    def enter_loop(self, loop: LoopIR.For, loop_entrance_placeholder: int):
        if isinstance(loop.loop_mode, LoopIR.Par):
            self.parallel_scopes.append(
                ParallelScope(loop.iter, loop_entrance_placeholder)
            )

    def exit_loop_body(self, loop: LoopIR.For):
        if isinstance(loop.loop_mode, LoopIR.Par):
            scope = self.parallel_scopes.pop()
            loop_tail_placeholder = (
                self.parent_state.parent_transpiler._make_placeholder()
            )
            for sym, sym_writes in scope.writes.items():
                for sym_write_idx, sym_write in enumerate(sym_writes):
                    for other_access in chain(
                        sym_writes[sym_write_idx:],
                        (scope.reads[sym] if sym in scope.reads else []),
                    ):
                        self.pairs.append(
                            ParallelAccessPair(
                                self.coverage_sym,
                                scope.iter_sym,
                                sym_write,
                                other_access,
                                (
                                    IndexedFiller(
                                        loop_tail_placeholder,
                                        f"{repr(scope.access_set_syms[sym])}_pw={repr(scope.access_set_syms[sym])}_pw.union({repr(scope.access_set_syms[sym])}_cw);",
                                    ),
                                    IndexedFiller(
                                        loop_tail_placeholder,
                                        f"{repr(scope.access_set_syms[sym])}_pr={repr(scope.access_set_syms[sym])}_pr.union({repr(scope.access_set_syms[sym])}_cr);",
                                    ),
                                ),
                            )
                        )

    def access_tensor(
        self,
        js_tensor: Tensor,
        js_idxs: tuple[str, ...],
        symbolic_idxs: tuple[Expression],
        access_placeholder: int,
        _access_cursor: Node,
        cov_node: CoverageSkeletonNode,
        is_write: bool,
    ):
        js_access = "+".join(
            f"Math.imul({js_idx},{js_dim.stride})"
            for js_idx, js_dim in zip(js_idxs, js_tensor.dims)
        )
        for parallel_scope in self.parallel_scopes:
            dest = parallel_scope.writes if is_write else parallel_scope.reads
            if js_tensor.name not in dest:
                dest[js_tensor.name] = []
            if js_tensor.name not in parallel_scope.access_set_syms:
                parallel_scope.access_set_syms[js_tensor.name] = Sym("access_set")
            access_set_sym = parallel_scope.access_set_syms[js_tensor.name]
            dest[js_tensor.name].append(
                ParallelAccess(
                    cov_node,
                    symbolic_idxs,
                    (
                        IndexedFiller(
                            parallel_scope.loop_entrance_placeholder,
                            "".join(
                                (
                                    f"let {repr(access_set_sym)}_pr=new Set();",
                                    f"let {repr(access_set_sym)}_pw=new Set();",
                                    f"let {repr(access_set_sym)}_cr=new Set();",
                                    f"let {repr(access_set_sym)}_cw=new Set();",
                                )
                            ),
                        ),
                        IndexedFiller(
                            self.parent_state.cov_placeholder,
                            f"let {repr(self.coverage_sym)}=false;",
                        ),
                        IndexedFiller(
                            access_placeholder,
                            "".join(
                                (
                                    f"{repr(access_set_sym)}{'_cw' if is_write else '_cr'}.add({js_access});",
                                    f"if({repr(access_set_sym)}_pw.has({js_access})){{{repr(self.coverage_sym)}=true}}",
                                    *(
                                        (
                                            f"if({repr(access_set_sym)}_pr.has({js_access})){{{repr(self.coverage_sym)}=true}}",
                                        )
                                        if is_write
                                        else ()
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            )

    def access_scalar(
        self,
        name: Sym,
        access_placeholder: int,
        _access_cursor: Node,
        cov_node: CoverageSkeletonNode,
        is_write: bool,
    ):
        self.access_tensor(
            Tensor(name, (Dimension("1", "0", Slice("0", "1")),)),
            ("0",),
            tuple(),
            access_placeholder,
            _access_cursor,
            cov_node,
            is_write,
        )

    def make_parallel_access_pairs(self) -> tuple[ParallelAccessPair, ...]:
        return tuple(self.pairs)


@dataclass
class CoverageArgs:
    cm: ConstraintMaker
    var_renaming: dict[Sym, Sym]
    failure_scope: Optional[Block] = None
    stage_mem_args: Optional[StageMemArgs] = None


class CoverageState:
    def __init__(self, args: CoverageArgs, parent_transpiler: "Transpiler"):
        self.cm: ConstraintMaker = args.cm
        self.var_renaming: dict[Sym, Sym] = args.var_renaming
        self.parent_transpiler: Transpiler = parent_transpiler
        self.cov_placeholder: int = parent_transpiler._make_placeholder()
        self.root: CoverageSkeletonNode = CoverageSkeletonNode(None, None, ())
        self.current_node: CoverageSkeletonNode = self.root
        self.symbolic_tensors: dict[Sym, SymbolicTensor] = {}
        self.scalar_symbols: dict[Sym, Sym] = {}
        self.ctxt_symbols: dict[tuple[Config, str], Sym] = {}
        self.aliasing_tracker = AliasingTracker(self)
        self.failure_tracker: Optional[FailureTracker] = (
            None
            if args.failure_scope is None
            else FailureTracker(args.failure_scope, self)
        )
        self.stage_mem_tracker: Optional[StageMemTracker] = (
            None
            if args.stage_mem_args is None
            else StageMemTracker(args.stage_mem_args, self)
        )
        self.parallel_access_tracker = ParallelAccessTracker(self)
        self.free_vars: list[Sym] = []

    def enter_loop(
        self,
        stmt: LoopIR.For,
        lo_js: str,
        hi_js: str,
        transpile_loop_body: Callable[[], None],
    ):
        body_sym = Sym("body")
        skip_sym = Sym("skip")
        loop_entrance_placeholder = self.parent_transpiler._make_placeholder()
        body_constraint = (
            self.cm.make_constraint_from_inequality(
                stmt.lo, stmt.iter, "<=", self.var_renaming
            )
            .lift_to_disjoint_constraint()
            .intersect(
                self.cm.make_constraint_from_inequality(
                    stmt.iter, stmt.hi, "<", self.var_renaming
                ).lift_to_disjoint_constraint()
            )
        )
        skip_constraint = self.cm.make_constraint_from_inequality(
            stmt.lo, stmt.hi, ">=", self.var_renaming
        ).lift_to_disjoint_constraint()
        parent_node = self.current_node
        body_child = CoverageSkeletonNode(
            body_sym,
            (parent_node, body_constraint),
            (
                IndexedFiller(
                    self.cov_placeholder,
                    f"let {repr(body_sym)}=false;",
                ),
                IndexedFiller(
                    loop_entrance_placeholder,
                    f"{repr(body_sym)}||=({lo_js}<{hi_js});",
                ),
            ),
        )
        skip_child = CoverageSkeletonNode(
            skip_sym,
            (parent_node, skip_constraint),
            (
                IndexedFiller(
                    self.cov_placeholder,
                    f"let {repr(skip_sym)}=false;",
                ),
                IndexedFiller(
                    loop_entrance_placeholder,
                    f"{repr(skip_sym)}||=({lo_js}>={hi_js});",
                ),
            ),
        )
        self.current_node.branches.append(
            CoverageSkeletonBranch(body_child, skip_child)
        )
        self.parallel_access_tracker.enter_loop(stmt, loop_entrance_placeholder)
        self.free_vars.append(stmt.iter)
        self.current_node = body_child
        transpile_loop_body()
        self.current_node = parent_node
        self.parallel_access_tracker.exit_loop_body(stmt)

    def enter_if(
        self,
        stmt: LoopIR.If,
        transpile_if_body: Callable[[], None],
        transpile_else_body: Callable[[], None],
    ):
        parent_node = self.current_node
        true_sym = Sym("true_case")
        false_sym = Sym("false_case")
        true_placeholder = self.parent_transpiler._make_placeholder()
        cond_constraint = self.cm.make_constraint(stmt.cond, self.var_renaming)
        true_node = CoverageSkeletonNode(
            true_sym,
            (parent_node, cond_constraint),
            (
                IndexedFiller(
                    self.cov_placeholder,
                    f"let {repr(true_sym)}=false;",
                ),
                IndexedFiller(true_placeholder, f"{repr(true_sym)}=true;"),
            ),
        )
        self.current_node = true_node
        transpile_if_body()
        false_placeholder = self.parent_transpiler._make_placeholder()
        false_node = CoverageSkeletonNode(
            false_sym,
            (parent_node, cond_constraint.invert()),
            (
                IndexedFiller(
                    self.cov_placeholder,
                    f"let {repr(false_sym)}=false;",
                ),
                IndexedFiller(false_placeholder, f"{repr(false_sym)}=true;"),
            ),
        )
        self.current_node = false_node
        transpile_else_body()
        self.current_node = parent_node
        new_branch = CoverageSkeletonBranch(true_node, false_node)
        self.current_node.branches.append(new_branch)

    def enter_stmt(self, stmt_cursor: Node):
        if self.failure_tracker is not None:
            self.failure_tracker.enter_stmt(stmt_cursor)
        if self.stage_mem_tracker is not None:
            self.stage_mem_tracker.enter_stmt(stmt_cursor)

    def exit_stmt(self, stmt_cursor: Node):
        if self.failure_tracker is not None:
            self.failure_tracker.exit_stmt(stmt_cursor)
        if self.stage_mem_tracker is not None:
            self.stage_mem_tracker.exit_stmt(stmt_cursor)

    def enter_proc_body(self):
        if self.failure_tracker is not None:
            self.failure_tracker.enter_proc_body()

    def exit_proc_body(self):
        if self.failure_tracker is not None:
            self.failure_tracker.exit_proc_body()

    def assert_shape_matches(
        self, tensor_sym: Sym, shape: list[LoopIR.expr], shape_matches_js: str
    ):
        match_cond = TRUE_CONSTRAINT
        for tensor_dim, shape_dim in zip(
            (
                dim
                for dim in self.symbolic_tensors[tensor_sym].dims
                if isinstance(dim, SymbolicSlice)
            ),
            shape,
        ):
            match_cond = match_cond.intersect(
                Constraint(
                    self.cm.make_expression(shape_dim, self.var_renaming)
                    .negate()
                    .add(tensor_dim.upper_bound)
                    .add(tensor_dim.lower_bound.negate()),
                    False,
                ).lift_to_disjoint_constraint(),
            )
        if self.failure_tracker is not None:
            self.failure_tracker.add_assertion(
                match_cond, shape_matches_js, self.parent_transpiler._make_placeholder()
            )

    def assert_predicate(self, pred: LoopIR.expr, js_pred: str):
        if self.failure_tracker is not None:
            self.failure_tracker.add_assertion(
                self.cm.make_constraint(pred, self.var_renaming),
                js_pred,
                self.parent_transpiler._make_placeholder(),
            )

    def make_tensor(self, sym: Sym, dims: list[LoopIR.expr], nonnegative_dims_js: str):
        symbolic_dims = tuple(
            self.cm.make_expression(dim, self.var_renaming) for dim in dims
        )
        nonnegative_constraint = TRUE_CONSTRAINT
        for symbolic_dim in symbolic_dims:
            nonnegative_constraint = nonnegative_constraint.intersect(
                Constraint(symbolic_dim, True).lift_to_disjoint_constraint()
            )
        if self.failure_tracker is not None:
            self.failure_tracker.add_assertion(
                nonnegative_constraint,
                nonnegative_dims_js,
                self.parent_transpiler._make_placeholder(),
            )
        self.symbolic_tensors[sym] = SymbolicTensor(
            sym,
            tuple(
                SymbolicSlice(Expression.from_constant(0), symbolic_dim)
                for symbolic_dim in symbolic_dims
            ),
            self.parent_transpiler._make_placeholder(),
        )

    def make_scalar(self, sym: Sym):
        self.scalar_symbols[sym] = sym

    def assign_tensor(self, arg_name: Sym, original_name: Sym):
        self.symbolic_tensors[arg_name] = self.symbolic_tensors[original_name]

    def assign_scalar(self, arg_name: Sym, original_name: Sym):
        self.scalar_symbols[arg_name] = self.scalar_symbols[original_name]

    def assign_scalar_from_context(self, scalar_sym: Sym, config: Config, field: str):
        config_key = (config, field)
        if config_key not in self.ctxt_symbols:
            self.ctxt_symbols[config_key] = Sym("ctxt")
        self.scalar_symbols[scalar_sym] = self.ctxt_symbols[config_key]

    def assign_window(
        self, sym: Sym, source_buf: Sym, access_cursor: Block, in_bounds_js: str
    ):
        base_tensor = self.symbolic_tensors[source_buf]
        in_bounds_constraint = TRUE_CONSTRAINT
        window_dims = []
        window_idx_iter = iter(access_cursor)
        for dim in base_tensor.dims:
            if isinstance(dim, SymbolicPoint):
                window_dims.append(dim)
            else:
                idx = next(window_idx_iter)._node
                if isinstance(idx, LoopIR.Interval):
                    new_dim = SymbolicSlice(
                        self.cm.make_expression(idx.lo, self.var_renaming).add(
                            dim.lower_bound
                        ),
                        self.cm.make_expression(idx.hi, self.var_renaming).add(
                            dim.lower_bound
                        ),
                    )
                    in_bounds_constraint = in_bounds_constraint.intersect(
                        Constraint(
                            new_dim.lower_bound.add(dim.lower_bound.negate()), True
                        ).lift_to_disjoint_constraint()
                    ).intersect(
                        Constraint(
                            dim.upper_bound.add(new_dim.upper_bound.negate()), True
                        ).lift_to_disjoint_constraint()
                    )
                    window_dims.append(new_dim)
                else:
                    new_dim = SymbolicPoint(
                        self.cm.make_expression(idx.pt, self.var_renaming).add(
                            dim.lower_bound
                        )
                    )
                    in_bounds_constraint = in_bounds_constraint.intersect(
                        Constraint(
                            new_dim.index.add(dim.lower_bound.negate()), True
                        ).lift_to_disjoint_constraint()
                    ).intersect(
                        Constraint(
                            dim.upper_bound.add(new_dim.index.negate()).add(
                                Expression.from_constant(-1)
                            ),
                            True,
                        ).lift_to_disjoint_constraint()
                    )
                    window_dims.append(new_dim)

        if self.failure_tracker is not None:
            self.failure_tracker.add_assertion(
                in_bounds_constraint,
                in_bounds_js,
                self.parent_transpiler._make_placeholder(),
            )
        self.symbolic_tensors[sym] = SymbolicTensor(
            base_tensor.name,
            tuple(window_dims),
            base_tensor.resize_placeholder,
        )

    def access_tensor(
        self,
        access_cursor: Node,
        js_idxs: tuple[str, ...],
        is_write: bool,
        in_bounds_js: str,
    ):
        js_tensor = self.parent_transpiler._lookup_sym(access_cursor._node.name)
        assert isinstance(js_tensor, Tensor)
        symbolic_tensor = self.symbolic_tensors[access_cursor._node.name]
        idx_expr_iter = iter(access_cursor._node.idx)
        symbolic_idxs = []
        in_bounds_constraint = TRUE_CONSTRAINT
        for dim in symbolic_tensor.dims:
            if isinstance(dim, SymbolicSlice):
                idx = self.cm.make_expression(
                    next(idx_expr_iter), self.var_renaming
                ).add(dim.lower_bound)
                in_bounds_constraint = in_bounds_constraint.intersect(
                    Constraint(
                        idx.negate().add(dim.upper_bound), True
                    ).lift_to_disjoint_constraint()
                )
                symbolic_idxs.append(idx)
            else:
                symbolic_idxs.append(dim.index)
        access_placeholder = self.parent_transpiler._make_placeholder()
        access_args = (
            js_tensor,
            js_idxs,
            tuple(symbolic_idxs),
            access_placeholder,
            access_cursor,
            self.current_node,
            is_write,
        )
        if self.failure_tracker is not None:
            self.failure_tracker.add_assertion(
                in_bounds_constraint,
                in_bounds_js,
                self.parent_transpiler._make_placeholder(),
            )
        self.aliasing_tracker.access_tensor(*access_args)
        if self.stage_mem_tracker is not None:
            self.stage_mem_tracker.access_tensor(*access_args)
        self.parallel_access_tracker.access_tensor(*access_args)

    def access_scalar(self, access_cursor: Node, is_write: bool):
        access_placeholder = self.parent_transpiler._make_placeholder()
        scalar_sym = self.scalar_symbols[access_cursor._node.name]
        access_args = (
            scalar_sym,
            access_placeholder,
            access_cursor,
            self.current_node,
            is_write,
        )
        self.aliasing_tracker.access_scalar(*access_args)
        self.parallel_access_tracker.access_scalar(*access_args)

    def access_context(self, access_cursor: Node, is_write: bool):
        access_placeholder = self.parent_transpiler._make_placeholder()
        config_key = (access_cursor._node.config, access_cursor._node.field)
        if config_key not in self.ctxt_symbols:
            self.ctxt_symbols[config_key] = Sym("ctxt")
        ctxt_sym = self.ctxt_symbols[config_key]
        access_args = (
            ctxt_sym,
            access_placeholder,
            access_cursor,
            self.current_node,
            is_write,
        )
        self.aliasing_tracker.access_scalar(*access_args)
        self.parallel_access_tracker.access_scalar(*access_args)

    def make_skeleton(self) -> CoverageSkeleton:
        return CoverageSkeleton(
            (self.root,),
            self.aliasing_tracker.make_aliasing_accesses(),
            (
                ()
                if self.failure_tracker is None
                else self.failure_tracker.make_failures()
            ),
            (
                ()
                if self.stage_mem_tracker is None
                else self.stage_mem_tracker.make_staging_overlaps()
            ),
            (
                ()
                if self.stage_mem_tracker is None
                else self.stage_mem_tracker.make_staging_bound_checks()
            ),
            self.parallel_access_tracker.make_parallel_access_pairs(),
            frozenset(self.free_vars),
        )


def get_shape_cursor(type_cursor: Node) -> Block:
    return (
        type_cursor._child_node("as_tensor")
        if isinstance(type_cursor._node, LoopIR.WindowType)
        else type_cursor
    )._child_block("hi")


class Transpiler:
    def __init__(self, proc: LoopIR.proc, coverage_args: Optional[CoverageArgs] = None):
        self._name_lookup: dict[Sym, ExoValue] = {}
        self._js_lines: list[str] = []
        self._configs: set[tuple[Config, str]] = set()
        self._buffer_args: list[Sym] = []
        self._coverage_state: Optional[CoverageState] = None
        self._skeleton: Optional[CoverageSkeleton] = None
        self._proc = proc
        self._transpile_proc(proc, coverage_args)

    def get_javascript_template(self) -> Template:
        return Template("\n".join(self._js_lines))

    def get_configs(self) -> frozenset[tuple[Config, str]]:
        return frozenset(self._configs)

    def get_buffer_arg_order(self) -> tuple[Sym, ...]:
        return tuple(self._buffer_args)

    def get_config_param_name(self, config: Config, field: str) -> str:
        return f"config_{config.name()}_{field}"

    def get_stride_param_name(self, tensor_name: Sym, dim_idx: int):
        return f"stride_{repr(tensor_name)}_{dim_idx}"

    def get_size_param_name(self, tensor_name: Sym, dim_idx: int):
        return f"size_{repr(tensor_name)}_{dim_idx}"

    def get_coverage_skeleton(self) -> Optional[CoverageSkeleton]:
        return self._skeleton

    def get_proc(self) -> LoopIR.proc:
        return self._proc

    def _assert_at_runtime(self, expr: str):
        self._js_lines.append(f"if(!{expr})return [1,{CONTEXT_OBJECT_NAME},{{}}];")

    # for CoverageState
    def _make_placeholder(self) -> int:
        placeholder_index = len(self._js_lines)
        self._js_lines.append("")
        return placeholder_index

    # for CoverageState
    def _lookup_sym(self, sym: Sym) -> ExoValue:
        return self._name_lookup[sym]

    def _transpile_proc(self, proc: LoopIR.proc, coverage_args: Optional[CoverageArgs]):
        self._buffer_args = [arg.name for arg in proc.args if arg.type.is_numeric()]
        root_cursor = Cursor.create(proc)
        self._js_lines.append(
            f'(({",".join(repr(arg) for arg in self._buffer_args)})=>{{'
        )
        ctxt_placeholder = self._make_placeholder()
        if coverage_args is not None:
            self._coverage_state = CoverageState(coverage_args, self)
        arg_values = []
        for arg_cursor in root_cursor._child_block("args"):
            arg = arg_cursor._node
            if arg.type.is_numeric():
                if isinstance(arg.type, LoopIR.Tensor):
                    value = Tensor(
                        arg.name,
                        tuple(
                            Dimension(
                                f"${size}",
                                f"${stride}",
                                Slice("0", f"${size}"),
                            )
                            for size, stride in map(
                                lambda dim_idx: (
                                    self.get_size_param_name(arg.name, dim_idx),
                                    self.get_stride_param_name(arg.name, dim_idx),
                                ),
                                range(len(arg.type.shape())),
                            )
                        ),
                    )
                    if self._coverage_state is not None:
                        self._coverage_state.make_tensor(
                            arg.name, arg.type.shape(), "true"
                        )
                elif isinstance(arg.type, LoopIR.WindowType):
                    value = self._transpile_window(
                        arg.name,
                        arg.type.src_buf,
                        arg_cursor._child_node("type")._child_block("idx"),
                    )
                else:
                    value = Reference(repr(arg.name), False)
                    if self._coverage_state is not None:
                        self._coverage_state.make_scalar(arg.name)
            else:
                value = Constant(f"${repr(arg.name)}")
            arg_values.append(value)
        self._call_proc(root_cursor, tuple(arg_values), True)
        coverage_object = ""
        if self._coverage_state is not None:
            skeleton = self._coverage_state.make_skeleton()
            self._skeleton = skeleton
            coverage_object = f"{{{','.join(sorted(repr(sym) for sym in self._skeleton.get_coverage_syms()))}}}"
            for indexed_filler in sorted(set(skeleton.get_indexed_fillers())):
                self._js_lines[indexed_filler.index] += indexed_filler.placefiller
        self._js_lines.append(f"return [0,{CONTEXT_OBJECT_NAME},{coverage_object}];}})")
        configs = ",".join(
            f"{self.get_config_param_name(config, field)}:${self.get_config_param_name(config, field)}"
            for config, field in self._configs
        )
        self._js_lines[ctxt_placeholder] = f"{CONTEXT_OBJECT_NAME}={{{configs}}}"

    def _transpile_window(
        self, name: Sym, source_buf: Sym, access_cursor: Block
    ) -> Tensor:
        base = self._name_lookup[source_buf]
        assert isinstance(base, Tensor)
        window_dims = []
        in_bounds_conds = []
        idx_cursor_iter = iter(access_cursor)
        for dim in base.dims:
            if isinstance(dim.window_idx, Point):
                window_dims.append(dim)
            else:
                idx_cursor = next(idx_cursor_iter)
                idx = idx_cursor._node
                if isinstance(idx, LoopIR.Interval):
                    lo_expr = self._transpile_expr(
                        idx_cursor._child_node("lo"),
                    )
                    hi_expr = self._transpile_expr(
                        idx_cursor._child_node("hi"),
                    )
                    lo_sym, hi_sym = Sym("lo"), Sym("hi")
                    self._js_lines.append(
                        f"let {repr(lo_sym)}=({lo_expr})+({dim.window_idx.lower_bound});"
                    )
                    self._js_lines.append(
                        f"let {repr(hi_sym)}=({hi_expr})+({dim.window_idx.lower_bound});"
                    )
                    in_bounds_conds.append(
                        f"(({dim.window_idx.lower_bound})<=({repr(lo_sym)})&&({repr(lo_sym)})<=({repr(hi_sym)})&&({repr(hi_sym)})<=({dim.window_idx.upper_bound}))"
                    )
                    window_dims.append(
                        Dimension(
                            dim.size, dim.stride, Slice(repr(lo_sym), repr(hi_sym))
                        )
                    )
                elif isinstance(idx, LoopIR.Point):
                    pt_expr = self._transpile_expr(
                        idx_cursor._child_node("pt"),
                    )
                    index_sym = Sym("idx")
                    self._js_lines.append(
                        f"let {repr(index_sym)}=({pt_expr})+({dim.window_idx.lower_bound});"
                    )
                    in_bounds_conds.append(
                        f"(({dim.window_idx.lower_bound})<={repr(index_sym)}&&{repr(index_sym)}<({dim.window_idx.upper_bound}))"
                    )
                    window_dims.append(
                        Dimension(dim.size, dim.stride, Point(repr(index_sym)))
                    )
                else:
                    assert False, "not a window index"
        in_bounds_js = "&&".join(in_bounds_conds)
        if self._coverage_state is not None:
            self._coverage_state.assign_window(
                name, source_buf, access_cursor, in_bounds_js
            )
        self._assert_at_runtime(in_bounds_js)
        return Tensor(base.name, tuple(window_dims))

    def _call_proc(
        self, proc_cursor: Node, arg_values: tuple[ExoValue, ...], top_level: bool
    ):
        for arg_cursor, arg_value in zip(proc_cursor._child_block("args"), arg_values):
            arg = arg_cursor._node
            self._name_lookup[arg.name] = arg_value
            if arg.type.is_tensor_or_window():
                assert isinstance(arg_value, Tensor)
                shape_matches_js = "&&".join(
                    f"((({arg_dim_slice.upper_bound})-({arg_dim_slice.lower_bound}))==({self._transpile_expr(arg_dim_cursor)}))"
                    for arg_dim_slice, arg_dim_cursor in zip(
                        (
                            dim.window_idx
                            for dim in arg_value.dims
                            if isinstance(dim.window_idx, Slice)
                        ),
                        get_shape_cursor(arg_cursor._child_node("type")),
                    )
                )
                if self._coverage_state is not None and not top_level:
                    self._coverage_state.assert_shape_matches(
                        arg.name, arg.type.shape(), shape_matches_js
                    )
                self._assert_at_runtime(shape_matches_js)

        for pred_cursor in proc_cursor._child_block("preds"):
            js_pred = self._transpile_expr(pred_cursor)
            if self._coverage_state is not None and not top_level:
                self._coverage_state.assert_predicate(pred_cursor._node, js_pred)
            self._assert_at_runtime(js_pred)

        if self._coverage_state is not None:
            self._coverage_state.enter_proc_body()
        for stmt_cursor in proc_cursor._child_block("body"):
            self._transpile_stmt(stmt_cursor)
        if self._coverage_state is not None:
            self._coverage_state.exit_proc_body()

    def _get_index_exprs(
        self,
        buf: Tensor,
        idxs: Block,
    ) -> tuple[str, ...]:
        idx_expr_iter = iter(self._transpile_expr(idx) for idx in idxs)
        idx_parts = []
        for dim in buf.dims:
            if isinstance(dim.window_idx, Slice):
                idx_expr = next(idx_expr_iter)
                idx_parts.append(f"(({idx_expr})+({dim.window_idx.lower_bound}))")
            else:
                idx_parts.append(f"({dim.window_idx.index})")
        return tuple(idx_parts)

    def _get_in_bounds_condition(
        self, index_exprs: tuple[str, ...], buf: Tensor
    ) -> str:
        return "&&".join(
            f"(({index_expr})>=({dim.window_idx.lower_bound})&&({index_expr})<({dim.window_idx.upper_bound}))"
            for index_expr, dim in zip(index_exprs, buf.dims)
            if isinstance(dim.window_idx, Slice)
        )

    def _transpile_stmt(
        self,
        stmt_cursor: Node,
    ):
        if self._coverage_state is not None:
            self._coverage_state.enter_stmt(stmt_cursor)
        stmt = stmt_cursor._node
        if isinstance(stmt, (LoopIR.Assign, LoopIR.Reduce)):
            lhs_buffer = self._name_lookup[stmt.name]
            rhs = self._transpile_expr(stmt_cursor._child_node("rhs"))
            if isinstance(lhs_buffer, Reference):
                lhs = (
                    lhs_buffer.name if lhs_buffer.is_config else f"{lhs_buffer.name}[0]"
                )
                if self._coverage_state is not None:
                    self._coverage_state.access_scalar(stmt_cursor, True)
            elif isinstance(lhs_buffer, Tensor):
                index_exprs = self._get_index_exprs(
                    lhs_buffer,
                    stmt_cursor._child_block("idx"),
                )
                index = f"+".join(
                    f"Math.imul({dim.stride},{index_expr})"
                    for dim, index_expr in zip(lhs_buffer.dims, index_exprs)
                )
                lhs = f"{repr(lhs_buffer.name)}[{index}]"
                in_bounds_js = self._get_in_bounds_condition(index_exprs, lhs_buffer)
                if self._coverage_state is not None:
                    self._coverage_state.access_tensor(
                        stmt_cursor, index_exprs, True, in_bounds_js
                    )
                self._assert_at_runtime(in_bounds_js)
            else:
                assert False
            if isinstance(stmt, LoopIR.Assign):
                self._js_lines.append(f"{lhs}={rhs};")
            else:
                self._js_lines.append(f"{lhs}+={rhs};")
        elif isinstance(stmt, LoopIR.WriteConfig):
            config_name = self.get_config_param_name(stmt.config, stmt.field)
            rhs = self._transpile_expr(stmt_cursor._child_node("rhs"))
            self._js_lines.append(f'{CONTEXT_OBJECT_NAME}["{config_name}"]={rhs};')
            self._configs.add((stmt.config, stmt.field))
            if self._coverage_state is not None:
                self._coverage_state.access_context(stmt_cursor, True)
        elif isinstance(stmt, LoopIR.Pass):
            pass
        elif isinstance(stmt, LoopIR.If):
            cond = self._transpile_expr(stmt_cursor._child_node("cond"))
            self._js_lines.append(f"if({cond}){{")

            def transpile_if_body():
                for body_cursor in stmt_cursor._child_block("body"):
                    self._transpile_stmt(body_cursor)
                self._js_lines.append("}else{")

            def transpile_else_body():
                for else_cursor in stmt_cursor._child_block("orelse"):
                    self._transpile_stmt(else_cursor)
                self._js_lines.append("}")

            if self._coverage_state is not None:
                self._coverage_state.enter_if(
                    stmt, transpile_if_body, transpile_else_body
                )
            else:
                transpile_if_body()
                transpile_else_body()
        elif isinstance(stmt, LoopIR.For):
            iter_name = repr(stmt.iter)
            iter_lo = self._transpile_expr(stmt_cursor._child_node("lo"))
            iter_hi = self._transpile_expr(stmt_cursor._child_node("hi"))
            self._name_lookup[stmt.iter] = Constant(iter_name)

            def transpile_loop_body():
                self._js_lines.append(
                    f"for(let {iter_name}={iter_lo};{iter_name}<{iter_hi};{iter_name}++){{"
                )
                for body_cursor in stmt_cursor._child_block("body"):
                    self._transpile_stmt(body_cursor)

            if self._coverage_state is not None:
                self._coverage_state.enter_loop(
                    stmt, iter_lo, iter_hi, transpile_loop_body
                )
            else:
                transpile_loop_body()

            self._js_lines.append("}")
        elif isinstance(stmt, LoopIR.Alloc):
            assert stmt.type.is_numeric()
            if stmt.type.is_tensor_or_window():
                tensor_name: Sym = stmt.name
                buffer_type = lookup_loopir_type(
                    stmt.type.basetype()
                ).javascript_array_type
                shape_cursor = get_shape_cursor(stmt_cursor._child_node("type"))
                dims = len(shape_cursor)
                self._js_lines.append(
                    "".join(
                        f"let {self.get_size_param_name(tensor_name, dim_idx)}={self._transpile_expr(dim_cursor)};"
                        for dim_idx, dim_cursor in enumerate(shape_cursor)
                    )
                )
                self._js_lines.append(
                    "".join(
                        f'let {self.get_stride_param_name(tensor_name, dim_idx)}={f"1" if dim_idx + 1 == dims else f"Math.imul({self.get_stride_param_name(tensor_name, dim_idx + 1)},{self.get_size_param_name(tensor_name,dim_idx + 1)})"};'
                        for dim_idx in reversed(range(dims))
                    )
                )
                nonnegative_dims_js = "&&".join(
                    f"({self.get_size_param_name(tensor_name, dim_idx)}>=0)"
                    for dim_idx in range(dims)
                )
                if self._coverage_state is not None:
                    self._coverage_state.make_tensor(
                        tensor_name, stmt.type.shape(), nonnegative_dims_js
                    )
                self._assert_at_runtime(nonnegative_dims_js)
                buffer_size = f"Math.imul({self.get_size_param_name(tensor_name, 0)},{self.get_stride_param_name(tensor_name, 0)})"
                self._js_lines.append(
                    f"let {repr(tensor_name)}=new {buffer_type}({buffer_size});for(let i=0;i<{buffer_size};i++){{{repr(tensor_name)}[i]=(Math.random()-0.5)*(1<<30);}}"
                )
                self._name_lookup[stmt.name] = Tensor(
                    stmt.name,
                    tuple(
                        Dimension(size, stride, Slice("0", size))
                        for size, stride in map(
                            lambda dim_idx: (
                                self.get_size_param_name(tensor_name, dim_idx),
                                self.get_stride_param_name(tensor_name, dim_idx),
                            ),
                            range(dims),
                        )
                    ),
                )
            else:
                ref_name = repr(stmt.name)
                buffer_type = lookup_loopir_type(stmt.type).javascript_array_type
                if self._coverage_state is not None:
                    self._coverage_state.make_scalar(stmt.name)
                self._js_lines.append(f"let {ref_name}=new {buffer_type}(1);")
                self._name_lookup[stmt.name] = Reference(ref_name, False)
        elif isinstance(stmt, LoopIR.Free):
            pass
        elif isinstance(stmt, LoopIR.Call):
            self._call_proc(
                stmt_cursor._child_node("f"),
                tuple(
                    (
                        self._transpile_buffer_arg(
                            arg_val_cursor, arg_name_cursor._node.name
                        )
                        if arg_val_cursor._node.type.is_numeric()
                        else Constant(self._transpile_expr(arg_val_cursor))
                    )
                    for arg_val_cursor, arg_name_cursor in zip(
                        stmt_cursor._child_block("args"),
                        stmt_cursor._child_node("f")._child_block("args"),
                    )
                ),
                False,
            )
        elif isinstance(stmt, LoopIR.WindowStmt):
            self._name_lookup[stmt.name] = self._transpile_buffer_arg(
                stmt_cursor._child_node("rhs"), stmt.name
            )
        else:
            assert False, "unsupported stmt"

        if self._coverage_state is not None:
            self._coverage_state.exit_stmt(stmt_cursor)

    def _transpile_buffer_arg(
        self, expr_cursor: Node, new_name: Sym
    ) -> Union[Tensor, Reference]:
        expr = expr_cursor._node
        if isinstance(expr, LoopIR.Read):
            assert len(expr.idx) == 0
            buf = self._name_lookup[expr.name]
            assert isinstance(buf, (Tensor, Reference))
            if self._coverage_state is not None:
                if isinstance(buf, Tensor):
                    self._coverage_state.assign_tensor(new_name, expr.name)
                else:
                    self._coverage_state.assign_scalar(new_name, expr.name)
            return buf
        elif isinstance(expr, LoopIR.WindowExpr):
            return self._transpile_window(
                new_name, expr.name, expr_cursor._child_block("idx")
            )
        elif isinstance(expr, LoopIR.ReadConfig):
            self._configs.add((expr.config, expr.field))
            if self._coverage_state is not None:
                self._coverage_state.assign_scalar_from_context(
                    new_name, expr.config, expr.field
                )
            return Reference(
                f'{CONTEXT_OBJECT_NAME}["{self.get_config_param_name(expr.config, expr.field)}"]',
                True,
            )
        else:
            assert False, "unsupported buffer expression"

    def _transpile_expr(
        self,
        expr_cursor: Node,
    ) -> str:
        expr = expr_cursor._node
        if isinstance(expr, LoopIR.Read):
            buf = self._name_lookup[expr.name]
            if isinstance(buf, Tensor):
                index_exprs = self._get_index_exprs(
                    buf,
                    expr_cursor._child_block("idx"),
                )
                index = f"+".join(
                    f"Math.imul({dim.stride},{index_expr})"
                    for dim, index_expr in zip(buf.dims, index_exprs)
                )
                in_bounds_js = self._get_in_bounds_condition(index_exprs, buf)
                if self._coverage_state is not None:
                    self._coverage_state.access_tensor(
                        expr_cursor, index_exprs, False, in_bounds_js
                    )
                self._assert_at_runtime(in_bounds_js)
                return f"{repr(buf.name)}[{index}]"
            elif isinstance(buf, Reference):
                if self._coverage_state is not None:
                    self._coverage_state.access_scalar(expr_cursor, False)
                return buf.name if buf.is_config else f"{buf.name}[0]"
            else:
                return buf.name
        elif isinstance(expr, LoopIR.Const):
            if isinstance(expr.val, bool):
                return "true" if expr.val else "false"
            elif isinstance(expr.val, (int, float)):
                return f"{expr.val}"
            else:
                assert False, "unexpected const type"
        elif isinstance(expr, LoopIR.USub):
            return f"(-{self._transpile_expr(expr_cursor._child_node('arg'))})"
        elif isinstance(expr, LoopIR.BinOp):
            lhs = self._transpile_expr(expr_cursor._child_node("lhs"))
            rhs = self._transpile_expr(expr_cursor._child_node("rhs"))
            is_int = (
                isinstance(expr.type, (T.INT8, T.UINT8, T.UINT16, T.INT32))
                or not expr.type.is_numeric()
            )
            if expr.op in ["+", "-", "%", "<", ">", "<=", ">=", "=="]:
                val = f"({lhs}{expr.op}{rhs})"
            elif expr.op == "*":
                val = f"Math.imul({lhs},{rhs})" if is_int else f"({lhs}*{rhs})"
            elif expr.op == "/":
                val = f"(({lhs}/{rhs})|0)" if is_int else f"({lhs}/{rhs})"
            elif expr.op == "and":
                val = f"({lhs}&&{rhs})"
            elif expr.op == "or":
                val = f"({lhs}||{rhs})"
            else:
                assert False, "invalid op"
            if isinstance(expr.type, T.INT8):
                return f"(({val}<<24)>>24)"
            elif isinstance(expr.type, T.UINT8):
                return f"({val}&0xFF)"
            elif isinstance(expr.type, T.UINT16):
                return f"({val}&0xFFFF)"
            else:
                return val
        elif isinstance(expr, LoopIR.Extern):
            return expr.f.transpile(
                tuple(
                    self._transpile_expr(arg_cursor)
                    for arg_cursor in expr_cursor._child_block("args")
                )
            )
        elif isinstance(expr, LoopIR.WindowExpr):
            assert False, "unexpected window expr"
        elif isinstance(expr, LoopIR.StrideExpr):
            buf = self._name_lookup[expr.name]
            assert isinstance(buf, Tensor)
            return tuple(dim for dim in buf.dims if isinstance(dim.window_idx, Slice))[
                expr.dim
            ].stride
        elif isinstance(expr, LoopIR.ReadConfig):
            self._configs.add((expr.config, expr.field))
            if self._coverage_state is not None:
                self._coverage_state.access_context(expr_cursor, False)
            return f'{CONTEXT_OBJECT_NAME}["{self.get_config_param_name(expr.config, expr.field)}"]'
        else:
            assert False, "unexpected expr"
