from itertools import chain
import time
from typing import Callable, Literal, Optional, Union

from ..core.internal_cursors import Cursor, Block, Node, NodePath

from ..backend.LoopIR_transpiler import CoverageArgs, StageMemArgs, Transpiler
from ..backend.coverage import CoverageSkeleton

from ..core.configs import Config

from ..core.LoopIR import LoopIR, T
from dataclasses import dataclass, field
from ..core.prelude import Sym, SrcInfo
from ..core.memory import DRAM, Memory
import numpy as np
from .new_eff import SchedulingError
from .constraint_solver import (
    TRUE_CONSTRAINT,
    Constraint,
    ConstraintClause,
    ConstraintMaker,
    ConstraintTerm,
    DisjointConstraint,
    Expression,
    Solution,
)

from pythonmonkey import eval as js_eval


class LoopIRVisitor:
    def visit(self, node):
        self.visit_generic(node)

    def visit_generic(self, node):
        if (
            isinstance(node, LoopIR.proc)
            or isinstance(node, LoopIR.instr)
            or isinstance(node, LoopIR.fnarg)
            or isinstance(node, LoopIR.stmt)
            or isinstance(node, LoopIR.loop_mode)
            or isinstance(node, LoopIR.expr)
            or isinstance(node, LoopIR.w_access)
            or isinstance(node, LoopIR.type)
        ):
            for field_name in dir(node):
                if not field_name.startswith("_"):
                    field = getattr(node, field_name)
                    if isinstance(field, list):
                        for child in field:
                            self.visit(child)
                    else:
                        self.visit(field)


@dataclass
class TypeVisitor(LoopIRVisitor):
    type_map: dict[Sym, LoopIR.type] = field(default_factory=lambda: {})
    mem_map: dict[Sym, Memory] = field(default_factory=lambda: {})

    def visit(self, node):
        if isinstance(node, LoopIR.For):
            self.type_map[node.iter] = T.Index()
            self.visit_generic(node)
        elif isinstance(node, LoopIR.Alloc):
            self.type_map[node.name] = node.type
            self.mem_map[node.name] = node.mem
        elif isinstance(node, LoopIR.WindowStmt):
            self.type_map[node.name] = node.rhs.type
        elif isinstance(node, LoopIR.fnarg):
            self.type_map[node.name] = node.type
            if node.mem:
                self.mem_map[node.name] = node.mem
        elif isinstance(node, LoopIR.Call):
            for arg_val, arg in zip(node.args, node.f.args):
                if isinstance(arg.type, LoopIR.Tensor) and arg.type.is_window:
                    self.type_map[arg.name] = arg_val.type
                else:
                    self.type_map[arg.name] = arg.type
            for stmt in node.f.body:
                self.visit(stmt)
        else:
            self.visit_generic(node)


@dataclass
class UsedVariableVisitor(LoopIRVisitor):
    used_vars: set[Sym] = field(default_factory=lambda: set())

    def visit(self, node):
        if isinstance(node, Sym):
            self.used_vars.add(node)
        else:
            self.visit_generic(node)


@dataclass
class ModifiedVariableVisitor(LoopIRVisitor):
    type_map: dict[Sym, LoopIR.type]
    modified_vars: set[Sym] = field(default_factory=lambda: set())

    def visit(self, node):
        if isinstance(node, (LoopIR.Assign, LoopIR.Reduce)):
            node_type = self.type_map[node.name]
            if isinstance(node_type, LoopIR.WindowType):
                self.modified_vars.add(node_type.src_buf)
            else:
                self.modified_vars.add(node.name)
        else:
            self.visit_generic(node)


class LoopIRModifier:
    def visit(self, node):
        return self.visit_generic(node)

    def visit_generic(self, node):
        if (
            isinstance(node, LoopIR.proc)
            or isinstance(node, LoopIR.instr)
            or isinstance(node, LoopIR.fnarg)
            or isinstance(node, LoopIR.stmt)
            or isinstance(node, LoopIR.loop_mode)
            or isinstance(node, LoopIR.expr)
            or isinstance(node, LoopIR.w_access)
            or isinstance(node, LoopIR.type)
        ):
            updates = {}
            for field_name in dir(node):
                if not field_name.startswith("_"):
                    field = getattr(node, field_name)
                    if isinstance(field, list):
                        new_field = field
                        for child_idx, child in enumerate(field):
                            new_child = self.visit(child)
                            if new_child != child:
                                if new_field == field:
                                    new_field = field.copy()
                                new_field[child_idx] = new_child
                    else:
                        new_field = self.visit(field)
                    if new_field != field:
                        updates[field_name] = new_field
            return node.update(**updates) if len(updates) != 0 else node


@dataclass
class ReadWriteSyms:
    reduced_syms: set[Sym]
    assigned_syms: set[Sym]
    written_configs: set[tuple[Config, str]]
    read_syms: set[Sym]


@dataclass
class Dimension:
    size: int
    stride: int


@dataclass
class Tensor:
    data: np.ndarray
    dims: tuple[Dimension, ...]


def get_free_variables(type_map, mem_map, fragment: Union[Block, Node]):
    fragment_type_visitor = TypeVisitor()
    fragment_var_visitor = UsedVariableVisitor()
    if isinstance(fragment, Block):
        for fragment_node in fragment.resolve_all():
            fragment_type_visitor.visit(fragment_node)
            fragment_var_visitor.visit(fragment_node)
    else:
        fragment_type_visitor.visit(fragment._node)
        fragment_var_visitor.visit(fragment._node)
    for var in fragment_var_visitor.used_vars - fragment_type_visitor.type_map.keys():
        fragment_var_visitor.visit(type_map[var])
    return {
        var: (type_map[var], mem_map[var] if var in mem_map else None)
        for var in fragment_var_visitor.used_vars
        - fragment_type_visitor.type_map.keys()
    }


def eval_tensor_dimension(
    dim_expr: LoopIR.expr, arg_values: dict[Sym, Union[int, bool, float, Tensor]]
) -> int:
    if isinstance(dim_expr, LoopIR.Read):
        return arg_values[dim_expr.name]
    elif isinstance(dim_expr, LoopIR.Const):
        return dim_expr.val
    elif isinstance(dim_expr, LoopIR.USub):
        return -eval_tensor_dimension(dim_expr.arg, arg_values)
    elif isinstance(dim_expr, LoopIR.BinOp):
        lhs, rhs = eval_tensor_dimension(
            dim_expr.lhs, arg_values
        ), eval_tensor_dimension(dim_expr.rhs, arg_values)
        if dim_expr.op == "+":
            return lhs + rhs
        elif dim_expr.op == "-":
            return lhs - rhs
        elif dim_expr.op == "*":
            return lhs * rhs
        elif dim_expr.op == "/":
            if isinstance(lhs, int) and isinstance(rhs, int):
                # this is what was here before and without the rhs check
                # counter example of why this is wrong -3 / 2 == -1 in C and 0 in this impl
                # return (lhs + rhs - 1) // rhs
                return int(lhs / rhs)
            else:
                return lhs / rhs
        elif dim_expr.op == "%":
            return lhs % rhs
        else:
            assert False, "unexpected binop in tensor dimension"
    else:
        assert False, "unexpected expression type in tensor dimension"


CONTROL_VAL_BOUND = 128
SEARCH_LIMIT = 10
INT_BOUND = 128
FLOAT_BOUND = 32


def collect_path_constraints(
    cursor: Union[Block, Node], cm: ConstraintMaker, type_map: dict[Sym, LoopIR.type]
) -> DisjointConstraint:
    if isinstance(cursor, Block):
        if len(cursor) > 0:
            cursor = cursor[0]
        else:
            cursor = cursor._anchor
    assert isinstance(cursor, Node)
    if len(cursor._path) == 0:
        return TRUE_CONSTRAINT
    last_attr, last_index = cursor._path[-1]
    cur = cursor.parent()
    result = TRUE_CONSTRAINT
    var_renaming = {}
    while cur.depth() != 0:
        if isinstance(cur, Node):
            if isinstance(cur._node, LoopIR.For):
                modified_variable_visitor = ModifiedVariableVisitor(type_map)
                for stmt in cur._node.body:
                    modified_variable_visitor.visit(stmt)
                for var_sym in modified_variable_visitor.modified_vars:
                    var_renaming[var_sym] = cm.copy_var(var_sym)
                result = result.intersect(
                    cm.make_constraint_from_inequality(
                        cur._node.iter, cur._node.lo, ">=", var_renaming
                    ).lift_to_disjoint_constraint()
                )
                result = result.intersect(
                    cm.make_constraint_from_inequality(
                        cur._node.iter, cur._node.hi, "<", var_renaming
                    ).lift_to_disjoint_constraint()
                )
            elif isinstance(cur._node, LoopIR.If):
                assert last_index is not None
                modified_variable_visitor = ModifiedVariableVisitor(type_map)
                for stmt, _ in zip(getattr(cur._node, last_attr), range(last_index)):
                    modified_variable_visitor.visit(stmt)
                for var_sym in modified_variable_visitor.modified_vars:
                    var_renaming[var_sym] = cm.copy_var(var_sym)
                constraint = cm.make_constraint(cur._node.cond, var_renaming)
                if last_attr == "orelse":
                    result = result.intersect(constraint.invert())
                else:
                    result = result.intersect(constraint)

        last_attr, last_index = cur._path[-1]
        cur = cur.parent()

    assert last_index is not None
    modified_variable_visitor = ModifiedVariableVisitor(type_map)
    for stmt, _ in zip(cur._node.body, range(last_index)):
        modified_variable_visitor.visit(stmt)
    for var_sym in modified_variable_visitor.modified_vars:
        var_renaming[var_sym] = cm.copy_var(var_sym)
    for pred in cur._node.preds:
        result = result.intersect(cm.make_constraint(pred, var_renaming))
    return result


@dataclass
class TestCase:
    arg_values: dict[Sym, Union[int, bool, float, Tensor]]
    ctxt: dict[tuple[Config, str], Union[int, bool, float, Tensor]]


def generate_control_value(var_type: LoopIR.type) -> Union[int, bool, float]:
    if isinstance(var_type, T.Bool):
        return np.random.rand() < 0.5
    elif isinstance(var_type, (T.Size, T.Stride)):
        return np.random.randint(1, CONTROL_VAL_BOUND)
    elif isinstance(var_type, (T.Int, T.Index)):
        return np.random.randint(-CONTROL_VAL_BOUND, CONTROL_VAL_BOUND)
    else:
        assert False, "not a control type"


def generate_numeric_value(var_type: LoopIR.type, shape: tuple[int, ...]) -> Tensor:
    if isinstance(var_type, (T.F32, T.Num)):
        dtype = np.float32
    elif isinstance(var_type, T.F16):
        dtype = np.float16
    elif isinstance(var_type, T.F64):
        dtype = np.float64
    elif isinstance(var_type, T.INT8):
        dtype = np.int8
    elif isinstance(var_type, T.INT32):
        dtype = np.int32
    elif isinstance(var_type, T.UINT8):
        dtype = np.uint8
    elif isinstance(var_type, T.UINT16):
        dtype = np.uint16
    else:
        assert False, "not a numeric type"

    if dtype in [np.int8, np.int32]:
        data = np.random.randint(-INT_BOUND, INT_BOUND, shape, dtype=dtype)
    elif dtype in [np.uint8, np.uint16]:
        data = np.random.randint(0, INT_BOUND, shape, dtype=dtype)
    elif dtype in [np.float16, np.float32, np.float64]:
        data = ((np.random.rand(*shape) * 2 - 1) * FLOAT_BOUND).astype(dtype)
    else:
        assert False, "unreachable"

    return Tensor(
        data.flatten(),
        tuple(
            Dimension(dim_size, dim_stride / data.dtype.itemsize)
            for dim_size, dim_stride in zip(data.shape, data.strides)
        ),
    )


def generate_test_case(
    arg_types: dict[Sym, LoopIR.type],
    config_fields: frozenset[tuple[Config, str]],
    constraint: DisjointConstraint,
    coverage_skeleton: CoverageSkeleton,
    cm: ConstraintMaker,
) -> Optional[TestCase]:
    ctxt = {}
    arg_values = {}
    solution = coverage_skeleton.solve_constraint_with_coverage(
        cm, constraint, bound=INT_BOUND, search_limit=SEARCH_LIMIT
    )
    if solution is None:
        return None
    for config, field in config_fields:
        if (config, field) in solution.ctxt:
            ctxt[(config, field)] = solution.ctxt[(config, field)]
        else:
            field_type = config.lookup_type(field)
            if field_type.is_numeric():
                val = generate_numeric_value(field_type, (1,))
            else:
                val = generate_control_value(field_type)
            ctxt[(config, field)] = val

    for arg_name, arg_type in arg_types.items():
        if not arg_type.is_numeric():
            if arg_name in solution.var_assignments:
                if isinstance(arg_type, T.Bool):
                    val = solution.var_assignments[arg_name] != 0
                else:
                    val = solution.var_assignments[arg_name]
            else:
                val = generate_control_value(arg_type)
            arg_values[arg_name] = val

    for arg_name, arg_type in arg_types.items():
        if arg_type.is_numeric() and not isinstance(arg_type, LoopIR.WindowType):
            if arg_type.is_real_scalar():
                shape = (1,)
            else:
                shape = tuple(
                    eval_tensor_dimension(dim_expr, arg_values)
                    for dim_expr in arg_type.shape()
                )
            arg_values[arg_name] = generate_numeric_value(arg_type.basetype(), shape)

    return TestCase(arg_values, ctxt)


@dataclass
class TestResult:
    buffer_values: dict[Sym, np.ndarray]
    ctxt_object: dict[str, Union[int, float]]
    coverage_result: Optional[dict[str, Union[bool, memoryview]]]


def run_test_case(
    test_case: TestCase,
    transpiled_proc: Transpiler,
) -> Union[TestResult, Literal["failed"]]:
    subs = {}
    for arg_name, arg_value in test_case.arg_values.items():
        if isinstance(arg_value, Tensor):
            for dim_idx, dim in enumerate(arg_value.dims):
                subs[transpiled_proc.get_size_param_name(arg_name, dim_idx)] = str(
                    dim.size
                )
                subs[transpiled_proc.get_stride_param_name(arg_name, dim_idx)] = str(
                    dim.stride
                )
        elif isinstance(arg_value, bool):
            subs[repr(arg_name)] = "true" if arg_value else "false"
        elif isinstance(arg_value, (int, float)):
            subs[repr(arg_name)] = str(arg_value)
        else:
            assert False
    for (config, field), config_value in test_case.ctxt.items():
        if isinstance(config_value, Tensor):
            assert config_value.data.shape == (1,)
            subs[transpiled_proc.get_config_param_name(config, field)] = str(
                config_value.data[0]
            )
        elif isinstance(config_value, bool):
            subs[transpiled_proc.get_config_param_name(config, field)] = (
                "true" if config_value else "false"
            )
        elif isinstance(config_value, (int, float)):
            subs[transpiled_proc.get_config_param_name(config, field)] = str(
                config_value
            )
        else:
            assert False

    buffer_args = tuple(
        test_case.arg_values[buffer_name].data.copy()
        for buffer_name in transpiled_proc.get_buffer_arg_order()
    )
    javascript = transpiled_proc.get_javascript_template().substitute(subs)
    try:
        eval_info = js_eval(javascript)(*buffer_args)
    except Exception as e:
        raise Exception(
            f"javascript:\n{javascript}\nproc:\n{transpiled_proc.get_proc()}"
        ) from e
    if transpiled_proc.get_coverage_skeleton() is None:
        [result, ctxt_object] = eval_info
        coverage_result = None
    else:
        [result, ctxt_object, coverage_result] = eval_info
    if result != 0:
        return "failed"
    return TestResult(
        {
            buffer_name: buffer_value
            for buffer_name, buffer_value in zip(
                transpiled_proc.get_buffer_arg_order(), buffer_args
            )
        },
        ctxt_object,
        coverage_result,
    )


@dataclass
class TestSpec:
    proc: LoopIR.proc
    constraint: DisjointConstraint
    arg_types: dict[Sym, LoopIR.type]
    original_scope: Block
    var_renaming: dict[Sym, Sym]

    def forward_to_test(self, cursor: Block) -> Optional[Block]:
        if cursor in self.original_scope:
            return Block(
                self.proc,
                Node(self.proc, []),
                "body",
                range(
                    cursor._range.start - self.original_scope._range.start,
                    cursor._range.stop - self.original_scope._range.stop,
                ),
            )
        for node_idx, node in enumerate(self.original_scope):
            if node.is_ancestor_of(cursor):
                return Block(
                    self.proc,
                    Node(
                        self.proc,
                        [("body", node_idx)] + cursor._anchor._path[len(node._path) :],
                    ),
                    cursor._attr,
                    cursor._range,
                )
        return None

    def backward_from_test(self, path: NodePath) -> NodePath:
        assert path.path[0][1] is not None
        return NodePath(
            tuple(self.original_scope._anchor._path)
            + (
                (
                    self.original_scope._attr,
                    self.original_scope._range.start + path.path[0][1],
                ),
            )
            + path.path[1:]
        )


@dataclass
class TestScope:
    scope: Block

    def broaden(self) -> Optional["TestScope"]:
        if self.scope._anchor.depth() == 0:
            new_scope = self.scope.expand()
            if (
                new_scope._range.start == self.scope._range.start
                and new_scope._range.stop == self.scope._range.stop
            ):
                return None
            else:
                return TestScope(new_scope)
        else:
            return TestScope(self.scope._anchor.as_block())

    def transform(self, forward: Callable[[Cursor], Cursor]) -> "TestScope":
        if self.broaden() is None:
            return TestScope(forward(self.scope._anchor)._child_block("body"))
        return TestScope(forward(self.scope))

    def get_type_map(self) -> dict[Sym, LoopIR.type]:
        root_proc = self.scope.get_root()
        proc_type_visitor = TypeVisitor()
        proc_type_visitor.visit(root_proc)
        return proc_type_visitor.type_map

    def get_test_spec(
        self, cm: ConstraintMaker, type_map: dict[Sym, LoopIR.type]
    ) -> TestSpec:
        root_proc = self.scope.get_root()
        proc_type_visitor = TypeVisitor()
        proc_type_visitor.visit(root_proc)

        constraint = TRUE_CONSTRAINT
        constraint = constraint.intersect(
            collect_path_constraints(self.scope, cm, type_map)
        )
        args = [
            LoopIR.fnarg(
                name=var,
                type=arg_type,
                mem=DRAM if arg_mem is None else arg_mem,
                srcinfo=SrcInfo("", 0),
            )
            for var, (arg_type, arg_mem) in get_free_variables(
                proc_type_visitor.type_map,
                proc_type_visitor.mem_map,
                self.scope,
            ).items()
        ]
        args = sorted(
            args,
            key=lambda arg: (
                (2 if isinstance(arg.type, LoopIR.WindowType) else 1)
                if arg.type.is_numeric()
                else 0
            ),
        )

        proc = LoopIR.proc(
            name=root_proc.name,
            args=args,
            preds=[],
            body=(self.scope.resolve_all()),
            instr=None,
            srcinfo=root_proc.srcinfo,
        )
        modified_variable_visitor = ModifiedVariableVisitor(type_map)
        modified_variable_visitor.visit(proc)
        arg_types = {arg.name: arg.type for arg in args}
        return TestSpec(
            proc,
            constraint,
            arg_types,
            self.scope,
            {sym: cm.top_var(sym) for sym in modified_variable_visitor.modified_vars},
        )


TEST_CASE_BOUND = 15
MAX_FAILS = 3
MAX_ITERS = 20
TIME_RECORDING_FILE = None  # "./times.csv"


@dataclass
class Timer:
    fuzz_start: Optional[int] = None
    transpile_start: Optional[int] = None
    constraint_start: Optional[int] = None
    test_start: Optional[int] = None
    scope_widen_count: int = 0
    fuzz_total: int = 0
    transpile_total: int = 0
    constraint_total: int = 0
    test_total: int = 0

    def start_fuzz(self):
        if TIME_RECORDING_FILE is None:
            return
        self.fuzz_start = time.process_time_ns()

    def widen_scope(self):
        self.scope_widen_count += 1

    def end_fuzz(self):
        if TIME_RECORDING_FILE is None:
            return
        assert self.fuzz_start is not None
        self.fuzz_total += time.process_time_ns() - self.fuzz_start
        self.fuzz_start = None

    def start_transpile(self):
        if TIME_RECORDING_FILE is None:
            return
        self.transpile_start = time.process_time_ns()

    def end_transpile(self):
        if TIME_RECORDING_FILE is None:
            return
        assert self.transpile_start is not None
        self.transpile_total += time.process_time_ns() - self.transpile_start
        self.transpile_start = None

    def start_constraint(self):
        if TIME_RECORDING_FILE is None:
            return
        self.constraint_start = time.process_time_ns()

    def end_constraint(self):
        if TIME_RECORDING_FILE is None:
            return
        assert self.constraint_start is not None
        self.constraint_total += time.process_time_ns() - self.constraint_start
        self.constraint_start = None

    def start_test(self):
        if TIME_RECORDING_FILE is None:
            return
        self.test_start = time.process_time_ns()

    def end_test(self):
        if TIME_RECORDING_FILE is None:
            return
        assert self.test_start is not None
        self.test_total += time.process_time_ns() - self.test_start
        self.test_start = None

    def record(self, failed: bool, unsolved: bool):
        if TIME_RECORDING_FILE is None:
            return
        with open(TIME_RECORDING_FILE, "a") as recording:
            recording.write(
                f"{self.fuzz_total},{self.transpile_total},{self.constraint_total},{self.test_total},{self.scope_widen_count},{1 if failed else 0},{1 if unsolved else 0}\n"
            )


def fuzz(starting_scope: Union[Block, Node], fwd: Callable[[Cursor], Cursor]):
    timer = Timer()
    timer.start_fuzz()
    starting_scope = Cursor.create(starting_scope.get_root())
    if isinstance(starting_scope, Node) and starting_scope.depth() == 0:
        starting_scope = starting_scope.body()

    starting_scope = (
        starting_scope.as_block()
        if isinstance(starting_scope, Node)
        else starting_scope
    )
    failure_scope = starting_scope
    failure_transformed_scope = fwd(failure_scope)
    assert isinstance(failure_transformed_scope, Block)
    cur_scope = TestScope(starting_scope)
    cur_type_map = cur_scope.get_type_map()
    transformed_type_map = cur_scope.transform(fwd).get_type_map()

    while cur_scope is not None:
        timer.start_transpile()
        transformed = cur_scope.transform(fwd)
        cm = ConstraintMaker(cur_type_map | transformed_type_map)

        spec1 = cur_scope.get_test_spec(cm, cur_type_map)
        spec2 = transformed.get_test_spec(cm, transformed_type_map)

        transpiled_test1 = Transpiler(
            spec1.proc,
            CoverageArgs(cm, spec1.var_renaming, spec1.forward_to_test(failure_scope)),
        )
        transpiled_test2 = Transpiler(
            spec2.proc,
            CoverageArgs(
                cm, spec2.var_renaming, spec2.forward_to_test(failure_transformed_scope)
            ),
        )

        config_fields = transpiled_test1.get_configs() | transpiled_test2.get_configs()

        arg_types = spec1.arg_types | spec2.arg_types
        constraint = spec1.constraint.union(spec2.constraint)
        skeleton1, skeleton2 = (
            transpiled_test1.get_coverage_skeleton(),
            transpiled_test2.get_coverage_skeleton(),
        )
        assert skeleton1 is not None and skeleton2 is not None
        coverage_skeleton = skeleton1.merge(skeleton2)
        tests_passed = True
        fails = 0
        iters = 0
        timer.end_transpile()
        while (
            not coverage_skeleton.get_coverage_progress().is_finished()
            and iters < MAX_ITERS
            and tests_passed
        ):
            timer.start_constraint()
            test_case = generate_test_case(
                arg_types,
                config_fields,
                constraint,
                coverage_skeleton,
                cm,
            )
            timer.end_constraint()
            if test_case is None:
                fails += 1
                if fails > MAX_FAILS:
                    timer.end_fuzz()
                    timer.record(False, True)
                    return
                else:
                    continue

            timer.start_test()
            out1 = run_test_case(test_case, transpiled_test1)
            out2 = run_test_case(test_case, transpiled_test2)
            if out1 == "failed" or out2 == "failed":
                tests_passed = False
                timer.end_test()
                break
            assert out1.coverage_result is not None and out2.coverage_result is not None
            coverage_skeleton.update_coverage(
                out1.coverage_result | out2.coverage_result
            )
            for buffer_name in out1.buffer_values.keys() & out2.buffer_values.keys():
                if not np.allclose(
                    out1.buffer_values[buffer_name], out2.buffer_values[buffer_name]
                ):
                    tests_passed = False
                    break
            if cur_scope.broaden() is not None:
                for ctxt_name in out1.ctxt_object & out2.ctxt_object.keys():
                    if not np.allclose(
                        out1.ctxt_object[ctxt_name], out2.ctxt_object[ctxt_name]
                    ):
                        tests_passed = False
                        break
            timer.end_test()
            iters += 1
        if tests_passed:
            timer.end_fuzz()
            timer.record(False, False)
            return
        else:
            timer.widen_scope()
            cur_scope = cur_scope.broaden()
    timer.end_fuzz()
    timer.record(True, False)
    raise SchedulingError("tests failed at broadest scope")


def fuzz_reorder_stmts(s1: Node, s2: Node):
    starting_scope = s1.as_block().expand(0, 1)
    _, fwd = s2._move(s1.before())
    patched_fwd = lambda cursor: (
        fwd(cursor) if isinstance(cursor, Node) else fwd(s2).as_block().expand(0, 1)
    )
    fuzz(starting_scope, patched_fwd)
