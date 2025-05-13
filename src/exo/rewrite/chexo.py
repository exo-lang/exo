from itertools import chain
from typing import Callable, Literal, Optional, Union

from ..core.internal_cursors import Cursor, Block, Node

from ..backend.LoopIR_transpiler import CoverageArgs, Transpiler
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


# @dataclass
# class LoopFlattener(LoopIRModifier):
#     universal_var_types: dict[Sym, LoopIR.type] = field(default_factory=lambda: {})
#     loop_syms: Optional[ReadWriteSyms] = None

#     def visit(self, node):
#         if isinstance(node, LoopIR.For):
#             old_loop_syms = self.loop_syms
#             new_node = self.visit_generic(node)
#             self.loop_syms = old_loop_syms
#         elif isinstance(node, LoopIR.Assign):
#         elif isinstance(node, LoopIR.Reduce):
#         elif isinstance(node, LoopIR.WriteConfig):


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
    cursor: Union[Block, Node], cm: ConstraintMaker
) -> DisjointConstraint:
    cur = cursor
    result = TRUE_CONSTRAINT
    last_attr = None
    while cur.depth() != 0:
        if isinstance(cur, Node):
            last_attr = cur._path[-1]
            if isinstance(cur._node, LoopIR.For):
                result = result.intersect(
                    cm.make_constraint_from_inequality(
                        cur._node.iter, cur._node.lo, ">="
                    ).lift_to_disjoint_constraint()
                )
                result = result.intersect(
                    cm.make_constraint_from_inequality(
                        cur._node.iter, cur._node.hi, "<"
                    ).lift_to_disjoint_constraint()
                )
            elif isinstance(cur._node, LoopIR.If):
                constraint = cm.make_constraint(cur._node.cond)
                if isinstance(last_attr, tuple) and last_attr[0] == "orelse":
                    result = result.intersect(constraint.invert())
                else:
                    result = result.intersect(constraint)

        cur = cur.parent()
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
        if arg_type.is_numeric():
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
    coverage_result: Optional[dict[str, Union[bool, memoryview, float]]]


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


@dataclass
class TestScope:
    scope: Union[Block, Node]
    flatten_loops: bool

    def broaden(self) -> Optional["TestScope"]:
        if self.scope.depth() == 0:
            return TestScope(self.scope, False) if self.flatten_loops else None
        else:
            return TestScope(self.scope.parent(), self.flatten_loops)

    def transform(self, forward: Callable[[Cursor], Cursor]) -> "TestScope":
        return TestScope(forward(self.scope), self.flatten_loops)

    def get_type_map(self) -> dict[Sym, LoopIR.type]:
        root_proc = self.scope.get_root()
        proc_type_visitor = TypeVisitor()
        proc_type_visitor.visit(root_proc)
        return proc_type_visitor.type_map

    def get_test_spec(self, cm: ConstraintMaker) -> TestSpec:
        root_proc = self.scope.get_root()
        proc_type_visitor = TypeVisitor()
        proc_type_visitor.visit(root_proc)

        constraint = TRUE_CONSTRAINT
        for pred in root_proc.preds:
            constraint = constraint.intersect(cm.make_constraint(pred))
        constraint = constraint.intersect(collect_path_constraints(self.scope, cm))
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
        args = [arg for arg in args if not arg.type.is_numeric()] + [
            arg for arg in args if arg.type.is_numeric()
        ]

        proc = LoopIR.proc(
            name=root_proc.name,
            args=args,
            preds=[],
            body=(
                [self.scope._node]
                if isinstance(self.scope, Node)
                else self.scope.resolve_all()
            ),
            instr=None,
            srcinfo=root_proc.srcinfo,
        )
        arg_types = {arg.name: arg.type for arg in args}
        return TestSpec(proc, constraint, arg_types)


TEST_CASE_BOUND = 15


def fuzz(starting_scope: Union[Block, Node], fwd: Callable[[Cursor], Cursor]):
    cur_scope = TestScope(starting_scope, True)
    transformed = cur_scope.transform(fwd)

    cm = ConstraintMaker(cur_scope.get_type_map() | transformed.get_type_map())

    spec1 = cur_scope.get_test_spec(cm)
    spec2 = transformed.get_test_spec(cm)

    failure_scope = (
        starting_scope.as_block()
        if isinstance(starting_scope, Node)
        else starting_scope
    )
    transpiled_test1 = Transpiler(spec1.proc, CoverageArgs(cm, failure_scope))
    transpiled_test2 = Transpiler(spec2.proc, CoverageArgs(cm, failure_scope))

    config_fields = transpiled_test1.get_configs() | transpiled_test2.get_configs()

    arg_types = spec1.arg_types | spec2.arg_types
    constraint = spec1.constraint.union(spec2.constraint)
    skeleton1, skeleton2 = (
        transpiled_test1.get_coverage_skeleton(),
        transpiled_test2.get_coverage_skeleton(),
    )
    assert skeleton1 is not None and skeleton2 is not None
    coverage_skeleton = skeleton1.merge(skeleton2)
    for _ in range(TEST_CASE_BOUND):
        test_case = generate_test_case(
            arg_types,
            config_fields,
            constraint,
            coverage_skeleton,
            cm,
        )
        if test_case is None:
            continue

        out1 = run_test_case(test_case, transpiled_test1)
        out2 = run_test_case(test_case, transpiled_test2)
        if out1 == "failed" or out2 == "failed":
            raise SchedulingError("domain mismatch")
        assert out1.coverage_result is not None and out2.coverage_result is not None
        coverage_skeleton.update_coverage(out1.coverage_result | out2.coverage_result)
        for buffer_name in out1.buffer_values.keys() & out2.buffer_values.keys():
            if not np.allclose(
                out1.buffer_values[buffer_name], out2.buffer_values[buffer_name]
            ):
                raise SchedulingError("mismatch found")
        for ctxt_name in out1.ctxt_object & out2.ctxt_object.keys():
            if not np.allclose(
                out1.ctxt_object[ctxt_name], out2.ctxt_object[ctxt_name]
            ):
                raise SchedulingError("context mismatch found")


def fuzz_reorder_stmts(s1: Node, s2: Node):
    starting_scope = s1.as_block().expand(0, 1)
    _, fwd = s2._move(s1.before())
    patched_fwd = lambda cursor: (
        fwd(cursor) if isinstance(cursor, Node) else fwd(s2).as_block().expand(0, 1)
    )
    fuzz(starting_scope, patched_fwd)
