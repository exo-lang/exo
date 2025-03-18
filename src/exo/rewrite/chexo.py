from typing import Optional, Union

from ..backend.LoopIR_transpiler import Transpiler

from ..core.configs import Config

from ..core.LoopIR import LoopIR, T
from dataclasses import dataclass, field
from ..core.prelude import Sym, SrcInfo
from ..core.memory import DRAM, Memory
from ..backend.LoopIR_interpreter import Interpreter, run_interpreter
import numpy as np
from .new_eff import SchedulingError
from .constraint_solver import (
    TRUE_CONSTRAINT,
    Constraint,
    ConstraintClause,
    ConstraintMaker,
    ConstraintTerm,
    DisjointConstraint,
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


@dataclass
class Dimension:
    size: int
    stride: int


@dataclass
class Tensor:
    data: np.ndarray
    dims: tuple[Dimension, ...]


def get_free_variables(type_map, mem_map, fragment):
    fragment_type_visitor = TypeVisitor()
    fragment_var_visitor = UsedVariableVisitor()
    for stmt in fragment:
        fragment_type_visitor.visit(stmt)
        fragment_var_visitor.visit(stmt)
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
MIN_BUFFER_SIZE_BOUND = 16**1
MAX_BUFFER_SIZE_BOUND = 16**6
SEARCH_LIMIT = 10
INT_BOUND = 128
FLOAT_BOUND = 32


def collect_path_constraints(cursor, cm: ConstraintMaker) -> DisjointConstraint:
    cur = cursor
    result = TRUE_CONSTRAINT
    last_attr = None
    while cur.depth() != 0:
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
        last_attr = cur._path[-1]

        cur = cur.parent()
    return result


def collect_arg_size_constraints(
    args: list[LoopIR.fnarg], cm: ConstraintMaker, buffer_size_bound: int
) -> DisjointConstraint:
    constraint = TRUE_CONSTRAINT
    for arg in args:
        if arg.type.is_tensor_or_window():
            dim_terms: tuple[ConstraintTerm, ...] = (ConstraintTerm(1, ()),)
            for dim_expr in arg.type.shape():
                dim_terms = tuple(
                    dim_term.multiply(rhs_term)
                    for dim_term in dim_terms
                    for rhs_term in cm.make_constraint_terms(dim_expr)
                )
            constraint = constraint.intersect(
                Constraint(
                    tuple(term.negate() for term in dim_terms)
                    + (ConstraintTerm(buffer_size_bound, ()),),
                    True,
                ).lift_to_disjoint_constraint()
            )
    return constraint


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
    args: list[LoopIR.fnarg],
    config_fields: frozenset[tuple[Config, str]],
    constraint: DisjointConstraint,
    cm: ConstraintMaker,
) -> Optional[TestCase]:
    ctxt = {}
    arg_values = {}
    solution = cm.solve_constraint(
        constraint, bound=CONTROL_VAL_BOUND, search_limit=SEARCH_LIMIT
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

    for arg in args:
        if not arg.type.is_numeric():
            if arg.name in solution.var_assignments:
                if isinstance(arg.type, T.Bool):
                    val = solution.var_assignments[arg.name] != 0
                else:
                    val = solution.var_assignments[arg.name]
            else:
                val = generate_control_value(arg.type)
            arg_values[arg.name] = val

    for arg in args:
        if arg.type.is_numeric():
            if arg.type.is_real_scalar():
                shape = (1,)
            else:
                shape = tuple(
                    eval_tensor_dimension(dim_expr, arg_values)
                    for dim_expr in arg.type.shape()
                )
            arg_values[arg.name] = generate_numeric_value(arg.type.basetype(), shape)

    return TestCase(arg_values, ctxt)


@dataclass
class TestResult:
    buffer_values: dict[Sym, np.ndarray]
    ctxt_object: dict[str, Union[int, float]]


def run_test_case(test_case: TestCase, transpiled_proc: Transpiler) -> TestResult:
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
        [result, ctxt_object] = js_eval(javascript)(*buffer_args)
    except Exception as e:
        print(e)

    assert result == 0
    return TestResult(
        {
            buffer_name: buffer_value
            for buffer_name, buffer_value in zip(
                transpiled_proc.get_buffer_arg_order(), buffer_args
            )
        },
        ctxt_object,
    )


TEST_CASE_BOUND = 15


def fuzz_reorder_stmts(s1, s2):
    proc = s1.get_root()
    proc_type_visitor = TypeVisitor()
    proc_type_visitor.visit(proc)

    cm = ConstraintMaker(proc_type_visitor.type_map)
    constraint = TRUE_CONSTRAINT
    for pred in proc.preds:
        constraint = constraint.intersect(cm.make_constraint(pred))
    constraint = constraint.intersect(collect_path_constraints(s1, cm))
    args = [
        LoopIR.fnarg(
            name=var,
            type=arg_type,
            mem=DRAM if arg_mem is None else arg_mem,
            srcinfo=SrcInfo("", 0),
        )
        for var, (arg_type, arg_mem) in get_free_variables(
            proc_type_visitor.type_map, proc_type_visitor.mem_map, [s1._node, s2._node]
        ).items()
    ]
    args = [arg for arg in args if not arg.type.is_numeric()] + [
        arg for arg in args if arg.type.is_numeric()
    ]

    transpiled_test1 = Transpiler(
        LoopIR.proc(
            name=proc.name,
            args=args,
            preds=[],
            body=[s1._node, s2._node],
            instr=None,
            srcinfo=proc.srcinfo,
        )
    )
    transpiled_test2 = Transpiler(
        LoopIR.proc(
            name=proc.name,
            args=args,
            preds=[],
            body=[s2._node, s1._node],
            instr=None,
            srcinfo=proc.srcinfo,
        )
    )

    config_fields = transpiled_test1.get_configs() | transpiled_test2.get_configs()
    buffer_size_bound = MIN_BUFFER_SIZE_BOUND
    for _ in range(TEST_CASE_BOUND):
        test_case = generate_test_case(
            args,
            config_fields,
            (
                constraint
                if buffer_size_bound is None
                else constraint.intersect(
                    collect_arg_size_constraints(args, cm, buffer_size_bound)
                )
            ),
            cm,
        )
        if test_case is None:
            if buffer_size_bound is None or buffer_size_bound >= MAX_BUFFER_SIZE_BOUND:
                assert buffer_size_bound is not None
                buffer_size_bound = None
            else:
                buffer_size_bound = min(MAX_BUFFER_SIZE_BOUND, buffer_size_bound * 4)
            continue

        out1 = run_test_case(test_case, transpiled_test1)
        out2 = run_test_case(test_case, transpiled_test2)
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
