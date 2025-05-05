from functools import reduce
from itertools import chain
from string import Template
from typing import Any, Iterable, Optional, Union

from ..core.configs import Config

from ..core.prelude import Sym
from ..core.LoopIR import LoopIR, T
from .coverage import (
    CoverageSkeleton,
    CoverageSkeletonNode,
    CoverageSkeletonBranch,
    IndexedFiller,
    MemoryAccess,
    MemoryAccessPair,
)
from ..rewrite.constraint_solver import ConstraintMaker
from dataclasses import dataclass

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
class Dimension:
    size: str
    stride: str


@dataclass
class Tensor:
    name: Sym
    offset: str
    dims: tuple[Dimension, ...]
    resize_placeholder: Optional[int]


ExoValue = Union[Constant, Reference, Tensor]


CONTEXT_OBJECT_NAME = "ctxt"
INITIAL_DYNAMIC_SIZE = 16


@dataclass
class CoverageArgs:
    cm: ConstraintMaker


class CoverageState:
    def __init__(self, args: CoverageArgs, cov_placeholder: int):
        self.cm: ConstraintMaker = args.cm
        self.root: CoverageSkeletonNode = CoverageSkeletonNode(None, None, ())
        self.buffer_writes: dict[Sym, list[MemoryAccess]] = {}
        self.buffer_reads: dict[Sym, list[MemoryAccess]] = {}
        self.free_vars: list[Sym] = []
        self.cov_placeholder = cov_placeholder

    def make_skeleton(self) -> CoverageSkeleton:
        aliasable_accesses: list[MemoryAccessPair] = []
        for sym, write_indices in self.buffer_writes.items():
            read_indices = self.buffer_reads[sym] if sym in self.buffer_reads else []
            for i, index1 in enumerate(write_indices):
                for index2 in chain(write_indices[i + 1 :], read_indices):
                    aliasable_accesses.append(MemoryAccessPair(index1, index2))

        return CoverageSkeleton(
            (self.root,), tuple(aliasable_accesses), frozenset(self.free_vars)
        )


class Transpiler:
    def __init__(self, proc: LoopIR.proc, coverage_args: Optional[CoverageArgs] = None):
        self._name_lookup: dict[Sym, ExoValue] = {}
        self._js_lines: list[str] = []
        self._configs: set[tuple[Config, str]] = set()
        self._buffer_args: list[Sym] = []
        self._coverage_state: Optional[CoverageState] = None
        self._skeleton: Optional[CoverageSkeleton] = None
        self.proc = proc  # debug
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

    def _assert_at_runtime(self, expr: str):
        self._js_lines.append(f"if(!{expr})return [1,{CONTEXT_OBJECT_NAME},{{}}];")

    def _make_placeholder(self) -> int:
        placeholder_index = len(self._js_lines)
        self._js_lines.append("")
        return placeholder_index

    def _transpile_proc(self, proc: LoopIR.proc, coverage_args: Optional[CoverageArgs]):
        arg_values = []
        for arg in proc.args:
            if arg.type.is_numeric():
                self._buffer_args.append(arg.name)
                if arg.type.is_tensor_or_window():
                    value = Tensor(
                        arg.name,
                        "0",
                        tuple(
                            Dimension(
                                f"${self.get_size_param_name(arg.name, dim_idx)}",
                                f"${self.get_stride_param_name(arg.name, dim_idx)}",
                            )
                            for dim_idx in range(len(arg.type.shape()))
                        ),
                        None,
                    )
                else:
                    value = Reference(repr(arg.name), False)
            else:
                value = Constant(f"${repr(arg.name)}")
            arg_values.append(value)
        self._js_lines.append(
            f'(({",".join(repr(arg) for arg in self._buffer_args)})=>{{'
        )
        ctxt_placeholder = self._make_placeholder()
        if coverage_args is not None:
            self._coverage_state = CoverageState(
                coverage_args, self._make_placeholder()
            )
        self._call_proc(
            proc,
            tuple(arg_values),
            None if self._coverage_state is None else self._coverage_state.root,
        )
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

    def _call_proc(
        self,
        proc: LoopIR.proc,
        arg_values: tuple[ExoValue, ...],
        coverage_node: Optional[CoverageSkeletonNode],
    ):
        for arg, arg_value in zip(proc.args, arg_values):
            self._name_lookup[arg.name] = arg_value
            if arg.type.is_tensor_or_window():
                assert isinstance(arg_value, Tensor)
                for arg_dim, arg_dim_expr in zip(arg_value.dims, arg.type.shape()):
                    self._assert_at_runtime(
                        f"({arg_dim.size}=={self._transpile_expr(arg_dim_expr, None)})",
                    )

        for pred in proc.preds:
            self._assert_at_runtime(self._transpile_expr(pred, None))

        for stmt in proc.body:
            self._transpile_stmt(stmt, coverage_node)

    def _get_index_expr(
        self,
        buf: Tensor,
        idxs: Iterable[LoopIR.expr],
        coverage_node: Optional[CoverageSkeletonNode],
    ):
        idx_exprs = tuple(self._transpile_expr(idx, coverage_node) for idx in idxs)
        for idx_expr, dim in zip(idx_exprs, buf.dims):
            self._assert_at_runtime(f"({idx_expr}<{dim.size}&&{idx_expr}>=0)")
        relative_idx = reduce(
            lambda dim1, dim2: f"{dim1}+{dim2}",
            (
                f"Math.imul({idx_expr},{dim.stride})"
                for idx_expr, dim in zip(idx_exprs, buf.dims)
            ),
        )
        return f"{relative_idx}+{buf.offset}"

    def _make_scalar_access_fillers(self, access_sym: Sym) -> tuple[IndexedFiller, ...]:
        assert self._coverage_state is not None
        mark_placeholder = self._make_placeholder()
        mark_stmt = f"{repr(access_sym)}=true;"
        decl_stmt = f"let {repr(access_sym)}=false;"
        return (
            IndexedFiller(mark_placeholder, mark_stmt),
            IndexedFiller(self._coverage_state.cov_placeholder, decl_stmt),
        )

    def _make_tensor_access_fillers(
        self, access_sym: Sym, buffer: Tensor, idx: Iterable[LoopIR.expr]
    ) -> tuple[IndexedFiller, ...]:
        assert self._coverage_state is not None
        mark_stmt = f"{repr(access_sym)}[{self._get_index_expr(buffer, idx, None)}]=1;"
        mark_placeholder = self._make_placeholder()
        base_buffer = self._name_lookup[buffer.name]
        assert isinstance(base_buffer, Tensor)
        base_dims = base_buffer.dims
        base_size = reduce(
            lambda dim1, dim2: f"Math.imul({dim1},{dim2})",
            (dim.size for dim in base_dims),
        )
        if buffer.resize_placeholder is None:
            decl_stmt = f"let {repr(access_sym)}=new ArrayBuffer({base_size});"
            return (
                IndexedFiller(mark_placeholder, mark_stmt),
                IndexedFiller(self._coverage_state.cov_placeholder, decl_stmt),
            )
        else:
            temp_sym = Sym("temp")
            decl_stmt = f"let {repr(access_sym)}=new ArrayBuffer(1,{{maxByteLength:{INITIAL_DYNAMIC_SIZE}}});"
            resize_stmt = f"while({base_size}>{repr(access_sym)}.maxByteLength){{let {repr(temp_sym)}=new ArrayBuffer({repr(access_sym)}.byteLength,{{maxByteLength:2*{repr(access_sym)}.maxByteLength}});for(let i=0;i<{repr(access_sym)}.byteLength;i++){repr(temp_sym)}[i]={repr(access_sym)}[i];{repr(access_sym)}={repr(temp_sym)}}};{repr(access_sym)}.resize(Math.max({base_size},{repr(access_sym)}.byteLength));"
            return (
                IndexedFiller(mark_placeholder, mark_stmt),
                IndexedFiller(self._coverage_state.cov_placeholder, decl_stmt),
                IndexedFiller(buffer.resize_placeholder, resize_stmt),
            )

    def _transpile_stmt(
        self, stmt: LoopIR.stmt, coverage_node: Optional[CoverageSkeletonNode]
    ):
        if isinstance(stmt, (LoopIR.Assign, LoopIR.Reduce)):
            lhs_buffer = self._name_lookup[stmt.name]

            if self._coverage_state is not None and coverage_node is not None:
                write_sym = Sym("write")
                if stmt.name not in self._coverage_state.buffer_writes:
                    self._coverage_state.buffer_writes[
                        lhs_buffer.name if isinstance(lhs_buffer, Tensor) else stmt.name
                    ] = []
                self._coverage_state.buffer_writes[
                    lhs_buffer.name if isinstance(lhs_buffer, Tensor) else stmt.name
                ].append(
                    MemoryAccess(
                        write_sym,
                        coverage_node,
                        tuple(
                            self._coverage_state.cm.make_expression(idx)
                            for idx in stmt.idx
                        ),
                        (
                            self._make_tensor_access_fillers(
                                write_sym, lhs_buffer, stmt.idx
                            )
                            if isinstance(lhs_buffer, Tensor)
                            else self._make_scalar_access_fillers(write_sym)
                        ),
                    )
                )
            rhs = self._transpile_expr(stmt.rhs, coverage_node)
            if isinstance(lhs_buffer, Reference):
                lhs = (
                    lhs_buffer.name if lhs_buffer.is_config else f"{lhs_buffer.name}[0]"
                )
            elif isinstance(lhs_buffer, Tensor):
                lhs = f"{repr(lhs_buffer.name)}[{self._get_index_expr(lhs_buffer, stmt.idx, coverage_node)}]"
            else:
                assert False
            if isinstance(stmt, LoopIR.Assign):
                self._js_lines.append(f"{lhs}={rhs};")
            else:
                self._js_lines.append(f"{lhs}+={rhs};")
        elif isinstance(stmt, LoopIR.WriteConfig):
            config_name = self.get_config_param_name(stmt.config, stmt.field)
            rhs = self._transpile_expr(stmt.rhs, coverage_node)
            self._js_lines.append(f'{CONTEXT_OBJECT_NAME}["{config_name}"]={rhs};')
            self._configs.add((stmt.config, stmt.field))
        elif isinstance(stmt, LoopIR.Pass):
            pass
        elif isinstance(stmt, LoopIR.If):
            cond = self._transpile_expr(stmt.cond, coverage_node)
            self._js_lines.append(f"if({cond}){{")

            if self._coverage_state is not None and coverage_node is not None:
                true_sym = Sym("true_case")
                false_sym = Sym("false_case")
                true_placeholder = self._make_placeholder()
                cond_constraint = self._coverage_state.cm.make_constraint(stmt.cond)
                true_node = CoverageSkeletonNode(
                    true_sym,
                    (coverage_node, cond_constraint),
                    (
                        IndexedFiller(
                            self._coverage_state.cov_placeholder,
                            f"let {repr(true_sym)}=false;",
                        ),
                        IndexedFiller(true_placeholder, f"{repr(true_sym)}=true;"),
                    ),
                )
                for body_stmt in stmt.body:
                    self._transpile_stmt(body_stmt, true_node)
                self._js_lines.append("}else{")
                false_placeholder = self._make_placeholder()
                false_node = CoverageSkeletonNode(
                    false_sym,
                    (coverage_node, cond_constraint.invert()),
                    (
                        IndexedFiller(
                            self._coverage_state.cov_placeholder,
                            f"let {repr(false_sym)}=false;",
                        ),
                        IndexedFiller(false_placeholder, f"{repr(false_sym)}=true;"),
                    ),
                )
                for else_stmt in stmt.orelse:
                    self._transpile_stmt(else_stmt, false_node)
                new_branch = CoverageSkeletonBranch(true_node, false_node)
                coverage_node.branches.append(new_branch)
            else:
                for body_stmt in stmt.body:
                    self._transpile_stmt(body_stmt, None)
                self._js_lines.append("}else{")
                for else_stmt in stmt.orelse:
                    self._transpile_stmt(else_stmt, None)
            self._js_lines.append("}")
        elif isinstance(stmt, LoopIR.For):
            iter_name = repr(stmt.iter)
            iter_lo = self._transpile_expr(stmt.lo, coverage_node)
            iter_hi = self._transpile_expr(stmt.hi, coverage_node)
            self._name_lookup[stmt.iter] = Constant(iter_name)

            body_child, skip_child = None, None
            if self._coverage_state is not None and coverage_node is not None:
                body_sym = Sym("body")
                skip_sym = Sym("skip")
                loop_placeholder = self._make_placeholder()
                body_constraint = (
                    self._coverage_state.cm.make_constraint_from_inequality(
                        stmt.lo, stmt.iter, "<="
                    )
                    .lift_to_disjoint_constraint()
                    .intersect(
                        self._coverage_state.cm.make_constraint_from_inequality(
                            stmt.iter, stmt.hi, "<"
                        ).lift_to_disjoint_constraint()
                    )
                )
                skip_constraint = (
                    self._coverage_state.cm.make_constraint_from_inequality(
                        stmt.lo, stmt.hi, ">="
                    ).lift_to_disjoint_constraint()
                )
                body_child = CoverageSkeletonNode(
                    body_sym,
                    (coverage_node, body_constraint),
                    (
                        IndexedFiller(
                            self._coverage_state.cov_placeholder,
                            f"let {repr(body_sym)}=false;",
                        ),
                        IndexedFiller(
                            loop_placeholder,
                            f"{repr(body_sym)}||=({iter_lo}<{iter_hi});",
                        ),
                    ),
                )
                skip_child = CoverageSkeletonNode(
                    skip_sym,
                    (coverage_node, skip_constraint),
                    (
                        IndexedFiller(
                            self._coverage_state.cov_placeholder,
                            f"let {repr(skip_sym)}=false;",
                        ),
                        IndexedFiller(
                            loop_placeholder,
                            f"{repr(skip_sym)}||=({iter_lo}>={iter_hi});",
                        ),
                    ),
                )
                self._coverage_state.free_vars.append(stmt.iter)
                new_loop = CoverageSkeletonBranch(body_child, skip_child)
                coverage_node.branches.append(new_loop)

            self._js_lines.append(
                f"for(let {iter_name}={iter_lo};{iter_name}<{iter_hi};{iter_name}++){{"
            )
            for body_stmt in stmt.body:
                self._transpile_stmt(body_stmt, body_child)
            self._js_lines.append("}")
        elif isinstance(stmt, LoopIR.Alloc):
            assert stmt.type.is_numeric()
            if stmt.type.is_tensor_or_window():
                tensor_name = repr(stmt.name)
                buffer_type = lookup_loopir_type(
                    stmt.type.basetype()
                ).javascript_array_type
                dim_exprs = tuple(
                    self._transpile_expr(dim, coverage_node)
                    for dim in stmt.type.shape()
                )
                for dim_expr in dim_exprs:
                    self._assert_at_runtime(f"({dim_expr}>=0)")
                buffer_size = reduce(
                    lambda dim1, dim2: f"Math.imul({dim1},{dim2})", dim_exprs
                )
                self._js_lines.append(
                    f"let {tensor_name}=new {buffer_type}({buffer_size});"
                )
                resize_placeholder = len(self._js_lines)
                self._js_lines.append("")
                dimensions: list[Dimension] = []
                for dim_idx, dim_expr in enumerate(dim_exprs):
                    self._assert_at_runtime(f"({dim_expr}>=0)")
                    stride_expr = reduce(
                        lambda dim1, dim2: f"Math.imul({dim1},{dim2})",
                        dim_exprs[dim_idx + 1 :],
                        "1",
                    )
                    dimensions.append(Dimension(dim_expr, stride_expr))
                self._name_lookup[stmt.name] = Tensor(
                    stmt.name, "0", tuple(dimensions), resize_placeholder
                )
            else:
                ref_name = repr(stmt.name)
                buffer_type = lookup_loopir_type(stmt.type).javascript_array_type
                self._js_lines.append(f"let {ref_name}=new {buffer_type}(1);")
                self._name_lookup[stmt.name] = Reference(ref_name, False)
        elif isinstance(stmt, LoopIR.Free):
            pass
        elif isinstance(stmt, LoopIR.Call):
            self._call_proc(
                stmt.f,
                tuple(
                    (
                        self._transpile_buffer_arg(arg_expr, coverage_node)
                        if arg_expr.type.is_numeric()
                        else Constant(self._transpile_expr(arg_expr, coverage_node))
                    )
                    for arg_expr in stmt.args
                ),
                coverage_node,
            )
        elif isinstance(stmt, LoopIR.WindowStmt):
            self._name_lookup[stmt.name] = self._transpile_buffer_arg(
                stmt.rhs, coverage_node
            )
        else:
            assert False, "unsupported stmt"

    def _transpile_buffer_arg(
        self, expr: LoopIR.expr, coverage_node: Optional[CoverageSkeletonNode]
    ) -> Union[Tensor, Reference]:
        if isinstance(expr, LoopIR.Read):
            assert len(expr.idx) == 0
            buf = self._name_lookup[expr.name]
            assert isinstance(buf, (Tensor, Reference))
            return buf
        elif isinstance(expr, LoopIR.WindowExpr):
            base = self._name_lookup[expr.name]
            assert isinstance(base, Tensor)
            offset_expr = base.offset
            window_dims = []
            for idx, dim in zip(expr.idx, base.dims):
                if isinstance(idx, LoopIR.Interval):
                    lo_expr = self._transpile_expr(idx.lo, coverage_node)
                    hi_expr = self._transpile_expr(idx.hi, coverage_node)
                    self._assert_at_runtime(
                        f"(0<={lo_expr}&&{lo_expr}<={hi_expr}&&{hi_expr}<={dim.size})"
                    )
                    offset_expr = f"({offset_expr}+Math.imul({lo_expr},{dim.stride}))"
                    size_expr = f"({hi_expr}-{lo_expr})"
                    window_dims.append(Dimension(size_expr, dim.stride))
                elif isinstance(idx, LoopIR.Point):
                    pt_expr = self._transpile_expr(idx.pt, coverage_node)
                    self._assert_at_runtime(f"(0<={pt_expr}&&{pt_expr}<{dim.size})")
                    offset_expr = f"({offset_expr}+Math.imul({pt_expr},{dim.stride}))"
                else:
                    assert False, "not a window index"
            return Tensor(
                base.name, offset_expr, tuple(window_dims), base.resize_placeholder
            )
        elif isinstance(expr, LoopIR.ReadConfig):
            self._configs.add((expr.config, expr.field))
            return Reference(
                f'{CONTEXT_OBJECT_NAME}["{self.get_config_param_name(expr.config, expr.field)}"]',
                True,
            )
        else:
            assert False, "unsupported buffer expression"

    def _transpile_expr(
        self, expr: LoopIR.expr, coverage_node: Optional[CoverageSkeletonNode]
    ) -> str:
        if isinstance(expr, LoopIR.Read):
            buf = self._name_lookup[expr.name]
            if self._coverage_state is not None and coverage_node is not None:
                read_sym = Sym("read")
                if expr.name not in self._coverage_state.buffer_reads:
                    self._coverage_state.buffer_reads[
                        buf.name if isinstance(buf, Tensor) else expr.name
                    ] = []
                self._coverage_state.buffer_reads[
                    buf.name if isinstance(buf, Tensor) else expr.name
                ].append(
                    MemoryAccess(
                        read_sym,
                        coverage_node,
                        tuple(
                            self._coverage_state.cm.make_expression(idx)
                            for idx in expr.idx
                        ),
                        (
                            self._make_tensor_access_fillers(read_sym, buf, expr.idx)
                            if isinstance(buf, Tensor)
                            else self._make_scalar_access_fillers(read_sym)
                        ),
                    )
                )
            if isinstance(buf, Tensor):
                return f"{repr(buf.name)}[{self._get_index_expr(buf, expr.idx, coverage_node)}]"
            elif isinstance(buf, Reference):
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
            return f"(-{self._transpile_expr(expr.arg, coverage_node)})"
        elif isinstance(expr, LoopIR.BinOp):
            lhs = self._transpile_expr(expr.lhs, coverage_node)
            rhs = self._transpile_expr(expr.rhs, coverage_node)
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
                tuple(self._transpile_expr(arg, coverage_node) for arg in expr.args)
            )
        elif isinstance(expr, LoopIR.WindowExpr):
            assert False, "unexpected window expr"
        elif isinstance(expr, LoopIR.StrideExpr):
            buf = self._name_lookup[expr.name]
            assert isinstance(buf, Tensor)
            return buf.dims[expr.dim].stride
        elif isinstance(expr, LoopIR.ReadConfig):
            self._configs.add((expr.config, expr.field))
            return f'{CONTEXT_OBJECT_NAME}["{self.get_config_param_name(expr.config, expr.field)}"]'
        else:
            assert False, "unexpected expr"
