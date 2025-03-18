from functools import reduce
from string import Template
from typing import Any, Iterable, Union

from ..core.configs import Config

from ..core.prelude import Sym
from ..core.LoopIR import LoopIR, T
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
    name: str
    offset: str
    dims: tuple[Dimension, ...]


ExoValue = Union[Constant, Reference, Tensor]


CONTEXT_OBJECT_NAME = "ctxt"


class Transpiler:
    def __init__(self, proc: LoopIR.proc):
        self._name_lookup: dict[Sym, ExoValue] = {}
        self._js_lines: list[str] = []
        self._configs: set[tuple[Config, str]] = set()
        self._buffer_args: list[Sym] = []
        self._transpile_proc(proc)

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

    def _assert_at_runtime(self, expr: str):
        self._js_lines.append(f"if(!{expr})return [1,{CONTEXT_OBJECT_NAME}];")

    def _transpile_proc(self, proc: LoopIR.proc):
        arg_values = []
        for arg in proc.args:
            if arg.type.is_numeric():
                self._buffer_args.append(arg.name)
                if arg.type.is_tensor_or_window():
                    value = Tensor(
                        repr(arg.name),
                        "0",
                        tuple(
                            Dimension(
                                f"${self.get_size_param_name(arg.name, dim_idx)}",
                                f"${self.get_stride_param_name(arg.name, dim_idx)}",
                            )
                            for dim_idx in range(len(arg.type.shape()))
                        ),
                    )
                else:
                    value = Reference(repr(arg.name), False)
            else:
                value = Constant(f"${repr(arg.name)}")
            arg_values.append(value)
        self._js_lines.append(
            f'(({",".join(repr(arg) for arg in self._buffer_args)})=>{{'
        )
        ctxt_placeholder = len(self._js_lines)
        self._js_lines.append(f"__placeholder__")
        self._call_proc(proc, tuple(arg_values))
        self._js_lines.append(f"return [0,{CONTEXT_OBJECT_NAME}];}})")
        configs = ",".join(
            f"{self.get_config_param_name(config, field)}:${self.get_config_param_name(config, field)}"
            for config, field in self._configs
        )
        self._js_lines[ctxt_placeholder] = f"{CONTEXT_OBJECT_NAME}={{{configs}}}"

    def _call_proc(self, proc: LoopIR.proc, arg_values: tuple[ExoValue, ...]):
        for arg, arg_value in zip(proc.args, arg_values):
            self._name_lookup[arg.name] = arg_value
            if arg.type.is_tensor_or_window():
                assert isinstance(arg_value, Tensor)
                for arg_dim, arg_dim_expr in zip(arg_value.dims, arg.type.shape()):
                    self._assert_at_runtime(
                        f"({arg_dim.size}=={self._transpile_expr(arg_dim_expr)})",
                    )

        for pred in proc.preds:
            self._assert_at_runtime(self._transpile_expr(pred))

        for stmt in proc.body:
            self._transpile_stmt(stmt)

    def _get_index_expr(self, buf: Tensor, idxs: Iterable[LoopIR.expr]):
        idx_exprs = tuple(self._transpile_expr(idx) for idx in idxs)
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

    def _transpile_stmt(self, stmt: LoopIR.stmt):
        if isinstance(stmt, (LoopIR.Assign, LoopIR.Reduce)):
            lhs_buffer = self._name_lookup[stmt.name]
            rhs = self._transpile_expr(stmt.rhs)
            if isinstance(lhs_buffer, Reference):
                lhs = (
                    lhs_buffer.name if lhs_buffer.is_config else f"{lhs_buffer.name}[0]"
                )
            elif isinstance(lhs_buffer, Tensor):
                lhs = f"{lhs_buffer.name}[{self._get_index_expr(lhs_buffer, stmt.idx)}]"
            else:
                assert False
            if isinstance(stmt, LoopIR.Assign):
                self._js_lines.append(f"{lhs}={rhs};")
            else:
                self._js_lines.append(f"{lhs}+={rhs};")
        elif isinstance(stmt, LoopIR.WriteConfig):
            config_name = self.get_config_param_name(stmt.config, stmt.field)
            rhs = self._transpile_expr(stmt.rhs)
            self._js_lines.append(f'{CONTEXT_OBJECT_NAME}["{config_name}"]={rhs};')
            self._configs.add((stmt.config, stmt.field))
        elif isinstance(stmt, LoopIR.Pass):
            pass
        elif isinstance(stmt, LoopIR.If):
            cond = self._transpile_expr(
                stmt.cond,
            )
            self._js_lines.append(f"if({cond}){{")
            for body_stmt in stmt.body:
                self._transpile_stmt(body_stmt)
            self._js_lines.append("}else{")
            for else_stmt in stmt.orelse:
                self._transpile_stmt(else_stmt)
            self._js_lines.append("}")
        elif isinstance(stmt, LoopIR.For):
            iter_name = repr(stmt.iter)
            iter_lo = self._transpile_expr(stmt.lo)
            iter_hi = self._transpile_expr(stmt.hi)
            self._name_lookup[stmt.iter] = Constant(iter_name)
            self._js_lines.append(
                f"for(let {iter_name}={iter_lo};{iter_name}<{iter_hi};{iter_name}++){{"
            )
            for body_stmt in stmt.body:
                self._transpile_stmt(body_stmt)
            self._js_lines.append("}")
        elif isinstance(stmt, LoopIR.Alloc):
            assert stmt.type.is_numeric()
            if stmt.type.is_tensor_or_window():
                tensor_name = repr(stmt.name)
                buffer_type = lookup_loopir_type(
                    stmt.type.basetype()
                ).javascript_array_type
                dim_exprs = tuple(
                    self._transpile_expr(dim) for dim in stmt.type.shape()
                )
                for dim_expr in dim_exprs:
                    self._assert_at_runtime(f"({dim_expr}>=0)")
                buffer_size = reduce(
                    lambda dim1, dim2: f"Math.imul({dim1},{dim2})", dim_exprs
                )
                self._js_lines.append(
                    f"let {tensor_name}=new {buffer_type}({buffer_size});"
                )
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
                    tensor_name, "0", tuple(dimensions)
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
                        self._transpile_buffer_arg(arg_expr)
                        if arg_expr.type.is_numeric()
                        else Constant(self._transpile_expr(arg_expr))
                    )
                    for arg_expr in stmt.args
                ),
            )
        elif isinstance(stmt, LoopIR.WindowStmt):
            self._name_lookup[stmt.name] = self._transpile_buffer_arg(stmt.rhs)
        else:
            assert False, "unsupported stmt"

    def _transpile_buffer_arg(self, expr: LoopIR.expr) -> Union[Tensor, Reference]:
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
                    lo_expr = self._transpile_expr(idx.lo)
                    hi_expr = self._transpile_expr(idx.hi)
                    self._assert_at_runtime(
                        f"(0<={lo_expr}&&{lo_expr}<={hi_expr}&&{hi_expr}<={dim.size})"
                    )
                    offset_expr = f"({offset_expr}+Math.imul({lo_expr},{dim.stride}))"
                    size_expr = f"({hi_expr}-{lo_expr})"
                    window_dims.append(Dimension(size_expr, dim.stride))
                elif isinstance(idx, LoopIR.Point):
                    pt_expr = self._transpile_expr(idx.pt)
                    self._assert_at_runtime(f"(0<={pt_expr}&&{pt_expr}<{dim.size})")
                    offset_expr = f"({offset_expr}+Math.imul({pt_expr},{dim.stride}))"
                else:
                    assert False, "not a window index"
            return Tensor(base.name, offset_expr, tuple(window_dims))
        elif isinstance(expr, LoopIR.ReadConfig):
            self._configs.add((expr.config, expr.field))
            return Reference(
                f'{CONTEXT_OBJECT_NAME}["{self.get_config_param_name(expr.config, expr.field)}"]',
                True,
            )
        else:
            assert False, "unsupported buffer expression"

    def _transpile_expr(
        self,
        expr: LoopIR.expr,
    ) -> str:
        if isinstance(expr, LoopIR.Read):
            buf = self._name_lookup[expr.name]
            if isinstance(buf, Tensor):
                return f"{buf.name}[{self._get_index_expr(buf, expr.idx)}]"
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
            return f"(-{self._transpile_expr(expr.arg)})"
        elif isinstance(expr, LoopIR.BinOp):
            lhs = self._transpile_expr(expr.lhs)
            rhs = self._transpile_expr(expr.rhs)
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
                tuple(self._transpile_expr(arg) for arg in expr.args)
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
