from .dataflow import DataflowIR
from .LoopIR_pprint import op_prec
from .prelude import Sym, SrcInfo, extclass
from collections import ChainMap
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# DataflowIR pretty printing
# --------------------------------------------------------------------------- #


@extclass(DataflowIR.proc)
def __str__(self):
    return "\n".join(_print_proc(self, PrintEnv(), ""))


# @extclass(DataflowIR.fnarg)
# def __str__(self):
#    return _print_fnarg(self, PrintEnv())


@extclass(DataflowIR.stmt)
def __str__(self):
    return "\n".join(_print_stmt(self, PrintEnv(), ""))


# @extclass(DataflowIR.expr)
# def __str__(self):
#    return _print_expr(self, PrintEnv())


# @extclass(DataflowIR.block)
# def __str__(self):
#     return "\n".join(_print_block(self, PrintEnv(), ""))


del __str__


@dataclass
class PrintEnv:
    env: ChainMap[Sym, str] = field(default_factory=ChainMap)
    names: ChainMap[str, int] = field(default_factory=ChainMap)

    def get_name(self, nm):
        if resolved := self.env.get(nm):
            return resolved

        candidate = str(nm)
        num = self.names.get(candidate, 1)
        while candidate in self.names:
            candidate = f"{nm}_{num}"
            num += 1

        self.env[nm] = candidate
        self.names[str(nm)] = num
        return candidate


def _print_proc(p, env: PrintEnv, indent: str) -> list[str]:
    args = [_print_fnarg(a, env) for a in p.args]

    lines = [f"{indent}def {p.name}({', '.join(args)}):"]

    indent = indent + "  "

    for pred in p.preds:
        lines.append(f"{indent}assert {_print_expr(pred, env)}")

    lines.extend(_print_block(p.body, env, indent))

    return lines


def _print_block(blk, env: PrintEnv, indent: str) -> list[str]:
    lines = []
    for stmt, absenv in zip(blk.stmts, blk.ctxts[:-1]):
        lines.extend(_print_absenv(absenv, env, indent))
        lines.extend(_print_stmt(stmt, env, indent))
    lines.extend(_print_absenv(blk.ctxts[-1], env, indent))

    return lines


def _print_absenv(absenv, env: PrintEnv, indent: str) -> list[str]:
    # Using indent actually makes it less readable
    p_res = ""
    for key, val in absenv.items():
        assert isinstance(key, Sym)
        p_res = p_res + ", " + f"{env.get_name(key)} : {val}"
    p_res = "{" + p_res[2:] + "}"

    return [f"{indent}" + f"{'-'*(25-len(indent))} {p_res}"]


def _print_stmt(stmt, env: PrintEnv, indent: str) -> list[str]:
    if isinstance(stmt, DataflowIR.Pass):
        return [f"{indent}pass"]

    elif isinstance(stmt, (DataflowIR.Assign, DataflowIR.Reduce)):
        # Assign( sym lhs, sym* dims, expr cond, expr body, expr orelse )
        op = "=" if isinstance(stmt, DataflowIR.Assign) else "+="
        lhs = env.get_name(stmt.lhs)
        dims = ["\\" + env.get_name(d) for d in stmt.dims]

        cond = _print_expr(stmt.cond, env)
        body = _print_expr(stmt.body, env)
        orelse = _print_expr(stmt.orelse, env)

        return [f"{indent}{lhs} {op}{' '.join(dims)} \phi({cond} ? {body} : {orelse})"]

    elif isinstance(stmt, DataflowIR.Alloc):
        ty = _print_type(stmt.type, env)
        his = f"[{', '.join(_print_expr(i, env) for i in stmt.hi)}]" if stmt.hi else ""
        return [f"{indent}{env.get_name(stmt.name)} : {ty}{his}"]

    elif isinstance(stmt, DataflowIR.If):
        cond = _print_expr(stmt.cond, env)
        lines = [f"{indent}if {cond}:"]
        lines.extend(_print_block(stmt.body, env, indent + "  "))
        if stmt.orelse:
            lines.append(f"{indent}else:")
            lines.extend(_print_block(stmt.orelse, env, indent + "  "))
        return lines

    elif isinstance(stmt, DataflowIR.For):
        lo = _print_expr(stmt.lo, env)
        hi = _print_expr(stmt.hi, env)
        lines = [f"{indent}for {env.get_name(stmt.iter)} in seq({lo}, {hi}):"]
        lines.extend(_print_block(stmt.body, env, indent + "  "))
        return lines

    assert False, f"unrecognized stmt: {type(stmt)}"


def _print_fnarg(a, env: PrintEnv) -> str:
    if a.type == DataflowIR.Size:
        return f"{env.get_name(a.name)} : size"
    elif a.type == DataflowIR.Index:
        return f"{env.get_name(a.name)} : index"
    else:
        his = f"[{', '.join(_print_expr(i, env) for i in a.hi)}]" if a.hi else ""
        ty = _print_type(a.type, env)
        return f"{env.get_name(a.name)} : {ty}{his}"


def _print_expr(e, env: PrintEnv, prec: int = 0) -> str:
    if isinstance(e, DataflowIR.Read):
        name = env.get_name(e.name)
        idx = f"[{', '.join(_print_expr(i, env) for i in e.idx)}]" if e.idx else ""
        return f"{name}{idx}"

    elif isinstance(e, DataflowIR.Const):
        return str(e.val)

    elif isinstance(e, DataflowIR.USub):
        return f'-{_print_expr(e.arg, env, prec=op_prec["~"])}'

    elif isinstance(e, DataflowIR.BinOp):
        local_prec = op_prec[e.op]
        # increment rhs by 1 to account for left-associativity
        lhs = _print_expr(e.lhs, env, prec=local_prec)
        rhs = _print_expr(e.rhs, env, prec=local_prec + 1)
        s = f"{lhs} {e.op} {rhs}"
        # if we have a lower precedence than the environment...
        if local_prec < prec:
            s = f"({s})"
        return s

    elif isinstance(e, DataflowIR.StrideExpr):
        return f"stride({env.get_name(e.name)}, {e.dim})"

    elif isinstance(e, DataflowIR.BuiltIn):
        pname = e.f.name() or "_anon_"
        args = [_print_expr(a, env) for a in e.args]
        return f"{pname}({', '.join(args)})"

    assert False, f"unrecognized expr: {type(e)}"


def _print_type(t, env: PrintEnv) -> str:
    if isinstance(t, DataflowIR.Num):
        return "R"
    elif isinstance(t, DataflowIR.F16):
        return "f16"
    elif isinstance(t, DataflowIR.F32):
        return "f32"
    elif isinstance(t, DataflowIR.F64):
        return "f64"
    elif isinstance(t, DataflowIR.INT8):
        return "i8"
    elif isinstance(t, DataflowIR.UINT8):
        return "ui8"
    elif isinstance(t, DataflowIR.UINT16):
        return "ui16"
    elif isinstance(t, DataflowIR.INT32):
        return "i32"
    elif isinstance(t, DataflowIR.Bool):
        return "bool"
    elif isinstance(t, DataflowIR.Int):
        return "int"
    elif isinstance(t, DataflowIR.Index):
        return "index"
    elif isinstance(t, DataflowIR.Size):
        return "size"
    elif isinstance(t, DataflowIR.Stride):
        return "stride"

    assert False, f"impossible type {type(t)}"
