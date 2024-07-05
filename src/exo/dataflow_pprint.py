from .dataflow import DataflowIR
from .LoopIR_pprint import PrintEnv, _print_type, op_prec
from .LoopIR import T
from .prelude import Sym, SrcInfo, extclass

# --------------------------------------------------------------------------- #
# DataflowIR pretty printing
# --------------------------------------------------------------------------- #


@extclass(DataflowIR.proc)
def __str__(self):
    return "\n".join(_print_proc(self, PrintEnv(), ""))


@extclass(DataflowIR.fnarg)
def __str__(self):
    return _print_fnarg(self, PrintEnv())


@extclass(DataflowIR.stmt)
def __str__(self):
    return "\n".join(_print_stmt(self, PrintEnv(), ""))


@extclass(DataflowIR.expr)
def __str__(self):
    return _print_expr(self, PrintEnv())


@extclass(DataflowIR.block)
def __str__(self):
    return "\n".join(_print_block(self, PrintEnv(), ""))


del __str__


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
        op = "=" if isinstance(stmt, DataflowIR.Assign) else "+="

        lhs = env.get_name(stmt.name)

        idx = [_print_expr(e, env) for e in stmt.idx]
        idx = f"[{', '.join(idx)}]" if idx else ""

        rhs = _print_expr(stmt.rhs, env)

        return [f"{indent}{lhs}{idx} {op} {rhs}"]

    elif isinstance(stmt, DataflowIR.WriteConfig):
        cname = env.get_name(stmt.config_field)
        rhs = _print_expr(stmt.rhs, env)
        return [f"{indent}{cname} = {rhs}"]

    elif isinstance(stmt, DataflowIR.Alloc):
        ty = _print_type(stmt.type, env)
        return [f"{indent}{env.get_name(stmt.name)} : {ty}"]

    elif isinstance(stmt, DataflowIR.If):
        cond = _print_expr(stmt.cond, env)
        lines = [f"{indent}if {cond}:"]
        lines.extend(_print_block(stmt.body, env.push(), indent + "  "))
        if stmt.orelse:
            lines.append(f"{indent}else:")
            lines.extend(_print_block(stmt.orelse, env.push(), indent + "  "))
        return lines

    elif isinstance(stmt, DataflowIR.For):
        lo = _print_expr(stmt.lo, env)
        hi = _print_expr(stmt.hi, env)
        body_env = env.push()
        lines = [f"{indent}for {body_env.get_name(stmt.iter)} in seq({lo}, {hi}):"]
        lines.extend(_print_block(stmt.body, body_env, indent + "  "))
        return lines

    elif isinstance(stmt, DataflowIR.WindowStmt):
        rhs = _print_expr(stmt.rhs, env)
        return [f"{indent}{env.get_name(stmt.name)} = {rhs}"]

    elif isinstance(stmt, DataflowIR.Call):
        args = [_print_expr(a, env) for a in stmt.args]
        return [f"{indent}{stmt.f.name}({', '.join(args)})"]

    assert False, f"unrecognized stmt: {type(stmt)}"


def _print_fnarg(a, env: PrintEnv) -> str:
    if a.type == T.size:
        return f"{env.get_name(a.name)} : size"
    elif a.type == T.index:
        return f"{env.get_name(a.name)} : index"
    else:
        ty = _print_type(a.type, env)
        return f"{env.get_name(a.name)} : {ty}"


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

    elif isinstance(e, DataflowIR.ReadConfig):
        name = env.get_name(e.config_field)
        return f"{name}"

    assert False, f"unrecognized expr: {type(e)}"
