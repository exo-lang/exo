from .dataflow import DataflowIR, D, V
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
    for stmt in blk.stmts:
        lines.extend(_print_stmt(stmt, env, indent))
    lines.extend(_print_absenv(blk.ctxt, env))

    return lines


def _print_absenv(absenv, env: PrintEnv) -> list[str]:
    # Using indent actually makes it less readable
    p_res = ""
    indent = " " * 25
    for key, val in absenv.items():
        assert isinstance(key, Sym)
        p_res = (
            p_res
            + "\n"
            + f"{indent}{env.get_name(key)} : {_print_absdom(val, env, indent)}"
        )

    return [f"{'-'*(len(indent)-1)} {p_res[len(indent)+1:]}"]


def _print_stmt(stmt, env: PrintEnv, indent: str) -> list[str]:
    if isinstance(stmt, DataflowIR.Pass):
        return [f"{indent}pass"]

    elif isinstance(
        stmt,
        (
            DataflowIR.Assign,
            DataflowIR.Reduce,
            DataflowIR.LoopStart,
            DataflowIR.LoopExit,
            DataflowIR.IfJoin,
        ),
    ):
        # Assign( sym lhs, sym* dims, expr cond, expr body, expr orelse )
        op = "+=" if isinstance(stmt, DataflowIR.Reduce) else "="
        lhs = env.get_name(stmt.lhs)
        dims = ["\\" + env.get_name(d) for d in stmt.iters + stmt.dims]

        cond = _print_expr(stmt.cond, env)
        body = _print_expr(stmt.body, env)
        orelse = _print_expr(stmt.orelse, env)

        comment = ""
        if isinstance(stmt, DataflowIR.LoopStart):
            comment = " # LoopStart"
        elif isinstance(stmt, DataflowIR.LoopExit):
            comment = " # LoopExit"
        elif isinstance(stmt, DataflowIR.IfJoin):
            comment = " # IfJoin"

        return [
            f"{indent}{lhs} {op}{' '.join(dims)} \phi({cond} ? {body} : {orelse}){comment}"
        ]

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


# --------------------------------------------------------------------------- #
# Abstact domain pretty printing
# --------------------------------------------------------------------------- #


@extclass(D.abs)
def __str__(self):
    return _print_absdom(self, PrintEnv(), "")


@extclass(D.node)
def __str__(self):
    return _print_tree(self, PrintEnv(), "")


@extclass(D.val)
def __str__(self):
    return _print_val(self, PrintEnv())


@extclass(D.aexpr)
def __str__(self):
    return _print_ae(self, PrintEnv())


@extclass(V.vabs)
def __str__(self):
    return _print_vabs(self, PrintEnv())


def _print_absdom(absdom, env: PrintEnv, indent: str):
    iter_strs = ". ".join(["\\" + env.get_name(i) for i in absdom.iterators])
    return iter_strs + "\n" + _print_tree(absdom.tree, env, indent + "    ")


def _print_tree(tree, env: PrintEnv, indent: str):
    if isinstance(tree, D.Leaf):
        return f"{indent}- {_print_val(tree.v, env)}"
    elif isinstance(tree, D.AffineSplit):
        nstr = _print_ae(tree.ae, env)
        newdent = indent + " " * (len(nstr) + 1)
        indent = indent + "- "
        return f"""{indent}{nstr}
{_print_tree(tree.ltz, env, newdent)}
{_print_tree(tree.eqz, env, newdent)}
{_print_tree(tree.gtz, env, newdent)}"""
    elif isinstance(tree, D.ModSplit):
        nstr = _print_ae(tree.ae, env) + f"%{tree.m}"
        newdent = indent + " " * (len(nstr) + 1)
        indent = indent + "- "
        return f"""{indent}{nstr}
{_print_tree(tree.neqz, env, newdent)}
{_print_tree(tree.eqz, env, newdent)}"""
    else:
        assert False, "bad case"


def _print_val(val, env: PrintEnv):
    if isinstance(val, D.SubVal):
        return _print_vabs(val.av, env)
    elif isinstance(val, D.ArrayConst):
        idxs = (
            "[" + ",".join([_print_ae(i, env) for i in val.idx]) + "]"
            if len(val.idx) > 0
            else ""
        )
        return f"{env.get_name(val.name)}{idxs}"
    assert False, "bad case"


def _print_ae(ae, env: PrintEnv):
    if isinstance(ae, D.Const):
        return str(ae.val)
    elif isinstance(ae, D.Var):
        return env.get_name(ae.name)
    elif isinstance(ae, D.Add):
        # clean printing for sanity
        if isinstance(ae.rhs, D.Mult) and ae.rhs.coeff == -1:
            return f"({_print_ae(ae.lhs, env)}-{_print_ae(ae.rhs.ae, env)})"
        elif isinstance(ae.rhs, D.Mult) and ae.rhs.coeff <= -1:
            return f"({_print_ae(ae.lhs, env)}{_print_ae(ae.rhs, env)})"
        return f"({_print_ae(ae.lhs, env)}+{_print_ae(ae.rhs, env)})"
    elif isinstance(ae, D.Mult):
        return f"{str(ae.coeff)}*{_print_ae(ae.ae, env)}"
    assert False, "bad case"


def _print_vabs(val, env: PrintEnv):
    if isinstance(val, V.Top):
        return "⊤"
    elif isinstance(val, V.Bot):
        return "⊥"
    elif isinstance(val, V.ValConst):
        return str(val.val)
    assert False, "bad case"


del __str__
