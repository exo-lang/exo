from .dataflow import DataflowIR, D, V
from ..core.LoopIR_pprint import op_prec
from ..core.prelude import Sym, SrcInfo, extclass
from collections import ChainMap
from dataclasses import dataclass, field
import sympy as sm

# --------------------------------------------------------------------------- #
# DataflowIR pretty printing
# --------------------------------------------------------------------------- #


@extclass(DataflowIR.proc)
def __str__(self):
    return "\n".join(_print_proc(self, PrintEnv(), ""))


@extclass(DataflowIR.stmt)
def __str__(self):
    return "\n".join(_print_stmt(self, PrintEnv(), ""))


@extclass(DataflowIR.expr)
def __str__(self):
    return _print_expr(self, PrintEnv())


@dataclass
class PrintEnv:
    env: ChainMap[Sym, str] = field(default_factory=ChainMap)
    names: ChainMap[str, int] = field(default_factory=ChainMap)
    sym_table = {}

    def get_name(self, nm):
        if isinstance(nm, sm.Symbol) and nm in self.sym_table:
            nm = self.sym_table[nm]

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
    env.sym_table = p.sym_table

    args = [_print_fnarg(a, env) for a in p.args]

    lines = [f"{indent}def {p.name}({', '.join(args)}):"]

    indent = indent + "  "

    for pred in p.preds:
        lines.append(f"{indent}assert {_print_expr(pred, env)}")

    lines.extend(_print_stmts(p.body, env, indent))
    lines.extend(_print_absenv(p.ctxt, env))

    return lines


def _print_stmts(stmts: list, env: PrintEnv, indent: str) -> list[str]:
    lines = []
    for stmt in stmts:
        lines.extend(_print_stmt(stmt, env, indent))
    return lines


def _print_absenv(absenv, env: PrintEnv) -> list[str]:
    # Using indent actually makes it less readable
    p_res = ""
    indent = " " * 25
    for key, val in absenv.items():
        # assert isinstance(key, Sym)
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
        lines.extend(_print_stmts(stmt.body, env, indent + "  "))
        if stmt.orelse:
            lines.append(f"{indent}else:")
            lines.extend(_print_stmts(stmt.orelse, env, indent + "  "))
        return lines

    elif isinstance(stmt, DataflowIR.For):
        lo = _print_expr(stmt.lo, env)
        hi = _print_expr(stmt.hi, env)
        lines = [f"{indent}for {env.get_name(stmt.iter)} in seq({lo}, {hi}):"]
        lines.extend(_print_stmts(stmt.body, env, indent + "  "))
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

    elif isinstance(e, DataflowIR.Extern):
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
# ArrayDomain pretty printer
# --------------------------------------------------------------------------- #
#
# Works with the updated ADT:
#
#   node = Leaf(val v, dict sample)
#        | LinSplit(cell *cells)
#
#   cell = Cell(rel eq, node tree)
#
# Produces output like:
#
#   (var x)
#   │   x < 0
#   │   │   y < x   ⟨x=-1, y=-2⟩
#   │   │   -x + y = 0   ⟨x=-1, y=-1⟩
#   │   │   y > x   ⟨x=-1, y=0⟩
#   │   -x = 0
#   │   │   y < x   ⟨x=0, y=-1⟩
#   │   │   -x + y = 0   ⟨x=0, y=0⟩
#   │   │   y > x   ⟨x=0, y=1⟩
#   │   x > 0
#   │   │   y < x   ⟨x=1, y=0⟩
#   │   │   -x + y = 0   ⟨x=1, y=1⟩
#   │   │   y > x   ⟨x=1, y=2⟩
#
# --------------------------------------------------------------------------- #

import sympy as sm
from sympy.core.relational import Relational

# ───────────────────────── helpers ────────────────────────────────────────────
def _expr_str(expr, env: PrintEnv) -> str:
    """
    Convert a raw SymPy Expr/Symbol to string, renaming every symbol through
    PrintEnv so the pretty-print is consistent with the rest of Exo.
    """
    if isinstance(expr, sm.Symbol):
        return env.get_name(expr)

    if isinstance(expr, sm.Expr):
        repl = {s: sm.Symbol(env.get_name(s)) for s in expr.free_symbols}
        return str(expr.xreplace(repl))

    # Fallback (should not happen for guards)
    return str(expr)


def _print_guard(expr, env: PrintEnv) -> str:
    """
    Pretty-print any SymPy Boolean expression:
        * Relational  →  “lhs < rhs”, “lhs = rhs”, … (with renamed vars)
        * And / Or    →  joined by  “∧”  /  “∨”
        * Not         →  “¬ …”
    """
    # Simple relational (Eq, Lt, Gt, …)
    if isinstance(expr, Relational):
        lhs = _expr_str(expr.lhs, env)
        rhs = _expr_str(expr.rhs, env)
        op = expr.rel_op.replace("==", "=")
        return f"{lhs} {op} {rhs}"

    # Boolean combinators
    if isinstance(expr, sm.And):
        return " ∧ ".join(_print_guard(arg, env) for arg in expr.args)

    if isinstance(expr, sm.Or):
        return " ∨ ".join(_print_guard(arg, env) for arg in expr.args)

    if isinstance(expr, sm.Not):
        return "¬" + _print_guard(expr.args[0], env)

    # Anything else – fall back to SymPy’s own str()
    return _expr_str(expr, env)


def _print_sample(sample: dict, env: PrintEnv) -> str:
    """Render the sample dict exactly as  ⟨x=1, y=2⟩  (order = insertion order)."""
    parts = [f"{env.get_name(k)}={v}" for k, v in sample.items()]
    return "⟨" + ", ".join(parts) + "⟩"


# ───────────────────────── Value printing helpers ───────────────────────────
def _print_vabs(vabs, env):
    """
    Pretty string for a ValueDomain.vabs element.
    """
    if isinstance(vabs, V.Top):
        return "⊤"
    if isinstance(vabs, V.Bot):
        return "⊥"
    if isinstance(vabs, V.ValConst):
        return str(vabs.val)  # already a Python or SymPy number
    raise TypeError(f"unknown vabs variant {type(vabs)}")


def _print_val(val, env):
    """
    Pretty string for an ArrayDomain.val (payload of a Leaf).
    """
    if isinstance(val, D.SubVal):
        return _print_vabs(val.av, env)

    if isinstance(val, D.ArrayVar):
        idxs = (
            "[" + ", ".join(_expr_str(i, env) for i in val.idx) + "]" if val.idx else ""
        )
        return f"{env.get_name(val.name)}{idxs}"

    raise TypeError(f"unknown val variant {type(val)}")


# ───────────────────────── core recursive printer ────────────────────────────
def _print_tree(node: D.node, env: PrintEnv, prefix: str = "") -> str:
    """
    Walk a LinSplit tree and build the ASCII art.
    `prefix` already contains any vertical bars (“│   ”) that must propagate
    from ancestor levels.
    """
    # ── Leaf ────────────────────────────────────────────────────────────────
    if isinstance(node, D.Leaf):
        val_str = _print_val(node.v, env)
        samp_str = _print_sample(node.sample, env)
        return prefix + (f" {samp_str} {val_str}" if node.sample else val_str)

    # ── LinSplit ────────────────────────────────────────────────────────────
    lines = []
    n_cells = len(node.cells)

    for i, cell in enumerate(node.cells):
        branch_prefix = "│   "  # always keep the guide
        guard_str = _print_guard(cell.eq, env)
        child = cell.tree

        # Guard line
        if isinstance(child, D.Leaf):
            lines.append(_print_tree(child, env, prefix + branch_prefix + guard_str))
        else:
            lines.append(f"{prefix}{branch_prefix}{guard_str}")
            lines.append(_print_tree(child, env, prefix + branch_prefix))

    return "\n".join(lines)


def _print_absdom(absdom: D.abs, env: PrintEnv, indent: str = "") -> str:
    """Entry point for an `ArrayDomain.abs` value."""
    iters = ", ".join(env.get_name(i) for i in absdom.iterators)
    # head   = f"(var {iters})"
    head = "(var)" if not iters else f"(var {iters})"
    body = _print_tree(absdom.tree, env, indent) if absdom.tree else ""
    return head + ("\n" + body if body else "")


# ───────────────────────── attach __str__ methods ────────────────────────────
@extclass(D.abs)
def __str__(self):
    return _print_absdom(self, PrintEnv())


@extclass(D.node)
def __str__(self):
    return _print_tree(self, PrintEnv())


# ───────────────────────── END OF MODULE ─────────────────────────────────────


del __str__
