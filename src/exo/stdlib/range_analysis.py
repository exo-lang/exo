from __future__ import annotations

from exo.API_cursors import *
from exo.LoopIR import get_reads_of_expr  # TODO: get rid of this
from exo.range_analysis import IndexRange

from .inspection import get_parents


def index_range_analysis(expr: ExprCursor, env: dict) -> IndexRange | int:
    """
    User-level implementation of range analysis implemented inside compiler.
    """
    assert isinstance(expr, ExprCursor)
    assert isinstance(env, dict)

    def analyze_range(expr) -> IndexRange | int:
        assert isinstance(expr, ExprCursor)

        if isinstance(expr, ReadCursor):
            sym = expr.name()
            if sym not in env:
                return IndexRange(expr._impl._node, 0, 0)
            lo, hi = env[sym]
            return IndexRange(None, lo, hi)
        elif isinstance(expr, LiteralCursor):
            return expr.value()
        elif isinstance(expr, UnaryMinusCursor):
            return -analyze_range(expr.arg())
        elif isinstance(expr, BinaryOpCursor):
            lhs_range = analyze_range(expr.lhs())
            rhs_range = analyze_range(expr.rhs())
            op = expr.op()
            if op == "+":
                return lhs_range + rhs_range
            elif op == "-":
                return lhs_range - rhs_range
            elif op == "*":
                return lhs_range * rhs_range
            elif op == "/":
                return lhs_range // rhs_range
            elif op == "%":
                return lhs_range % rhs_range
            else:
                assert False, "invalid binop in index expression"
        else:
            assert False, "invalid expr in index expression"

    return analyze_range(expr)


def constant_bound(expr: ExprCursor, env: dict) -> (int, int) | None:
    """
    This is an exact copy and paste of the constant_bound function
    in the compiler's range_analysis. The only difference is that
    it uses the user-level index_range_analysis implemented above.
    """
    if isinstance(expr, int):
        return (expr, expr)

    idx_rng = index_range_analysis(expr, env)
    if isinstance(idx_rng, int):
        return (idx_rng, idx_rng)

    if idx_rng.base is not None:
        return (None, None)
    return (idx_rng.lo, idx_rng.hi)


def infer_range(idx_expr: Cursor, scope: Cursor):
    assert isinstance(idx_expr, Cursor)
    assert isinstance(scope, Cursor)
    env = dict()

    # Only add bound variables to the env (which excludes scope)
    ancestors = list(get_parents(idx_expr.proc(), idx_expr, up_to=scope))[:-1]
    for c in filter(lambda x: isinstance(x, ForCursor), ancestors):
        lo, _ = constant_bound(c.lo(), env)
        _, hi = constant_bound(c.hi(), env)
        if hi is not None:
            hi -= 1  # loop upper bound is exclusive
        env[c.name()] = (lo, hi)

    bounds = index_range_analysis(idx_expr, env)
    return bounds


def get_affected_dim(proc, buffer_name: str, iter_sym):
    """
    Return which dimension of buffer are affected by [iter_sym]. Raises
    an error if there are multiple.
    """
    dims = set()
    # TODO: this only matches against writes
    for c in proc.find(f"{buffer_name}[_] = _", many=True):
        for idx, idx_expr in enumerate(c.idx()):
            idx_vars = [
                name.name() for (name, _) in get_reads_of_expr(idx_expr._impl._node)
            ]
            if iter_sym in idx_vars:
                dims.add(idx)

    if len(dims) > 1:
        raise ValueError(f"{iter_sym} affects multiple indices into {buffer_name}")

    return list(dims)[0]


# TODO: fix this include interface to be something better
def bounds_inference(proc, loop, buffer_name: str, buffer_dim: int, include=["W"]):
    loop = proc.forward(loop)
    alloc = proc.find_alloc_or_arg(buffer_name)
    dim = alloc.shape()[buffer_dim]

    matches = []
    # TODO: also want probably reduces... for both read and write
    if "R" in include:
        # TODO: proc.find doesn't take a scope. Either write a variant or add that as an optional arg
        # TODO: Also, proc.find fails if no matches are found...but we really just want it to return []
        matches += proc.find(f"{buffer_name}[_]", many=True)
    if "W" in include:
        matches += proc.find(f"{buffer_name}[_] = _", many=True)

    # TODO: This implementation is slower than tree traversal, but maybe easier to understand
    bound = None  # None is basically bottom
    for c in matches:
        idx_expr = c.idx()[buffer_dim]
        cur_bounds = infer_range(idx_expr, loop)

        if bound is None:
            # This is effectively joining the bounds w/ Bottom
            bound = cur_bounds
        else:
            bound |= cur_bounds
    return bound
