from __future__ import annotations
from typing import Tuple

from exo.API_cursors import *
from exo.range_analysis import IndexRange

from .inspection import get_parents


def index_range_analysis(expr: ExprCursor, env: dict) -> IndexRange | int:
    """
    User-level implementation of range analysis implemented inside compiler. This
    implementation is more a proof of concept, since we could really just write
    a wrapper around the compiler's range analysis.
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
            return IndexRange.create_constant_range(lo, hi)
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


def constant_bound(expr: ExprCursor, env: dict) -> Tuple[int, int] | None:
    """
    Returns constant integer bounds for [expr], if possible, and
    None otherwise. The bounds are inclusive.

    This is an exact copy and paste of the constant_bound function
    in the compiler's range_analysis. The only difference is that
    it calls the user-level index_range_analysis defined above.
    """
    if isinstance(expr, int):
        return (expr, expr)

    idx_rng = index_range_analysis(expr, env)
    if isinstance(idx_rng, int):
        return (idx_rng, idx_rng)

    if idx_rng.base is not None:
        return (None, None)
    return (idx_rng.lo, idx_rng.hi)


def infer_range(idx_expr: Cursor, scope: Cursor) -> IndexRange:
    """
    Infers the range of possible values of [idx_expr] within [scope].
    """
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

    if isinstance(bounds, int):
        return IndexRange.create_int(bounds, bounds)

    return bounds


def bounds_inference(loop, buffer_name: str, buffer_dim: int, include=["W"]):
    """ """
    matches = []
    # TODO: also probably want reduces... for both read and write
    if "R" in include:
        matches += loop.find(f"{buffer_name}[_]", many=True)
    if "W" in include:
        matches += loop.find(f"{buffer_name}[_] = _", many=True)

    bound = None  # None is basically bottom in abstract interpretation
    for c in matches:
        idx_expr = c.idx()[buffer_dim]
        # TODO: This implementation is slower, but easier to understand than a tree traversal
        # because we have to rebuild the environment for each infer_range call.
        cur_bounds = infer_range(idx_expr, loop)

        if bound is None:
            bound = cur_bounds
        else:
            bound |= cur_bounds
    return bound
