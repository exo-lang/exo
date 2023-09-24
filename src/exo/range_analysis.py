from .LoopIR import LoopIR


def index_range_analysis(expr, env):
    """
    Performs range-analysis on an index expression.

    Class takes in an environment in which the expression exists.
    The environment is a mapping from `Sym` -> range.
    It calculates the range of possible values that the index
    expression could be in.

    Any range, in the environment mapping or the result of the
    analysis is either:
        1. A tuple `T` of length 2 s.t. it represents the range
        `[T[0], T[1]]` (both inclusive).
        2. A `None` representing no knowledge of the value range
        or a failure to perform the analysis.
    """

    def merge_add(lhs_range, rhs_range):
        return (lhs_range[0] + rhs_range[0], lhs_range[1] + rhs_range[1])

    def merge_sub(lhs_range, rhs_range):
        return (lhs_range[0] - rhs_range[1], lhs_range[1] - rhs_range[0])

    def merge_mul(lhs_range, rhs_range):
        if lhs_range[0] < 0 or rhs_range[0] < 0:
            return None

        a = [i * j for i in lhs_range for j in rhs_range]
        return (min(a), max(a))

    def merge_div(lhs_range, rhs_range):
        assert rhs_range[0] == rhs_range[1]
        assert rhs_range[0] > 0

        if lhs_range[0] < 0:
            return None

        d = rhs_range[0]
        return (lhs_range[0] // d, lhs_range[1] // d)

    def merge_mod(lhs_range, rhs_range):
        assert rhs_range[0] == rhs_range[1]
        assert rhs_range[0] > 0

        m = rhs_range[0]
        if lhs_range[0] // m == lhs_range[1] // m:
            return (lhs_range[0] % m, lhs_range[1] % m)

        # We can be a bit smarter here when lhs_range[1] - lhs_range[0] < m
        # but we would need to keep track of two ranges for the expression

        return (0, m)

    e_symbols = set()

    def analyze_range(expr):
        assert isinstance(expr, LoopIR.expr)

        if not expr.type.is_indexable():
            return None

        if isinstance(expr, LoopIR.Read):
            sym = expr.name
            if sym in e_symbols:
                # It is unclear how to do range analysis when a symbol
                # is read twice within an expression. In most cases,
                # this won't matter since the expression are normalized
                # before we try to do range analysis on them
                return None
            e_symbols.add(sym)
            return env.get(sym)
        elif isinstance(expr, LoopIR.Const):
            return (expr.val, expr.val)
        elif isinstance(expr, LoopIR.USub):
            e_range = analyze_range(expr.arg)
            if e_range is None:
                return None
            return (-e_range[1], -e_range[0])
        elif isinstance(expr, LoopIR.BinOp):
            lhs_range = analyze_range(expr.lhs)
            rhs_range = analyze_range(expr.rhs)
            if lhs_range is None or rhs_range is None:
                return None
            merge_binop = {
                "+": merge_add,
                "-": merge_sub,
                "*": merge_mul,
                "/": merge_div,
                "%": merge_mod,
            }
            return merge_binop[expr.op](lhs_range, rhs_range)
        else:
            return None

    return analyze_range(expr)
