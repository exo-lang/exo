from collections import ChainMap

from .LoopIR import LoopIR, T
from .new_eff import Check_ExprBound, Check_ExprBound_Options


def index_range_analysis(expr, env):
    """
    Performs range-analysis on an index expression.

    Function takes in an environment in which the expression exists.
    The environment is a mapping from `Sym` -> range.
    It calculates the range of possible values that the index
    expression could be in.

    Any range, in the environment mapping or the result of the
    analysis is either:
        A tuple `T` of length 2 s.t. it represents the range
        `[T[0], T[1]]` (both inclusive).
        If `T[0]` or `T[1]` is `None` it represents no knowledge
        of the value of that side of the range.
    """
    if isinstance(expr, int):
        return (expr, expr)

    def merge_add(lhs_range, rhs_range):
        new_lhs = None
        new_rhs = None
        if lhs_range[0] is not None and rhs_range[0] is not None:
            new_lhs = lhs_range[0] + rhs_range[0]
        if lhs_range[1] is not None and rhs_range[1] is not None:
            new_rhs = lhs_range[1] + rhs_range[1]
        return (new_lhs, new_rhs)

    def merge_sub(lhs_range, rhs_range):
        new_lhs = None
        new_rhs = None
        if lhs_range[0] is not None and rhs_range[1] is not None:
            new_lhs = lhs_range[0] - rhs_range[1]
        if lhs_range[1] is not None and rhs_range[0] is not None:
            new_rhs = lhs_range[1] - rhs_range[0]
        return (new_lhs, new_rhs)

    def merge_mul(lhs_range, rhs_range):
        if any(i is None for i in lhs_range + rhs_range):
            return (None, None)

        if lhs_range[0] < 0 or rhs_range[0] < 0:
            return (None, None)

        a = [i * j for i in lhs_range for j in rhs_range]
        return (min(a), max(a))

    def merge_div(lhs_range, rhs_range):
        assert isinstance(rhs_range[0], int)
        assert isinstance(rhs_range[1], int)
        assert rhs_range[0] == rhs_range[1]
        assert rhs_range[0] > 0

        if lhs_range[0] is None or lhs_range[0] < 0:
            return (None, None)

        d = rhs_range[0]

        new_lhs = None
        new_rhs = None
        if lhs_range[0] is not None:
            new_lhs = lhs_range[0] // d
        if lhs_range[1] is not None:
            new_rhs = lhs_range[1] // d

        return (new_lhs, new_rhs)

    def merge_mod(lhs_range, rhs_range):
        assert isinstance(rhs_range[0], int)
        assert isinstance(rhs_range[1], int)
        assert rhs_range[0] == rhs_range[1]
        assert rhs_range[0] > 0

        m = rhs_range[0]
        if (
            lhs_range[0] is not None
            and lhs_range[0] is not None
            and lhs_range[0] // m == lhs_range[1] // m
        ):
            return (lhs_range[0] % m, lhs_range[1] % m)

        # We can be a bit smarter here when lhs_range[1] - lhs_range[0] < m
        # but we would need to keep track of two ranges for the expression

        return (0, m)

    e_symbols = set()

    def analyze_range(expr):
        assert isinstance(expr, LoopIR.expr)

        if not expr.type.is_indexable():
            return (None, None)

        if isinstance(expr, LoopIR.Read):
            sym = expr.name
            if sym in e_symbols:
                # It is unclear how to do range analysis when a symbol
                # is read twice within an expression. In most cases,
                # this won't matter since the expression are normalized
                # before we try to do range analysis on them
                return (None, None)
            e_symbols.add(sym)
            assert sym in env
            return env[sym]
        elif isinstance(expr, LoopIR.Const):
            return (expr.val, expr.val)
        elif isinstance(expr, LoopIR.USub):
            e_range = analyze_range(expr.arg)
            negate = lambda x: -x if x is not None else x
            return (negate(e_range[1]), negate(e_range[0]))
        elif isinstance(expr, LoopIR.BinOp):
            lhs_range = analyze_range(expr.lhs)
            rhs_range = analyze_range(expr.rhs)
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


def arg_range_analyize(proc, arg):
    """
    Try to find a bounding range on the arguments
    of a proc.

    Returns: A tuple `T` of length 2 s.t. it represents the range
    `[T[0], T[1]]` (both inclusive).
    If `T[0]` or `T[1]` is `None` it represents no knowledge
    of the value of that side of the range.
    """
    assert isinstance(arg.type, LoopIR.Size)

    def binary_search(left, right, check):
        result = None

        while left <= right:
            mid = left + (right - left) // 2
            if check(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1

        return result

    # Let's try to find a bounding range on args between
    # 0 and 2^31 - 1. The upper bound on the range is a
    # reasonable large value so that if an answer exists
    # it will probably be within this.
    size_type_max = 2**31 - 1

    def lower_bound_check(value):
        return Check_ExprBound(
            proc,
            [],
            LoopIR.Read(name=arg.name, idx=[], type=T.size, srcinfo=proc.srcinfo),
            value,
            Check_ExprBound_Options.GEQ,
            exception=False,
        )

    def upper_bound_check(value):
        return not Check_ExprBound(
            proc,
            [],
            LoopIR.Read(name=arg.name, idx=[], type=T.size, srcinfo=proc.srcinfo),
            value,
            Check_ExprBound_Options.LT,
            exception=False,
        )

    def try_range(left, right, func):
        # We need to check this to make sure our
        # upper-bound is a correct over-approximation.
        # It is also a good way to prune analysis: e.g. most of the
        # time there won't actually be an upper bound on the arg.
        if func(right + 1):
            return None
        else:
            return binary_search(left, right, func)

    def split_search(func):
        # Try to split the search into multiple ranges
        # then run a binary search for each if the previous
        # smaller range didn't work. This is useful to prune
        # some trials since most of the time if there is an
        # answer, it will be for small values
        bound = try_range(1, 128, func)
        if bound is None:
            bound = try_range(129, 2048, func)
            if bound is None:
                bound = try_range(2049, size_type_max, func)
        return bound

    return (split_search(lower_bound_check), split_search(upper_bound_check))


class IndexRangeEnvironment:
    lt = "<"
    leq = "<="
    eq = "=="

    def __init__(self, proc) -> None:
        assert isinstance(proc, LoopIR.proc)

        self.proc = proc
        self.env = ChainMap()
        for arg in proc.args:
            if isinstance(arg.type, LoopIR.Size):
                self.env[arg.name] = arg_range_analyize(proc, arg)

    def enter_scope(self):
        self.env = self.env.new_child()

    def exit_scope(self):
        self.env = self.env.parents

    def add_sym(self, sym, lo_expr=None, hi_expr=None):
        range_lo = None
        range_hi = None
        if lo_expr is not None:
            range_lo = index_range_analysis(lo_expr, self.env)
        if hi_expr is not None:
            range_hi = index_range_analysis(hi_expr, self.env)

        sym_range = (range_lo[0], range_hi[1])

        # This means that this variable's range is invalid e.g.
        # `for i in seq(4, 4)`
        if (
            sym_range[0] is not None
            and sym_range[1] is not None
            and sym_range[0] > sym_range[1]
        ):
            sym_range = (None, None)

        self.env[sym] = sym_range

    @staticmethod
    def _check_range(range0, op, range1):
        if range0[1] is None or range1[0] is None:
            return False
        if op == IndexRangeEnvironment.lt:
            return range0[1] < range1[0]
        elif op == IndexRangeEnvironment.leq:
            return range0[0] <= range1[0]
        else:
            if range0[0] is None or range1[1] is None:
                return False
            return range0[0] == range0[1] == range1[0] == range1[1]

    def check_expr_bound(self, expr0, op, expr1):
        expr0_range = index_range_analysis(expr0, self.env)
        expr1_range = index_range_analysis(expr1, self.env)
        return IndexRangeEnvironment._check_range(expr0_range, op, expr1_range)

    def check_expr_bounds(self, expr0, op0, expr1, op1, expr2):
        expr0_range = index_range_analysis(expr0, self.env)
        expr1_range = index_range_analysis(expr1, self.env)
        expr2_range = index_range_analysis(expr2, self.env)
        return IndexRangeEnvironment._check_range(
            expr0_range, op0, expr1_range
        ) and IndexRangeEnvironment._check_range(expr1_range, op1, expr2_range)
