from __future__ import annotations
from collections import ChainMap
from dataclasses import dataclass
from typing import Optional

from .LoopIR import LoopIR, T, is_const_zero, LoopIR_Compare
from .new_eff import Check_ExprBound, Check_ExprBound_Options


def binop(op: str, e1, e2):
    return LoopIR.BinOp(op, e1, e2, e1.type, e1.srcinfo)


@dataclass
class IndexRange:
    """
    Represents a range of possible values between [base + lo, base + hi]
        - [base] contains all expressions that we don't want to bounds infer (e.g. free
        variables). If base is None, it signifies a base of 0.
        - [lo] and [hi] are bounds. If either is None, it means that there is no
        constant bound
    """

    base: Optional[LoopIR.expr]
    lo: Optional[int]
    hi: Optional[int]

    def __str__(self):
        base = "0" if self.base is None else str(self.base)
        lo = "-inf" if self.lo is None else str(self.lo)
        hi = "inf" if self.hi is None else str(self.hi)
        return f"({base}, {lo}, {hi})"

    def __add__(self, other: int | IndexRange) -> IndexRange:
        if isinstance(other, int):
            new_lo, new_hi = None, None
            if self.lo is not None:
                new_lo = self.lo + other
            if self.hi is not None:
                new_hi = self.hi + other
            return IndexRange(self.base, new_lo, new_hi)
        else:
            if self.base is None:
                new_base = other.base
            elif other.base is None:
                new_base = self.base
            else:
                new_base = binop("+", self.base, other.base)

            new_lo, new_hi = None, None
            if self.lo is not None and other.lo is not None:
                new_lo = self.lo + other.lo
            if self.hi is not None and other.hi is not None:
                new_hi = self.hi + other.hi
            return IndexRange(new_base, new_lo, new_hi)

    def __neg__(self) -> IndexRange:
        new_base, new_lo, new_hi = None, None, None
        if self.base is not None:
            new_base = LoopIR.USub(self.base, self.base.type, self.base.srcinfo)
        if self.lo is not None:
            new_hi = -self.lo
        if self.hi is not None:
            new_lo = -self.hi
        return IndexRange(new_base, new_lo, new_hi)

    def __sub__(self, other: int | IndexRange) -> IndexRange:
        # TODO: see if I should manually implement this
        return self + (-other)

    def __mul__(self, c: int) -> IndexRange:
        if c == 0:
            return 0

        new_base, new_lo, new_hi = None, None, None
        if self.base is not None:
            const = LoopIR.Const(c, T.index, self.base.srcinfo)
            new_base = binop("*", self.base, const)
        if self.lo is not None:
            new_lo = self.lo * c
        if self.hi is not None:
            new_hi = self.hi * c

        if c > 0:
            return IndexRange(new_base, new_lo, new_hi)
        else:
            return IndexRange(new_base, new_hi, new_lo)

    def __rmul__(self, c: int) -> IndexRange:
        return self.__mul__(c)

    def __div__(self, c: int) -> IndexRange:
        if other <= 0:
            return IndexRange(None, None, None)

        new_lo, new_hi = None, None
        if self.base is None:
            if self.lo is not None:
                new_lo = self.lo // c
            if self.hi is not None:
                new_hi = self.hi // c

            return IndexRange(None, new_lo, new_hi)
        else:
            # TODO: Maybe can do some reasoning about base and lo if new_lo is not None
            if self.lo is not None and self.hi is not None:
                new_lo = 0
                new_hi = (self.hi - self.lo) // c

            return IndexRange(None, new_lo, new_hi)

    def __mod__(self, c: int) -> IndexRange:
        # TODO: improve this
        return IndexRange(None, 0, c - 1)

    # join
    def __or__(self, other: IndexRange) -> IndexRange:
        compare_ir = LoopIR_Compare()
        if compare_ir.match_e(self.base, other.base):
            if self.lo is None:
                new_lo = other.lo
            elif other.lo is None:
                new_lo = self.lo
            else:
                new_lo = min(self.lo, other.lo)
            if self.hi is None:
                new_hi = other.hi
            elif other.hi is None:
                new_hi = self.hi
            else:
                new_hi = max(self.hi, other.hi)
            return IndexRange(self.base, new_lo, new_hi)
        else:
            return IndexRange(None, None, None)


def index_range_analysis_v2(expr, env):
    def analyze_range(expr):
        assert isinstance(expr, LoopIR.expr)

        if not expr.type.is_indexable():
            return (None, None)

        if isinstance(expr, LoopIR.Read):
            sym = expr.name
            if sym not in env:
                return IndexRange(expr, 0, 0)
            lo, hi = env[sym]
            return IndexRange(None, lo, hi)
        elif isinstance(expr, LoopIR.Const):
            return expr.val
        elif isinstance(expr, LoopIR.USub):
            return -analyze_range(expr.arg)
        elif isinstance(expr, LoopIR.BinOp):
            lhs_range = analyze_range(expr.lhs)
            rhs_range = analyze_range(expr.rhs)
            if expr.op == "+":
                return lhs_range + rhs_range
            elif expr.op == "-":
                return lhs_range - rhs_range
            elif expr.op == "*":
                return lhs_range * rhs_range
            elif expr.op == "/":
                return lhs_range / rhs_range
            elif expr.op == "%":
                return lhs_range % rhs_range
            else:
                assert False, "invalid binop in index expression"
        else:
            assert False, "invalid expr in index expression"

    return analyze_range(expr)


def infer_range(expr, scope):
    c = expr
    ancestors = []
    while c != c.parent():  # Only False if c is InvalidCursor
        ancestors.append(c)
        c = c.parent()
    ancestors.reverse()
    i = ancestors.index(scope)

    proc = expr._impl.get_root()
    env = IndexRangeEnvironment(proc, fast=False)

    # Only add bound variables to the env
    for c in ancestors[i:]:
        env.enter_scope()
        s = c._impl._node
        if isinstance(s, LoopIR.Seq):
            lo = s.lo
            hi = LoopIR.BinOp(
                "-", s.hi, LoopIR.Const(1, T.int, s.srcinfo), T.index, s.srcinfo
            )
            env.add_sym(s.iter, lo, hi)
    bounds = index_range_analysis_v2(expr._impl._node, env.env)
    return bounds


# TODO: fix this include interface to be something better
def bounds_inference(proc, loop, buffer, include=["R", "W"]):
    dims = proc.find_alloc(buffer).shape()
    bounds = [None for _ in dims]  # None is basically bottom

    matches = []
    # TODO: proc.find doesn't take a scope. Either write a variant or add that as an optional arg
    # TODO: Also, proc.find fails if no matches are found...but we really just want it to return []
    if "R" in include:
        matches += proc.find(f"{buffer}[_]", many=True)
    if "W" in include:
        matches += proc.find(f"{buffer}[_] = _", many=True)

    for c in matches:
        idxs = c.idx()
        for i, dim in enumerate(dims):
            cur_bounds = infer_range(idxs[i], loop)

            # This is a hacky way of joining the bounds w/o a representation of Bottom
            if bounds[i] is None:
                bounds[i] = cur_bounds
            else:
                bounds[i] |= cur_bounds
    return bounds


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

    TODO: modify this to ignore variables which are free and
    instead also return a "base" for the range.
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
        # We make sure numbers aren't negative here,
        # there is probably a way to come up with a correct
        # range even when the range contains negative numbers
        if (lhs_range[0] is not None and lhs_range[0] < 0) or (
            rhs_range[0] is not None and rhs_range[0] < 0
        ):
            return (None, None)

        new_lhs = None
        new_rhs = None
        if lhs_range[0] is not None and rhs_range[0] is not None:
            new_lhs = lhs_range[0] * rhs_range[0]
        if lhs_range[1] is not None and rhs_range[1] is not None:
            new_rhs = lhs_range[1] * rhs_range[1]
        return (new_lhs, new_rhs)

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
            and lhs_range[1] is not None
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
            binop_range = merge_binop[expr.op](lhs_range, rhs_range)
            return binop_range
        else:
            return None

    return analyze_range(expr)


def arg_range_analysis(proc, arg, fast=True):
    """
    Try to find a bounding range on the arguments
    of a proc.

    Returns: A tuple `T` of length 2 s.t. it represents the range
    `[T[0], T[1]]` (both inclusive).
    If `T[0]` or `T[1]` is `None` it represents no knowledge
    of the value of that side of the range.

    NOTE: This is analysis can be slow. We should
    consider doing it when constructing a new proc
    and storing the result within LoopIR.fnarg.
    """
    assert arg.type.is_indexable()

    if fast:
        if isinstance(arg.type, LoopIR.Size):
            return (1, None)
        else:
            return (None, None)

    def lower_bound_check(value):
        return Check_ExprBound(
            proc,
            [proc.body[0]],
            LoopIR.Read(name=arg.name, idx=[], type=T.size, srcinfo=proc.srcinfo),
            value,
            Check_ExprBound_Options.GEQ,
            exception=False,
        )

    def upper_bound_check(value):
        return Check_ExprBound(
            proc,
            [proc.body[0]],
            LoopIR.Read(name=arg.name, idx=[], type=T.size, srcinfo=proc.srcinfo),
            value,
            Check_ExprBound_Options.LEQ,
            exception=False,
        )

    def binary_search_lower_bound(left, right):
        result = None

        while left <= right:
            mid = left + (right - left) // 2
            if lower_bound_check(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1

        return result

    def binary_search_upper_bound(left, right):
        result = None

        while left <= right:
            mid = left + (right - left) // 2
            if upper_bound_check(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1

        return result

    # Let's try to find a bounding range on args.
    # The upper bound on the absolute value of the range is a
    # reasonable large value so that if an answer exists
    # it will probably be within this.
    max_abs_search = 2**15

    min_search = 1 if isinstance(arg.type, LoopIR.Size) else -max_abs_search

    lower_bound = binary_search_lower_bound(min_search, max_abs_search)
    upper_bound = binary_search_upper_bound(min_search, max_abs_search)

    return (lower_bound, upper_bound)


class IndexRangeEnvironment:
    lt = "<"
    leq = "<="
    eq = "=="

    @staticmethod
    def get_pred_reads(expr):
        if isinstance(expr, LoopIR.Read):
            return {expr.name}
        elif isinstance(expr, LoopIR.USub):
            return IndexRangeEnvironment.get_pred_reads(expr.arg)
        elif isinstance(expr, LoopIR.BinOp):
            return IndexRangeEnvironment.get_pred_reads(
                expr.lhs
            ) | IndexRangeEnvironment.get_pred_reads(expr.rhs)
        else:
            return set()

    def __init__(self, proc, fast=True) -> None:
        assert isinstance(proc, LoopIR.proc)

        preds_reads = set()
        if not fast:
            # Get parameters referenced in predicates
            # to only analyze those
            for pred in proc.preds:
                preds_reads = preds_reads | IndexRangeEnvironment.get_pred_reads(pred)

        self.proc = proc
        self.env = ChainMap()
        for arg in proc.args:
            if isinstance(arg.type, LoopIR.Size):
                self.env[arg.name] = arg_range_analysis(
                    proc, arg, fast=arg.name not in preds_reads
                )

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


"""
Want to bound an expr within a scope:
 - determine which variables are free/bound in this scope

Difference between
- free = don't want to reason about
- bound, non-constant boudnable = can't reason about

"""
