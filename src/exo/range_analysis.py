from __future__ import annotations
from collections import ChainMap
from dataclasses import dataclass
from typing import Optional

from .LoopIR import LoopIR, T, LoopIR_Compare, get_reads_of_expr
from .new_eff import Check_ExprBound, Check_ExprBound_Options


def binop(op: str, e1: LoopIR.expr, e2: LoopIR.expr):
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

    def get_bounds(self):
        """
        Returns a tuple of the form (base + lo, base + hi + 1). If either
        endpoint is unboundable, returns inf or -inf.
        """
        if self.base is None:
            lo = "-inf" if self.lo is None else str(self.lo)
            hi = "inf" if self.hi is None else str(self.hi)
        else:
            lo = "-inf" if self.lo is None else f"{self.base} + {self.lo}"
            hi = "inf" if self.hi is None else f"{self.base} + {self.hi + 1}"
        return lo, hi

    def get_size(self):
        if self.lo is None or self.hi is None:
            return None
        # +1 because bounds are inclusive
        return self.hi - self.lo + 1

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
        elif isinstance(other, IndexRange):
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
        else:
            raise ValueError(f"Invalid type for add: {type(other)}")

    def __radd__(self, c: int) -> IndexRange:
        assert isinstance(c, int)
        return self.__add__(c)

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
        assert isinstance(other, (int, IndexRange))
        return self + (-other)

    def __rsub__(self, c: int) -> IndexRange:
        assert isinstance(c, int)
        return -self + c

    def __mul__(self, c: int) -> IndexRange:
        assert isinstance(c, int)
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
        assert isinstance(c, int)
        return self.__mul__(c)

    def __floordiv__(self, c: int) -> IndexRange:
        assert isinstance(c, int)
        if c == 0:
            return ValueError("Cannot divide by 0.")
        elif c < 0:
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
        assert isinstance(c, int)
        if self.base is None and self.lo is not None and self.hi is not None:
            if self.lo // c == self.hi // c:
                return IndexRange(None, self.lo % c, self.hi % c)
        return IndexRange(None, 0, c - 1)

    # join
    def __or__(self, other: IndexRange) -> IndexRange:
        assert isinstance(other, IndexRange)
        compare_ir = LoopIR_Compare()
        if (self.base is None and other.base is None) or compare_ir.match_e(
            self.base, other.base
        ):
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
                return lhs_range // rhs_range
            elif expr.op == "%":
                return lhs_range % rhs_range
            else:
                assert False, "invalid binop in index expression"
        else:
            assert False, "invalid expr in index expression"

    return analyze_range(expr)


def get_ancestors(c, up_to=None):
    """
    Returns all ancestors of `c` below `up_to` from oldest to youngest.
    If `up_to` is `None`, returns all ancestors of `c` from oldest to youngest.
    """
    ancestors = []
    if up_to is not None:
        while c != up_to:
            ancestors.append(c)
            c = c.parent()
    else:
        while c != c.parent():  # Only False if c is InvalidCursor
            ancestors.append(c)
            c = c.parent()
    ancestors.reverse()
    return ancestors


def infer_range(expr, scope):
    proc = expr._impl.get_root()
    env = IndexRangeEnvironment(proc, fast=False)

    # Only add bound variables to the env
    ancestors = get_ancestors(expr, up_to=scope)
    for c in ancestors:
        env.enter_scope()
        s = c._impl._node
        if isinstance(s, LoopIR.Seq):
            env.add_loop_iter(s.iter, s.lo, s.hi)
    bounds = index_range_analysis_v2(expr._impl._node, env.env)
    return bounds


def get_affected_idxs(proc, buffer_name, iter_sym):
    idxs = set()
    # TODO: this only matches against writes
    for c in proc.find(f"{buffer_name}[_] = _", many=True):
        for idx, idx_expr in enumerate(c.idx()):
            idx_vars = [
                name.name() for (name, typ) in get_reads_of_expr(idx_expr._impl._node)
            ]
            if iter_sym in idx_vars:
                idxs.add(idx)

    return idxs


# TODO: fix this include interface to be something better
def bounds_inference(proc, loop, buffer_name: str, buffer_idx: int, include=["R", "W"]):
    # TODO: check that loop is a cursor of proc, and try to forward if not
    alloc = proc.find_alloc_or_arg(buffer_name)
    dim = alloc.shape()[buffer_idx]

    matches = []
    if "R" in include:
        # TODO: proc.find doesn't take a scope. Either write a variant or add that as an optional arg
        # TODO: Also, proc.find fails if no matches are found...but we really just want it to return []
        matches += proc.find(f"{buffer_name}[_]", many=True)
    if "W" in include:
        matches += proc.find(f"{buffer_name}[_] = _", many=True)

    # TODO: This implementation is slower than tree traversal, but maybe easier to understand
    bound = None  # None is basically bottom
    for c in matches:
        idx_expr = c.idx()[buffer_idx]
        cur_bounds = infer_range(idx_expr, loop)

        # This is effectively joining the bounds w/ Bottom
        if bound is None:
            bound = cur_bounds
        else:
            bound |= cur_bounds
    return bound


def index_range_analysis(expr, env):
    """
    Returns constant integer bounds for [expr], if possible, and
    None otherwise. The bounds are inclusive.
    """
    if isinstance(expr, int):
        return (expr, expr)

    idx_rng = index_range_analysis_v2(expr, env)
    if isinstance(idx_rng, int):
        return (idx_rng, idx_rng)

    if idx_rng.base is not None:
        return (None, None)
    return (idx_rng.lo, idx_rng.hi)


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

    def add_loop_iter(self, sym, lo_expr, hi_expr):
        lo, _ = index_range_analysis(lo_expr, self.env)
        _, hi = index_range_analysis(hi_expr, self.env)
        if hi is not None:
            hi = hi - 1

        sym_range = (lo, hi)

        # This means that this variable's range is invalid e.g.
        # `for i in seq(4, 4)`
        if (
            sym_range[0] is not None
            and sym_range[1] is not None
            and sym_range[0] > sym_range[1]
        ):
            # TODO: this probably shouldn't get added as (None, None), since that
            # means it could take on any possible value.
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
