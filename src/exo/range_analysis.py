from __future__ import annotations
from collections import ChainMap
from dataclasses import dataclass
from typing import Optional, Tuple

from .LoopIR import LoopIR, T, LoopIR_Compare
from .new_eff import Check_ExprBound
from .prelude import Sym, _null_srcinfo_obj


# TODO: we should implement a more general index analysis which
# will leverage SMT solver to prove comparisons. But we should still
# keep all the code below for fast, constant bounds inference.


def binop(op: str, e1: LoopIR.expr, e2: LoopIR.expr):
    return LoopIR.BinOp(op, e1, e2, e1.type, e1.srcinfo)


def zero():
    return LoopIR.Const(0, T.index, _null_srcinfo_obj)


def is_zero(e):
    return isinstance(e, LoopIR.Const) and e.val == 0


@dataclass
class IndexRange:
    """
    Represents a range of possible values between [base + lo, base + hi]
        - [base] contains all expressions that we don't want to bounds infer (e.g. free
        variables.
        - [lo] and [hi] are bounds. If either is None, it means that there is no
        constant bound
    """

    base: LoopIR.expr
    lo: Optional[int]
    hi: Optional[int]

    def create_unbounded() -> IndexRange:
        return IndexRange(zero(), None, None)

    def create_int(x: int) -> IndexRange:
        return IndexRange(zero(), x, x)

    def create_constant_range(lo: int, hi: int) -> IndexRange:
        return IndexRange(zero(), lo, hi)

    def get_bounds(self) -> Tuple[str, str]:
        """
        Returns a tuple of the form (base + lo, base + hi + 1). If either
        endpoint is unboundable, returns inf or -inf.
        """
        if is_zero(self.base):
            lo = "-inf" if self.lo is None else str(self.lo)
            hi = "inf" if self.hi is None else str(self.hi)
        else:
            lo = "-inf" if self.lo is None else f"{self.base} + {self.lo}"
            hi = "inf" if self.hi is None else f"{self.base} + {self.hi + 1}"
        return lo, hi

    def get_stride_of(self, idx: Sym) -> int:
        assert isinstance(idx, Sym)

        def get_coeff(e):
            """
            This implementation assumes that self.base is a linear combination of index
            variables WITHOUT a constant term, e.g. ax - by + cz where a, b, c are constants
            and x, y, z are index expression. This holds because constant terms are always
            folded into self.lo and self.hi
            """
            if isinstance(e, LoopIR.Read):
                return 1 if e.name == idx else 0
            elif isinstance(e, LoopIR.Const):
                return e.val
            elif isinstance(e, LoopIR.BinOp):
                lhs = get_coeff(e.lhs)
                rhs = get_coeff(e.rhs)
                op = e.op
                if op == "+":
                    return lhs + rhs
                elif op == "-":
                    return lhs - rhs
                elif op == "*":
                    return lhs * rhs
                else:
                    raise ValueError(
                        f"cannot get stride of {idx} because {e} contains an unsupported operand '{op}'"
                    )
            elif isinstance(e, LoopIR.USub):
                return -get_coeff(e.arg)
            return 0

        return get_coeff(self.base)

    def partial_eval_with_range(self, var: Sym, rng: IndexRange) -> IndexRange:
        c = self.get_stride_of(var)
        if c == 0:
            return self

        new_bounds = index_range_analysis(self.base, {var: (rng.lo, rng.hi)})

        if is_zero(rng.base):
            return new_bounds
        else:
            c_expr = LoopIR.Const(c, T.index, self.base.srcinfo)
            new_base = LoopIR.BinOp("*", c_expr, rng.base, T.index, self.base.srcinfo)
            return new_bounds + IndexRange(new_base, 0, 0)

    def get_size(self) -> int | None:
        if self.lo is None or self.hi is None:
            return None
        return self.hi - self.lo + 1  # +1 because bounds are inclusive

    def __str__(self) -> str:
        base = "0" if is_zero(self.base) else str(self.base)
        lo = "-inf" if self.lo is None else str(self.lo)
        hi = "inf" if self.hi is None else str(self.hi)
        return f"({base}, {lo}, {hi})"

    def __add__(self, other: int | IndexRange) -> IndexRange:
        assert isinstance(other, (int, IndexRange))
        if isinstance(other, int):
            new_lo, new_hi = None, None
            if self.lo is not None:
                new_lo = self.lo + other
            if self.hi is not None:
                new_hi = self.hi + other
            return IndexRange(self.base, new_lo, new_hi)
        else:
            if is_zero(self.base):
                new_base = other.base
            elif is_zero(other.base):
                new_base = self.base
            else:
                new_base = binop("+", self.base, other.base)

            new_lo, new_hi = None, None
            if self.lo is not None and other.lo is not None:
                new_lo = self.lo + other.lo
            if self.hi is not None and other.hi is not None:
                new_hi = self.hi + other.hi
            return IndexRange(new_base, new_lo, new_hi)

    def __radd__(self, c: int) -> IndexRange:
        assert isinstance(c, int)
        return self.__add__(c)

    def __neg__(self) -> IndexRange:
        new_base, new_lo, new_hi = zero(), None, None
        if not is_zero(self.base):
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

        new_base, new_lo, new_hi = zero(), None, None
        if not is_zero(self.base):
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
            return IndexRange.create_unbounded()

        new_lo, new_hi = None, None
        if is_zero(self.base):
            if self.lo is not None:
                new_lo = self.lo // c
            if self.hi is not None:
                new_hi = self.hi // c

            return IndexRange.create_constant_range(new_lo, new_hi)
        else:
            # TODO: Maybe can do some reasoning about base and lo if new_lo is not None
            c_expr = LoopIR.Const(c, T.index, self.base.srcinfo)
            new_base = LoopIR.BinOp(
                "/", self.base, c_expr, self.base.type, self.base.srcinfo
            )
            if self.lo is not None and self.hi is not None:
                new_lo = self.lo // c
                new_hi = self.hi // c + 1

            return IndexRange(new_base, new_lo, new_hi)

    def __mod__(self, c: int) -> IndexRange:
        assert isinstance(c, int)
        if (
            is_zero(self.base)
            and self.lo is not None
            and self.hi is not None
            and self.lo // c == self.hi // c
        ):
            return IndexRange.create_constant_range(self.lo % c, self.hi % c)
        return IndexRange.create_constant_range(0, c - 1)

    # join
    def __or__(self, other: IndexRange) -> IndexRange:
        assert isinstance(other, IndexRange)
        compare_ir = LoopIR_Compare()
        if (is_zero(self.base) and other.base is None) or compare_ir.match_e(
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
            return IndexRange.create_unbounded()


def index_range_analysis(
    expr: LoopIR.expr, env: ChainMap | dict = {}
) -> IndexRange | int:
    """
    Based on the supplied [env], recursively performs range analysis on
    the possible values for [expr]. Returns either an int if the value
    is constant, or an IndexRange of the form [offset+lo, offset+hi].

    When env is empty, this function effectively just separates the
    constants from the non-constants index expressions.
    """
    assert isinstance(expr, LoopIR.expr)
    assert isinstance(env, (ChainMap, dict))

    def analyze_range(expr) -> IndexRange | int:
        assert isinstance(expr, LoopIR.expr)

        if not expr.type.is_indexable():
            raise ValueError("Cannot analyze range of non-index expression")

        if isinstance(expr, LoopIR.Read):
            sym = expr.name
            if sym not in env:
                return IndexRange(expr, 0, 0)
            lo, hi = env[sym]
            return IndexRange.create_constant_range(lo, hi)
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


def constant_bound(expr, env) -> Tuple[int, int] | None:
    """
    Returns constant integer bounds for [expr], if possible, and
    None otherwise. The bounds are inclusive.
    """
    if isinstance(expr, int):
        return (expr, expr)

    idx_rng = index_range_analysis(expr, env)
    if isinstance(idx_rng, int):
        return (idx_rng, idx_rng)

    if not is_zero(idx_rng.base):
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
            ">=",
            value,
            exception=False,
        )

    def upper_bound_check(value):
        return Check_ExprBound(
            proc,
            [proc.body[0]],
            LoopIR.Read(name=arg.name, idx=[], type=T.size, srcinfo=proc.srcinfo),
            "<=",
            value,
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
        lo, _ = constant_bound(lo_expr, self.env)
        _, hi = constant_bound(hi_expr, self.env)
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
            return range0[1] <= range1[0]
        else:
            if range0[0] is None or range1[1] is None:
                return False
            return range0[0] == range0[1] == range1[0] == range1[1]

    def check_expr_bound(self, expr0, op, expr1):
        expr0_range = constant_bound(expr0, self.env)
        expr1_range = constant_bound(expr1, self.env)
        return IndexRangeEnvironment._check_range(expr0_range, op, expr1_range)

    def check_expr_bounds(self, expr0, op0, expr1, op1, expr2):
        expr0_range = constant_bound(expr0, self.env)
        expr1_range = constant_bound(expr1, self.env)
        expr2_range = constant_bound(expr2, self.env)
        return IndexRangeEnvironment._check_range(
            expr0_range, op0, expr1_range
        ) and IndexRangeEnvironment._check_range(expr1_range, op1, expr2_range)
