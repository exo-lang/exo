from __future__ import annotations
from fractions import Fraction
from typing import Dict, Optional, Tuple


class CollParam(object):
    __slots__ = ["name"]

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name + "_param"


class CollSizeExpr(object):
    __slots__ = ["scalar", "coll_params"]

    scalar: Fraction
    coll_params: Tuple[CollParam]

    def __init__(self, scalar: Fraction | int, coll_params: Tuple[CollParam]):
        self.scalar = Fraction(scalar)
        self.coll_params = coll_params

    def __call__(self, env: Dict[CollParam, int]):
        n = self.scalar.numerator
        for p in self.coll_params:
            n *= env[p]
        assert n % self.scalar.denominator == 0  # TODO better message
        n //= self.scalar.denominator
        assert isinstance(n, int)
        return n

    def __mul__(self, other):
        if isinstance(other, CollSizeExpr):
            return CollSizeExpr(
                self.scalar * other.scalar, self.coll_params + other.coll_params
            )
        else:
            return CollSizeExpr(Fraction(other) * self.scalar, self.coll_params)

    def __truediv__(self, other):
        return CollSizeExpr(self.scalar / Fraction(other), self.coll_params)

    def __floordiv__(self, other):
        return CollSizeExpr(self.scalar / Fraction(other), self.coll_params)

    def __repr__(self):
        if not self.coll_params:
            return f"CollSizeExpr({self.scalar}, {self.coll_params})"
        s = " * ".join([p.name for p in self.coll_params])
        if self.scalar.numerator != 1:
            s += f" * {self.scalar.numerator}"
        if self.scalar.denominator != 1:
            s += f" / {self.scalar.denominator}"
        return s


blockDim_param = CollParam("blockDim")
blockDim = CollSizeExpr(1, (blockDim_param,))
clusterDim_param = CollParam("clusterDim")
clusterDim = CollSizeExpr(1, (clusterDim_param,))


class CollUnit(object):
    __slots__ = ["partial_domain", "tile"]

    partial_domain: Tuple[CollSizeExpr | int]
    tile: Tuple[CollSizeExpr | int]

    def __init__(self, partial_domain, tile):
        assert len(partial_domain) + 1 == len(tile)
        self.partial_domain = partial_domain
        self.tile = tile

    def __repr__(self):
        return f"CollUnit({self.partial_domain}, {self.tile})"


class CollSpecialize(object):
    __slots__ = ["partial_domain", "offset", "box"]

    partial_domain: Tuple[CollSizeExpr | int]
    offset: Tuple[CollSizeExpr | int]
    box: Tuple[CollSizeExpr | int]

    def __init__(self, partial_domain, offset, box):
        assert len(partial_domain) + 1 == len(offset)
        assert len(partial_domain) + 1 == len(box)
        self.partial_domain = partial_domain
        self.offset = offset
        self.box = box

    def __repr__(self):
        return f"CollSpecialize({self.partial_domain}, {self.offset}, {self.box})"


class CollIndexExpr(object):
    __slots__ = ["base_expr", "ops", "hash"]

    base_expr: str  # Target language (e.g. C) expression, e.g. "threadIdx"
    ops: Tuple[str, int]  # Sequence of (operator, int) pairs to apply
    hash: int

    def __init__(self, base_expr, ops=()):
        self.base_expr = base_expr
        self.ops = ops
        self.hash = hash((base_expr, ops))

    def __eq__(self, other):
        """Note, this is just syntactic equality, not algebraic equality"""
        return (
            type(other) is CollIndexExpr
            and self.ops == other.ops
            and self.base_expr == other.base_expr
        )

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self._codegen_impl(f"CollIndexExpr({repr(self.base_expr)})", False)

    def __sub__(self, v: int):
        assert isinstance(v, int)
        if self.ops and self.ops[-1][0] == "-":
            # Merge with the prior subtract op if possible
            new_ops = self.ops[:-1] + (("-", self.ops[-1][1] + v),)
        else:
            new_ops = self.ops + (("-", v),)
        return CollIndexExpr(self.base_expr, new_ops)

    def __truediv__(self, v: int):
        assert isinstance(v, int)
        if self.ops and self.ops[-1][0] == "/":
            # Merge with the prior divide op if possible
            new_ops = self.ops[:-1] + (("/", self.ops[-1][1] * v),)
        else:
            new_ops = self.ops + (("/", v),)
        return CollIndexExpr(self.base_expr, new_ops)

    def __floordiv__(self, v: int):
        return self.__truediv__(v)

    def __mod__(self, v: int):
        assert isinstance(v, int)
        if self.ops and self.ops[-1][0] == "%":
            # Merge with the prior modulo op if possible
            # If a divides b, then x % a % b => x % a
            u = self.ops[-1][1]
            if u % v == 0:
                return CollIndexExpr(self.base_expr, self.ops[:-1] + (("%", v),))
            elif v % u == 0:
                return CollIndexExpr(self.base_expr, self.ops[:-1] + (("%", u),))
        return CollIndexExpr(self.base_expr, self.ops + (("%", v),))

    def codegen(self):
        """Assuming C for now. Should be usable downstream without further parenthesization"""
        need_parens = not self.base_expr.isalnum()
        expr = self._codegen_impl(self.base_expr, need_parens)

        # Wrap final expression in parens if not trivial
        if not expr.isalnum():
            expr = f"({expr})"
        return expr

    def _codegen_impl(self, expr, need_parens):
        for op, value in self.ops:
            if need_parens:
                expr = f"({expr})"
            if op == "-":
                expr = f"{expr} - {value}"
                need_parens = True
            elif op == "/":
                expr = f"{expr} / {value}"
                need_parens = False
            elif op == "%":
                expr = f"{expr} % {value}"
                need_parens = False
            else:
                assert False
        return expr


class CollTiling(object):
    __slots__ = ["parent", "domain", "tile", "offset", "box", "intra_box_exprs", "hash"]

    parent: Optional[CollTiling]
    domain: Tuple[int]
    tile: Tuple[int]
    offset: Tuple[int]
    box: Tuple[int]
    intra_box_exprs: Tuple[CollIndexExpr]
    hash: int

    def __init__(self, parent, domain, tile, offset, box, intra_box_exprs):
        assert parent is None or isinstance(parent, CollTiling)
        self.parent = parent
        self.domain = domain
        self.tile = tile
        self.offset = offset
        self.box = box
        for tup in (domain, tile, offset, box):
            assert (
                isinstance(tup, tuple)
                and all(isinstance(c, int) for c in tup)
                and len(tup) == len(domain)
            )

        self.intra_box_exprs = intra_box_exprs
        assert isinstance(intra_box_exprs, tuple)
        assert all(isinstance(c, CollIndexExpr) for c in intra_box_exprs)
        assert len(intra_box_exprs) == len(box)

        self.hash = hash((parent, domain, tile, offset, box, intra_box_exprs))

    def __repr__(self):
        return f"CollTiling({self.parent}, {self.domain}, {self.tile}, {self.offset}, {self.box}, {self.intra_box_exprs})"

    def __eq__(self, other: CollTiling):
        return (
            type(other) is CollTiling
            and self.parent is other.parent
            and self.domain == other.domain
            and self.tile == other.tile
            and self.offset == other.offset
            and self.box == other.box
        )

    def __hash__(self):
        return self.hash


class DomainCompletionOp(object):
    __slots__ = ["idx_factors", "input_dim", "domain"]
    idx_factors: Tuple[int, int]
    input_dim: int
    domain: Tuple[int]

    def __init__(
        self,
        source_domain: Tuple[int],
        target_domain: Tuple[int],
        source_is_partial: bool,
    ):
        def cumulative_thread_counts(domain):
            tmp = [1]
            for c in domain[::-1]:
                tmp.append(tmp[-1] * c)
            tmp.reverse()
            return tmp

        cumulative_s = cumulative_thread_counts(source_domain)
        cumulative_t = cumulative_thread_counts(target_domain)
        if source_is_partial:
            assert cumulative_t[0] % cumulative_s[0] == 0  # TODO message
            source_domain = (cumulative_t[0] // cumulative_s[0],) + source_domain
            cumulative_s = [cumulative_t[0]] + cumulative_s
        else:
            assert cumulative_s[0] % cumulative_t[0] == 0  # TODO message

        idx_factors = []

        for i_s in range(len(source_domain) - 1, -1, -1):
            s0 = cumulative_s[i_s] if i_s >= 0 else float("inf")
            s1 = cumulative_s[i_s + 1]
            for i_t in range(len(target_domain) - 1, -1, -1):
                t0 = cumulative_t[i_t]
                if s0 > t0 > s1:
                    t1 = cumulative_t[i_t + 1]
                    divisor = max(t1, s1)
                    split = t0 // divisor
                    if i_s >= 0:
                        assert source_domain[i_s] % split == 0  # TODO message
                    idx_factors.append((i_s, split))

        self.idx_factors = idx_factors
        self.input_dim = len(source_domain)
        self.domain = self.new_size(source_domain)

    def new_size(self, size: Tuple):
        return self._new_coords(size, lambda c, factor: factor)

    def new_offset(self, offset: Tuple):
        return self._new_coords(offset, lambda c, factor: 0)

    def new_intra_box_exprs(self, coords: Tuple):
        return self._new_coords(coords, lambda c, factor: c % factor)

    def _new_coords(self, coords: Tuple, inner_op):
        coords = list(coords)
        assert len(coords) == self.input_dim
        for idx, factor in self.idx_factors:
            assert idx >= 0
            assert idx < self.input_dim
            c = coords[idx]
            if isinstance(c, int):
                assert c % factor == 0  # TODO message
            coords[idx : idx + 1] = [c // factor, inner_op(c, factor)]
        return tuple(coords)
