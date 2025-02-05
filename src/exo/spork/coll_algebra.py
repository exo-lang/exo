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


def coll_size_tuple(tup):
    result = []
    for n in tup:
        if isinstance(n, int):
            result.append(CollSizeExpr(n, ()))
        else:
            assert isinstance(n, CollSizeExpr)
            result.append(n)
    return tuple(result)


def int_size_tuple(tup: Tuple[CollSizeExpr], env: Dict[CollParam, int]):
    return tuple(n(env) for n in tup)


class CollUnit(object):
    __slots__ = ["partial_domain", "tile"]

    partial_domain: Tuple[CollSizeExpr]
    tile: Tuple[CollSizeExpr]

    def __init__(self, partial_domain, tile):
        assert len(partial_domain) == len(tile)
        self.partial_domain = coll_size_tuple(partial_domain)
        self.tile = coll_size_tuple(tile)

    def __repr__(self):
        return f"CollUnit({self.partial_domain}, {self.tile})"

    def int_partial_domain(self, env: Dict[CollParam, int]):
        return int_size_tuple(self.partial_domain, env)

    def int_tile(self, env: Dict[CollParam, int]):
        return int_size_tuple(self.tile, env)


class CollIndexExpr(object):
    __slots__ = ["base_expr", "ops", "hash"]

    # Target language (e.g. C) expression, e.g. "threadIdx" as str
    # or constant value as int
    base_expr: str | int
    # Sequence of (operator, int) pairs to apply.
    # Only allowed if the base_expr is a str.
    ops: Tuple[str, int]
    # Pre-computed hash
    hash: int

    def __init__(self, base_expr, ops=()):
        if isinstance(base_expr, int):
            assert not ops
        else:
            assert isinstance(base_expr, str)
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

        if isinstance(self.base_expr, int):
            return CollIndexExpr(self.base_expr - v)
        elif v == 0:
            return self
        elif self.ops and self.ops[-1][0] == "-":
            # Merge with the prior subtract op if possible
            new_ops = self.ops[:-1] + (("-", self.ops[-1][1] + v),)
        else:
            new_ops = self.ops + (("-", v),)

        return CollIndexExpr(self.base_expr, new_ops)

    def __truediv__(self, v: int):
        assert isinstance(v, int)
        if isinstance(self.base_expr, int):
            return CollIndexExpr(self.base_expr // v)
        elif v == 1:
            return self
        elif self.ops and self.ops[-1][0] == "/":
            # Merge with the prior divide op if possible
            new_ops = self.ops[:-1] + (("/", self.ops[-1][1] * v),)
        else:
            new_ops = self.ops + (("/", v),)
        return CollIndexExpr(self.base_expr, new_ops)

    def __floordiv__(self, v: int):
        return self.__truediv__(v)

    def __mod__(self, v: int):
        assert isinstance(v, int)
        if isinstance(self.base_expr, int):
            return CollIndexExpr(self.base_expr % v)
        elif v == 1:
            return CollIndexExpr(0)
        elif self.ops and self.ops[-1][0] == "%":
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
        need_parens = isinstance(self.base_expr, str) and not self.base_expr.isalnum()
        expr = self._codegen_impl(self.base_expr, need_parens)

        # Wrap final expression in parens if not trivial
        if not expr.isalnum():
            expr = f"({expr})"
        return expr

    def _codegen_impl(self, expr, need_parens):
        expr = str(expr)
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
        self.domain = tuple(domain)
        self.tile = tuple(tile)
        self.offset = tuple(offset)
        self.box = tuple(box)
        for tup in (domain, tile, offset, box):
            assert all(isinstance(c, int) for c in tup)
            assert len(tup) == len(domain)

        self.intra_box_exprs = tuple(intra_box_exprs)
        assert all(isinstance(c, CollIndexExpr) for c in intra_box_exprs)
        assert len(intra_box_exprs) == len(box)

        self.hash = hash(
            (
                self.parent,
                self.domain,
                self.tile,
                self.offset,
                self.box,
                self.intra_box_exprs,
            )
        )

    def __repr__(self):
        return f"CollTiling({self.parent}, {self.domain}, {self.tile}, {self.offset}, {self.box}, {self.intra_box_exprs})"

    def __eq__(self, other: CollTiling):
        return self is other or (
            type(other) is CollTiling
            and self.parent is other.parent
            and self.domain == other.domain
            and self.tile == other.tile
            and self.offset == other.offset
            and self.box == other.box
        )

    def __hash__(self):
        return self.hash

    def tiled(self, unit: CollUnit, env: Dict[CollParam, int]):
        # Translate unit domain and tiling to concrete integers
        unit_partial_domain = unit.int_partial_domain(env)
        unit_tile = unit.int_tile(env)

        # Determine the common domain between us and the given unit
        unit_completion = DomainCompletionOp(
            unit_partial_domain, self.domain, allow_partial_source=True
        )
        self_completion = DomainCompletionOp(
            self.domain, unit_partial_domain, allow_partial_source=False
        )
        common_domain = unit_completion.domain
        assert unit_completion.domain == self_completion.domain

        # Translate ourself to common domain
        new_exprs = self_completion.new_intra_box_exprs(self.intra_box_exprs)
        new_tile = self_completion.new_size(self.box)  # May be modified later
        old_box = tuple(new_tile)  # Constant

        # Tiling will be the same as the box dimension of the parent
        # except along the dimension being tiled.
        # Count tiles and update tmp_tile size
        # Must only have change (tiling) on up to one dimension
        tiled_dim_idx = None
        tile_count = 1
        tile_remainder = 0
        for dim_idx, unit_tile_coord in enumerate(unit_completion.new_size(unit_tile)):
            domain_coord = common_domain[dim_idx]
            box_coord = old_box[dim_idx]
            if (
                unit_tile_coord is not None
                and unit_tile_coord != domain_coord
                and unit_tile_coord != box_coord
            ):
                assert unit_tile_coord < box_coord  # TODO message
                assert tiled_dim_idx is None  # TODO message
                tiled_dim_idx = dim_idx
                tile_count = box_coord // unit_tile_coord
                tile_remainder = box_coord % unit_tile_coord
                new_tile[dim_idx] = unit_tile_coord
                new_exprs[dim_idx] = new_exprs[dim_idx] % unit_tile_coord

        new_parent = self
        new_offset = (0,) * len(common_domain)
        new_tile = tuple(new_tile)
        new_box = new_tile

        return CollTiling(
            new_parent,
            common_domain,
            new_tile,
            new_offset,
            new_box,
            new_exprs,
        )

    def specialized(self, unit: CollUnit, lo: int, hi: int, env: Dict[CollParam, int]):
        # Translate unit domain and tiling to concrete integers
        unit_partial_domain = unit.int_partial_domain(env)
        unit_tile = unit.int_tile(env)

        # Determine the common domain between us and the given unit
        unit_completion = DomainCompletionOp(
            unit_partial_domain, self.domain, allow_partial_source=True
        )
        self_completion = DomainCompletionOp(
            self.domain, unit_partial_domain, allow_partial_source=False
        )
        common_domain = unit_completion.domain
        assert unit_completion.domain == self_completion.domain

        # Translate ourself to common domain
        # These may be modified to get the derived CollTiling
        new_exprs = self_completion.new_intra_box_exprs(self.intra_box_exprs)
        new_offset = self_completion.new_offset(self.offset)
        new_box = self_completion.new_size(self.box)

        common_tile = tuple(self_completion.new_size(self.tile))

        # Count tiles when tiled by unit
        # Must only have change (tiling) on up to one dimension
        tiled_dim_idx = None
        stride = None
        tile_count = 1
        for dim_idx, unit_tile_coord in enumerate(unit_completion.new_size(unit_tile)):
            domain_coord = common_domain[dim_idx]
            common_tile_coord = common_tile[dim_idx]
            if (
                unit_tile_coord is not None
                and unit_tile_coord != domain_coord
                and unit_tile_coord != common_tile_coord
            ):
                tile_count = common_tile_coord // unit_tile_coord

                # TODO messages
                assert tiled_dim_idx is None
                assert hi - lo <= tile_count
                assert new_box[dim_idx] == common_tile[dim_idx]

                tiled_dim_idx = dim_idx
                stride = unit_tile_coord
                new_exprs[dim_idx] -= lo * stride
                new_offset[dim_idx] = lo * stride
                new_box[dim_idx] = (hi - lo) * stride

        if tiled_dim_idx is None:
            assert (lo, hi) == (0, 1)  # TODO message

        new_parent = self.parent

        return CollTiling(
            new_parent,
            common_domain,
            common_tile,
            new_offset,
            new_box,
            new_exprs,
        )


class DomainCompletionOp(object):
    __slots__ = ["idx_factors", "input_dim", "domain", "source_partial"]
    idx_factors: Tuple[int, int]
    input_dim: int
    domain: Tuple[int]
    source_partial: bool

    def __init__(
        self,
        source_domain: Tuple[int],
        target_domain: Tuple[int],
        allow_partial_source: bool,
    ):
        def cumulative_thread_counts(domain):
            tmp = [1]
            for c in domain[::-1]:
                tmp.append(tmp[-1] * c)
            tmp.reverse()
            return tmp

        cumulative_s = cumulative_thread_counts(source_domain)
        cumulative_t = cumulative_thread_counts(target_domain)
        if allow_partial_source and cumulative_s[0] != cumulative_t[0]:
            assert cumulative_t[0] % cumulative_s[0] == 0  # TODO message
            source_domain = (cumulative_t[0] // cumulative_s[0],) + source_domain
            cumulative_s = [cumulative_t[0]] + cumulative_s
            self.source_partial = True
        else:
            assert cumulative_s[0] % cumulative_t[0] == 0  # TODO message
            self.source_partial = False

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
        self.domain = self._new_coords(
            source_domain,
            None,
            lambda c, factor: c // factor,
            lambda c, factor: factor,
            allow_prefix=False,
        )

    def new_size(self, size: Tuple, defaults=None):
        def outer_op(c, factor):
            if c < factor:
                return 1
            else:
                assert c % factor == 0
                return c // factor

        def inner_op(c, factor):
            return min(c, factor)

        return self._new_coords(size, defaults, outer_op, inner_op)

    def new_offset(self, offset: Tuple, defaults=None):
        def outer_op(c, factor):
            return c // factor

        def inner_op(c, factor):
            assert c % factor == 0  # TODO message
            return 0

        return self._new_coords(offset, defaults, outer_op, inner_op)

    def new_intra_box_exprs(self, coords: Tuple, defaults=None):
        def outer_op(c, factor):
            return c // factor

        def inner_op(c, factor):
            return c % factor

        return self._new_coords(coords, defaults, outer_op, inner_op)

    def _new_coords(
        self, coords: Tuple, defaults, outer_op, inner_op, allow_prefix=True
    ):
        if allow_prefix and self.source_partial:
            coords = [None] + list(coords)
        else:
            coords = list(coords)
        assert len(coords) == self.input_dim
        for idx, factor in self.idx_factors:
            assert idx >= 0
            assert idx < self.input_dim
            c = coords[idx]
            if c is None:
                coords[idx : idx + 1] = [None, None]
            else:
                coords[idx : idx + 1] = [outer_op(c, factor), inner_op(c, factor)]
        if defaults is not None:
            assert len(defaults) == len(coords)
            for i, c in enumerate(coords):
                if c is None:
                    coords[i] = defaults[i]
        return coords
