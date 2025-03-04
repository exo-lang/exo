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
    __slots__ = ["partial_domain", "tile", "name", "scaled_dim_idx", "repr_scale"]

    partial_domain: Tuple[CollSizeExpr]
    tile: Tuple[CollSizeExpr]
    name: str
    scaled_dim_idx: Optional[int]
    repr_scale: int

    def __init__(self, partial_domain, tile, name, scaled_dim_idx):
        assert len(partial_domain) == len(tile)
        self.partial_domain = coll_size_tuple(partial_domain)
        self.tile = coll_size_tuple(tile)
        self.name = name
        self.scaled_dim_idx = scaled_dim_idx
        self.repr_scale = 1
        assert scaled_dim_idx is None or scaled_dim_idx < len(tile)

    def scaled(self, scale):
        try:
            tmp = int(scale)
            if tmp != scale or scale <= 0:
                raise ValueError
            scale = tmp
        except Exception:
            raise TypeError(
                f"Expected {self.name} to be scaled by positive int, not {scale!r}"
            )
        i_scale = self.scaled_dim_idx
        if i_scale is None:
            raise ValueError(f"{self.name} cannot be scaled")

        new_tile = [n * scale if i == i_scale else n for i, n in enumerate(self.tile)]
        res = CollUnit(self.partial_domain, new_tile, self.name, i_scale)
        res.repr_scale = self.repr_scale * scale
        return res

    def __mul__(self, scale):
        return self.scaled(scale)

    def __rmul__(self, scale):
        return self.scaled(scale)

    def __repr__(self):
        nm = self.name
        scale = self.repr_scale
        return nm if scale == 1 else f"{scale} * {nm}"

    def int_partial_domain(self, env: Dict[CollParam, int]):
        return int_size_tuple(self.partial_domain, env)

    def int_tile(self, env: Dict[CollParam, int]):
        return int_size_tuple(self.tile, env)

    def int_threads(self, env: Dict[CollParam, int]):
        n = 1
        for c in self.tile:
            n *= c(env)
        return n


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
        simple = lambda s: all(c.isalnum() or c == "." for c in s)
        need_parens = isinstance(self.base_expr, str) and not simple(self.base_expr)
        expr = self._codegen_impl(self.base_expr, need_parens)

        # Wrap final expression in parens if not trivial
        if not simple(expr):
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
            and self.parent == other.parent
            and self.domain == other.domain
            and self.tile == other.tile
            and self.offset == other.offset
            and self.box == other.box
        )

    def __hash__(self):
        return self.hash

    def tiled(self, unit: CollUnit, tiles_needed: int, env: Dict[CollParam, int]):
        advice = CollLoweringAdvice()

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
        # Count tiles; we will check against tiles_needed later.
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

                advice.coll_index = new_exprs[dim_idx] // unit_tile_coord
                new_exprs[dim_idx] = new_exprs[dim_idx] % unit_tile_coord

                if tile_remainder != 0 or tile_count != tiles_needed:
                    advice.hi = tiles_needed

        if tiled_dim_idx is None:
            advice.coll_index = CollIndexExpr(0)

        assert tile_count >= tiles_needed  # TODO message

        new_parent = self
        new_offset = (0,) * len(common_domain)
        new_tile = tuple(new_tile)
        new_box = new_tile

        return (
            CollTiling(
                new_parent,
                common_domain,
                new_tile,
                new_offset,
                new_box,
                new_exprs,
            ),
            advice,
        )

    def specialized(self, unit: CollUnit, lo: int, hi: int, env: Dict[CollParam, int]):
        advice = CollLoweringAdvice()

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
                tile_remainder = common_tile_coord % unit_tile_coord

                # TODO messages
                assert tiled_dim_idx is None
                assert 0 <= lo <= hi <= tile_count
                assert new_box[dim_idx] == common_tile[dim_idx]

                tiled_dim_idx = dim_idx
                stride = unit_tile_coord
                new_exprs[dim_idx] -= lo * stride
                new_offset[dim_idx] = lo * stride
                new_box[dim_idx] = (hi - lo) * stride

                advice.coll_index = new_exprs[dim_idx]
                if lo != 0:
                    advice.lo = lo * unit_tile_coord
                if hi != tile_count or tile_remainder != 0:
                    advice.hi = hi * unit_tile_coord

        if tiled_dim_idx is None:
            advice.coll_index = CollIndexExpr(0)
            assert (lo, hi) == (0, 1)  # TODO message

        new_parent = self.parent

        return (
            CollTiling(
                new_parent,
                common_domain,
                common_tile,
                new_offset,
                new_box,
                new_exprs,
            ),
            advice,
        )

    def box_threads(self):
        n = 1
        for c in self.box:
            n *= c
        return n


class CollLoweringAdvice(object):
    """Advice for lowering a collective tiling or specialization

    Translate the coll_index to C code, and test
    coll_index >= lo  [skip if lo is None]
    coll_index < hi [skip if hi is None]"""

    __slots__ = ["coll_index", "lo", "hi"]
    coll_index: CollIndexExpr
    lo: Optional[int]
    hi: Optional[int]

    def __init__(self, coll_index=None, lo=None, hi=None):
        self.coll_index = coll_index
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return f"CollLoweringAdvice({self.coll_index}, {self.lo}, {self.hi})"


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
            lambda c, factor: c // factor,
            lambda c, factor: factor,
            allow_prefix=False,
        )

    def new_size(self, size: Tuple):
        def outer_op(c, factor):
            if c < factor:
                return 1
            else:
                assert c % factor == 0
                return c // factor

        def inner_op(c, factor):
            return min(c, factor)

        return self._new_coords(size, outer_op, inner_op)

    def new_offset(self, offset: Tuple):
        def outer_op(c, factor):
            return c // factor

        def inner_op(c, factor):
            assert c % factor == 0  # TODO message
            return 0

        return self._new_coords(offset, outer_op, inner_op)

    def new_intra_box_exprs(self, coords: Tuple):
        def outer_op(c, factor):
            return c // factor

        def inner_op(c, factor):
            return c % factor

        return self._new_coords(coords, outer_op, inner_op)

    def _new_coords(self, coords: Tuple, outer_op, inner_op, allow_prefix=True):
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
        return coords


cuda_thread = CollUnit((blockDim,), (1,), "cuda_thread", 0)
cuda_quadpair = CollUnit((blockDim / 16, 16), (2, 4), "cuda_quadpair", None)
cuda_warp = CollUnit((blockDim,), (32,), "cuda_warp", 0)
cuda_warpgroup = CollUnit((blockDim,), (128,), "cuda_warpgroup", 0)
cuda_cta_in_cluster = CollUnit(
    (clusterDim * blockDim,), (blockDim,), "cuda_cta_in_cluster", 0
)
