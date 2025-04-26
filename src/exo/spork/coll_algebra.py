from __future__ import annotations
from fractions import Fraction
from typing import Dict, Optional, Tuple
from math import prod


class CollParam(object):
    """Collective parameter: a variable like blockDim, clusterDim, etc.

    This will be substituted with an integer value during code lowering,
    via a collective environment (just env in this module).
    This is Dict[CollParam, int]"""

    __slots__ = ["name"]

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name + "_param"


class CollSizeExpr(object):
    """Scalar coordinate type for CollUnit (collective unit)

    Product of CollParam(s) and a fraction.
    You should not create this directly, but use overloaded * and /
    e.g. clusterDim * blockDim * 3 / 4"""

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
    """Coerce ints in tuple to CollSizeExpr"""
    result = []
    for n in tup:
        if isinstance(n, int):
            result.append(CollSizeExpr(n, ()))
        else:
            assert isinstance(n, CollSizeExpr)
            result.append(n)
    return tuple(result)


def int_size_tuple(tup: Tuple[CollSizeExpr], env: Dict[CollParam, int]):
    """Translate Tuple[CollSizeExpr] to Tuple[int]"""
    return tuple(n(env) for n in tup)


class CollUnit(object):
    """Collective unit, e.g. a cuda warp, cuda CTA

    Consists of a possibly-partial domain and box size (identical-length
    tuples of CollSizeExpr) which should be described in
    collective algebra documentation.

    As a convenience, if repr_scale is not None, we support multiplying by an
    int; this scales the repr_scale-th coordinate of the box size.
    This is intended syntax for the Exo end user (e.g. 8 * cuda_thread)
    """

    __slots__ = ["domain", "box", "name", "scaled_dim_idx", "repr_scale"]

    domain: Tuple[CollSizeExpr]
    box: Tuple[CollSizeExpr]
    name: str
    scaled_dim_idx: Optional[int]
    repr_scale: int

    def __init__(self, domain, box, name, scaled_dim_idx):
        assert len(domain) == len(box)
        self.domain = coll_size_tuple(domain)
        self.box = coll_size_tuple(box)
        self.name = name
        self.scaled_dim_idx = scaled_dim_idx
        self.repr_scale = 1
        assert scaled_dim_idx is None or scaled_dim_idx < len(box)

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

        new_box = [n * scale if i == i_scale else n for i, n in enumerate(self.box)]
        res = CollUnit(self.domain, new_box, self.name, i_scale)
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

    def int_domain(self, env: Dict[CollParam, int]):
        return int_size_tuple(self.domain, env)

    def int_box(self, env: Dict[CollParam, int]):
        return int_size_tuple(self.box, env)

    def int_threads(self, env: Dict[CollParam, int]):
        n = 1
        for c in self.box:
            n *= c(env)
        return n


class CollIndexExpr(object):
    """This is mainly intended to aid codegen, with intra_box_exprs

    We need to be able to deduce the coordinates of each thread in
    a given thread box of a CollTiling.
    For example, in the top-level collective (clusterDim, blockDim),
    the intra_box_exprs are (blockIdx % clusterDim, threadIdx).

    These CollIndexExpr (collective index expressions) will be an
    expression of blockIdx/threadIdx (or the equivalent if we support
    non-CUDA), plus division, modulo, and subtract by integers.
    """

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
            # We rely on this in DomainCompletionOp.new_intra_box_exprs!
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
    """Immutable collective tiling. See collective algebra documentation."""

    __slots__ = [
        "parent",
        "full_domain",
        "tile",
        "offset",
        "box",
        "intra_box_exprs",
        "tile_count",
        "tile_expr",
    ]

    parent: Optional[CollTiling]
    full_domain: Tuple[int]
    tile: Tuple[int]
    offset: Tuple[int]
    box: Tuple[int]
    intra_box_exprs: Tuple[CollIndexExpr]
    tile_count: int
    tile_expr: CollIndexExpr

    def __init__(
        self,
        parent,
        full_domain,
        tile,
        offset,
        box,
        intra_box_exprs,
        tile_count,
        tile_expr,
    ):
        assert parent is None or isinstance(parent, CollTiling)
        self.parent = parent
        self.full_domain = tuple(full_domain)
        self.tile = tuple(tile)
        self.offset = tuple(offset)
        self.box = tuple(box)
        for tup in (full_domain, tile, offset, box):
            assert all(isinstance(c, int) for c in tup)
            assert len(tup) == len(full_domain)

        self.intra_box_exprs = tuple(intra_box_exprs)
        assert all(isinstance(c, CollIndexExpr) for c in intra_box_exprs)
        assert len(intra_box_exprs) == len(box)

        self.tile_count = tile_count
        self.tile_expr = tile_expr
        assert isinstance(tile_count, int)
        assert isinstance(tile_expr, CollIndexExpr)

    def __repr__(self):
        return f"CollTiling({self.parent}, {self.full_domain}, {self.tile}, {self.offset}, {self.box}, {self.intra_box_exprs}, {self.tile_count}, {self.tile_expr})"

    def equiv(self, other: Optional[CollTiling]):
        if other is None:
            return False
        assert isinstance(other, CollTiling)
        if self.parent is None:
            parent_match = other.parent is None
        else:
            parent_match = self.parent.equiv(other.parent)
        return (
            parent_match
            and self.full_domain == other.full_domain
            and self.tile == other.tile
            and self.offset == other.offset
            and self.box == other.box
        )

    def tiled(self, unit: CollUnit, tiles_needed: int, env: Dict[CollParam, int]):
        """Tile the CollTiling with the given collective unit.

        Returns (CollTiling, CollLoweringAdvice).
        Produces the given number of tiles (or throws if not possible).
        self is the parent of the resulting CollTiling.
        """
        advice = CollLoweringAdvice()

        # Translate unit domain and tiling to concrete integers
        unit_domain = unit.int_domain(env)
        unit_box = unit.int_box(env)

        # Determine the common domain between us and the given unit
        unit_completion = DomainCompletionOp(
            unit_domain, self.full_domain, allow_partial_source=True
        )
        self_completion = DomainCompletionOp(
            self.full_domain, unit_domain, allow_partial_source=False
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
        max_tile_count = 1
        tile_remainder = 0
        for dim_idx, unit_box_coord in enumerate(unit_completion.new_size(unit_box)):
            domain_coord = common_domain[dim_idx]
            box_coord = old_box[dim_idx]
            if (
                unit_box_coord is not None
                and unit_box_coord != domain_coord
                and unit_box_coord != box_coord
            ):
                assert unit_box_coord < box_coord  # TODO message
                assert tiled_dim_idx is None  # TODO message
                tiled_dim_idx = dim_idx
                max_tile_count = box_coord // unit_box_coord
                tile_remainder = box_coord % unit_box_coord
                new_tile[dim_idx] = unit_box_coord

                advice.coll_index = new_exprs[dim_idx] // unit_box_coord
                new_exprs[dim_idx] = new_exprs[dim_idx] % unit_box_coord

                if tile_remainder != 0 or max_tile_count != tiles_needed:
                    advice.hi = tiles_needed

        if tiled_dim_idx is None:
            advice.coll_index = CollIndexExpr(0)
            advice.hi = tiles_needed  # In case tiles_needed = 0

        assert max_tile_count >= tiles_needed  # TODO message

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
                tiles_needed,
                advice.coll_index,
            ),
            advice,
        )

    def specialized(self, unit: CollUnit, lo: int, hi: int, env: Dict[CollParam, int]):
        """Specialize the CollTiling.

        Returns (CollTiling, CollLoweringAdvice).

        self and the resulting CollTiling share a common parent."""
        advice = CollLoweringAdvice()

        # Translate unit domain and tiling to concrete integers
        unit_domain = unit.int_domain(env)
        unit_box = unit.int_box(env)

        # Determine the common domain between us and the given unit
        unit_completion = DomainCompletionOp(
            unit_domain, self.full_domain, allow_partial_source=True
        )
        self_completion = DomainCompletionOp(
            self.full_domain, unit_domain, allow_partial_source=False
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
        for dim_idx, unit_box_coord in enumerate(unit_completion.new_size(unit_box)):
            domain_coord = common_domain[dim_idx]
            common_tile_coord = common_tile[dim_idx]
            if (
                unit_box_coord is not None
                and unit_box_coord != domain_coord
                and unit_box_coord != common_tile_coord
            ):
                tile_count = common_tile_coord // unit_box_coord
                tile_remainder = common_tile_coord % unit_box_coord

                # TODO messages
                assert tiled_dim_idx is None
                assert 0 <= lo <= hi <= tile_count
                assert new_box[dim_idx] == common_tile[dim_idx]

                tiled_dim_idx = dim_idx
                stride = unit_box_coord
                advice.coll_index = new_exprs[dim_idx]  # before -= below
                new_exprs[dim_idx] -= lo * stride
                new_offset[dim_idx] = lo * stride
                new_box[dim_idx] = (hi - lo) * stride

                if lo != 0:
                    advice.lo = lo * unit_box_coord
                if hi != tile_count or tile_remainder != 0:
                    advice.hi = hi * unit_box_coord

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
                self.tile_count,
                self.tile_expr,
            ),
            advice,
        )

    def box_num_threads(self):
        """Total number of threads in the thread box"""
        return prod(self.box)

    def tile_num_threads(self):
        """Total number of threads in the thread tile.

        Unlike the box, this includes threads that are inactive
        """
        return prod(self.tile)

    def unit_mismatch(
        self, unit: CollUnit, env: Dict[CollParam, int], no_message=False
    ):
        """Return False iff the thread boxes match the given collective unit

        Matching requires both size match and alignment match.

        If mismatched, return a str reason, unless
        no_message=True (return True if so).
        """
        assert isinstance(unit, CollUnit)

        self_n_threads = self.box_num_threads()
        # TODO explain: tile = box for unit but not CollTiling
        unit_box_raw = unit.int_box(env)
        unit_n_threads = unit.int_threads(env)
        unit_domain = unit.int_domain(env)
        if self_n_threads != unit_n_threads:
            return no_message or (
                f"Have {self_n_threads} threads {self.box}; "
                f"expected {unit_n_threads} ({unit})"
            )
        try:
            tiling = self
            while tiling is not None:
                unit_completion = DomainCompletionOp(
                    unit_domain, tiling.full_domain, allow_partial_source=True
                )
                tiling_completion = DomainCompletionOp(
                    tiling.full_domain, unit_domain, allow_partial_source=False
                )
                assert unit_completion.domain == tiling_completion.domain

                new_unit_box = unit_completion.new_size(unit_box_raw, 1)

                # Check box size for leaf CollTiling
                if self is tiling:
                    new_tiling_box = tiling_completion.new_size(tiling.box)
                    if new_unit_box != new_tiling_box:
                        return no_message or (
                            f"Have threads in shape {new_tiling_box}; "
                            f"expected {new_unit_box} "
                            f"({unit}); domain={unit_completion.domain}"
                        )

                # Check alignment for all CollTiling to root
                new_tiling_offset = tiling_completion.new_offset(tiling.offset, 0)
                assert len(new_tiling_offset) == len(new_unit_box)
                for off_c, box_c in zip(new_tiling_offset, new_unit_box):
                    if off_c % box_c != 0:
                        return no_message or f"Incorrect alignment for {unit}"

                # Traverse to root
                tiling = tiling.parent

        except DomainCompletionError as e:
            # TODO no one is going to understand this failure mode...
            return no_message or "domain completion failed: " + str(e)

        return False  # False => match


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


class DomainCompletionError(Exception):
    pass


class DomainCompletionOp(object):
    __slots__ = ["input_dim", "source_partial", "remove_idx", "idx_factors", "domain"]
    input_dim: int
    source_partial: bool
    remove_idx: Tuple[int]
    idx_factors: Tuple[int, int]
    domain: Tuple[int]

    def __init__(
        self,
        source_domain: Tuple[int],
        target_domain: Tuple[int],
        allow_partial_source: bool,
    ):
        # Record the original source domain dimension, before modifications
        assert isinstance(source_domain, tuple)
        original_source_domain = source_domain
        self.input_dim = len(original_source_domain)

        # Calculate total number of threads in source and target domain
        # and deduce if the source domain is partial.
        # NB we have to do this early, before removing 1s.
        def total_threads(domain):
            tmp = 1
            for c in domain:
                tmp *= c
            return tmp

        threads_s = total_threads(source_domain)
        threads_t = total_threads(target_domain)
        s_to_t_multiplier = threads_t // threads_s
        if allow_partial_source and threads_s != threads_t:
            # Complete source domain if needed (prepend coordinate to domain
            # so total thread count matches that of target domain).
            # self.source_partial means we will do a matching prepend when
            # translating coordinates
            if threads_t % threads_s != 0:
                raise DomainCompletionError()  # TODO message
            self.source_partial = True
            source_domain = (s_to_t_multiplier,) + source_domain
        else:
            if threads_s % threads_t != 0:
                raise DomainCompletionError()  # TODO message
            self.source_partial = False

        # Remove 1s in source and target domains
        def remove_1s_impl(domain):
            new_domain = []
            remove_idx = []
            for i, c in enumerate(domain):
                assert c >= 1, "Expected domain to consist only of positive numbers"
                if c == 1:
                    remove_idx.append(i - len(remove_idx))
                else:
                    new_domain.append(c)
            return tuple(new_domain), tuple(remove_idx)

        source_domain, self.remove_idx = remove_1s_impl(source_domain)
        target_domain, _ = remove_1s_impl(target_domain)

        # Generate list of splitting commands (i, f) to apply in order
        # "split the (current) i-th dimension by f".
        idx_factors = []

        def cumulative_thread_counts(domain):
            tmp = [1]
            for c in domain[::-1]:
                tmp.append(tmp[-1] * c)
            tmp.reverse()
            return tmp

        cumulative_s = cumulative_thread_counts(source_domain)
        cumulative_t = cumulative_thread_counts(target_domain)

        for i_s in range(len(source_domain) - 1, -1, -1):
            s0 = cumulative_s[i_s] if i_s >= 0 else float("inf")
            s1 = cumulative_s[i_s + 1]
            for i_t in range(len(target_domain) - 1, -1, -1):
                t0 = cumulative_t[i_t]
                if s0 > t0 > s1:
                    t1 = cumulative_t[i_t + 1]
                    divisor = max(t1, s1)
                    split = t0 // divisor
                    if i_s >= 0 and source_domain[i_s] % split != 0:
                        raise DomainCompletionError()  # TODO message
                    idx_factors.append((i_s, split))

        self.idx_factors = idx_factors

        # Generate the new domain
        self.domain = self._new_coords(
            original_source_domain,
            lambda c, factor: c // factor,
            lambda c, factor: factor,
            1,
            partial_prepend=s_to_t_multiplier,
        )

    def new_size(self, size: Tuple, partial_prepend=None):
        def outer_op(c, factor):
            if c < factor:
                return 1
            else:
                if c % factor != 0:
                    raise DomainCompletionError()  # TODO message
                return c // factor

        def inner_op(c, factor):
            return min(c, factor)

        return self._new_coords(
            size, outer_op, inner_op, 1, partial_prepend=partial_prepend
        )

    def new_offset(self, offset: Tuple, partial_prepend=None):
        def outer_op(c, factor):
            return c // factor

        def inner_op(c, factor):
            if c % factor != 0:
                raise DomainCompletionError()  # TODO message
            return 0

        return self._new_coords(
            offset, outer_op, inner_op, 0, partial_prepend=partial_prepend
        )

    def new_intra_box_exprs(self, coords: Tuple):
        def outer_op(c, factor):
            return c // factor

        def inner_op(c, factor):
            return c % factor

        # NB rely on CollIndexExpr(...) % 1 to be 0 for expected_removed_coord
        return self._new_coords(coords, outer_op, inner_op, CollIndexExpr(0))

    def _new_coords(
        self,
        coords: Tuple,
        outer_op,
        inner_op,
        expected_removed_coord,
        partial_prepend=None,
    ):
        # We have to do translation in the same order as we initialized the
        # DomainCompletionOp.
        #   a. prepend if completing a partial domain
        #   b. remove indices (1s)
        #   c. split coordinates
        # Wrong order would e.g. cause index values to lose intended meaning.
        assert len(coords) == self.input_dim

        if self.source_partial:
            coords = [partial_prepend] + list(coords)
        else:
            coords = list(coords)

        for i in self.remove_idx:
            assert coords[i] == expected_removed_coord
            del coords[i]

        for idx, factor in self.idx_factors:
            assert idx >= 0
            assert idx < len(coords)
            c = coords[idx]
            if c is None:
                coords[idx : idx + 1] = [None, None]
            else:
                coords[idx : idx + 1] = [outer_op(c, factor), inner_op(c, factor)]
        return coords


standalone_thread = CollUnit((1,), (1,), "standalone_thread", 0)
cuda_thread = CollUnit((blockDim,), (1,), "cuda_thread", 0)
cuda_quadpair = CollUnit((blockDim / 16, 16), (2, 4), "cuda_quadpair", None)
cuda_warp = CollUnit((blockDim,), (32,), "cuda_warp", 0)
cuda_warpgroup = CollUnit((blockDim,), (128,), "cuda_warpgroup", 0)
cuda_cta_in_cluster = CollUnit(
    (clusterDim * blockDim,), (blockDim,), "cuda_cta_in_cluster", 0
)
