from __future__ import annotations

# Note: no imports from the rest of Exo so it's easy to run side experiments
# on coll_algebra and do demos of it as a sort of "type system"

from dataclasses import dataclass, replace, field
from enum import Enum, auto
from fractions import Fraction
from typing import Dict, Optional, Tuple, Set
from math import prod


class CollTilingError(Exception):
    pass


class CollParam(object):
    """Collective parameter: a variable like blockDim, clusterDim, etc.

    This will be substituted with an integer value during code lowering,
    via a collective environment (just env in this module).
    This is Dict[CollParam, int]"""

    __slots__ = ["name", "hash"]

    def __init__(self, name):
        self.name = name
        self.hash = hash(name)

    def __repr__(self):
        return self.name + "_param"

    def __eq__(self, param):
        return isinstance(param, CollParam) and self.name == param.name

    def __hash__(self):
        return self.hash


class CollSizeExpr(object):
    """Scalar coordinate type for CollUnit (collective unit)

    Product of CollParam(s) and a fraction.
    You should not create this directly, but use overloaded * and /
    e.g. clusterDim * blockDim * 3 / 4"""

    __slots__ = ["scalar", "coll_params"]

    scalar: Fraction
    coll_params: Tuple[CollParam]

    def __init__(self, scalar: Fraction | int, coll_params: Tuple[CollParam] = ()):
        self.scalar = Fraction(scalar)
        self.coll_params = coll_params

    def __call__(self, env: Dict[CollParam, int]):
        n = self.scalar.numerator
        for p in self.coll_params:
            n *= env[p]
        if n % self.scalar.denominator != 0:
            envstr = " ".join(f"{k}={v}" for k, v in env.items())
            raise CollTilingError(f"{self!r} has divisibility issue with {envstr}")
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
            return f"CollSizeExpr({self.scalar})"
        s = " * ".join([p.name for p in self.coll_params])
        if self.scalar.numerator != 1:
            s += f" * {self.scalar.numerator}"
        if self.scalar.denominator != 1:
            s += f" / {self.scalar.denominator}"
        return s

    def equals_const(self, c):
        return not self.coll_params and self.scalar == c


blockDim_param = CollParam("blockDim")
blockDim = CollSizeExpr(1, (blockDim_param,))
clusterDim_param = CollParam("clusterDim")
clusterDim = CollSizeExpr(1, (clusterDim_param,))


def coll_size_tuple_helper(tup):
    """To (Tuple[CollSizeExpr], is_agnostic: bool)"""
    agnostic = False
    result = []
    for n in tup:
        if isinstance(n, int):
            result.append(CollSizeExpr(n, ()))
        elif n is None:
            result.append(None)
            agnostic = True
        else:
            assert isinstance(n, CollSizeExpr)
            result.append(n)
    return tuple(result), agnostic


def int_size_tuple(tup: Tuple[Optional[CollSizeExpr]], env: Dict[CollParam, int]):
    """Translate Tuple[Optional[CollSizeExpr]] to Tuple[Optional[int]]"""
    return tuple(None if n is None else n(env) for n in tup)


def format_tuple(tup: Tuple[Optional[CollSizeExpr]]):
    return "(" + ", ".join("*" if n is None else str(n) for n in tup) + ")"


class CollUnit(object):
    """Collective unit, e.g. a cuda warp, cuda CTA

    Consists of a possibly-partial domain and box size (identical-length
    tuples of CollSizeExpr) which should be described in
    collective algebra documentation.

    The box may also contain None (agnostic dimension)

    As a convenience, if scaled_dim_idx is not None, we support
    multiplying by an int; this scales the scaled_dim_idx-th
    coordinate of the box size.  This is intended syntax for the Exo
    end user (e.g. 8 * cuda_thread)

    """

    __slots__ = ["domain", "box", "name", "scaled_dim_idx", "repr_scale", "agnostic"]

    domain: Tuple[CollSizeExpr]
    box: Tuple[Optional[CollSizeExpr]]
    name: str
    scaled_dim_idx: Optional[int]
    repr_scale: int
    agnostic: bool

    def __init__(self, domain, box, name, scaled_dim_idx=None):
        assert len(domain) == len(box)
        self.domain, None_in_domain = coll_size_tuple_helper(domain)
        self.box, self.agnostic = coll_size_tuple_helper(box)
        assert not None_in_domain
        self.name = name
        self.scaled_dim_idx = scaled_dim_idx
        self.repr_scale = 1
        if scaled_dim_idx is not None:
            assert scaled_dim_idx < len(box)
            assert box[scaled_dim_idx] is not None

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

        new_box = list(self.box)
        new_box[i_scale] *= scale
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
        assert not self.agnostic
        n = 1
        for c in self.box:
            n *= c(env)
        return n

    def is_always_single_thread(self):
        return all(c.equals_const(1) for c in self.box)


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

    __slots__ = ["base_expr", "base_hi", "current_range", "ops"]

    # Target language (e.g. C) expression, e.g. "threadIdx" as str
    # or constant value as int
    base_expr: str | int

    # base_expr known to be in range [0, base_hi-1], or [0, inf) if None
    base_hi: Optional[int]

    # Known range of full expression [lo, hi-1].
    # NOTE: divide and modulo are treated as in Python (floor division,
    # always non-negative modulo result), but we codegen to C without
    # handling this, because negative values should never arise in the
    # C code (these values correspond to masked-out threads).
    current_range: Optional[Tuple[int, int]]

    # Sequence of (operator, int) pairs to apply.
    # Only allowed if the base_expr is a str.
    ops: Tuple[Tuple[str, int]]

    def __init__(self, base_expr, base_hi=None, _current_range=None, _ops=()):
        self.base_expr = base_expr
        if isinstance(base_expr, int):
            self.base_hi = base_expr + 1
            self.current_range = (base_expr, base_expr + 1)
            assert not _ops
        else:
            assert isinstance(base_expr, str)
            self.base_hi = base_hi
            if _ops:  # Internal-use constructor
                self.current_range = _current_range
            elif base_hi is None:
                self.current_range = None
            else:
                self.current_range = (0, base_hi)
        self.ops = _ops

    def __repr__(self):
        expr = f"CollIndexExpr({self.base_expr!r}, {self.base_hi})"
        return self._codegen_impl(expr, False)

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
        _range = self.current_range
        if _range:
            _range = (_range[0] - v, _range[1] - v)
        return CollIndexExpr(self.base_expr, self.base_hi, _range, new_ops)

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
        _range = self.current_range
        if _range:
            # Ceiling division required to calculate new hi
            lo, hi = _range
            _range = (lo // v, (hi + v - 1) // v)
        return CollIndexExpr(self.base_expr, self.base_hi, _range, new_ops)

    def __floordiv__(self, v: int):
        return self.__truediv__(v)

    def __mod__(self, v: int):
        assert isinstance(v, int)
        _range = self.current_range
        if isinstance(self.base_expr, int):
            return CollIndexExpr(self.base_expr % v)
        elif v == 1:
            # We rely on this in DomainCompletionOp.new_intra_box_exprs!
            return CollIndexExpr(0)
        elif _range and _range[0] >= 0 and _range[1] <= v:
            # Modulo with no effect
            return self

        # Result of x % v is is in range [0, v-1]
        # We will append "% v" to the ops list, but try to remove redundant ops.
        _range = (0, v)
        ops = self.ops
        while ops:
            op, u = ops[-1]
            # A prior mod or subtract by a constant divisible by v
            # will have its effect eliminated by a subsequent `% v`.
            if (op == "%" or op == "-") and u % v == 0:
                ops = ops[:-1]
                continue
            break
        return CollIndexExpr(self.base_expr, self.base_hi, _range, ops + (("%", v),))

    def __call__(self, var_value):
        base_expr = self.base_expr
        if isinstance(base_expr, int):
            return base_expr
        assert isinstance(base_expr, str)
        result = var_value
        for op, v in self.ops:
            if op == "%":
                result %= v
            elif op == "-":
                result -= v
            else:
                assert op == "/"
                result //= v
        assert isinstance(result, int)
        return result

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


coll_index_0 = CollIndexExpr(0)


@dataclass(slots=True, init=False)
class CollCodegen:
    """For lowering a collective tiling or specialization:

    Translate the codegen_expr to C code, and test
    codegen_expr >= codegen_static_lo [skip if None]
    codegen_expr < codegen_static_hi [skip if None]

    thread_pitch is the distance in # of threads 0th
    thread in a thread collective and the 0th thread of the
    next-adjacent thread collective in a tiling (Cf. "seat pitch")
    This gives some notion of what "axis" the tiling is performed on,
    separate from the size of the collective unit.  For example,
    "adjacent warps in a CTA" has pitch 32, while
    "warp 0 of each CTA in a cluster" has pitch blockDim.

    thread_pitch = 0 when there are fewer than 2 tiles in the tiling.

    """

    # Iterator variable name
    iter: object

    dim_idx_factors: List[Tuple[int, int]]

    # Advice for compiling to abstract machine ThreadsFor loop (am_threads).
    # ThreadsFor(..., dim_idx, offset, box)
    dim_idx: Optional[int]
    offset: int
    box: Optional[int]

    tile_count: int
    thread_pitch: int
    codegen_expr: CollIndexExpr
    codegen_static_lo: Optional[int]
    codegen_static_hi: Optional[int]

    def __init__(self, _iter):
        # Initialized with defaults for a "do nothing" tiling i.e.
        # for i in cuda_threads(0, 1, unit=current_coll_unit):
        self.iter = _iter
        self.dim_idx_factors = []
        self.dim_idx = None
        self.offset = 0
        self.box = None
        self.tile_count = 1
        self.thread_pitch = 0
        self.codegen_expr = coll_index_0
        self.codegen_static_lo = None
        self.codegen_static_hi = None


@dataclass(slots=True, frozen=True)
class CollDimOp:
    iter: object
    offset: int
    box: int
    tile_count: int
    intra_box_expr: CollIndexExpr
    thread_pitch: int
    tree_depth: int

    def divided_by(self, factor):
        old_offset = self.offset
        old_box = self.box
        new_offset, r_offset = divmod(old_offset, factor)
        new_box, r_box = divmod(old_box, factor)
        new_expr = self.intra_box_expr // factor
        if r_offset or r_box:
            # TODO this error isn't very good without context.
            raise DomainCompletionError(
                f"thread alignment issue with {self.iter}-loop:\n"
                f"with thread_pitch={self.thread_pitch}, "
                f"offset={old_offset}, box={old_box}, "
                f"offset or box wasn't divisible by {factor}"
            )
        return replace(self, offset=new_offset, box=new_box, intra_box_expr=new_expr)


class CollDimExpectation(Enum):
    agnostic = auto()
    intact = auto()
    subdiv = auto()


@dataclass(slots=True, frozen=True)
class CollDim:
    dim_extent: int
    dim_thread_pitch: int
    dim_ops: Tuple[CollDimOp]
    dim_iter_expr: CollIndexExpr

    # Relevant when the CollTiling is completed for a certain unit.
    dim_expectation: DimExpectation = CollDimExpectation.agnostic

    def __post_init__(self):
        p = self.dim_thread_pitch

    def get_base_offset_coord(self, subdiv_only: bool) -> int:
        if subdiv_only:
            x = self.dim_expectation
            assert (
                x != CollDimExpectation.agnostic
            ), "need unit_completion(...) with non-agnostic unit"
            if x == CollDimExpectation.intact:
                return 0
        return sum(op.offset for op in self.dim_ops)

    def get_box_coord(self) -> int:
        if ops := self.dim_ops:
            return ops[-1].box
        return self.dim_extent

    def get_base_box_coord(self, subdiv_only: bool) -> int:
        if subdiv_only:
            x = self.dim_expectation
            assert (
                x != CollDimExpectation.agnostic
            ), "need unit_completion(...) with non-agnostic unit"
            if x == CollDimExpectation.intact:
                return self.dim_extent
        return self.get_box_coord()

    def get_leaf_intra_box_expr(self) -> CollIterExpr:
        if ops := self.dim_ops:
            return ops[-1].intra_box_expr
        return self.dim_iter_expr

    def get_expected_box_coord(self) -> Optional[int]:
        x = self.dim_expectation
        if x == CollDimExpectation.subdiv:
            return 1
        if x == CollDimExpectation.intact:
            return self.dim_extent
        return None


@dataclass(slots=True)
class CollTiling:
    """Immutable collective tiling."""

    _tree_depth: int
    _codegen: CollCodegen
    _dims: List[CollDim]
    _unit_cache: dict = field(default_factory=dict)

    def get_tree_depth(self) -> int:
        return self._tree_depth

    def get_codegen(self) -> CollCodegen:
        return self._codegen

    def get_domain(self) -> Tuple[int]:
        return tuple(d.dim_extent for d in self._dims)

    def get_base_offset(self, subdiv_only: bool) -> Tuple[int]:
        return tuple(d.get_base_offset_coord(subdiv_only) for d in self._dims)

    def get_box(self) -> Tuple[int]:
        return tuple(d.get_box_coord() for d in self._dims)

    def get_base_box(self, subdiv_only: bool) -> Tuple[int]:
        return tuple(d.get_base_box_coord(subdiv_only) for d in self._dims)

    def get_expected_box(self) -> Tuple[Optional[int]]:
        return tuple(d.get_expected_box_coord() for d in self._dims)

    def get_box_num_threads(self) -> int:
        return prod(d.get_box_coord() for d in self._dims)

    def get_domain_num_threads(self) -> int:
        return prod(d.dim_extent for d in self._dims)

    def get_dim_ops(self) -> List[CollDimOp]:
        return [op for d in self._dims for op in d.dim_ops]

    def get_dims(self) -> Tuple[CollDim]:
        return tuple(self._dims)

    def get_subdiv_dim_ops(self) -> List[CollDimOp]:
        def is_subdiv(d: CollDim):
            x = d.dim_expectation
            assert (
                x != CollDimExpectation.agnostic
            ), "need unit_completion(...) with non-agnostic unit"
            return x == CollDimExpectation.subdiv

        return [op for d in self._dims for op in d.dim_ops if is_subdiv(d)]

    def err(
        self,
        unit: CollUnit,
        tile_count: int,
        msg: str,
        new_domain: Optional[Tuple[int, ...]] = None,
        new_self_box: Optional[Tuple[int, ...]] = None,
        new_unit_box: Optional[Tuple[int, ...]] = None,
    ):
        def fmt_tup(t):
            return "[" + ",".join("%4i" % n for n in t) + "]"

        suffix = ""
        if new_domain is not None:
            assert len(new_domain) == len(new_self_box) == len(new_unit_box)
            suffix = f"""
  D = {fmt_tup(new_domain)}
ω.B = {fmt_tup(new_self_box)}
δ.B = {fmt_tup(new_unit_box)}"""

        raise CollTilingError(
            f"Bad CollTiling ({msg}); "
            f"box={self.get_box()}, tried tiling {tile_count}-many {unit}, "
            f"i.e. CollUnit({unit.domain}, {unit.box}, ...)" + suffix
        )

    def tiled(
        self,
        _iter: object,
        unit: CollUnit,
        tiles_needed: int,
        env: Dict[CollParam, int],
    ) -> "CollTiling":
        """Tile the CollTiling with the given collective unit.

        Returns a new CollTiling with a higher tree_depth.

        Produces the given number of tiles (or throws if not possible).
        The _iter is passed through to the generated CollTiling
        ("iterator variable name").

        """
        return self._make_derived(_iter, unit, 0, tiles_needed, env, True)

    def specialized(
        self, _iter: object, unit: CollUnit, lo: int, hi: int, env: Dict[CollParam, int]
    ) -> "CollTiling":
        """Specialize the CollTiling

        Returns a new CollTiling with a higher tree_depth.

        We add another level of "tiling", where the tiling just yields
        one thread box of possibly fewer threads ("warp specialization")

        """
        return self._make_derived(_iter, unit, lo, hi, env, False)

    def unit_completion(self, unit: CollUnit, env: Dict[CollParam, int]):
        """Return a new CollTiling with domain completion done for the unit.

        For unit matching, each domain dimension will be labelled one of:

        * subdiv: expect a box coordinate of 1 on this dimension.
        * intact: expect a box coordinate equal to the extent
        * agnostic: no requirements

        """

        unit_domain = unit.int_domain(env)
        unit_box = unit.int_box(env)
        key = (unit_domain, unit_box)
        try:
            return self._unit_cache[key]
        except KeyError:
            new = self._unit_completion_impl(*key)
            self._unit_cache[key] = new
            return new

    def unit_mismatch(
        self,
        unit: CollUnit,
        env: Dict[CollParam, int],
        *,
        no_message=False,
    ) -> Optional[str]:
        """Unpack the CollUnit with alignment and 1-padding to get δ (collective type).
        Are statements with this CollTiling at δ-scope?
        If so, return None, else, give a reason this is not the case.
        """

        # Do domain completion on the current CollTiling to get completed new_tiling.
        try:
            new_tiling = self.unit_completion(unit, env)
        except DomainCompletionError as e:
            # TODO may make more sense with context
            return no_message or str(e)

        # Check that the box of the new_tiling is as expected.
        actual_box = new_tiling.get_box()
        expected_box = new_tiling.get_expected_box()
        assert len(actual_box) == len(expected_box)
        for a, e in zip(actual_box, expected_box):
            if e is not None and a != e:
                return (
                    no_message
                    or f"domain={new_tiling.get_domain()}, box={actual_box} ({prod(actual_box)} threads); expected box={expected_box}"
                )
        return None  # i.e. no mismatch detected.

    def base_mismatch(self, other, *, subdiv_only: bool) -> Optional[str]:
        """Check that the two CollTiling have the same 0th thread collective.

        Assume that all control values are 0.
        Do the two CollTiling instances generate the same thread collective?
        If so, return None, else, give a reason this is not the case.

        """
        first_domain = self.get_domain()
        first_box = self.get_base_box(subdiv_only)
        first_offset = self.get_base_offset(subdiv_only)
        second_domain = other.get_domain()
        second_box = other.get_base_box(subdiv_only)
        second_offset = other.get_base_offset(subdiv_only)

        def linear_offset(domain, offset):
            n = 0
            for dc, oc in zip(domain, offset):
                n *= dc
                n += oc
            return n

        try:
            # Check base offset equality.
            first_linear_offset = linear_offset(first_domain, first_offset)
            second_linear_offset = linear_offset(second_domain, second_offset)
            if first_linear_offset != second_linear_offset:
                raise CollTilingError("Incompatible offsets")
            # Check base box equality, with domain completion.
            op1 = DomainCompletionOp(
                first_domain, second_domain, allow_partial_source=False
            )
            op2 = DomainCompletionOp(
                second_domain, first_domain, allow_partial_source=False
            )
            assert op1.domain == op2.domain
            if op1.new_size(first_box) != op2.new_size(second_box):
                raise CollTilingError("Incompatible box size")

        except CollTilingError as e:
            return f"""{e}
    First  domain={first_domain}, base_offset={first_offset}, base_box={first_box}
    Second domain={second_domain}, base_offset={second_offset}, base_box={second_box}"""

        return None  # i.e. no mismatch detected.

    def _apply_split_dim(self, dim_idx, factor):
        assert dim_idx < len(self._dims)
        self._codegen.dim_idx_factors.append((dim_idx, factor))
        old_dim = self._dims[dim_idx]
        old_iter_expr = old_dim.dim_iter_expr

        outer_ops = []
        for op in old_dim.dim_ops:
            if op.box >= factor:
                outer_ops.append(op.divided_by(factor))
            else:
                break
        outer_ops = tuple(outer_ops)
        inner_ops = old_dim.dim_ops[len(outer_ops) :]

        outer_dim = CollDim(
            old_dim.dim_extent // factor,
            old_dim.dim_thread_pitch * factor,
            outer_ops,
            old_iter_expr // factor,
        )
        inner_dim = CollDim(
            factor, old_dim.dim_thread_pitch, inner_ops, old_iter_expr % factor
        )
        self._dims[dim_idx : dim_idx + 1] = [outer_dim, inner_dim]

    def _unit_completion_impl(
        self, unit_original_domain: Tuple[int], unit_original_box: Tuple[int]
    ):
        # Create new CollTiling.
        # We don't consider this a new level of tree depth, as no subdivision
        # from an added variable occured. We just re-shaped the domain threads.
        codegen = CollCodegen(None)
        new = CollTiling(self._tree_depth, codegen, list(self._dims))

        # Create temporary collective unit based on input unit.
        # It has all box coordinates either None, 1, or equal to the
        # corresponding domain coordinate. This enforces alignment.
        tmp_unit_domain = ()
        tmp_unit_box = ()
        assert len(unit_original_domain) == len(unit_original_box)
        for domain_c, box_c in zip(unit_original_domain, unit_original_box):
            if box_c is None or domain_c == box_c:
                tmp_unit_domain += (domain_c,)
                tmp_unit_box += (box_c,)
            else:
                if box_c is not None and domain_c % box_c != 0:
                    raise DomainCompletionError(
                        f"Invalid alignment for CollUnit({unit_original_domain}, {unit_original_box})\n"
                        f"{box_c} doesn't divide {domain_c} (change clusterDim/blockDim?)\n"
                        f"domain={self.get_domain()}"
                    )
                tmp_unit_domain += (domain_c // box_c, box_c)
                tmp_unit_box += (1, box_c)

        # Perform domain completion on the unit and new CollTiling.
        # Note partial_prepend=1; if the unit is subject to domain completion,
        # we want any added dimensions to be 1, so the unit doesn't grow.
        self_original_domain = self.get_domain()
        unit_completion = DomainCompletionOp(
            tmp_unit_domain, self_original_domain, allow_partial_source=True
        )
        self_completion = DomainCompletionOp(
            self_original_domain, tmp_unit_domain, allow_partial_source=False
        )
        assert unit_completion.domain == self_completion.domain
        assert not self_completion.remove_idx
        unit_box = unit_completion.new_size(tmp_unit_box, partial_prepend=1)
        for dim_idx, split_factor in self_completion.idx_factors:
            new._apply_split_dim(dim_idx, split_factor)
        new_domain = tuple(self_completion.domain)
        assert new.get_domain() == new_domain, f"{new.get_domain()} == {new_domain}"

        # Label dimensions as intact, subdivided, or agnostic.
        assert len(new._dims) == len(unit_box)
        for i, unit_c in enumerate(unit_box):
            if unit_c is None:
                dim_expectation = CollDimExpectation.agnostic
            elif unit_c == 1:
                dim_expectation = CollDimExpectation.subdiv
            else:
                assert unit_c == new_domain[i]
                dim_expectation = CollDimExpectation.intact
            new._dims[i] = replace(new._dims[i], dim_expectation=dim_expectation)

        return new

    def _make_derived(
        self,
        _iter: object,
        unit: CollUnit,
        lo: int,
        hi: int,
        env: Dict[CollParam, int],
        is_tiled: bool,
    ):
        assert 0 <= lo <= hi
        codegen = CollCodegen(_iter)  # Update later
        new = CollTiling(self._tree_depth + 1, codegen, list(self._dims))

        # Translate unit domain and tiling to concrete integers
        unit_original_domain = unit.int_domain(env)
        unit_original_box = unit.int_box(env)

        # Determine the common domain between us and the given unit,
        # and translate the box size of the unit and self CollTiling.
        self_domain = self.get_domain()
        unit_completion = DomainCompletionOp(
            unit_original_domain, self_domain, allow_partial_source=True
        )
        self_completion = DomainCompletionOp(
            self_domain, unit_original_domain, allow_partial_source=False
        )
        new_domain = tuple(unit_completion.domain)
        self_box = self_completion.new_size(self.get_box())
        unit_box = unit_completion.new_size(unit_original_box)
        assert unit_completion.domain == self_completion.domain

        # Update the new CollTiling in-place based on domain completion.
        assert not self_completion.remove_idx
        for dim_idx, split_factor in self_completion.idx_factors:
            new._apply_split_dim(dim_idx, split_factor)
        assert new.get_domain() == new_domain, f"{new.get_domain()} == {new_domain}"

        # Select the unique dimension on which to tile or specialize.
        # If this dimension is found, overwrite the default codegen object,
        # and add the CollDimOp representing the thread box change caused
        # by this tile/specialize.
        assert len(new_domain) == len(self_box) == len(unit_box)
        tiled_dim_idx = None
        for i in range(len(new_domain)):
            domain_c = new_domain[i]
            unit_c = unit_box[i]
            self_c = self_box[i]

            if unit_c == self_c or unit_c == domain_c or unit_c is None:
                continue
            if tiled_dim_idx is not None:
                self.err(
                    unit,
                    hi,
                    "ambiguous dimension to tile",
                    new_domain,
                    self_box,
                    unit_box,
                )

            mod_dim = new._dims[i]  # This CollDim in `new` will be replaced.

            tiled_dim_idx = i
            offset = lo * unit_c
            box = unit_c if is_tiled else (hi - lo) * unit_c
            tile_count = (hi - lo) if is_tiled else 1
            if is_tiled and hi - lo > 1:
                thread_pitch = mod_dim.dim_thread_pitch * unit_c
            else:
                thread_pitch = 0
            old_intra_box_expr = mod_dim.get_leaf_intra_box_expr()
            intra_box_expr = old_intra_box_expr - offset
            if tile_count > 1:
                intra_box_expr %= unit_c
            box_size_needed = hi * unit_c
            if box_size_needed > self_c:
                self.err(
                    unit,
                    hi,
                    f"not enough threads available; max_tile_count={self_c // unit_c}",
                    new_domain,
                    self_box,
                    unit_box,
                )

            codegen.dim_idx = tiled_dim_idx
            codegen.offset = offset
            codegen.box = box
            codegen.tile_count = tile_count
            codegen.thread_pitch = thread_pitch
            if is_tiled:
                codegen.codegen_expr = old_intra_box_expr // unit_c
                codegen.codegen_static_lo = None if lo == 0 else lo
                codegen.codegen_static_hi = None if self_c == box_size_needed else hi
            else:
                codegen.codegen_expr = old_intra_box_expr
                codegen.codegen_static_lo = None if lo == 0 else lo * unit_c
                codegen.codegen_static_hi = (
                    None if self_c == box_size_needed else hi * unit_c
                )

            op = CollDimOp(
                _iter,
                offset,
                box,
                tile_count,
                intra_box_expr,
                thread_pitch,
                new.get_tree_depth(),
            )
            new._dims[i] = replace(mod_dim, dim_ops=mod_dim.dim_ops + (op,))

        # Should we support hi = 0 as well in this case?
        if tiled_dim_idx is None and (lo != 0 or hi != 1):
            self.err(
                unit,
                hi,
                f'expect lo=0, hi=1 when CollUnit matches current one in scope ("trivial tiling")',
            )

        return new


def top_level_coll_tiling(
    domain: Tuple[int], intra_box_exprs: Tuple[CollIndexExpr]
) -> CollTiling:
    assert len(domain) == len(intra_box_exprs)
    dims = []
    for i, e in enumerate(intra_box_exprs):
        extent = domain[i]
        thread_pitch = prod(domain[i + 1 :])
        dims.append(CollDim(extent, thread_pitch, (), e))
    tiling = CollTiling(0, CollCodegen(None), dims)
    return tiling


class DomainCompletionError(CollTilingError):
    pass


@dataclass(slots=True)
class DomainCompletionOp:
    source_domain: Tuple[int]
    target_domain: Tuple[int]
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
        self.source_domain = source_domain
        self.target_domain = target_domain
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
            self.require_divides(threads_t, threads_s)
            self.source_partial = True
            source_domain = (s_to_t_multiplier,) + source_domain
        else:
            self.require_divides(threads_s, threads_t)
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

        if not allow_partial_source:
            assert (
                not self.remove_idx
            ), f"Unexpected 1 in complete domain {source_domain}"

        # Generate list of splitting commands (i, f) to apply in order
        # "split the (current) i-th dimension by f".
        idx_factors = []

        def thread_pitch_tuple(domain):
            tmp = [1]
            for c in domain[::-1]:
                tmp.append(tmp[-1] * c)
            tmp.reverse()
            return tmp

        thread_pitch_s = thread_pitch_tuple(source_domain)
        thread_pitch_t = thread_pitch_tuple(target_domain)

        for i_s in range(len(source_domain) - 1, -1, -1):
            s0 = thread_pitch_s[i_s] if i_s >= 0 else float("inf")
            s1 = thread_pitch_s[i_s + 1]
            for i_t in range(len(target_domain) - 1, -1, -1):
                t0 = thread_pitch_t[i_t]
                if s0 > t0 > s1:
                    t1 = thread_pitch_t[i_t + 1]
                    divisor = max(t1, s1)
                    split = t0 // divisor
                    if i_s >= 0:
                        self.require_divides(source_domain[i_s], split)
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

    def err(self, msg):
        raise DomainCompletionError(
            f"Domain completion for {self.source_domain} and {self.target_domain} failed: {msg}"
        )

    def require_divides(self, a, b):
        if a % b != 0:
            # This will not make much sense without context but better than nothing.
            # In general these errors are unlikely for the vast majority of programs.
            # So currently I don't invest in better messages.
            self.err(f"{b} doesn't divide {a}")

    def new_size(self, size: Tuple, partial_prepend=None):
        def outer_op(c, factor):
            if c is None:
                return None
            elif c < factor:
                return 1
            else:
                self.require_divides(c, factor)
                return c // factor

        def inner_op(c, factor):
            return None if c is None else min(c, factor)

        return self._new_coords(
            size, outer_op, inner_op, 1, partial_prepend=partial_prepend
        )

    def new_offset(self, offset: Tuple, partial_prepend=None):
        def outer_op(c, factor):
            return c // factor

        def inner_op(c, factor):
            self.require_divides(c, factor)
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
        return self._new_coords(coords, outer_op, inner_op, coll_index_0)

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
            c = coords[i]
            if c is None or c == expected_removed_coord:
                del coords[i]
            else:
                self.err(
                    f"expected coords[{i}]={expected_removed_coord} in coords={coords}"
                )

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

# Matches collective units that reside within one CTA
cuda_agnostic_sub_cta = CollUnit(
    (clusterDim, blockDim), (1, None), "cuda_agnostic_sub_cta", None
)

# Matches collective units that consist of any number of non-subdivided CTAs
cuda_agnostic_intact_cta = CollUnit(
    (clusterDim, blockDim), (None, blockDim), "cuda_agnostic_intact_cta", None
)

# Questionable, these may change later
cuda_warp_in_cluster = CollUnit(
    (clusterDim, blockDim), (1, 32), "cuda_warp_in_cluster", 0
)
cuda_cta_in_cluster = CollUnit(
    (clusterDim * blockDim,), (blockDim,), "cuda_cta_in_cluster", 0
)
cuda_cluster = CollUnit(
    (clusterDim * blockDim,), (clusterDim * blockDim,), "cuda_cluster", 0
)


def cuda_cta_in_cluster_strided(cta_stride):
    name = f"cuda_cta_in_cluster_strided({cta_stride})"
    return CollUnit(
        (clusterDim / cta_stride, cta_stride, blockDim), (1, 1, blockDim), name, 0
    )


def cuda_warp_in_cluster_strided(cta_stride):
    name = f"cuda_warp_in_cluster_strided({cta_stride})"
    return CollUnit(
        (clusterDim / cta_stride, cta_stride, blockDim), (1, 1, 32), name, 0
    )


def cuda_threads_strided(num_threads, thread_stride):
    name = f"cuda_threads_strided({num_threads}, {thread_stride})"
    return CollUnit(
        (blockDim / thread_stride, thread_stride), (1, num_threads), name, 0
    )


assert standalone_thread.is_always_single_thread()
assert cuda_thread.is_always_single_thread()
assert not cuda_warp.is_always_single_thread()
assert not cuda_cta_in_cluster.is_always_single_thread()
