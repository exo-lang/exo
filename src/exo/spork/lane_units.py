from typing import Optional


class LaneUnit(object):
    """Unit type of a "parallel lane" (parlane).

    Physical hardware resource used to execute a single iteration of a
    parallel for loop.

    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<exo.spork.lane_unit.LaneUnit {self.name}>"

    def __str__(self):
        return self.name


cpu_thread = LaneUnit("cpu_thread")
cuda_cluster = LaneUnit("cuda_cluster")
cuda_block = LaneUnit("cuda_block")
cuda_warpgroup = LaneUnit("cuda_warpgroup")
cuda_warp = LaneUnit("cuda_warp")
cuda_thread = LaneUnit("cuda_thread")

lane_unit_dict = {
    unit.name: unit
    for unit in [
        cpu_thread,
        cuda_cluster,
        cuda_block,
        cuda_warpgroup,
        cuda_warp,
        cuda_thread,
    ]
}


class LaneSpecialization(object):
    """Valid only as the condition of an if statement with no orelse

    This instructs the backend compiler, when targetting multi-threaded
    semantics, to distribute the child for loops over only the named
    subset of resources (locally indexed within the parent parlane).

    Example:

    for blk_id in cuda_blocks(0, x, warps = 4):  # x blocks of 4*32 threads
        if cuda_thread in (0, 64):
            for y in cuda_threads(0, 16):
                for x in cuda_threads(0, 16):
                    # Code for threads [0, 63] of the block
        if cuda_thread in (64, 128):
            for y in cuda_threads(0, 16):
                for x in cuda_threads(0, 16):
                    # Code for threads [64, 127] of the block

        for warp_id in cuda_warps(0, 4):
            if cuda_thread in (0, 16):
                for y in cuda_threads(0, 4):
                    for x in cuda_threads(0, 4):
                        # Code for threads [0, 15] of each _warp_

    When interpreting exo code under single-threaded semantics, this
    should be treated as unconditionally true, since all child for
    loops are interpreted as serially executed by the one thread.

    """

    __slots__ = ["unit", "lo", "hi"]

    def __init__(self, unit: LaneUnit, lo: int, hi: int):
        assert isinstance(unit, LaneUnit)
        self.unit = unit
        self.lo = int(lo)
        self.hi = int(hi)

    def __repr__(self):
        return f"exo.spork.lane_units.LaneSpecialization({self.unit}, {self.lo}, {self.hi})"

    def __str__(self):
        """Syntax of the lane specialization as it appears in Exo code.

        The backend compiler needs to implement valid C/cuda code generation itself."""
        return f"{self.unit} in ({self.lo}, {self.hi})"

    def __eq__(self, other):
        if isinstance(other, LaneSpecializationPattern):
            return other == self
        return self.unit == other.unit and self.lo == other.lo and self.hi == other.hi


class LaneSpecializationPattern(object):
    """Helper for pattern matching LaneSpecialization.

    Any None attributes are assumed holes"""

    __slots__ = ["unit", "lo", "hi"]

    def __init__(self, unit: Optional[LaneUnit], lo: Optional[int], hi: Optional[int]):
        assert unit is None or isinstance(unit, LaneUnit)
        self.unit = unit
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return f"exo.spork.lane_units.LaneSpecializationPattern({self.unit}, {self.lo}, {self.hi})"

    def strattr(self, name):
        value = getattr(self, name)
        return "_" if value is None else str(value)

    def __str__(self):
        return f"{self.strattr('unit')} in ({self.strattr('lo')}, {self.strattr('hi')})"

    def matchattr(self, other, name):
        x = getattr(self, name)
        y = getattr(other, name)
        return x is None or y is None or x == y

    def __eq__(self, other):
        """Pattern match. Non-transitive, so we are abusing the meaning of =="""
        if isinstance(other, LaneSpecialization) or isinstance(
            other, LaneSpecializationPattern
        ):
            return (
                self.matchattr(other, "unit")
                and self.matchattr(other, "lo")
                and self.matchattr(other, "hi")
            )
        else:
            return False
