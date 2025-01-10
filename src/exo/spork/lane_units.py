from __future__ import annotations

from typing import Optional

from .base_with_context import BaseWithContext


class LaneUnit(object):
    """Unit type of a "parallel lane" (parlane).

    Physical hardware resource used to execute a single iteration of a
    parallel for loop.

    """

    name: str
    parent: LaneUnit
    thread_count: Optional[int]  # None if runtime sized (e.g. CTA)

    def __init__(self, name, parent, thread_count):
        self.name = name
        self.parent = parent
        self.thread_count = thread_count

    def __repr__(self):
        return f"<exo.spork.lane_unit.LaneUnit {self.name}>"

    def __str__(self):
        return self.name

    def contains(self, child):
        """Evaluate nesting relationship

        e.g. cuda_block.contains(cuda_thread) -> True"""
        if self == child.parent:
            return True
        else:
            return child.parent is not None and self.contains(child.parent)


cpu_thread = LaneUnit("cpu_thread", None, 1)
cuda_cluster = LaneUnit("cuda_cluster", None, None)
cuda_block = LaneUnit("cuda_block", cuda_cluster, None)
cuda_warpgroup = LaneUnit("cuda_warpgroup", cuda_block, 128)
cuda_warp = LaneUnit("cuda_warp", cuda_warpgroup, 32)
cuda_thread = LaneUnit("cuda_thread", cuda_warp, 1)

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


class LaneSpecialization(BaseWithContext):
    """Valid as the context of an Exo with statement.

    This instructs the backend compiler, when targetting multi-threaded
    semantics, to distribute the child for loops over only the named
    subset of resources (locally indexed within the parent parlane).

    Example:

    for blk_id in cuda_blocks(0, x):
        with LaneSpecialization(cuda_thread, 0, 64):
            for y in cuda_threads(0, 16):
                for x in cuda_threads(0, 16):
                    # Code for threads [0, 63] of the block
        with LaneSpecialization(cuda_thread, 64, 128):
            for y in cuda_threads(0, 16):
                for x in cuda_threads(0, 16):
                    # Code for threads [64, 127] of the block

        for warp_id in cuda_warps(0, 4):
            with LaneSpecialization(cuda_thread, 0, 16):
                for y in cuda_threads(0, 4):
                    for x in cuda_threads(0, 4):
                        # Code for threads [0, 15] of each _warp_

    When interpreting exo code under single-threaded semantics, this
    should be treated as unconditionally true, since all child for
    loops are interpreted as serially executed by the one thread.

    NOTE: legacy syntax `if <lane_unit> in (<lo>, <hi>)`
    may appear in old documents from 2024.

    """

    __slots__ = ["unit", "lo", "hi"]

    def __init__(self, unit: LaneUnit, lo: int, hi: int):
        assert isinstance(unit, LaneUnit)
        self.unit = unit
        self.lo = int(lo)
        self.hi = int(hi)
        if lo < 0:
            raise ValueError("LaneSpecialization.lo must be non-negative")
        if hi <= lo:
            raise ValueError("LaneSpecialization [lo, hi) must be non-empty")

    def __repr__(self):
        return f"LaneSpecialization({self.unit}, {self.lo}, {self.hi})"

    def __eq__(self, other):
        return (
            isinstance(other, LaneSpecialization)
            and self.unit == other.unit
            and self.lo == other.lo
            and self.hi == other.hi
        )
