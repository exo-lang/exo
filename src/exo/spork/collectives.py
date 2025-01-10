from __future__ import annotations

from typing import Optional

from .base_with_context import BaseWithContext


class CollectiveUnit(object):
    """Unit type of a grouping of 1 or more hardware lanes.

    Physical hardware resource used to execute a single iteration of a
    parallel for loop.

    """

    name: str
    parent: CollectiveUnit
    thread_count: Optional[int]  # None if runtime sized (e.g. CTA)

    def __init__(self, name, parent, thread_count):
        self.name = name
        self.parent = parent
        self.thread_count = thread_count

    def __repr__(self):
        return f"<exo.spork.collectives.CollectiveUnit {self.name}>"

    def __str__(self):
        return self.name

    def contains(self, child):
        """Evaluate nesting relationship

        e.g. cuda_block.contains(cuda_thread) -> True"""
        if self == child.parent:
            return True
        else:
            return child.parent is not None and self.contains(child.parent)


cpu_thread = CollectiveUnit("cpu_thread", None, 1)
cuda_cluster = CollectiveUnit("cuda_cluster", None, None)
cuda_block = CollectiveUnit("cuda_block", cuda_cluster, None)
cuda_warpgroup = CollectiveUnit("cuda_warpgroup", cuda_block, 128)
cuda_warp = CollectiveUnit("cuda_warp", cuda_warpgroup, 32)
cuda_thread = CollectiveUnit("cuda_thread", cuda_warp, 1)

collective_unit_dict = {
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


class SpecializeCollective(BaseWithContext):
    """Valid as the context of an Exo with statement.

    This instructs the backend compiler, when targetting multi-threaded
    semantics, to distribute the child for loops over only the named
    subset of resources (locally indexed within the parent collective lane).

    Example:

    for blk_id in cuda_blocks(0, x):
        with SpecializeCollective(cuda_thread, 0, 64):
            for y in cuda_threads(0, 16):
                for x in cuda_threads(0, 16):
                    # Code for threads [0, 63] of the block
        with SpecializeCollective(cuda_thread, 64, 128):
            for y in cuda_threads(0, 16):
                for x in cuda_threads(0, 16):
                    # Code for threads [64, 127] of the block

        for warp_id in cuda_warps(0, 4):
            with SpecializeCollective(cuda_thread, 0, 16):
                for y in cuda_threads(0, 4):
                    for x in cuda_threads(0, 4):
                        # Code for threads [0, 15] of each _warp_

    When interpreting exo code under single-threaded semantics, this
    should be treated as unconditionally true, since all child for
    loops are interpreted as serially executed by the one thread.

    NOTE: legacy syntax `if <unit> in (<lo>, <hi>)`
    may appear in old documents from 2024.

    """

    __slots__ = ["unit", "lo", "hi"]

    def __init__(self, unit: CollectiveUnit, lo: int, hi: int):
        assert isinstance(unit, CollectiveUnit)
        self.unit = unit
        self.lo = int(lo)
        self.hi = int(hi)
        if lo < 0:
            raise ValueError("SpecializeCollective.lo must be non-negative")
        if hi <= lo:
            raise ValueError("SpecializeCollective [lo, hi) must be non-empty")

    def __repr__(self):
        return f"SpecializeCollective({self.unit}, {self.lo}, {self.hi})"

    def __eq__(self, other):
        return (
            isinstance(other, SpecializeCollective)
            and self.unit == other.unit
            and self.lo == other.lo
            and self.hi == other.hi
        )
