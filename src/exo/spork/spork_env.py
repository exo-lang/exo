from typing import List, Optional, Tuple

from ..core.prelude import Sym, SrcInfo, extclass

from .lane_units import LaneUnit, LaneSpecialization
from .loop_modes import LoopMode
from .actor_kinds import ActorKind, cpu

from ..core.LoopIR import LoopIR


class ParIndexRecord(object):
    __slots__ = ["exo_iter", "c_iter", "c_range", "static_range"]

    exo_iter: str  # Iteration variable name in Exo code
    c_iter: str  # Iteration variable name in C code
    # [c[0], c[1]) -- iteration bounds as compiled C code fragments
    c_range: Tuple[str, str]
    # [static_range[0], static_range[1]) -- statically analyzed maximal bounds
    static_range: Tuple[int, int]


# This is where most of the backend logic for CUDA should go.
class SporkEnv(object):
    kernel_name: str
    kernel_lines: List[str]
    _actor_kind_stack: List[ActorKind]

    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self.kernel_lines = []
        self._actor_kind_stack = []

    def __bool__(self):
        return True

    def get_actor_kind(self):
        assert self._actor_kind_stack
        return self._actor_kind_stack[-1]

    def push_actor_kind(self, actor_kind: ActorKind):
        if self._actor_kind_stack:
            assert actor_kind.allows_parent(self._actor_kind_stack[-1])

        self._actor_kind_stack.append(actor_kind)

    def pop_actor_kind(self):
        self._actor_kind_stack.pop()

    def push_parallel_for(self, s: LoopIR.For) -> bool:
        assert isinstance(s, LoopIR.For)
        assert s.loop_mode.is_par

        self.kernel_lines.append(f"// TODO parallel-for {s.loop_mode}")
        return True

    def pop_parallel_for(self):
        pass

    def push_lane_specialization(self, lane_specialization: LaneSpecialization) -> str:
        assert isinstance(lane_specialization, LaneSpecialization)
        return "true  /* TODO compile LaneSpecialization */"

    def pop_lane_specialization(self):
        pass

    def on_comp_s(self, s):
        pass
