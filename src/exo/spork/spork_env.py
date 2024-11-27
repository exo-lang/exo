from __future__ import annotations

from typing import List, Optional, Tuple

from ..core.prelude import Sym, SrcInfo, extclass

from .lane_units import LaneUnit, LaneSpecialization, cuda_cluster, cuda_block
from .loop_modes import LoopMode, CudaClusters, CudaBlocks
from .actor_kinds import ActorKind, cpu

from ..core.LoopIR import LoopIR


class ParallelForRecord(object):
    __slots__ = ["exo_iter", "c_iter", "c_range", "static_range"]

    exo_iter: str  # Iteration variable name in Exo code
    c_iter: str  # Iteration variable name in C code
    # [c_range[0], c_range[1]) -- iteration bounds as compiled C code fragments
    c_range: Tuple[str, str]
    # [static_range[0], static_range[1]) -- statically analyzed maximal bounds
    static_range: Tuple[int, int]

    def __init__(self, exo_iter, c_iter, c_range, static_range):
        self.exo_iter = exo_iter
        self.c_iter = c_iter
        self.c_range = c_range
        self.static_range = static_range

    def __repr__(self):
        return f"ParallelForRecord(exo_iter={self.exo_iter}, c_iter={self.c_iter}, c_range={self.c_range}, static_range={self.static_range})"


class ForRecord(object):
    __slots__ = ["actor_kind", "loop_mode"]

    actor_kind: ActorKind
    loop_mode: LoopMode

    def __init__(self, actor_kind, loop_mode):
        self.actor_kind = actor_kind
        self.loop_mode = loop_mode

    def __repr__(self):
        return f"ForRecord(actor_kind={self.actor_kind}, loop_mode={self.loop_mode})"


class ParscopeEnv(object):
    __slots__ = [
        "parent",
        "root_node",
        "relative_thread_range",
        "lane_unit",
        "loop_records",
        "partial",
    ]

    parent: Optional[ParscopeEnv]
    root_node: (LoopIR.For, LoopIR.If)
    relative_thread_range: Tuple[int, int]
    lane_unit: LaneUnit
    loop_records: List[ParallelForRecord]
    partial: bool

    def __init__(self, parent, root_node, relative_thread_range, lane_unit):
        assert parent is None or isinstance(parent, ParscopeEnv)
        assert isinstance(root_node, (LoopIR.For, LoopIR.If))

        self.parent = parent
        self.root_node = root_node
        self.relative_thread_range = relative_thread_range
        self.lane_unit = lane_unit
        self.loop_records = []
        self.partial = True

        if relative_thread_range[1] <= relative_thread_range[0]:
            raise ValueError(
                f"{root_node.srcinfo}: Invalid lane specialization (0 threads allocated)"
            )
        if parent is not None:
            parent_thread_count = (
                parent.relative_thread_range[1] - parent.relative_thread_range[0]
            )
            max_thread_index = relative_thread_range[1] - 1
            if max_thread_index >= parent_thread_count:
                raise ValueError(
                    f"{root_node.srcinfo}: Invalid lane specialization (maximum relative thread index {max_thread_index} requested overflows {parent_thread_count} threads available from parent {str(parent.lane_unit)} scope"
                )

    def static_shape(self):
        return [record.static_range for record in self.loop_records]

    def __repr__(self):
        return f"<ParscopeEnv {type(self.root_node)} {str(self.lane_unit)} threads:{self.relative_thread_range}>"


# This is where most of the backend logic for CUDA should go.
class SporkEnv(object):
    _base_kernel_name: str
    _future_lines: List
    _parallel_for_stack: Optional[ParallelForRecord]
    _for_stack: List[ForRecord]
    _parscope: ParscopeEnv
    _cluster_size: Optional[int]
    _blockDim: int

    __slots__ = {
        "_base_kernel_name": "Prefix of name of the CUDA kernel being generated",
        "_future_lines": "List of str lines, or objects that can be converted to str TODO",
        "_parallel_for_stack": "Info pushed for each parallel for",
        "_for_stack": "Info pushed for each parallel or temporal for loop",
        "_parscope": "Current Parscope information",
        "_cluster_size": "Cuda blocks per cuda cluster (None if feature unused)",
        "_blockDim": "Cuda threads per cuda block",
    }

    def __init__(self, base_kernel_name: str, for_stmt: LoopIR.For):
        self._base_kernel_name = base_kernel_name
        self._future_lines = []
        self._parallel_for_stack = []
        self._for_stack = []
        self._parscope = None

        assert isinstance(for_stmt, LoopIR.For)
        loop_mode = for_stmt.loop_mode
        assert loop_mode.is_par
        if isinstance(loop_mode, CudaClusters):
            self._cluster_size = loop_mode.blocks
            self._blockDim = loop_mode.blockDim
        elif isinstance(loop_mode, CudaBlocks):
            self._cluster_size = None
            if loop_mode.blockDim is None:
                raise ValueError(
                    f"{for_stmt.srcinfo}: CudaBlocks object of {for_stmt.iter} loop must have explicit blockDim"
                )
            self._blockDim = loop_mode.blockDim
        else:
            raise ValueError(
                f"{for_stmt:srcinfo}: CUDA kernel must be defined by a `for cuda_clusters` or `for cuda_blocks` loop"
            )

    def __bool__(self):
        return True

    def add_line(self, line: str):
        assert isinstance(line, str)
        self._future_lines.append(line)

    def get_actor_kind(self):
        assert self._for_stack
        return self._for_stack[-1].actor_kind

    def push_for(
        self,
        s: LoopIR.For,
        new_actor_kind: ActorKind,
        c_iter: str,
        c_range,
        sym_range,
        tabs,
    ) -> bool:
        emit_loop = True
        if s.loop_mode.is_par:
            static_range = (
                sym_range[0],
                None if sym_range[1] is None else sym_range[1] + 1,
            )
            parallel_for_record = ParallelForRecord(
                s.iter, c_iter, c_range, static_range
            )
            self._parallel_for_stack.append(parallel_for_record)
            self._push_parscope_update(s, parallel_for_record)
            emit_loop = True  # XXX
        self._for_stack.append(ForRecord(new_actor_kind, s.loop_mode))
        return emit_loop

    def pop_for(self, s: LoopIR.For):
        assert self._for_stack
        self._for_stack.pop()
        if s.loop_mode.is_par:
            assert self._parallel_for_stack
            self._parallel_for_stack.pop()
            self._pop_parscope_update(s)

    def push_lane_specialization(self, s: LoopIR.If) -> str:
        spec = s.cond.val
        assert isinstance(spec, LaneSpecialization)
        self._push_parscope_update(s)
        return "true  /* TODO compile LaneSpecialization */"

    def pop_lane_specialization(self, s: LoopIR.If):
        assert self._parscope.root_node is s
        self._parscope = self._parscope.parent

    def get_lines(self):
        return self._future_lines

    def _push_parscope_update(self, s, parallel_for_record=None):
        parent_parscope = self._parscope

        if isinstance(s, LoopIR.If):
            assert parallel_for_record is None
            spec = s.cond.val
            assert isinstance(spec, LaneSpecialization)
            new_lane_unit = spec.unit
            if new_lane_unit.thread_count is None:
                # Currently can't specialize blocks within clusters. Assume
                # lane specialization is implemented with threadIdx for now.
                raise ValueError(
                    "f{s.srcinfo}: Can't specialize lanes of unit type {str(new_lane_unit)}"
                )
            relative_thread_range = (
                spec.lo * new_lane_unit.thread_count,
                spec.hi * new_lane_unit.thread_count,
            )
        elif isinstance(s, LoopIR.For):
            assert isinstance(parallel_for_record, ParallelForRecord)
            loop_mode = s.loop_mode
            assert loop_mode.is_par
            new_lane_unit = loop_mode.lane_unit()
            if new_lane_unit == cuda_cluster:
                relative_thread_range = (0, self._cluster_size * self._blockDim)
            elif new_lane_unit == cuda_block:
                relative_thread_range = (0, self._blockDim)
            else:
                assert isinstance(new_lane_unit.thread_count, int)
                relative_thread_range = (0, new_lane_unit.thread_count)
        else:
            assert 0

        if parent_parscope is not None:
            parent_lane_unit = parent_parscope.lane_unit
            if not parent_lane_unit.contains(new_lane_unit):
                if isinstance(s, LoopIR.If):
                    raise ValueError(
                        f"{s.srcinfo}: Can't specialize {new_lane_unit} in {parent_lane_unit} scope"
                    )
                if isinstance(s, LoopIR.For):
                    if new_lane_unit == parent_lane_unit and parent_parscope.partial:
                        pass
                    else:
                        note = ""
                        if new_lane_unit == parent_lane_unit:
                            note = "(to define a multidimensional iteration space, this loop must be the only statement in the body of the parent loop) "
                        parent_node = parent_parscope.root_node
                        if isinstance(parent_node, LoopIR.If):
                            defined_by = f"lane specialization @ {parent_node.srcinfo}"
                        else:
                            defined_by = (
                                f"{parent_node.iter} loop @ {parent_node.srcinfo}"
                            )
                        raise ValueError(
                            f"{s.srcinfo}: {note}Can't nest {new_lane_unit} loop ({s.iter}) in {parent_lane_unit} scope (defined by {defined_by})"
                        )

        if parent_parscope is None or not parent_parscope.partial:
            self._parscope = ParscopeEnv(
                parent_parscope, s, relative_thread_range, new_lane_unit
            )

        if parallel_for_record is not None:
            self._parscope.loop_records.append(parallel_for_record)
            # Detect multidimensional iteration space for parscope
            self._parscope.partial = False
            if len(s.body) == 1:
                body_stmt = s.body[0]
                if isinstance(body_stmt, LoopIR.For):
                    loop_mode = body_stmt.loop_mode
                    self._parscope.partial = (
                        loop_mode.is_par and loop_mode.lane_unit() == new_lane_unit
                    )

    def _pop_parscope_update(self, s):
        assert isinstance(self._parscope, ParscopeEnv)

        if isinstance(s, LoopIR.If):
            assert self._parscope.root_node is s
            self._parscope = self._parscope.parent
        elif isinstance(s, LoopIR.For):
            assert s.loop_mode.is_par
            loop_record = self._parscope.loop_records.pop()
            assert loop_record.exo_iter == s.iter
            self._parscope.partial = True
            if self._parscope.root_node is s:
                self._parscope = self._parscope.parent
        else:
            assert 0
