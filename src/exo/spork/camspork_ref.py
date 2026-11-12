"""PROTOTYPE Python reference for camspork sync env traces.

Plan: endgame_plan/plan_camspork_test.md (sketch: plan_sketch_camspork_test.md).
Implements the amspork3 synchronization environment (spork/docs/amspork3/SyncEnvConcat.tex)
directly on sets, plus the camspork deviations/optimizations we compare against, each
marked DEVIATION or OPT. Thread IDs are hand-computed by the test author (like excut
reference generators), so the Exo -> camspork lowering of parallel loops is checked
independently.

Task IDs: a (possibly multi-dimensional) cuda_tasks loop nest is linearized in
lexicographic order, as camspork's lazy task_index does (every device task bumps it
once, through at least its task-end JoinThreads).
Managed ring buffers and mbarriers follow amspork3 SmemAppendix.tex (allocTrDetail /
freeTrDetail); freeTrDetail is only observable through pass/fail of the free check.
Atomics follow the spec (Tid x Q_atom, masked; needs exo 7c82d4a0's QtlMask).
The trace only samples sync statements, so a test must never leave an access without
a later sync (otherwise the access is invisible).


Names
-----

    x          StrVarName     str(Sym) of a proc variable: what probes, camspork's trace,
                              and error remarks use.
    z*         ElemIdx        index tuple (() for scalars).
    x[z*]      StrElemName    StrVarName + ElemIdx. Probes, and barriers in traced pending awaits.
    x#n        AllocVarName   the n-th allocation of x in the reference (counted per x).
                              Distinguishes allocations in time (each loop iteration's
                              allocation is a new one, like amspork3's alpha-renaming).
                              Allocations of different Syms with the same str(Sym) can't
                              be told apart by camspork's trace (see sync_check.py).
    x#n[z*]    AllocElemName  AllocVarName + ElemIdx. Keys the reference's sync env and
                              barrier state; projected to StrElemName in traces.


Typical test (tests/cuda/test_camspork_trace.py)
------------------------------------------------

    x, g = StrVarName("x"), StrVarName("g")

    def mkproc_foo():                     # Exo proc under test
        @proc
        def proc_foo(g: f32[2, 32] @ CudaGmemLinear):
            with CudaDeviceFunction(blockDim=64):
                for task in cuda_tasks(0, 1):
                    ...
        return proc_foo

    def mkref_foo():                      # hand-written reference
        ref = SyncEnvRef([x[0, 0], g[1]], threads_per_task=64)  # probes
        task = ref.task_threads()

        def task_body():
            ref.alloc(x, (2, 32))
            ref.mutate(x[w, t], [w * 32 + t],
                       quals=access_quals(CudaSmemLinear, cuda_in_order_instr, convergent=True))
            ref.fence(task, L1_bits, L2_bits)
            ...
            ref.free(x, check_tids=task, precondition_bits=...)  # SMEM free check

        ref.cuda_device_function(lambda: ref.device_task(0, task_body))
        return ref

    def test_foo(compiler, tmp_path):
        errors = check_trace(mkproc_foo(), mkref_foo(), tmp_path, **sizes)
        assert not errors, "\\n".join(errors)

    def test_bar_negative(compiler):      # both camspork and the reference must reject;
        check_both_fail(mkproc_bar(), mkref_bar)   # the trace is not inspected

* Probes are StrElemNames; the StrVarName must name exactly one Sym in the proc.
  Only sync-checked variables have records: written, non-sync-exempt memory.
  Barrier variables additionally report arrive/await counts.
* The reference must issue the same sync statements, in the same order, as camspork
  executes them. That includes the implicit ones: cuda_device_function() emits the
  C-CudaDeviceFunction Fences plus the device-end JoinThreads, and device_task() emits
  the task-end JoinThreads.
* Access qual-tls come from Exo's qual-tl tables via access_quals(mem, instr_tl); those
  tables are configuration, not under test. Convergence follows sync_check.py:
  Assign/Reduce always convergent; expression reads convergent iff one thread executes
  them; instr args use the initial qual-tl's default (out-of-order ones also take tag=TAG).
* Windows: access(..., extent=...) applies the access to each element independently.
* Barriers: alloc_barrier/free_barrier, arrive/await_ by StrElemName (resolved to the
  current allocation).
* mbarriers: alloc_mbarrier (always a managed ring buffer guarding itself). Unlike other
  barriers, an mbarrier array has its own sync env: arrive/await_ first do a read access
  of it (cuda_mbarrier_qual; Arrive: non-convergent, every multicast target; Await:
  convergent, home barrier only), as sync_check.py lowers them.
* SMEM data ring buffers: alloc(..., ring=RingGuard(...)) / free(var).
* Sync statements record trace entries with src = the calling Python line; helpers
  pass depth= so entries point at the mkref line that called them.


Trace files (in tmp_path; same JSONL schema on both sides, excut-style round trip)
--------------------------------------------------------------------------------

camspork_trace.jsonl (camspork exec.cpp trace_sync) and reference_trace.jsonl
(SyncEnvRef.write_trace) are both read back by parse_trace into TraceEntry objects and
compared by compare_traces. camspork_trace.txt / reference_trace.txt are readable dumps
of the parsed traces (same printer, so they diff line by line; only "@ src" differs when
the traces match).

One line per sync statement phase, a TraceEntryJson:

    TraceEntryJson = {
        "phase": "before" | "after",                             TracePhase
        "stmt": "Fence" | "Arrive" | "Await" | "JoinThreads",    SyncStmtKind
        "stmt_id": int | null,     camspork StmtRef raw data; null in the reference
        "src": str | null,         null in camspork (parse_trace maps stmt_id to the Exo
                                   source line); "file:line" of the mkref call in the reference
        "threads": [[int, int], ...],   Tids of the executing threads, each [lo, hi)
        "cuboid": object | null,   camspork ThreadCuboid (informational; null in the reference)
        "probes": [TraceProbeJson, ...]  one per probe, in probe order
    }

    TraceProbeJson = {
        "var": str,                StrVarName
        "idx": [int, ...],         ElemIdx
        "mutate": [VisRecordJson, ...] | null,   AccessKind.MUTATE records; null if the
        "read": [VisRecordJson, ...] | null,     variable is not allocated, idx is out of
                                                 range, or it is a scalar with no records
        "arrive_count": int,       optional, both or neither: present only for a live
        "await_count": int         barrier element (BarrierCounts)
    }

    VisRecordJson = {
        "ob": [[int, int, int], ...],   observers [tid_lo, tid_hi, qual_bits], tid_hi exclusive
        "aw": [[str, int], ...],        PendingAwaitRays [StrElemName as "x[i, j]" ("x" if
                                        scalar), arrive_count]
        "mask": int,                    qual_tl_mask bits
        "flags": int                    camspork vis_record_before_alloc_flag; ignored
                                        (it only names the use_before_alloc hazard)
    }

Comparison: per entry, same (phase, stmt) and thread set; per probe and AccessKind, the
SET of TraceVisRecords must match (order and duplicates ignored; observers merged per
qual-tl; pending awaits reduced to the minimum arrive count per barrier), plus barrier
counts. Traced barriers are StrElemNames, so two allocations of the same name are
indistinguishable in the trace (identity is only checked through its effects).


Representation (in memory)
--------------------------

  Tids       sorted, merged tuple of [lo, hi) intervals. tid is camspork's linearized
             task_index * threads_per_task + local tid, task_index being the lexicographic
             position of the task in the cuda_tasks loop nest (task_shape). Local tids
             are numbered by CTA rank in the cluster, then thread index in the CTA.
             All threads (and camspork's "CPU") are [0, CPU_HI).
  Ob         frozenset of QualObservers (observers Tids x {q}), at most one per qual-tl q.
  Aw         frozenset of PendingAwaitRay, at most one per barrier; each stands for the
             infinite set {PendingAwait(bar, z) | z >= arrive_count}.
  envVis     VisRecord(ob, aw, mask); TraceVisRecord is the traced form.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from ..core.prelude import get_srcinfo

Tids = Tuple[Tuple[int, int], ...]
ElemIdx = Tuple[int, ...]

CPU_HI = 2**32 - 1  # camspork models the CPU as threads [0, UINT32_MAX)


# *** Names ***


def as_elem_idx(idx: Union[int, Iterable[int]]) -> ElemIdx:
    return (idx,) if isinstance(idx, int) else tuple(idx)


@dataclass(slots=True, frozen=True)
class StrVarName:
    """x: str(Sym) of a proc variable. x[i, j] gives a StrElemName (x[()] for scalars)."""

    name: str

    def __getitem__(self, idx) -> StrElemName:
        return StrElemName(self, as_elem_idx(idx))

    def __str__(self):
        return self.name


@dataclass(slots=True, frozen=True)
class StrElemName:
    """x[z*]"""

    var: StrVarName
    idx: ElemIdx

    def __str__(self):
        # Matches camspork exec.cpp print_idx_helper (no brackets for scalars).
        return f"{self.var}{fmt_elem_idx(self.idx)}"

    @staticmethod
    def parse(s: str) -> StrElemName:
        """Inverse of str(): "x[0, 1]" or "x"."""
        m = re.fullmatch(r"([^\[\]]+)(?:\[([0-9, ]*)\])?", s)
        assert m, f"not an element name: {s!r}"
        idx = tuple(int(i) for i in m.group(2).split(",")) if m.group(2) else ()
        return StrElemName(StrVarName(m.group(1)), idx)


@dataclass(slots=True, frozen=True)
class AllocVarName:
    """x#n: the n-th allocation of x (counted per x)."""

    var: StrVarName
    n: int

    def __getitem__(self, idx) -> AllocElemName:
        return AllocElemName(self, as_elem_idx(idx))

    def __str__(self):
        return f"{self.var}#{self.n}"


@dataclass(slots=True, frozen=True)
class AllocElemName:
    """x#n[z*]"""

    alloc_var: AllocVarName
    idx: ElemIdx

    def str_elem(self) -> StrElemName:
        """Projection to the traced name (drops #n)."""
        return StrElemName(self.alloc_var.var, self.idx)

    def __str__(self):
        return f"{self.alloc_var}{fmt_elem_idx(self.idx)}"


def fmt_elem_idx(idx: ElemIdx) -> str:
    return f"[{', '.join(str(i) for i in idx)}]" if idx else ""


def elem_idxs(shape: Tuple[int, ...]) -> Iterable[ElemIdx]:
    return itertools.product(*(range(n) for n in shape))


# *** Enums ***


class AccessKind(Enum):
    """Values are the TraceProbeJson keys."""

    MUTATE = "mutate"
    READ = "read"


class SyncStmtKind(Enum):
    """Values are TraceEntryJson "stmt"."""

    FENCE = "Fence"
    ARRIVE = "Arrive"
    AWAIT = "Await"
    JOIN_THREADS = "JoinThreads"


class TracePhase(Enum):
    """Values are TraceEntryJson "phase"."""

    BEFORE = "before"
    AFTER = "after"


# *** Thread sets ***


def bits_to_indices(bits: int) -> FrozenSet[int]:
    return frozenset(i for i in range(bits.bit_length()) if bits >> i & 1)


def tids_norm(intervals: Iterable[Tuple[int, int]]) -> Tids:
    out = []
    for lo, hi in sorted(iv for iv in intervals if iv[0] < iv[1]):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return tuple(out)


def tids_of(ints: Iterable[int]) -> Tids:
    return tids_norm((t, t + 1) for t in ints)


def as_tids(x) -> Tids:
    if isinstance(x, tuple) and all(isinstance(iv, tuple) for iv in x):
        return tids_norm(x)
    if isinstance(x, range) and x.step == 1:
        return tids_norm([(x.start, x.stop)])
    return tids_of(x)


def tids_intersects(a: Tids, b: Tids) -> bool:
    return any(lo1 < hi2 and lo2 < hi1 for lo1, hi1 in a for lo2, hi2 in b)


def tids_split_per_thread(t: Tids) -> List[Tids]:
    """Per-thread singletons (non-convergent case); only for finite GPU tid sets."""
    assert sum(hi - lo for lo, hi in t) < 1 << 20, "too many threads to split"
    return [((x, x + 1),) for lo, hi in t for x in range(lo, hi)]


# *** Observers and pending awaits ***


@dataclass(slots=True, frozen=True)
class QualObservers:
    """Observers tids x {qual-tl number qual_index}."""

    qual_index: int
    tids: Tids


Ob = FrozenSet[QualObservers]


def obs(tids, qual_bits: int) -> Ob:
    """Observer set tids x qual-tl bits"""
    t = as_tids(tids)
    if not t:
        return frozenset()
    return frozenset(QualObservers(qi, t) for qi in bits_to_indices(qual_bits))


def ob_union(a: Ob, b: Ob) -> Ob:
    d: Dict[int, Tids] = {o.qual_index: o.tids for o in a}
    for o in b:
        d[o.qual_index] = tids_norm(d.get(o.qual_index, ()) + o.tids)
    return frozenset(QualObservers(qi, t) for qi, t in d.items())


def ob_intersects(a: Ob, b: Ob) -> bool:
    d = {o.qual_index: o.tids for o in a}
    return any(
        o.qual_index in d and tids_intersects(d[o.qual_index], o.tids) for o in b
    )


def ob_masked(ob: Ob, mask: int) -> Ob:
    m = bits_to_indices(mask)
    return frozenset(o for o in ob if o.qual_index in m)


@dataclass(slots=True, frozen=True)
class PendingAwaitRay:
    """{PendingAwait(bar, z) | z >= arrive_count} (camspork PendingAwaitRay)."""

    bar: AllocElemName
    arrive_count: int


@dataclass(slots=True, frozen=True)
class TracePendingAwaitRay:
    """PendingAwaitRay as traced: the barrier is only known by its StrElemName."""

    bar: StrElemName
    arrive_count: int


Aw = FrozenSet[PendingAwaitRay]
TraceAw = FrozenSet[TracePendingAwaitRay]


def aw_union(a: FrozenSet, b: FrozenSet) -> FrozenSet:
    """Union of the infinite sets: keep the min arrive count per barrier.
    Works on Aw and TraceAw (same field names)."""
    d = {r.bar: r for r in a}
    for r in b:
        if r.bar not in d or r.arrive_count < d[r.bar].arrive_count:
            d[r.bar] = r
    return frozenset(d.values())


# *** Visibility records ***


@dataclass(slots=True, frozen=True)
class VisRecord:
    """envVis in the reference's sync env."""

    ob: Ob
    aw: Aw
    mask: int

    def mask_ob(self, ob: Ob) -> Ob:
        return ob_masked(ob, self.mask)

    def syncs_with(self, ob_in: Ob, aw_in: Iterable[PendingAwaitRay] = ()) -> bool:
        """syncsWith (SyncEnvConcat.tex); each aw_in ray is a concrete
        PendingAwait(bar, arrive_count)."""
        if ob_intersects(self.mask_ob(self.ob), ob_in):
            return True
        mins = {r.bar: r.arrive_count for r in self.aw}
        return any(r.bar in mins and mins[r.bar] <= r.arrive_count for r in aw_in)

    def traced(self) -> TraceVisRecord:
        return TraceVisRecord(
            self.ob,
            frozenset(
                TracePendingAwaitRay(r.bar.str_elem(), r.arrive_count) for r in self.aw
            ),
            self.mask,
        )


@dataclass(slots=True, frozen=True)
class TraceVisRecord:
    """A VisRecord as traced (either side), in canonical form."""

    ob: Ob
    aw: TraceAw
    mask: int

    def to_json(self) -> dict:
        """VisRecordJson"""
        return {
            "ob": [
                [lo, hi, 1 << o.qual_index]
                for o in sorted(self.ob, key=lambda o: o.qual_index)
                for lo, hi in o.tids
            ],
            "aw": [[str(r.bar), r.arrive_count] for r in self.aw],
            "mask": self.mask,
            "flags": 0,
        }

    @staticmethod
    def from_json(j: dict) -> TraceVisRecord:
        """VisRecordJson -> canonical form (flags ignored)."""
        ob = frozenset()
        for lo, hi, bits in j["ob"]:
            ob = ob_union(ob, obs(((lo, hi),), bits))
        aw = aw_union(
            frozenset(),
            frozenset(
                TracePendingAwaitRay(StrElemName.parse(bar), z) for bar, z in j["aw"]
            ),
        )
        return TraceVisRecord(ob, aw, j["mask"])


@dataclass(slots=True)
class ElemSyncEnv:
    """Sync env of one array element: its VisRecords of each AccessKind."""

    records: Dict[AccessKind, List[VisRecord]] = field(
        default_factory=lambda: {kind: [] for kind in AccessKind}
    )

    def checked_by(self, kind: AccessKind) -> List[VisRecord]:
        """mutReqs: a mutate checks prior mutates and reads; a read checks prior mutates."""
        if kind == AccessKind.MUTATE:
            return self.all_records()
        return self.records[AccessKind.MUTATE]

    def all_records(self) -> List[VisRecord]:
        return self.records[AccessKind.MUTATE] + self.records[AccessKind.READ]


@dataclass(slots=True, frozen=True)
class AccessQuals:
    """Qual-tl arguments of an access (TrPreAccess / TrPostAccess)."""

    initial_bit: int
    precondition_bits: int
    mask: int
    convergent: bool


class SyncEnvRefError(Exception):
    """The reference rejects the program (expected to coincide with a camspork error)."""


# *** Barrier counting ***


@dataclass(slots=True, frozen=True)
class BarrierCounts:
    arrive_count: int
    await_count: int


@dataclass(slots=True)
class BarrierState:
    expected: int
    per_thread: bool
    one_shot_await: bool
    arrive_count: int
    await_count: int = 0


class Barriers:
    """ALL barrier counting lives here: arrive/await counts, which PendingAwaits an
    Arrive produces, which PendingAwait an Await consumes, and the expected-count checks.
    amspork3 leaves the count increment to barrier-mechanism-specific de-sugaring;
    this follows camspork (HamsterBarrierState), see plan_multicast_barrier_counts.md:
    * Each target barrier's own count is read and incremented (no home barrier).
    * per_thread (mbarrier): an Arrive adds its thread count, else 1.
    * expected != 0: Await requires count == expected (forward progress),
      count must not exceed it, and it must be reached by the end of the lifetime.
    Barriers are identified per allocation (AllocElemName).
    """

    def __init__(self):
        self.states: Dict[AllocElemName, BarrierState] = {}

    def alloc(
        self,
        bar: AllocElemName,
        *,
        expected=0,
        per_thread=False,
        one_shot_await=False,
        pre_arrived=False,
    ):
        """pre_arrived: start with arrive count = expected (no PendingAwaits)."""
        assert expected or not pre_arrived
        assert bar not in self.states, bar
        arrive_count = expected if pre_arrived else 0
        self.states[bar] = BarrierState(
            expected, per_thread, one_shot_await, arrive_count
        )

    def free(self, bar: AllocElemName):
        s = self.states.pop(bar)
        if s.expected and s.arrive_count != s.expected:
            raise SyncEnvRefError(
                f"Arrive count {s.arrive_count} != expected {s.expected} at free of {bar}"
            )

    def expected(self, bar: AllocElemName) -> int:
        return self.states[bar].expected

    def counts(self, bar: AllocElemName) -> Optional[BarrierCounts]:
        s = self.states.get(bar)
        return None if s is None else BarrierCounts(s.arrive_count, s.await_count)

    def arrive(self, targets: List[AllocElemName], n_threads: int) -> Aw:
        """PendingAwait(b, z) for z >= b's own count before arriving, for each target b."""
        aw = []
        for bar in dict.fromkeys(targets):
            s = self.states[bar]
            z = s.arrive_count
            s.arrive_count = z + (n_threads if s.per_thread else 1)
            if s.expected and s.arrive_count > s.expected:
                raise SyncEnvRefError(f"Too many arrives on {bar}")
            aw.append(PendingAwaitRay(bar, z))
        return frozenset(aw)

    def await_(self, bar: AllocElemName, N: int) -> PendingAwaitRay:
        """Tr-Await's inbound PendingAwait, with z_await = arrive_count - lag - 1."""
        s = self.states[bar]
        s.await_count += 1
        if s.one_shot_await and s.await_count != 1:
            raise SyncEnvRefError(f"Awaited twice on one-shot await barrier {bar}")
        if s.expected and s.arrive_count != s.expected:
            raise SyncEnvRefError(
                f"No forward progress: Await on {bar} with arrive count "
                f"{s.arrive_count} of expected {s.expected}"
            )
        return PendingAwaitRay(bar, s.arrive_count - N - 1)


# *** Managed ring buffers ***


@dataclass(slots=True, frozen=True)
class RingGuard:
    """Managed ring buffer of SMEM data guarded by mbarrier array `guard`
    (.ring_guarded_by(guard)), with ring_depth on dimension ring_dim of the data.
    guard_multicast: the multicast pattern of the Arrives on the guard (hand-computed,
    like thread IDs), in the same format as LoopIR SyncStmt.multicasts(): one
    BarrierExpr.multicast_flags() tuple per barrier expression (is each index an interval).
    E.g. `>> war[m, n, k] >> war[m, :, k] >> war[:, n, k]` is
    ((False, False, False), (False, True, False), (True, False, False)).
    guards(x[z]) is the home guard plus, per expression, all guards that differ from it
    only along that expression's multicast dimensions (a union, not a product)."""

    guard: StrVarName
    ring_dim: int
    ring_depth: int
    guard_multicast: Tuple[Tuple[bool, ...], ...] = ()


@dataclass(slots=True, frozen=True)
class AllocRing:
    """A RingGuard with the guard resolved to its allocation at the data's allocation."""

    spec: RingGuard
    guard: AllocVarName


# *** Reference sync env ***


class SyncEnvRef:
    """Reference sync env. Sync statements record before/after TraceEntries for
    `probes`; src is the Python line of the call (depth= to skip helpers).
    """

    all_threads: Tids = ((0, CPU_HI),)  # amspork3 Tid; camspork's "CPU" cuboid

    def __init__(
        self,
        probes: List[StrElemName],
        threads_per_task: int,
        task_shape: Tuple[int, ...] = (1,),
    ):
        """task_shape: iteration counts of the cuda_tasks loop nest, outermost first.
        Methods taking `task` accept an index tuple into task_shape, or an int for a 1-D nest.
        """
        assert all(isinstance(p, StrElemName) for p in probes)
        self.probes = list(probes)
        self.threads_per_task = threads_per_task
        self.task_shape = tuple(task_shape)
        self.live: Dict[StrVarName, AllocVarName] = {}
        self.n_allocs: Dict[StrVarName, int] = {}
        self.shapes: Dict[AllocVarName, Tuple[int, ...]] = {}
        self.env: Dict[AllocElemName, ElemSyncEnv] = {}
        self.rings: Dict[AllocVarName, AllocRing] = {}  # incl. mbarriers (self-guarded)
        self.mbarriers: Set[AllocVarName] = set()
        self.barriers = Barriers()
        self.trace: List[TraceEntry] = []

    def task_linear_index(self, task) -> int:
        """Task index tuple (or int, for a 1-D nest) -> camspork task_index (lexicographic)."""
        task_idx = as_elem_idx(task)
        assert len(task_idx) == len(self.task_shape), (task_idx, self.task_shape)
        linear = 0
        for i, n in zip(task_idx, self.task_shape):
            assert 0 <= i < n, (task_idx, self.task_shape)
            linear = linear * n + i
        return linear

    def task_idxs(self) -> List[Tuple[int, ...]]:
        """All task index tuples, in camspork's (lexicographic) execution order."""
        return list(elem_idxs(self.task_shape))

    def tids(self, lo: int, hi: int, task=0) -> Tids:
        base = self.task_linear_index(task) * self.threads_per_task
        return ((base + lo, base + hi),)

    def task_threads(self, task=0) -> Tids:
        return self.tids(0, self.threads_per_task, task)

    # *** Names ***

    def _new_alloc_var(self, var: StrVarName, shape: Tuple[int, ...]) -> AllocVarName:
        assert var not in self.live, f"{var} already allocated"
        n = self.n_allocs.get(var, 0)
        self.n_allocs[var] = n + 1
        alloc_var = AllocVarName(var, n)
        self.live[var] = alloc_var
        self.shapes[alloc_var] = tuple(shape)
        return alloc_var

    def alloc_elem(self, elem: StrElemName) -> AllocElemName:
        """The current allocation of elem's variable."""
        return self.live[elem.var][elem.idx]

    # *** Allocation ***

    def alloc(
        self,
        var: StrVarName,
        shape: Tuple[int, ...],
        *,
        ring: Optional[RingGuard] = None,
        mask=0,
    ) -> AllocVarName:
        """D-Alloc. With ring: allocTrDetail for SMEM data guarded by mbarriers, i.e. a
        mutate record (no observers, QtlMask `mask`) with PendingAwait(g, z) for
        z >= z_eac - 1 for EACH guard g in guards(x[z]) (Awaiting any one suffices)."""
        alloc_var = self._new_alloc_var(var, shape)
        if ring is not None:
            assert mask, "ring buffer alloc record needs the data's QtlMask"
            self.rings[alloc_var] = AllocRing(ring, self.live[ring.guard])
        for data_idx in elem_idxs(shape):
            elem_env = ElemSyncEnv()
            if ring is not None:
                aw = self._gawait_aw(self.rings[alloc_var], data_idx)
                elem_env.records[AccessKind.MUTATE].append(
                    VisRecord(frozenset(), aw, mask)
                )
            self.env[alloc_var[data_idx]] = elem_env
        return alloc_var

    def free(self, var: StrVarName, *, check_tids=None, precondition_bits=0):
        """D-Free. With check_tids: freeTrDetail for (non-ring) SMEM, i.e.
        TrPreAccess(mutate, x, z, {tid}, Q_pre, {}) for every element and tid.
        Ring buffers: freeTrDetail with gNext (see _ring_free_check)."""
        alloc_var = self.live[var]
        if check_tids is not None:
            assert alloc_var not in self.rings
            for idx in elem_idxs(self.shapes[alloc_var]):
                self._check_prior_records(
                    AccessKind.MUTATE,
                    alloc_var[idx],
                    tids_split_per_thread(as_tids(check_tids)),
                    precondition_bits,
                )
        if alloc_var in self.rings:
            self._ring_free_check(alloc_var)
        self._free_alloc_var(var)

    def alloc_barrier(
        self,
        var: StrVarName,
        shape: Tuple[int, ...],
        *,
        ring_dim=None,
        pre_arrive=0,
        **flags,
    ) -> AllocVarName:
        """Barrier variable without its own sync env (commit groups, cluster sync):
        Barriers state only. mbarriers use alloc_mbarrier.
        Pre-arrive: elements with idx[ring_dim] < pre_arrive start at the expected count.
        """
        assert not flags.get("per_thread"), "mbarrier: use alloc_mbarrier"
        alloc_var = self._new_alloc_var(var, shape)
        self._alloc_barrier_state(alloc_var, ring_dim, pre_arrive, flags)
        return alloc_var

    def alloc_mbarrier(
        self,
        var: StrVarName,
        shape: Tuple[int, ...],
        *,
        expected: int,
        ring_dim: int,
        ring_depth: int,
        pre_arrive: int = 0,
    ) -> AllocVarName:
        """mbarrier array, a managed ring buffer guarding itself (SmemAppendix "mbarrier").
        allocTrDetail: element x_b[z_b] gets a mutate record (no observers, QtlMask =
        {cuda_mbarrier_qual}) with PendingAwait(x_b[z_b - off], z) for z >= z_eac - 1,
        omitted when z_b - off is out of range (first phase of each physical mbarrier).
        """
        assert 0 <= pre_arrive <= ring_depth
        alloc_var = self._new_alloc_var(var, shape)
        self._alloc_barrier_state(
            alloc_var, ring_dim, pre_arrive, dict(expected=expected, per_thread=True)
        )
        self.mbarriers.add(alloc_var)
        self.rings[alloc_var] = AllocRing(
            RingGuard(var, ring_dim, ring_depth), alloc_var
        )
        mb = mbarrier_qual_bit()
        for idx in elem_idxs(shape):
            elem_env = ElemSyncEnv()
            prev_phase_idx = ring_shifted_idx(idx, ring_dim, -ring_depth)
            if prev_phase_idx[ring_dim] >= 0:
                aw = frozenset(
                    [self._pending_await_eac_minus_1(alloc_var[prev_phase_idx])]
                )
                elem_env.records[AccessKind.MUTATE].append(
                    VisRecord(frozenset(), aw, mb)
                )
            self.env[alloc_var[idx]] = elem_env
        return alloc_var

    def _alloc_barrier_state(
        self, alloc_var: AllocVarName, ring_dim, pre_arrive, flags
    ):
        for idx in elem_idxs(self.shapes[alloc_var]):
            pre_arrived = bool(pre_arrive) and idx[ring_dim] < pre_arrive
            self.barriers.alloc(alloc_var[idx], pre_arrived=pre_arrived, **flags)

    def free_barrier(self, var: StrVarName):
        """Ring free check first (sync_check.py order), then the arrive count checks."""
        alloc_var = self.live[var]
        if alloc_var in self.rings:
            self._ring_free_check(alloc_var)
        for idx in elem_idxs(self.shapes[alloc_var]):
            self.barriers.free(alloc_var[idx])
        self._free_alloc_var(var)

    def _free_alloc_var(self, var: StrVarName):
        alloc_var = self.live.pop(var)
        for idx in elem_idxs(self.shapes.pop(alloc_var)):
            self.env.pop(alloc_var[idx], None)
        self.rings.pop(alloc_var, None)
        self.mbarriers.discard(alloc_var)

    # *** Managed ring buffers (SmemAppendix.tex) ***

    def _pending_await_eac_minus_1(self, guard: AllocElemName) -> PendingAwaitRay:
        E = self.barriers.expected(guard)
        assert E >= 1, "guard mbarrier needs an expected arrive count"
        return PendingAwaitRay(guard, E - 1)

    def _multicast_guards(
        self, ring: AllocRing, home_guard_idx: ElemIdx
    ) -> List[AllocElemName]:
        """guards(): the home guard expanded by the multicast reverse mapping."""
        guard_shape = self.shapes[ring.guard]
        guard_idxs = {tuple(home_guard_idx)}
        for flags in ring.spec.guard_multicast:
            assert len(flags) == len(guard_shape)
            guard_idxs.update(
                itertools.product(
                    *(
                        range(n) if multicast else (i,)
                        for i, n, multicast in zip(home_guard_idx, guard_shape, flags)
                    )
                )
            )
        return [ring.guard[idx] for idx in sorted(guard_idxs)]

    def _gawait_aw(self, ring: AllocRing, data_idx: ElemIdx) -> Aw:
        """gAwait(x[z]) for SMEM data: each guard of x[z], z >= z_eac - 1."""
        home_guard_idx = data_idx[: len(self.shapes[ring.guard])]
        return frozenset(
            self._pending_await_eac_minus_1(guard)
            for guard in self._multicast_guards(ring, home_guard_idx)
        )

    def _ring_free_check(self, alloc_var: AllocVarName):
        """freeTrDetail: TrPreAccess(mutate, x, z, {}, {}, {PendingAwait(g, z_eac - 1)})
        for each g in gNext(x[z]) = guards of the next ring slot aliased to x[z]
        (mbarrier: just x_b[z_b + off]). Skipped when that slot is out of range, which
        only happens for a self-guarded mbarrier (the last phase of each physical mbarrier).
        """
        ring = self.rings[alloc_var]
        self_guarded = ring.guard == alloc_var
        guard_shape = self.shapes[ring.guard]
        for idx in elem_idxs(self.shapes[alloc_var]):
            next_home_guard_idx = ring_shifted_idx(
                idx[: len(guard_shape)], ring.spec.ring_dim, ring.spec.ring_depth
            )
            if (
                next_home_guard_idx[ring.spec.ring_dim]
                >= guard_shape[ring.spec.ring_dim]
            ):
                assert (
                    self_guarded
                ), f"{alloc_var}: guard {ring.guard} too short for free check"
                continue
            gnext = (
                [ring.guard[next_home_guard_idx]]
                if self_guarded
                else self._multicast_guards(ring, next_home_guard_idx)
            )
            elem_env = self.env[alloc_var[idx]]
            for guard in gnext:
                aw_in = [self._pending_await_eac_minus_1(guard)]
                for r in elem_env.all_records():
                    if not r.syncs_with(frozenset(), aw_in):
                        raise SyncEnvRefError(
                            f"free_missing_arrive @ {alloc_var[idx]}: needs {guard}"
                        )

    # *** Access (TrPreAccess then TrPostAccess, Tr-ConvergentTid / Tr-SeparateTid) ***

    def _check_prior_records(
        self,
        kind: AccessKind,
        elem: AllocElemName,
        tid_groups: List[Tids],
        precondition_bits: int,
    ):
        for rec in self.env[elem].checked_by(kind):
            for g in tid_groups:
                if not rec.syncs_with(obs(g, precondition_bits)):
                    raise SyncEnvRefError(f"hazard @ {elem} tids {g}")

    def access(
        self,
        kind: AccessKind,
        elem: StrElemName,
        tids,
        *,
        quals: AccessQuals,
        tag: int = 1,
        atomic_bits: int = 0,
        aw_post: Aw = frozenset(),
        extent: Optional[Tuple[int, ...]] = None,
    ):
        """Access elem, or the window elem.var[elem.idx : elem.idx + extent] (each element independently)."""
        if extent is not None:
            for off in elem_idxs(extent):
                self.access(
                    kind,
                    elem.var[tuple(i + o for i, o in zip(elem.idx, off))],
                    tids,
                    quals=quals,
                    tag=tag,
                    atomic_bits=atomic_bits,
                    aw_post=aw_post,
                )
            return
        tids = as_tids(tids)
        alloc_elem = self.alloc_elem(elem)
        # Pre: one TrPreAccess for all threads (convergent) or one per thread.
        self._check_prior_records(
            kind,
            alloc_elem,
            [tids] if quals.convergent else tids_split_per_thread(tids),
            quals.precondition_bits
            | atomic_bits,  # atomic accesses also accept atomic observers
        )
        # Post: ObOut = Tid_post x {q_init} U Tid x Q_atom, masked (envConcatPost).
        if quals.convergent:
            post_groups = [tids]
        else:
            # OPT (gOooOpt): per-thread records widened to aligned groups of `tag` threads.
            assert tag == 1 or tag & (tag - 1) == 0
            post_groups = sorted(
                {
                    ((t - t % tag, t - t % tag + tag),)
                    for ((t, _),) in tids_split_per_thread(tids)
                }
            )
        new = []
        for g in post_groups:
            ob_out = ob_union(
                obs(g, quals.initial_bit), obs(self.all_threads, atomic_bits)
            )
            new.append(VisRecord(ob_masked(ob_out, quals.mask), aw_post, quals.mask))
        # Spec semantics: prior reads and mutates are kept (camspork no longer
        # discards prior reads upon a mutate; that was unsound across tasks).
        records = self.env[alloc_elem].records
        records[kind] = records[kind] + new

    def read(self, elem: StrElemName, tids, **kw):
        self.access(AccessKind.READ, elem, tids, **kw)

    def mutate(self, elem: StrElemName, tids, **kw):
        self.access(AccessKind.MUTATE, elem, tids, **kw)

    # *** Synchronization (envConcatSync); each records before/after trace entries ***

    def _lambda(self, pred: Callable[[VisRecord], bool], ob_out: Ob, aw_out: Aw):
        for elem_env in self.env.values():
            for kind in AccessKind:
                elem_env.records[kind] = [
                    (
                        replace(
                            r,
                            ob=ob_union(r.ob, r.mask_ob(ob_out)),
                            aw=aw_union(r.aw, aw_out),
                        )
                        if pred(r)
                        else r
                    )
                    for r in elem_env.records[kind]
                ]

    def _traced_sync(
        self, stmt: SyncStmtKind, tids, fn: Callable[[], None], depth: int
    ):
        srcinfo = get_srcinfo(depth + 3)
        src = f"{srcinfo.filename}:{srcinfo.lineno}"
        for phase in TracePhase:
            if phase == TracePhase.AFTER:
                fn()
            self.trace.append(
                TraceEntry(phase, stmt, src, "", as_tids(tids), self.probe_snapshot())
            )

    def fence(self, tids, L1: int, L2: int, *, depth=0):
        tids = as_tids(tids)
        ob_in = obs(tids, L1)
        self._traced_sync(
            SyncStmtKind.FENCE,
            tids,
            lambda: self._lambda(
                lambda r: r.syncs_with(ob_in), obs(tids, L2), frozenset()
            ),
            depth,
        )

    def arrive(
        self, tids, L1: int, bar: Union[StrElemName, List[StrElemName]], *, depth=0
    ):
        """bar: one barrier element, or a list of them (multicast targets)."""
        tids = as_tids(tids)
        ob_in = obs(tids, L1)
        targets = [bar] if isinstance(bar, StrElemName) else list(bar)
        n_threads = sum(hi - lo for lo, hi in tids)
        for t in targets:
            self._mbarrier_access(t, tids, convergent=False)

        def fn():
            aw_out = self.barriers.arrive(
                [self.alloc_elem(t) for t in targets], n_threads
            )
            self._lambda(lambda r: r.syncs_with(ob_in), frozenset(), aw_out)

        self._traced_sync(SyncStmtKind.ARRIVE, tids, fn, depth)

    def await_(self, tids, L2: int, bar: StrElemName, N: int = 0, *, depth=0):
        tids = as_tids(tids)
        self._mbarrier_access(bar, tids, convergent=True)

        def fn():
            aw_in = [self.barriers.await_(self.alloc_elem(bar), N)]
            self._lambda(
                lambda r: r.syncs_with(frozenset(), aw_in), obs(tids, L2), frozenset()
            )

        self._traced_sync(SyncStmtKind.AWAIT, tids, fn, depth)

    def _mbarrier_access(self, bar: StrElemName, tids: Tids, convergent: bool):
        """sync_check.py models Arrive/Await on an mbarrier as a read ("concurrent
        access is allowed") of the mbarrier array element, before the sync statement."""
        if self.live[bar.var] not in self.mbarriers:
            return
        mb = mbarrier_qual_bit()
        self.read(bar, tids, quals=AccessQuals(mb, mb, mb, convergent))

    def join_threads(self, tids, *, depth=0):
        # DEVIATION (not in amspork3; David Zhao Akeley: matches camspork, which is assumed correct):
        # records with any observer among tids get tids x (union of those observers' qual-tls), masked.
        tids = as_tids(tids)

        def fn():
            for elem_env in self.env.values():
                for kind in AccessKind:
                    out = []
                    for r in elem_env.records[kind]:
                        q = {
                            o.qual_index for o in r.ob if tids_intersects(o.tids, tids)
                        }
                        bits = sum(1 << qi for qi in q)
                        out.append(
                            replace(r, ob=ob_union(r.ob, r.mask_ob(obs(tids, bits))))
                            if q
                            else r
                        )
                    elem_env.records[kind] = out

        self._traced_sync(SyncStmtKind.JOIN_THREADS, tids, fn, depth)

    # *** Kernel structure ***

    def cuda_device_function(self, body: Callable[[], None], *, depth=0):
        """C-CudaDeviceFunction (amspork3 ControlSemantics.tex): Fence(Q_stream + cpu_in_order, Q_stream)
        by all threads; body; Fence(Q_stream, Q_stream) by all threads. Q_stream = cuda_stream_sync's
        qual-tl (David Zhao Akeley). DEVIATION (camspork lowering): JoinThreads by all threads before the second Fence.
        """
        from .timelines import cpu_in_order_qual, cuda_stream_sync

        cpu_bit = cpu_in_order_qual.as_bit()
        q_stream = cuda_stream_sync.get_timeline_set_bits()
        self.fence(self.all_threads, q_stream | cpu_bit, q_stream, depth=depth + 1)
        body()
        self.join_threads(self.all_threads, depth=depth + 1)
        self.fence(self.all_threads, q_stream, q_stream, depth=depth + 1)

    def device_task(self, task, body: Callable[[], None], *, depth=0):
        """Body of the inner-most cuda_tasks loop. DEVIATION (camspork lowering): JoinThreads by the task's threads at the end."""
        body()
        self.join_threads(self.task_threads(task), depth=depth + 1)

    # *** Trace ***

    def probe_snapshot(self) -> List[TraceProbe]:
        out = []
        for elem in self.probes:
            alloc_var = self.live.get(elem.var)
            alloc_elem = None if alloc_var is None else alloc_var[elem.idx]
            elem_env = self.env.get(alloc_elem)
            if not elem.idx and elem_env is not None and not elem_env.all_records():
                elem_env = (
                    None  # matches camspork: a scalar without records prints as null
                )
            records = {
                kind: (
                    None
                    if elem_env is None
                    else frozenset(r.traced() for r in elem_env.records[kind])
                )
                for kind in AccessKind
            }
            counts = None if alloc_elem is None else self.barriers.counts(alloc_elem)
            out.append(TraceProbe(elem, records, counts))
        return out

    def write_trace(self, filename: str):
        """Write the reference trace; read it back with parse_trace (excut-style round trip)."""
        with open(filename, "w") as f:
            for e in self.trace:
                f.write(json.dumps(e.to_json()) + "\n")


def mbarrier_qual_bit() -> int:
    from .timelines import cuda_mbarrier_qual

    return cuda_mbarrier_qual.as_bit()


def ring_shifted_idx(idx: ElemIdx, ring_dim: int, delta: int) -> ElemIdx:
    return tuple(i + delta if d == ring_dim else i for d, i in enumerate(idx))


# *** Trace entries, parsing (both sides), comparison, and readable dumps ***


@dataclass(slots=True)
class TraceProbe:
    """TraceProbeJson. records[kind] is None for "null"."""

    elem: StrElemName
    records: Dict[AccessKind, Optional[FrozenSet[TraceVisRecord]]]
    barrier_counts: Optional[BarrierCounts]

    def to_json(self) -> dict:
        j = {"var": str(self.elem.var), "idx": list(self.elem.idx)}
        for kind, recs in self.records.items():
            j[kind.value] = None if recs is None else [r.to_json() for r in recs]
        if self.barrier_counts is not None:
            j["arrive_count"] = self.barrier_counts.arrive_count
            j["await_count"] = self.barrier_counts.await_count
        return j

    @staticmethod
    def from_json(j: dict) -> TraceProbe:
        records = {
            kind: (
                None
                if j[kind.value] is None
                else frozenset(TraceVisRecord.from_json(r) for r in j[kind.value])
            )
            for kind in AccessKind
        }
        counts = None
        if "arrive_count" in j:
            counts = BarrierCounts(j["arrive_count"], j["await_count"])
        elem = StrElemName(StrVarName(j["var"]), tuple(j["idx"]))
        return TraceProbe(elem, records, counts)


@dataclass(slots=True)
class TraceEntry:
    """TraceEntryJson, parsed. where: "file:line" of the JSONL line ("" before writing)."""

    phase: TracePhase
    stmt: SyncStmtKind
    src: Optional[str]
    where: str
    tids: Tids
    probes: List[TraceProbe]

    def to_json(self) -> dict:
        return {
            "phase": self.phase.value,
            "stmt": self.stmt.value,
            "stmt_id": None,
            "src": self.src,
            "threads": [list(iv) for iv in self.tids],
            "cuboid": None,
            "probes": [p.to_json() for p in self.probes],
        }


@dataclass(slots=True)
class CamsporkTraceRequest:
    """Passed as Procedure.sync_check(_camspork_trace=...).

    sync_check sets stmt_srcinfo (camspork stmt_id -> Exo source location str)
    for parse_trace."""

    filename: str
    probes: List[StrElemName]
    stmt_srcinfo: Optional[Callable[[int], str]] = None


def parse_trace(
    filename: str, stmt_srcinfo: Optional[Callable[[int], str]] = None
) -> List[TraceEntry]:
    """Parse a camspork or reference trace (same schema) into canonical form:
    sets of TraceVisRecord with merged observer intervals."""
    entries = []
    with open(filename) as f:
        for line_no, line in enumerate(f, 1):
            j = json.loads(line)
            src = j["src"]
            if src is None and j["stmt_id"] is not None and stmt_srcinfo:
                src = stmt_srcinfo(j["stmt_id"])
            entries.append(
                TraceEntry(
                    TracePhase(j["phase"]),
                    SyncStmtKind(j["stmt"]),
                    src,
                    f"{filename}:{line_no}",
                    tids_norm(tuple(iv) for iv in j["threads"]),
                    [TraceProbe.from_json(p) for p in j["probes"]],
                )
            )
    return entries


def fmt_tids(t: Tids) -> str:
    return ",".join(
        (
            f"{lo}"
            if hi == lo + 1
            else ("cpu" if (lo, hi) == (0, CPU_HI) else f"{lo}..{hi - 1}")
        )
        for lo, hi in t
    )


def fmt_record(r: TraceVisRecord, qual_names: List[str]) -> str:
    parts = [
        f"{qual_names[o.qual_index]}:{{{fmt_tids(o.tids)}}}"
        for o in sorted(r.ob, key=lambda o: o.qual_index)
    ]
    aw = sorted((str(a.bar), a.arrive_count) for a in r.aw)
    return f"Ob[{' '.join(parts)}] Aw{aw} mask={r.mask:#x}"


def fmt_records(
    recs: Optional[FrozenSet[TraceVisRecord]], qual_names: List[str], sep: str
) -> str:
    if recs is None:
        return "null"
    return sep.join(sorted(fmt_record(r, qual_names) for r in recs))


def compare_traces(
    actual: List[TraceEntry], expected: List[TraceEntry], qual_names: List[str]
) -> List[str]:
    """Return list of mismatch messages (empty: traces match as sets)."""
    errors = []
    for i, (a, e) in enumerate(zip(actual, expected)):
        where = (
            f"#{i} {a.phase.value} {a.stmt.value}\n"
            f"  actual   @ {a.src} ({a.where})\n"
            f"  expected @ {e.src} ({e.where})"
        )
        if (a.phase, a.stmt) != (e.phase, e.stmt):
            errors.append(f"{where}\n  expected {e.phase.value} {e.stmt.value}")
            break
        if a.tids != e.tids:
            errors.append(
                f"{where}\n  threads {{{fmt_tids(a.tids)}}} != expected {{{fmt_tids(e.tids)}}}"
            )
        for pa, pe in zip(a.probes, e.probes):
            assert pa.elem == pe.elem, (pa.elem, pe.elem)
            for kind in AccessKind:
                if pa.records[kind] != pe.records[kind]:
                    sep = "\n              "
                    errors.append(
                        f"{where}\n  {pa.elem} {kind.value}\n"
                        f"    actual:   {fmt_records(pa.records[kind], qual_names, sep)}\n"
                        f"    expected: {fmt_records(pe.records[kind], qual_names, sep)}"
                    )
            if pa.barrier_counts != pe.barrier_counts:
                errors.append(
                    f"{where}\n  {pa.elem} barrier counts {pa.barrier_counts} != expected {pe.barrier_counts}"
                )
    if len(actual) != len(expected):
        errors.append(f"trace length {len(actual)} != expected {len(expected)}")
    return errors


def write_trace_txt(filename: str, entries: List[TraceEntry], qual_names: List[str]):
    """Readable dump of a parsed trace; same printer for both sides, so the .txt files diff."""
    with open(filename, "w") as f:
        for e in entries:
            f.write(
                f"{e.phase.value} {e.stmt.value} threads={{{fmt_tids(e.tids)}}} @ {e.src}\n"
            )
            for p in e.probes:
                c = p.barrier_counts
                if c is not None:
                    f.write(
                        f"  {p.elem} arrive_count={c.arrive_count} await_count={c.await_count}\n"
                    )
                for kind in AccessKind:
                    recs = fmt_records(p.records[kind], qual_names, "\n    ")
                    f.write(f"  {p.elem} {kind.value}: {recs}\n")


# *** Test helpers ***


def access_quals(mem, instr_tl, *, convergent: Optional[bool] = None) -> AccessQuals:
    """Access qual-tls from Exo's qual-tl tables (configuration, not under test).
    convergent defaults to the initial qual-tl's default (instr args); Assign/Reduce are
    always convergent, and expression reads are convergent iff executed by one thread.
    """
    q = mem.qual_tl_dict[instr_tl]
    if convergent is None:
        convergent = q.initial.get_default_convergent_access()
    return AccessQuals(
        q.initial_bit, q.precondition_bits, mem.make_qual_tl_mask(), convergent
    )


def check_trace(p, ref: SyncEnvRef, tmp_path, **sizes) -> List[str]:
    """sync_check p with ref.probes traced, then compare with the reference trace.

    Like excut: both traces go to disk in the same JSONL schema and are read back
    with the same parser; readable .txt dumps of both sit next to them.
    Returns mismatch messages; also complains about probes that are null in every
    camspork entry (vacuous probe)."""
    from .timelines import Qual_tl

    request = CamsporkTraceRequest(str(tmp_path / "camspork_trace.jsonl"), ref.probes)
    p.sync_check(_camspork_trace=request, **sizes)
    ref_path = str(tmp_path / "reference_trace.jsonl")
    ref.write_trace(ref_path)

    actual = parse_trace(request.filename, request.stmt_srcinfo)
    expected = parse_trace(ref_path)
    qual_names = [str(q) for q in Qual_tl.get_all()]
    write_trace_txt(str(tmp_path / "camspork_trace.txt"), actual, qual_names)
    write_trace_txt(str(tmp_path / "reference_trace.txt"), expected, qual_names)
    errors = compare_traces(actual, expected, qual_names)
    for i, elem in enumerate(ref.probes):
        if not any(
            e.probes[i].records[AccessKind.MUTATE]
            or e.probes[i].records[AccessKind.READ]
            or e.probes[i].barrier_counts is not None
            for e in actual
        ):
            errors.append(f"vacuous probe {elem}: never has records or barrier counts")
    return errors


def check_both_fail(p, mkref: Callable[[], SyncEnvRef], **sizes):
    """Negative test (David Zhao Akeley): passes iff both camspork and the reference
    reject the program; the trace is not inspected."""
    import pytest

    with pytest.raises(Exception):
        p.sync_check(**sizes)
    with pytest.raises(SyncEnvRefError):
        mkref()
