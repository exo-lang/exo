"""PROTOTYPE Python reference for camspork sync env traces.

Plan: endgame_plan/plan_camspork_test.md (sketch: plan_sketch_camspork_test.md).
Implements the amspork3 synchronization environment (spork/docs/amspork3/SyncEnvConcat.tex)
directly on sets, plus the camspork deviations/optimizations we compare against, each
marked DEVIATION or OPT. Thread IDs are hand-computed by the test author (like excut
reference generators), so the Exo -> camspork lowering of parallel loops is checked
independently.

Current scope: no mbarriers, managed ring buffers, or multi-dimensional tasks.
Atomics follow the spec (Tid x Q_atom, masked; needs exo 7c82d4a0's QtlMask).
The trace only samples sync statements, so a test must never leave an access without
a later sync (otherwise the access is invisible).


Typical test (tests/cuda/test_camspork_trace.py)
------------------------------------------------

    def mkproc_foo():                     # Exo proc under test
        @proc
        def proc_foo(g: f32[2, 32] @ CudaGmemLinear):
            with CudaDeviceFunction(blockDim=64):
                for task in cuda_tasks(0, 1):
                    ...
        return proc_foo

    def mkref_foo():                      # hand-written reference
        ref = SyncEnvRef([("x", (0, 0)), ("g", (1,))], threads_per_task=64)  # probes
        task = ref.task_threads()

        def task_body():
            ref.alloc("x", (2, 32))
            ref.mutate("x", (w, t), [w * 32 + t], **qual_info(CudaSmemLinear, cuda_in_order_instr,
                                                             convergent=True))
            ref.fence(task, L1_bits, L2_bits)
            ...
            ref.free("x", check_tids=task, precondition_bits=...)  # SMEM free check

        ref.cuda_device_function(lambda: ref.device_task(0, task_body))
        return ref

    def test_foo(compiler, tmp_path):
        errors = check_trace(mkproc_foo(), mkref_foo(), tmp_path, **sizes)
        assert not errors, "\n".join(errors)

    def test_bar_negative(compiler):      # both camspork and the reference must reject;
        check_both_fail(mkproc_bar(), mkref_bar)   # the trace is not inspected

* Probes are (var name, idx) pairs, with var named by str(Sym) in the proc (must be
  unambiguous). Only sync-checked variables have records: written, non-sync-exempt memory.
  Barrier variables additionally report arrive/await counts.
* The reference must issue the same sync statements, in the same order, as camspork
  executes them. That includes the implicit ones: cuda_device_function() emits the
  C-CudaDeviceFunction Fences plus the device-end JoinThreads, and device_task() emits
  the task-end JoinThreads.
* Access arguments come from Exo's qual-tl tables via qual_info(mem, instr_tl); those
  tables are configuration, not under test. Convergence follows sync_check.py:
  Assign/Reduce always convergent; expression reads convergent iff one thread executes
  them; instr args use the initial qual-tl's default (out-of-order ones also take tag=TAG).
* Windows: access(..., extent=...) applies the access to each element independently.
* Barriers: alloc_barrier/free_barrier, arrive/await_ by "name[idx]"; identity is per
  allocation (each loop iteration's allocation is a new barrier).
* Sync statements record trace entries with "src" = the calling Python line; helpers
  pass depth= so entries point at the mkref line that called them.


Trace files (in tmp_path; same JSONL schema on both sides, excut-style round trip)
--------------------------------------------------------------------------------

camspork_trace.jsonl (camspork exec.cpp trace_sync) and reference_trace.jsonl
(SyncEnvRef.write_trace) are both read back by parse_trace and compared by
compare_traces. camspork_trace.txt / reference_trace.txt are readable dumps of the
parsed traces (same printer, so they diff line by line; only "@ src" differs when the
traces match).

One JSON line per sync statement phase:

    {"phase": "before" | "after",
     "stmt": "Fence" | "Arrive" | "Await" | "JoinThreads",
     "stmt_id": camspork StmtRef raw data (null in the reference),
     "src": null in camspork (parse_trace fills in the Exo proc line from stmt_id),
            "file:line" of the mkref call in the reference,
     "threads": [[lo, hi), ...]   executing threads,
     "cuboid": camspork ThreadCuboid (null in the reference; informational),
     "probes": [{"var", "idx",
                 "mutate": [record, ...] | null,   null: not allocated / out of range
                 "read": [record, ...] | null,     (a scalar with no records is null too)
                 "arrive_count", "await_count"}]}  (only for live barriers)

    record = {"ob": [[tid_lo, tid_hi, qual_bits], ...],   observers (tid_hi exclusive)
              "aw": [["name[idx]", arrive_count], ...],   pending awaits
              "mask": qual_tl_mask, "flags": 2 = vis_record_freed_flag}

Comparison: per entry, same (phase, stmt) and thread set; per probe and kind, the SET
of records must match (order and duplicates ignored; observers merged per qual-tl;
pending awaits reduced to the minimum arrive count per barrier name), plus barrier counts.
Barrier names drop the allocation instance, so two allocations of the same name are
indistinguishable in the trace (identity is only checked through its effects).


Representation (in memory)
--------------------------

  tids       Tids: sorted, merged tuple of [lo, hi) intervals. tid is camspork's linearized
             task_index * threads_per_task + local tid (papered-over task IDs). Local tids
             are numbered by CTA rank in the cluster, then thread index in the CTA.
             All threads (and camspork's "CPU") are [0, CPU_HI).
  Ob         frozenset of (qual_bit_index, Tids): observers Tids x {q}, one entry per q
  Aw         frozenset of (barrier instance, min_arrive_count), at most one per barrier;
             stands for the infinite set {PendingAwait(b, z) | z >= min_arrive_count}
  envVis     VisRecord(ob, aw, mask, freed)
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from ..core.prelude import get_srcinfo

Tids = Tuple[Tuple[int, int], ...]
Ob = FrozenSet[Tuple[int, Tids]]
Aw = FrozenSet[Tuple[str, int]]

CPU_HI = 2**32 - 1  # camspork models the CPU as threads [0, UINT32_MAX)


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


def tids_split(t: Tids) -> List[Tids]:
    """Per-thread singletons (non-convergent case); only for finite GPU tid sets."""
    assert sum(hi - lo for lo, hi in t) < 1 << 20, "too many threads to split"
    return [((x, x + 1),) for lo, hi in t for x in range(lo, hi)]


def obs(tids, bits: int) -> Ob:
    """Observer set tids x qual-tl bits"""
    t = as_tids(tids)
    return frozenset((qi, t) for qi in bits_to_indices(bits)) if t else frozenset()


def ob_union(a: Ob, b: Ob) -> Ob:
    d: Dict[int, Tids] = dict(a)
    for qi, t in b:
        d[qi] = tids_norm(d.get(qi, ()) + t)
    return frozenset(d.items())


def ob_intersects(a: Ob, b: Ob) -> bool:
    d = dict(a)
    return any(qi in d and tids_intersects(d[qi], t) for qi, t in b)


def aw_union(a: Aw, b: Aw) -> Aw:
    # Union of the infinite sets: keep the min arrive count per barrier.
    d = dict(a)
    for bar, z in b:
        d[bar] = min(d.get(bar, z), z)
    return frozenset(d.items())


@dataclass(frozen=True)
class VisRecord:
    ob: Ob
    aw: Aw
    mask: int
    freed: bool = False

    def mask_ob(self, ob: Ob) -> Ob:
        m = bits_to_indices(self.mask)
        return frozenset(o for o in ob if o[0] in m)

    def syncs_with(self, ob_in: Ob, aw_in: Iterable[Tuple[str, int]] = ()) -> bool:
        # syncsWith (SyncEnvConcat.tex), with aw_in given as concrete PendingAwait(b, z).
        if ob_intersects(self.mask_ob(self.ob), ob_in):
            return True
        mins = dict(self.aw)
        return any(bar in mins and mins[bar] <= z for bar, z in aw_in)


class SyncEnvRefError(Exception):
    """The reference rejects the program (expected to coincide with a camspork error)."""


class Barriers:
    """ALL barrier counting lives here (plan_multicast_barrier_counts.md may change it):
    arrive/await counts, which PendingAwaits an Arrive produces, which PendingAwait
    an Await consumes, and the one-shot checks. Follows camspork (HamsterBarrierState);
    amspork3 leaves the count increment to barrier-mechanism-specific de-sugaring.
    """

    # Barriers are identified per allocation (alpha-renaming per loop iteration in amspork3;
    # distinct barrier ids in camspork): instance "name[idx]#n". Traces print just "name[idx]".

    def __init__(self):
        self.current: Dict[str, str] = {}  # "name[idx]" -> live instance
        self.n_instances = 0
        self.arrive_count: Dict[str, int] = {}  # keyed by instance
        self.await_count: Dict[str, int] = {}
        self.one_shot_arrive: Dict[str, bool] = {}
        self.one_shot_await: Dict[str, bool] = {}

    def alloc(self, name: str, *, one_shot_arrive=False, one_shot_await=False):
        bar = f"{name}#{self.n_instances}"
        self.n_instances += 1
        self.current[name] = bar
        self.arrive_count[bar] = 0
        self.await_count[bar] = 0
        self.one_shot_arrive[bar] = one_shot_arrive
        self.one_shot_await[bar] = one_shot_await

    def free(self, name: str):
        del self.current[name]

    def counts(self, name: str) -> Optional[Tuple[int, int]]:
        bar = self.current.get(name)
        return None if bar is None else (self.arrive_count[bar], self.await_count[bar])

    def arrive(self, name: str) -> Aw:
        """Non-multicast Arrive: PendingAwait(bar, z) for z >= the count before arriving."""
        bar = self.current[name]
        z = self.arrive_count[bar]
        self.arrive_count[bar] = z + 1
        if self.one_shot_arrive[bar] and z != 0:
            raise SyncEnvRefError(f"Arrived twice on one-shot arrive barrier {name}")
        return frozenset([(bar, z)])

    def arrive_multicast_mbarrier(self, home: str, targets: List[str]) -> Aw:
        # DEVIATION (current camspork, mbarrier only; out of scope for now): only the home
        # barrier's count is read and incremented; every target gets a PendingAwait tagged with it.
        home_bar = self.current[home]
        z = self.arrive_count[home_bar]
        self.arrive_count[home_bar] = z + 1
        return frozenset((self.current[t], z) for t in targets)

    def await_(self, name: str, N: int) -> Tuple[str, int]:
        # Tr-Await: z_await = arrive_count - lag - 1
        bar = self.current[name]
        self.await_count[bar] += 1
        if self.one_shot_await[bar] and self.await_count[bar] != 1:
            raise SyncEnvRefError(f"Awaited twice on one-shot await barrier {name}")
        if self.one_shot_arrive[bar] and self.arrive_count[bar] == 0:
            raise SyncEnvRefError(f"No forward progress: Await before Arrive on {name}")
        return (bar, self.arrive_count[bar] - N - 1)


class SyncEnvRef:
    """Reference sync env. Sync statements record before/after trace entries for
    `probes` ([(var, idx)]); "src" is the Python line of the call (depth= to skip helpers).
    """

    all_threads: Tids = ((0, CPU_HI),)  # amspork3 Tid; camspork's "CPU" cuboid

    def __init__(self, probes, threads_per_task: int, *, discard_reads_on_mutate=True):
        self.probes = [(var, tuple(idx)) for var, idx in probes]
        self.threads_per_task = threads_per_task
        self.discard_reads_on_mutate = discard_reads_on_mutate
        # (var, idx) -> {"mutate": [VisRecord], "read": [VisRecord]}
        self.env: Dict[Tuple[str, tuple], Dict[str, List[VisRecord]]] = {}
        self.shapes: Dict[str, tuple] = {}
        self.barriers = Barriers()
        self.trace: List[dict] = []

    def tids(self, lo: int, hi: int, task: int = 0) -> Tids:
        base = task * self.threads_per_task
        return ((base + lo, base + hi),)

    def task_threads(self, task: int = 0) -> Tids:
        return self.tids(0, self.threads_per_task, task)

    # *** Allocation ***

    def alloc(self, var: str, shape: tuple):
        assert var not in self.shapes, var
        self.shapes[var] = tuple(shape)
        for idx in itertools.product(*(range(n) for n in shape)):
            self.env[(var, idx)] = {"mutate": [], "read": []}

    def free(self, var: str, *, check_tids=None, precondition_bits=0):
        """D-Free. With check_tids: freeTrDetail for (non-ring) SMEM, i.e.
        TrPreAccess(mutate, x, z, {tid}, Q_pre, {}) for every element and tid."""
        if check_tids is not None:
            for idx in itertools.product(*(range(n) for n in self.shapes[var])):
                self._check(
                    True, var, idx, tids_split(as_tids(check_tids)), precondition_bits
                )
        del self.shapes[var]
        for key in [k for k in self.env if k[0] == var]:
            del self.env[key]

    def alloc_barrier(self, var: str, shape: tuple, **flags):
        """Barrier variable. Only sync-exempt mechanisms (commit groups, cluster sync)
        for now: no sync env entries, just Barriers state."""
        self.shapes[var] = tuple(shape)
        for idx in itertools.product(*(range(n) for n in shape)):
            self.barriers.alloc(barrier_str(var, idx), **flags)

    def free_barrier(self, var: str):
        for idx in itertools.product(*(range(n) for n in self.shapes[var])):
            self.barriers.free(barrier_str(var, idx))
        del self.shapes[var]

    # *** Access (TrPreAccess then TrPostAccess, Tr-ConvergentTid / Tr-SeparateTid) ***

    def _check(self, is_mutate, var, idx, groups, precondition_bits):
        state = self.env[(var, tuple(idx))]
        # mutReqs: mutate checks prior mutates and reads; read checks prior mutates.
        checked = state["mutate"] + (state["read"] if is_mutate else [])
        for rec in checked:
            for g in groups:
                if not rec.syncs_with(obs(g, precondition_bits)):
                    raise SyncEnvRefError(f"hazard @ {var}{list(idx)} tids {g}")

    def access(
        self,
        is_mutate: bool,
        var: str,
        idx: tuple,
        tids,
        *,
        initial_bit: int,
        precondition_bits: int,
        mask: int,
        convergent: bool,
        tag: int = 1,
        atomic_bits: int = 0,
        aw_post: Aw = frozenset(),
        extent: Optional[tuple] = None,
    ):
        """Access var[idx], or the window var[idx : idx + extent] (each element independently)."""
        if extent is not None:
            for off in itertools.product(*(range(n) for n in extent)):
                elem = tuple(i + o for i, o in zip(idx, off))
                self.access(
                    is_mutate,
                    var,
                    elem,
                    tids,
                    initial_bit=initial_bit,
                    precondition_bits=precondition_bits,
                    mask=mask,
                    convergent=convergent,
                    tag=tag,
                    atomic_bits=atomic_bits,
                    aw_post=aw_post,
                )
            return
        tids = as_tids(tids)
        state = self.env[(var, tuple(idx))]
        # Pre: one TrPreAccess for all threads (convergent) or one per thread.
        self._check(
            is_mutate,
            var,
            idx,
            [tids] if convergent else tids_split(tids),
            precondition_bits
            | atomic_bits,  # atomic accesses also accept atomic observers
        )
        # Post: ObOut = Tid_post x {q_init} U Tid x Q_atom, masked (envConcatPost).
        if convergent:
            post_groups = [tids]
        else:
            # OPT (gOooOpt): per-thread records widened to aligned groups of `tag` threads.
            assert tag == 1 or tag & (tag - 1) == 0
            post_groups = sorted(
                {((t - t % tag, t - t % tag + tag),) for ((t, _),) in tids_split(tids)}
            )
        new = []
        for g in post_groups:
            rec = VisRecord(frozenset(), aw_post, mask)
            ob_out = ob_union(obs(g, initial_bit), obs(self.all_threads, atomic_bits))
            new.append(replace(rec, ob=rec.mask_ob(ob_out)))
        # OPT (camspork): a mutate discards prior reads. UNSOUND across tasks
        # (test_read_then_reduce_*); discard_reads_on_mutate=False gives spec semantics.
        # Prior mutates are kept.
        if is_mutate:
            if self.discard_reads_on_mutate:
                state["read"] = []
            state["mutate"] = state["mutate"] + new
        else:
            state["read"] = state["read"] + new

    def read(self, var, idx, tids, **kw):
        self.access(False, var, idx, tids, **kw)

    def mutate(self, var, idx, tids, **kw):
        self.access(True, var, idx, tids, **kw)

    # *** Synchronization (envConcatSync); each records before/after trace entries ***

    def _lambda(self, pred, ob_out: Ob, aw_out: Aw):
        for state in self.env.values():
            for kind in ("mutate", "read"):
                state[kind] = [
                    (
                        replace(
                            r,
                            ob=ob_union(r.ob, r.mask_ob(ob_out)),
                            aw=aw_union(r.aw, aw_out),
                        )
                        if pred(r)
                        else r
                    )
                    for r in state[kind]
                ]

    def _sync(self, stmt: str, tids, fn, depth: int):
        srcinfo = get_srcinfo(depth + 3)
        src = f"{srcinfo.filename}:{srcinfo.lineno}"
        threads = [list(iv) for iv in as_tids(tids)]
        for phase in ("before", "after"):
            if phase == "after":
                fn()
            self.trace.append(
                {
                    "phase": phase,
                    "stmt": stmt,
                    "stmt_id": None,
                    "src": src,
                    "threads": threads,
                    "cuboid": None,
                    "probes": self.snapshot(),
                }
            )

    def fence(self, tids, L1: int, L2: int, *, depth=0):
        tids = as_tids(tids)
        ob_in = obs(tids, L1)
        self._sync(
            "Fence",
            tids,
            lambda: self._lambda(
                lambda r: r.syncs_with(ob_in), obs(tids, L2), frozenset()
            ),
            depth,
        )

    def arrive(self, tids, L1: int, bar: str, *, depth=0):
        tids = as_tids(tids)
        ob_in = obs(tids, L1)

        def fn():
            aw_out = self.barriers.arrive(bar)
            self._lambda(lambda r: r.syncs_with(ob_in), frozenset(), aw_out)

        self._sync("Arrive", tids, fn, depth)

    def await_(self, tids, L2: int, bar: str, N: int = 0, *, depth=0):
        tids = as_tids(tids)

        def fn():
            aw_in = [self.barriers.await_(bar, N)]
            self._lambda(
                lambda r: r.syncs_with(frozenset(), aw_in), obs(tids, L2), frozenset()
            )

        self._sync("Await", tids, fn, depth)

    def join_threads(self, tids, *, depth=0):
        # DEVIATION (not in amspork3; David Zhao Akeley: matches camspork, which is assumed correct):
        # records with any observer among tids get tids x (union of those observers' qual-tls), masked.
        tids = as_tids(tids)

        def fn():
            for state in self.env.values():
                for kind in ("mutate", "read"):
                    out = []
                    for r in state[kind]:
                        q = {qi for (qi, t) in r.ob if tids_intersects(t, tids)}
                        bits = sum(1 << qi for qi in q)
                        out.append(
                            replace(r, ob=ob_union(r.ob, r.mask_ob(obs(tids, bits))))
                            if q
                            else r
                        )
                    state[kind] = out

        self._sync("JoinThreads", tids, fn, depth)

    # *** Kernel structure ***

    def cuda_device_function(self, body, *, depth=0):
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

    def device_task(self, task: int, body, *, depth=0):
        """Body of the inner-most cuda_tasks loop. DEVIATION (camspork lowering): JoinThreads by the task's threads at the end."""
        body()
        self.join_threads(self.task_threads(task), depth=depth + 1)

    # *** Trace (same JSONL schema as camspork's exec.cpp trace_sync) ***

    def snapshot(self) -> List[dict]:
        out = []
        for var, idx in self.probes:
            state = self.env.get((var, idx))
            if (
                not idx
                and state is not None
                and not state["mutate"]
                and not state["read"]
            ):
                state = (
                    None  # matches camspork: a scalar without records prints as null
                )
            p = {"var": var, "idx": list(idx)}
            for kind in ("mutate", "read"):
                p[kind] = (
                    None if state is None else [record_to_json(r) for r in state[kind]]
                )
            if (counts := self.barriers.counts(barrier_str(var, idx))) is not None:
                p["arrive_count"], p["await_count"] = counts
            out.append(p)
        return out

    def write_trace(self, filename: str):
        """Write the reference trace; read it back with parse_trace (excut-style round trip)."""
        with open(filename, "w") as f:
            for e in self.trace:
                f.write(json.dumps(e) + "\n")


def barrier_str(var: str, idx) -> str:
    # Matches camspork exec.cpp print_idx_helper (no brackets for scalars).
    return var + (f"[{', '.join(str(i) for i in idx)}]" if len(idx) else "")


def record_to_json(r: VisRecord) -> dict:
    return {
        "ob": [[lo, hi, 1 << qi] for qi, t in sorted(r.ob) for lo, hi in t],
        "aw": [
            [bar.split("#")[0], z] for bar, z in sorted(r.aw)
        ],  # drop instance suffix
        "mask": r.mask,
        "flags": 2 if r.freed else 0,  # vis_record_freed_flag
    }


def record_from_json(r: dict) -> VisRecord:
    ob = frozenset()
    for lo, hi, bits in r["ob"]:
        ob = ob_union(ob, obs(((lo, hi),), bits))
    return VisRecord(
        ob,
        aw_union(frozenset(), frozenset(tuple(a) for a in r["aw"])),
        r["mask"],
        bool(r["flags"] & 2),  # vis_record_freed_flag
    )


# *** Trace parsing (both sides), comparison, and readable dumps ***


@dataclass
class CamsporkTraceRequest:
    """Passed as Procedure.sync_check(_camspork_trace=...).

    probes: [(var name str, idx tuple)]. sync_check sets stmt_srcinfo
    (camspork stmt_id -> Exo source location str) for parse_trace."""

    filename: str
    probes: List[Tuple[str, tuple]]
    stmt_srcinfo: Optional[object] = None


def parse_trace(filename: str, stmt_srcinfo=None) -> List[dict]:
    """Parse a camspork or reference trace (same schema) into canonical form:
    sets of VisRecord with merged observer intervals."""
    entries = []
    with open(filename) as f:
        for line_no, line in enumerate(f, 1):
            e = json.loads(line)
            src = e["src"]
            if src is None and e["stmt_id"] is not None and stmt_srcinfo:
                src = stmt_srcinfo(e["stmt_id"])
            probes = []
            for p in e["probes"]:
                q = {
                    "var": p["var"],
                    "idx": p["idx"],
                    "arrive_count": p.get("arrive_count"),
                    "await_count": p.get("await_count"),
                }
                for kind in ("mutate", "read"):
                    recs = p[kind]
                    q[kind] = (
                        None if recs is None else {record_from_json(r) for r in recs}
                    )
                probes.append(q)
            entries.append(
                {
                    "phase": e["phase"],
                    "stmt": e["stmt"],
                    "src": src,
                    "where": f"{filename}:{line_no}",
                    "tids": tids_norm(tuple(iv) for iv in e["threads"]),
                    "probes": probes,
                }
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


def fmt_record(r: VisRecord, qual_names) -> str:
    parts = [f"{qual_names[qi]}:{{{fmt_tids(t)}}}" for qi, t in sorted(r.ob)]
    return f"Ob[{' '.join(parts)}] Aw{sorted(r.aw)} mask={r.mask:#x}{' FREED' if r.freed else ''}"


def fmt_records(recs, qual_names, sep) -> str:
    if recs is None:
        return "null"
    return sep.join(sorted(fmt_record(r, qual_names) for r in recs))


def compare_traces(actual: List[dict], expected: List[dict], qual_names) -> List[str]:
    """Return list of mismatch messages (empty: traces match as sets)."""
    errors = []
    for i, (a, e) in enumerate(zip(actual, expected)):
        where = (
            f"#{i} {a['phase']} {a['stmt']}\n"
            f"  actual   @ {a['src']} ({a['where']})\n"
            f"  expected @ {e['src']} ({e['where']})"
        )
        if (a["phase"], a["stmt"]) != (e["phase"], e["stmt"]):
            errors.append(f"{where}\n  expected {e['phase']} {e['stmt']}")
            break
        if a["tids"] != e["tids"]:
            errors.append(
                f"{where}\n  threads {{{fmt_tids(a['tids'])}}} != expected {{{fmt_tids(e['tids'])}}}"
            )
        for pa, pe in zip(a["probes"], e["probes"]):
            name = f"{pa['var']}{pa['idx']}"
            for kind in ("mutate", "read"):
                if pa[kind] != pe[kind]:
                    sep = "\n              "
                    errors.append(
                        f"{where}\n  {name} {kind}\n"
                        f"    actual:   {fmt_records(pa[kind], qual_names, sep)}\n"
                        f"    expected: {fmt_records(pe[kind], qual_names, sep)}"
                    )
            for count in ("arrive_count", "await_count"):
                if pa[count] != pe[count]:
                    errors.append(
                        f"{where}\n  {name} {count} {pa[count]} != expected {pe[count]}"
                    )
    if len(actual) != len(expected):
        errors.append(f"trace length {len(actual)} != expected {len(expected)}")
    return errors


def write_trace_txt(filename: str, entries: List[dict], qual_names):
    """Readable dump of a parsed trace; same printer for both sides, so the .txt files diff."""
    with open(filename, "w") as f:
        for e in entries:
            f.write(
                f"{e['phase']} {e['stmt']} threads={{{fmt_tids(e['tids'])}}} @ {e['src']}\n"
            )
            for p in e["probes"]:
                name = f"{p['var']}{p['idx']}"
                if p["arrive_count"] is not None:
                    f.write(
                        f"  {name} arrive_count={p['arrive_count']} await_count={p['await_count']}\n"
                    )
                for kind in ("mutate", "read"):
                    f.write(
                        f"  {name} {kind}: {fmt_records(p[kind], qual_names, chr(10) + '    ')}\n"
                    )


# *** Test helpers ***


def qual_info(mem, instr_tl, *, convergent=None) -> dict:
    """access(...) keyword args from Exo's qual-tl tables (configuration, not under test).
    convergent defaults to the initial qual-tl's default (instr args); Assign/Reduce are
    always convergent, and expression reads are convergent iff executed by one thread.
    """
    q = mem.qual_tl_dict[instr_tl]
    if convergent is None:
        convergent = q.initial.get_default_convergent_access()
    return dict(
        initial_bit=q.initial_bit,
        precondition_bits=q.precondition_bits,
        mask=mem.make_qual_tl_mask(),
        convergent=convergent,
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
    for i, (var, idx) in enumerate(ref.probes):
        if not any(
            e["probes"][i]["mutate"]
            or e["probes"][i]["read"]
            or e["probes"][i]["arrive_count"] is not None
            for e in actual
        ):
            errors.append(
                f"vacuous probe {var}{list(idx)}: never has records or barrier counts"
            )
    return errors


def check_both_fail(p, mkref, **sizes):
    """Negative test (David Zhao Akeley): passes iff both camspork and the reference
    reject the program; the trace is not inspected."""
    import pytest

    with pytest.raises(Exception):
        p.sync_check(**sizes)
    with pytest.raises(SyncEnvRefError):
        mkref()
