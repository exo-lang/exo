"""PROTOTYPE Python reference for camspork sync env traces.

See endgame_plan/plan_sketch_camspork_test.md. Implements the amspork3
synchronization environment (spork/docs/amspork3/SyncEnvConcat.tex) directly
on sets, plus the camspork deviations/optimizations we compare against
(each marked DEVIATION or OPT). The test author drives this with hand-computed
thread IDs (like excut reference generators), so that the Exo -> camspork
lowering of parallel loops is checked independently.

Representation
  tids       Tids: sorted, merged tuple of [lo, hi) intervals. tid is camspork's linearized
             task_index * threads_per_task + local tid (papered-over task IDs);
             the CPU is [0, CPU_HI).
  Ob         frozenset of (qual_bit_index, Tids): observers Tids x {q}, one entry per q
  Aw         frozenset of (barrier_str, min_arrive_count), at most one per barrier;
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
    pass


class Mbarriers:
    """ALL mbarrier-specific behavior lives here (plan_multicast_barrier_counts.md
    may change it): arrive counts, which PendingAwaits an Arrive produces, and
    which PendingAwait an Await consumes."""

    def __init__(self):
        self.arrive_count: Dict[str, int] = {}
        self.await_count: Dict[str, int] = {}

    def alloc(self, bar: str):
        self.arrive_count[bar] = 0
        self.await_count[bar] = 0

    def free(self, bar: str):
        del self.arrive_count[bar]
        del self.await_count[bar]

    def arrive(self, home: str, targets: List[str]) -> Aw:
        # DEVIATION (current camspork): only the home barrier's count is read and
        # incremented; every multicast target gets a PendingAwait tagged with it.
        z = self.arrive_count[home]
        self.arrive_count[home] = z + 1
        return frozenset((t, z) for t in targets)

    def await_(self, bar: str, N: int) -> Tuple[str, int]:
        # Tr-Await: z_await = arrive_count - lag - 1
        self.await_count[bar] += 1
        return (bar, self.arrive_count[bar] - N - 1)


class SyncEnvRef:
    def __init__(self, threads_per_task: int):
        self.threads_per_task = threads_per_task
        # (var, idx) -> {"mutate": [VisRecord], "read": [VisRecord]}
        self.env: Dict[Tuple[str, tuple], Dict[str, List[VisRecord]]] = {}
        self.shapes: Dict[str, tuple] = {}
        self.mbarriers = Mbarriers()
        self.trace: List[dict] = []

    # *** Allocation ***

    def alloc(self, var: str, shape: tuple):
        self.shapes[var] = tuple(shape)
        for idx in itertools.product(*(range(n) for n in shape)):
            self.env[(var, idx)] = {"mutate": [], "read": []}

    def free(self, var: str):
        del self.shapes[var]
        for key in [k for k in self.env if k[0] == var]:
            del self.env[key]

    def tids(self, lo: int, hi: int, task: int = 0) -> Tids:
        base = task * self.threads_per_task
        return ((base + lo, base + hi),)

    cpu = ((0, CPU_HI),)

    # *** Access (TrPreAccess then TrPostAccess, Tr-ConvergentTid / Tr-SeparateTid) ***

    def access(
        self,
        is_mutate: bool,
        var: str,
        idx: tuple,
        tids: Iterable[int],
        *,
        initial_bit: int,
        precondition_bits: int,
        mask: int,
        convergent: bool,
        atomic_bits: int = 0,
        aw_post: Aw = frozenset(),
    ):
        tids = as_tids(tids)
        state = self.env[(var, tuple(idx))]
        # Pre: mutate checks prior mutates and reads (mutReqs); read checks prior mutates.
        checked = state["mutate"] + (state["read"] if is_mutate else [])
        groups = [tids] if convergent else tids_split(tids)
        for rec in checked:
            for g in groups:
                if not rec.syncs_with(obs(g, precondition_bits)):
                    raise SyncEnvRefError(f"hazard @ {var}{list(idx)} tids {g}")
        # Post: ObOut = Tid_post x {q_init} U Tid x Q_atom, masked (envConcatPost).
        new = []
        for g in groups:
            ob = obs(g, initial_bit)
            rec = VisRecord(ob, aw_post, mask)
            ob = rec.mask_ob(ob)
            if atomic_bits:
                # Spec (envConcatPost) masks Tid x Q_atom; camspork wrongly doesn't, and
                # Exo's QtlMask lacks atomic qual-tls (plan_camspork_test.md, deviation 2).
                raise NotImplementedError("atomics: blocked on QtlMask fix")
            new.append(replace(rec, ob=ob))
        # OPT (SyncEnvImpl.tex): mutate discards prior reads; non-atomic mutate
        # also discards prior mutates.
        if is_mutate:
            state["read"] = []
            if not atomic_bits:
                state["mutate"] = []
            state["mutate"] = state["mutate"] + new
        else:
            state["read"] = state["read"] + new

    # *** Synchronization (envConcatSync) ***

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

    def fence(self, tids, L1: int, L2: int):
        tids = as_tids(tids)
        ob_in = obs(tids, L1)
        self._lambda(lambda r: r.syncs_with(ob_in), obs(tids, L2), frozenset())

    def arrive(self, tids, L1: int, home: str, targets: Optional[List[str]] = None):
        tids = as_tids(tids)
        aw_out = self.mbarriers.arrive(home, targets or [home])
        ob_in = obs(tids, L1)
        self._lambda(lambda r: r.syncs_with(ob_in), frozenset(), aw_out)

    def await_(self, tids, L2: int, bar: str, N: int):
        aw_in = [self.mbarriers.await_(bar, N)]
        self._lambda(
            lambda r: r.syncs_with(frozenset(), aw_in), obs(tids, L2), frozenset()
        )

    def join_threads(self, tids):
        # DEVIATION (not in amspork3): records with any observer among tids get
        # tids x (union of those observers' qual-tls).
        tids = as_tids(tids)
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

    def alloc_barrier(self, var: str, shape: tuple):
        """Barrier variables: sync env entries (Arrive/Await access them) plus Mbarriers state."""
        self.alloc(var, shape)
        for idx in itertools.product(*(range(n) for n in shape)):
            self.mbarriers.alloc(barrier_str(var, idx))

    def free_barrier(self, var: str):
        for idx in itertools.product(*(range(n) for n in self.shapes[var])):
            self.mbarriers.free(barrier_str(var, idx))
        self.free(var)

    # *** Trace (same JSONL schema as camspork's exec.cpp trace_sync) ***

    def snapshot(self, probes) -> List[dict]:
        out = []
        for var, idx in probes:
            state = self.env.get((var, tuple(idx)))
            p = {"var": var, "idx": list(idx)}
            for kind in ("mutate", "read"):
                p[kind] = (
                    None if state is None else [record_to_json(r) for r in state[kind]]
                )
            bar = barrier_str(var, idx)
            if bar in self.mbarriers.arrive_count:
                p["arrive_count"] = self.mbarriers.arrive_count[bar]
                p["await_count"] = self.mbarriers.await_count[bar]
            out.append(p)
        return out

    def sync(self, stmt: str, tids, probes, fn, *, depth=0):
        """Record before/after snapshots around a sync event, like camspork's trace.

        The entry's "src" is the caller's Python line (like excut's xrg(...));
        pass depth to attribute it to a caller further up (e.g. a helper's caller).
        """
        srcinfo = get_srcinfo(depth + 2)
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
                    "probes": self.snapshot(probes),
                }
            )

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
        "aw": [[bar, z] for bar, z in sorted(r.aw)],
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
