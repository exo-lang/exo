from dataclasses import dataclass
from typing import Optional, List, Dict, Type, Tuple

from ..core.LoopIR import LoopIR, LoopIR_Do
from ..core.memory import BarrierType, BarrierTypeTraits
from ..core.prelude import Sym
from .base_with_context import is_if_holding_with
from .sync_types import SyncType
from .timelines import Sync_tl


@dataclass(slots=True)
class SyncInfo:
    sync_tl: Sync_tl
    stmts: List[LoopIR.stmt]
    min_N: int
    max_N: int
    multicasts: Tuple[Tuple[bool]]

    def get_srcinfo(self):
        return self.stmts[0].srcinfo


@dataclass(slots=True)
class BarrierUsage:
    # None iff this barrier is a Fence()
    barrier_type: Optional[Type[BarrierType]]

    decl_stmt: LoopIR.stmt  # barrier alloc, or Fence

    # Information for Arrive/Await statements, split by usage.
    # Fence() stmts are decomposed as an Arrive + Await
    Arrive: Optional[SyncInfo]
    Await: Optional[SyncInfo]

    guards: Sym
    guarded_by: Sym

    def __init__(self, s):
        self.decl_stmt = s
        self.Arrive = None
        self.Await = None
        if isinstance(s, LoopIR.SyncStmt):
            sync_type = s.sync_type
            assert not sync_type.is_split()
            self.barrier_type = None
            self._init_Fence_impl(s)
        else:
            assert isinstance(s, LoopIR.Alloc)
            self.barrier_type = s.mem
            self.guards = s.name
            self.guarded_by = s.name
            assert issubclass(s.mem, BarrierType)

    def get_srcinfo(self):
        return self.decl_stmt.srcinfo

    def is_fence(self):
        return self.barrier_type is None

    def get_arrive(self) -> Optional[SyncInfo]:
        return self.Arrive

    def get_await(self) -> Optional[SyncInfo]:
        return self.Await

    def visit_Arrive(self, s: LoopIR.SyncStmt):
        # We do not enforce pairing, but we enforce other traits
        mem = self.barrier_type
        assert mem
        sync_type = s.sync_type
        sync_tl = sync_type.first_sync_tl
        N = sync_type.N
        assert sync_type.is_arrive()

        # home_barrier_expr() enforces usage of the same queue barrier array
        home_barrier = s.home_barrier_expr()
        nm = home_barrier.name

        traits = mem.traits()
        multicasts = s.multicasts()

        def kvetch_invalid(reason):
            raise ValueError(f"{s.srcinfo}: invalid {s}; {reason}")

        def kvetch_incompatible(thing):
            raise ValueError(
                f"{s.srcinfo}: incompatible {thing} with previous Arrive\n{old} @ {old.srcinfo}\n{s} @ {s.srcinfo}"
            )

        # Save new SyncInfo, or check with previously saved SyncInfo.
        # Must have identical sync-tl and multicasting as any
        # other Arrives to the same queue barrier array.
        info = self.Arrive
        if info is None:
            info = SyncInfo(sync_tl, [s], N, N, multicasts)
            self.Arrive = info
        else:
            old = info.stmts[0]
            info.stmts.append(info)
            info.min_N = min(N, info.min_N)
            info.max_N = max(N, info.max_N)
            if info.sync_tl != sync_tl:
                kvetch_incompatible(f"sync-tl ({sync_tl})")
            if info.multicasts != multicasts:
                kvetch_incompatible("multicasts")

        # Enforce traits
        if not traits.supports_arrive_multicast:
            s.forbid_multicast(f"{mem.name()} does not support multicast")

        # Enforce N = 1
        if N != 1:
            kvetch_invalid("Need N = 1")

    def visit_Await(self, s: LoopIR.SyncStmt):
        # We do not enforce requirements on pairing, but we enforce other traits
        mem = self.barrier_type
        assert mem
        sync_type = s.sync_type
        sync_tl = sync_type.second_sync_tl
        N = sync_type.N
        assert sync_type.is_await()

        assert len(s.barriers) == 1
        e0 = s.home_barrier_expr()
        nm = e0.name
        traits = mem.traits()
        multicasts = s.multicasts()

        def kvetch_invalid(reason):
            raise ValueError(f"{s.srcinfo}: invalid {s}; {reason}")

        def kvetch_incompatible(thing):
            raise ValueError(
                f"{s.srcinfo}: incompatible {thing} with previous Await\n{old} @ {old.srcinfo}\n{s} @ {s.srcinfo}"
            )

        # Save new SyncInfo, or check with previously saved SyncInfo.
        # Must have identical sync-tl as any
        # other Awaits to the same queue barrier array.
        info = self.Await
        if info is None:
            info = SyncInfo(sync_tl, [s], N, N, multicasts)
            self.Await = info
        else:
            old = info.stmts[0]
            info.stmts.append(info)
            info.min_N = min(N, info.min_N)
            info.max_N = max(N, info.max_N)
            if info.sync_tl != sync_tl:
                kvetch_incompatible(f"sync-tl ({sync_tl})")

        # Enforce traits
        if traits.negative_await_N:
            assert not traits.non_negative_await_N
            if N >= 0:
                kvetch_invalid(f"{mem.name()} requires N < 0 (e.g. N = ~0)")
        elif traits.non_negative_await_N:
            if N < 0:
                kvetch_invalid(f"{mem.name()} requires N >= 0")
        else:
            if N != 0:
                kvetch_invalid(f"{mem.name()} requires N = 0")

        if traits.uniform_await_N and info.min_N != info.max_N:
            kvetch_incompatible(f"N ({mem.name()} uniform-N requirement)")

        # Enforce no multicast for any Await
        s.forbid_multicast("multicast is for Arrive, not Await")

    fence_multicasts = (False,)

    def _init_Fence_impl(self, s: LoopIR.SyncStmt):
        sync_type = s.sync_type
        assert not sync_type.is_split()
        # Decompose Fence
        self.Arrive = SyncInfo(
            sync_type.first_sync_tl, [s], 1, 1, self.fence_multicasts
        )
        self.Await = SyncInfo(
            sync_type.second_sync_tl, [s], 0, 0, self.fence_multicasts
        )
        self.guards = s.barriers[0].name
        self.guarded_by = s.barriers[0].name


class BarrierUsageAnalysis(LoopIR_Do):
    __slots__ = ["proc", "uses", "_explicit_guarded_by"]
    proc: LoopIR.proc
    uses: Dict[Sym, BarrierUsage]
    _explicit_guarded_by: Dict[Sym, Optional[Sym]]

    def __init__(self, proc):
        self.proc = proc
        self.uses = {}
        self._explicit_guarded_by = {}
        self.do_stmts(proc.body)

    def do_stmts(self, stmts):
        barriers_here = []
        for i, s in enumerate(stmts):
            barrier_type = self.do_s(s)
            if barrier_type is not None:
                barriers_here.append((s.name, barrier_type, i))

        # Check barriers declared in this scope now that the full
        # scope has been scanned. Ignore Fence(s) which are trivially
        # correct for our purposes here (only later in CUDA code
        # lowering can we meaningfully inspect sync-tl, CollTiling).
        for name, barrier_type, i in barriers_here:
            self.check_split_barrier(name, barrier_type, stmts, i)

    def do_s(self, s):
        if isinstance(s, LoopIR.Alloc):
            if s.type.is_barrier():
                mem = s.mem
                assert mem and issubclass(mem, BarrierType)
                assert s.name not in self.uses
                self.uses[s.name] = BarrierUsage(s)
                self.add_barrier_guard_edge(s)
                return mem  # Indicates to do_stmts() that we found a barrier decl
        elif isinstance(s, LoopIR.SyncStmt):
            sync_type: SyncType = s.sync_type
            # Arrive
            if sync_type.is_arrive():
                if not s.barriers:
                    raise ValueError(
                        f"{s.srcinfo}: {s} missing >> trailing barrier exprs"
                    )
                usage = self.uses.get(s.barriers[0].name)
                assert isinstance(usage, BarrierUsage)
                usage.visit_Arrive(s)
            # Await
            elif sync_type.is_await():
                assert len(s.barriers) == 1
                usage = self.uses.get(s.barriers[0].name)
                assert isinstance(usage, BarrierUsage)
                usage.visit_Await(s)
            # Fence
            else:
                assert (
                    len(s.barriers) == 1
                ), "exocc internal error: Fence internal barrier not initialized"
                nm = s.barriers[0].name
                assert (
                    nm not in self.uses
                ), "exocc internal error, invalid Fence Sym {nm!r}"
                self.uses[nm] = BarrierUsage(s)
        elif hasattr(s, "body"):
            super().do_s(s)
        return None

    def do_e(self, e):
        return None  # speed things up

    def add_barrier_guard_edge(self, s: LoopIR.Alloc):
        gb = s.type.guarded_by
        g_new = s.name
        self._explicit_guarded_by[g_new] = gb
        if gb is None:
            return
        gb_uses = self.uses[gb]
        gs = gb_uses.guards
        g_new_uses = self.uses[g_new]
        gs_uses = self.uses[gs]
        # We have an edge gb -> gs
        # Replace with gb -> g_new -> gs
        # Where x -> y means "x guards y"
        g_new_uses.guards = gs
        g_new_uses.guarded_by = gb
        gs_original_gb = self._explicit_guarded_by[gs]
        if gs_original_gb is not None:
            raise ValueError(
                f"{s.srcinfo}: {s}, cannot have guarded_by={gb} as it guards {gs} already"
            )
        assert gs_uses.guarded_by == gb
        gb_uses.guards = g_new
        gs_uses.guarded_by = g_new

    def check_split_barrier(
        self,
        name: Sym,
        barrier_type: Type[BarrierType],
        in_stmts: List[LoopIR.stmt],
        alloc_idx: int,
    ):
        usage: BarrierUsage = self.uses[name]
        assert not usage.is_fence()
        alloc = in_stmts[alloc_idx]
        assert isinstance(alloc, LoopIR.Alloc)
        assert alloc.name == name
        traits: BarrierTypeTraits = barrier_type.traits()

        # Boilerplate for missing Arrive/Await pairs.
        _arrive = usage.Arrive
        _await = usage.Await

        def kvetch_missing(info, whats_missing: str):
            s = info.stmts[0]
            raise ValueError(f"{s.srcinfo}: missing {whats_missing} for {s}")

        if _arrive is None:
            if _await is None:
                raise ValueError(
                    f"{alloc.srcinfo}: missing Arrive(...) >> +{name} and Await(+{name})"
                )
            else:
                kvetch_missing(_await, f"Arrive(...) >> +{name}")
        if _await is None:
            kvetch_missing(_arrive, f"Await(+{name}, ...)")

        # XXX TODO FIXME
        # # Check pairing requirements only if barrier type traits require it.
        # if traits.requires_pairing:
        #     self.check_pairing(name, traits, in_stmts, alloc_idx)

    def check_pairing(
        self,
        name: Sym,
        traits: BarrierTypeTraits,
        in_stmts: List[LoopIR.stmt],
        alloc_idx: int,
    ):
        # XXX this needs to be rewritten XXX
        usage: BarrierUsage = self.uses[name]
        has_back = usage.has_back_array()
        await_first = None  # Set to True/False when we know.
        nesting_level_if = 1  # Search for matched Arrive in if stmts, but not Await
        nesting_level_for = 2  # Never search for matches in for stmts.

        if has_back:
            # We only support {+/-}Await -> {-/+}Arrive for 2-way barriers
            await_first = True
        if traits.requires_arrive_first:
            assert not await_first
            await_first = False

        def paired_expected(s: LoopIR.SyncStmt):
            back = s.barriers[0].back ^ has_back
            sign = "-" if back else "+"
            if s.sync_type.is_await():
                return f"Arrive(...) >> {sign}{name}"
            else:
                return f"Await({sign}{name}, ...)"

        def recurse(
            sub_stmts: List[LoopIR.stmt],
            unmatched_sync: Optional[LoopIR.SyncStmt],
            nesting,
        ):
            nonlocal await_first

            # with statement and parallel-for loop bodies are inlined into the surrounding body.
            flattened_stmts = []

            def add_flatten(stmts):
                for s in stmts:
                    if is_if_holding_with(s, LoopIR):
                        add_flatten(s.body)
                    elif isinstance(s, LoopIR.For) and s.loop_mode.is_par():
                        add_flatten(s.body)
                    else:
                        flattened_stmts.append(s)

            add_flatten(sub_stmts)

            # Inspect flattened stmts.
            for s in flattened_stmts:
                if is_if_holding_with(s, LoopIR):  # Before if case
                    pass  # flattened
                elif isinstance(s, LoopIR.If):
                    sub_nesting = max(
                        nesting, 0 if unmatched_sync is None else nesting_level_if
                    )
                    # NB |= and not short-circuit `or`,
                    # as orelse body always needs to be scanned anyway
                    found_match = recurse(s.body, unmatched_sync, sub_nesting)
                    found_match |= recurse(s.orelse, unmatched_sync, sub_nesting)
                    if found_match:
                        unmatched_sync = None
                elif isinstance(s, LoopIR.For):
                    if s.loop_mode.is_par():
                        pass  # flattened
                    else:
                        sub_nesting = max(
                            nesting, 0 if unmatched_sync is None else nesting_level_for
                        )
                        recurse(s.body, unmatched_sync, sub_nesting)
                elif isinstance(s, LoopIR.SyncStmt) and s.barriers[0].name == name:
                    # Inspect SyncStmt that uses the `name` barrier.
                    sync_type = s.sync_type
                    assert sync_type.is_split()

                    # Nesting check (see comments)
                    if nesting:
                        # Forbid Arrive/Await in for loop when we had an unmatched sync outside
                        if nesting == nesting_level_for:
                            kwd = "for loop"
                            ok = False
                        # Forbid Await in if/else body when we had an unmatched sync outside
                        else:
                            assert nesting == nesting_level_if
                            kwd = "if/else"
                            ok = sync_type.is_arrive()
                        if not ok:
                            raise ValueError(
                                f"{s.srcinfo}: Invalid nesting; {s} inside {kwd} with unmatched {unmatched_sync} outside @ {unmatched_sync.srcinfo}"
                            )

                    # No unmatched statement.
                    # Retain this as the unmatched_sync; and check/update await_first.
                    if unmatched_sync is None:
                        if await_first is None:
                            await_first = sync_type.is_await()
                        elif await_first != sync_type.is_await():
                            expected = paired_expected(s)
                            raise ValueError(
                                f"{s.srcinfo}: {s} not paired with previous {expected} (note: those guarded by if/seq-for don't count)"
                            )
                        unmatched_sync = s

                    # Have unmatched statement (and nesting check OK)
                    # Look for exact matching statement
                    else:
                        expected = paired_expected(unmatched_sync)
                        matches = (
                            sync_type.is_arrive() == unmatched_sync.sync_type.is_await()
                        )
                        if has_back:
                            matches &= (
                                s.barriers[0].back != unmatched_sync.barriers[0].back
                            )
                        if not matches:
                            raise ValueError(
                                f"{s.srcinfo}: expected {expected} to match {unmatched_sync} @ {unmatched_sync.srcinfo}"
                            )
                        unmatched_sync = None

                else:
                    assert not hasattr(s, "body")
            # end for s in flattened_stmts

            # Arrives must be matched with a subsequent Await,
            # but we forgive unmatched Awaits.
            # TODO rethink this if we stop supporting conditional Arrive.
            if unmatched_sync is not None and unmatched_sync.sync_type.is_arrive():
                raise ValueError(
                    f"{unmatched_sync.srcinfo}: {unmatched_sync} without corresponding Await({name}) in same block (not split by if/seq-for)"
                )

            return unmatched_sync is None  # -> found_match

        recurse(in_stmts[alloc_idx + 1 :], None, 0)
