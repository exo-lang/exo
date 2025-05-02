from dataclasses import dataclass
from typing import Optional, List, Dict, Type, Tuple

from ..core.LoopIR import LoopIR, LoopIR_Do
from ..core.memory import BarrierType, BarrierTypeTraits
from ..core.prelude import Sym
from .actor_kinds import ActorKind
from .base_with_context import is_if_holding_with
from .sync_types import SyncType


@dataclass(slots=True)
class SyncInfo:
    actor_kind: ActorKind
    stmts: List[LoopIR.stmt]
    min_N: int
    max_N: int

    def get_srcinfo(self):
        return self.stmts[0].srcinfo


@dataclass(slots=True)
class BarrierUsage:
    # None iff this barrier is a Fence()
    barrier_type: Optional[Type[BarrierType]]

    decl_stmt: LoopIR.stmt  # barrier alloc, or Fence

    # Information for up to 4 types of SyncStmt.
    # Fence() stmts are decomposed as an Arrive+Await
    Arrive: Optional[SyncInfo] = None
    Await: Optional[SyncInfo] = None
    ReverseArrive: Optional[SyncInfo] = None
    ReverseAwait: Optional[SyncInfo] = None

    def get_srcinfo(self):
        return self.decl_stmt.srcinfo

    def is_fence(self):
        return self.barrier_type is None

    def has_reverse(self):
        assert (self.ReverseArrive is None) == (self.ReverseAwait is None)
        return self.ReverseArrive is not None


class BarrierUsageAnalysis(LoopIR_Do):
    __slots__ = ["proc", "uses"]

    def __init__(self, proc):
        self.proc = proc
        self.uses = {}
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
        # lowering can we meaningfully inspect actor kind, CollTiling).
        for name, barrier_type, i in barriers_here:
            self.check_split_barrier(name, barrier_type, stmts, i)

    def do_s(self, s):
        if isinstance(s, LoopIR.Alloc):
            if s.type.is_barrier():
                mem = s.mem
                assert mem and issubclass(mem, BarrierType)
                assert s.name not in self.uses
                self.uses[s.name] = BarrierUsage(mem, s)
                return mem  # Indicates we found a barrier decl
        elif isinstance(s, LoopIR.SyncStmt):
            sync_type: SyncType = s.sync_type
            N = sync_type.N
            # Arrive/Await
            if sync_type.is_split():
                usage = self.uses.get(s.name)
                assert isinstance(usage, BarrierUsage)
                assert not usage.is_fence()
                # Set or update usage.Arrive, usage.Await
                # usage.ReverseArrive, usage.ReverseAwait
                if sync_type.is_arrive():
                    actor_kind = sync_type.first_actor_kind
                else:
                    assert sync_type.is_await()
                    actor_kind = sync_type.second_actor_kind
                attr = sync_type.fname()
                sync_info: SyncInfo = getattr(usage, attr)
                if sync_info is None:
                    setattr(usage, attr, SyncInfo(actor_kind, [s], N, N))
                else:
                    if sync_info.actor_kind != actor_kind:
                        sus = sync_info.stmts[0]
                        raise ValueError(
                            f"{s.srcinfo}: {s} mismatches actor kind of {sus} at {sus.srcinfo}"
                        )
                    sync_info.stmts.append(s)
                    sync_info.min_N = min(sync_info.min_N, N)
                    sync_info.max_N = max(sync_info.max_N, N)
            # Fence, but we ignore any with stmt.lowered set as a
            # debug backdoor for now.
            elif s.lowered is None:
                assert (
                    s.name not in self.uses
                ), "exocc internal error, invalid Fence Sym"
                usage = BarrierUsage(None, s)
                usage.Arrive = SyncInfo(sync_type.first_actor_kind, [s], 1, 1)
                usage.Await = SyncInfo(sync_type.second_actor_kind, [s], 0, 0)
                self.uses[s.name] = usage
                assert usage.is_fence()
        elif hasattr(s, "body"):
            super().do_s(s)
        return None

    def do_e(self, e):
        return None  # speed things up

    def check_n(
        self,
        s: LoopIR.SyncStmt,
        barrier_type: Type[BarrierType],
        traits: BarrierTypeTraits,
    ):
        sync_type = s.sync_type
        assert sync_type.is_split()
        N = sync_type.N
        requires = None

        if sync_type.is_arrive():
            if N != 1:
                if traits.negative_arrive:
                    if N != ~0:
                        requires = "N = 1 or N = ~0"
                else:
                    requires = "N = 1"
        else:
            if traits.negative_await:
                if N >= 0:
                    requires = "N <= ~0"
            else:
                if N < 0:
                    requires = "N >= 0"

        if requires:
            raise ValueError(
                f"{s.srcinfo}: {barrier_type.name()} requires {requires} in {s})"
            )

    def check_split_barrier(
        self,
        name: Sym,
        barrier_type: Type[BarrierType],
        in_stmts: List[LoopIR.stmt],
        i: int,
    ):
        usage: BarrierUsage = self.uses[name]
        assert not usage.is_fence()
        assert in_stmts[i].name == name
        traits: BarrierTypeTraits = barrier_type.traits()

        # Boilerplate for missing Arrive/Await,
        # or unpaired or unsupported ReverseArrive/ReverseAwait;
        # also check N values.
        if usage.Arrive is None and usage.Await is None:
            raise ValueError(
                f"{usage.decl_stmt.srcinfo}: missing Arrive({name}) and Await({name})"
            )
        if usage.Arrive is None:
            s = usage.Await.stmts[0]
            raise ValueError(f"{s.srcinfo}: {s} missing corresponding Arrive({name})")
        if usage.Await is None:
            s = usage.Arrive.stmts[0]
            raise ValueError(f"{s.srcinfo}: {s} missing corresponding Await({name})")
        for s in usage.Arrive.stmts:
            self.check_n(s, barrier_type, traits)
        for s in usage.Await.stmts:
            self.check_n(s, barrier_type, traits)
        if usage.ReverseArrive is not None:
            s = usage.ReverseArrive.stmts[0]
            if not traits.supports_reverse:
                raise ValueError(
                    f"{s.srcinfo}: ReverseArrive not supported by {name} @ {barrier_type.name()}"
                )
            if usage.ReverseAwait is None:
                raise ValueError(
                    f"{s.srcinfo}: {s} missing corresponding ReverseAwait({name})"
                )
            for s in usage.ReverseArrive.stmts:
                self.check_n(s, barrier_type, traits)
        if usage.ReverseAwait is not None:
            s = usage.ReverseAwait.stmts[0]
            if not traits.supports_reverse:
                raise ValueError(
                    f"{s.srcinfo}: ReverseAwait not supported by {name} @ {barrier_type.name()}"
                )
            if usage.ReverseArrive is None:
                raise ValueError(
                    f"{s.srcinfo}: {s} missing corresponding ReverseArrive({name})"
                )
            for s in usage.ReverseAwait.stmts:
                self.check_n(s, barrier_type, traits)

        # Check pairing requirements only if barrier type traits require it.
        if not traits.requires_pairing:
            return

        has_reverse = usage.has_reverse()
        await_first = None  # Set to True/False when we know.
        nesting_level_if = 1  # Search for matched Arrive in if stmts, but not Await
        nesting_level_for = 2  # Never search for matches in for stmts.

        if has_reverse:
            # We only support {Reverse}Await -> {!Reverse}Arrive for 2-way barriers
            await_first = True
        if traits.requires_arrive_first:
            assert not await_first
            await_first = False

        def paired_fname(sync_type):
            fname = "Await" if sync_type.is_arrive() else "Arrive"
            if has_reverse and not sync_type.is_reversed:
                fname = "Reverse" + fname
            return fname

        def recurse(
            sub_stmts: List[LoopIR.stmt],
            unmatched_sync: Optional[LoopIR.SyncStmt],
            nesting,
        ):
            nonlocal await_first

            # with statement bodies are inlined into the surrounding body.
            flattened_stmts = []

            def add_flatten(stmts):
                for s in stmts:
                    if is_if_holding_with(s, LoopIR):
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
                    # NB |= and not or, as orelse body always needs to be scanned anyway
                    found_match = recurse(s.body, unmatched_sync, sub_nesting)
                    found_match |= recurse(s.orelse, unmatched_sync, sub_nesting)
                    if found_match:
                        unmatched_sync = None
                elif isinstance(s, LoopIR.For):
                    sub_nesting = max(
                        nesting, 0 if unmatched_sync is None else nesting_level_for
                    )
                    recurse(s.body, unmatched_sync, sub_nesting)
                elif isinstance(s, LoopIR.SyncStmt) and s.name == name:
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
                        if sync_type.is_arrive() and sync_type.N == ~0:
                            # Arrive with N = ~0 special case: does not need to be matched.
                            pass
                        else:
                            if await_first is None:
                                await_first = sync_type.is_await()
                            elif await_first != sync_type.is_await():
                                expected = paired_fname(sync_type)
                                raise ValueError(
                                    f"{s.srcinfo}: {s} not paired with previous {expected} (note: those guarded by if/for don't count)"
                                )
                            unmatched_sync = s

                    # Have unmatched statement (and nesting check OK)
                    # Look for exact matching statement
                    else:
                        expected = paired_fname(unmatched_sync.sync_type)
                        if expected != sync_type.fname():
                            raise ValueError(
                                f"{s.srcinfo}: expected {expected} to match {unmatched_sync} @ {unmatched_sync.srcinfo}"
                            )
                        unmatched_sync = None

                else:
                    assert not hasattr(s, "body")
            # end for s in flattened_stmts

            # Arrives must be matched with a subsequent Await,
            # but we forgive unmatched Awaits.
            if unmatched_sync is not None and unmatched_sync.sync_type.is_arrive():
                raise ValueError(
                    f"{unmatched_sync.srcinfo}: {unmatched_sync} without corresponding Await({name}) in same block (not split by if/for)"
                )

            return unmatched_sync is None  # -> found_match

        recurse(in_stmts[i + 1 :], None, 0)
