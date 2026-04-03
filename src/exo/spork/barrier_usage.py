from dataclasses import dataclass
from typing import Optional, List, Dict, Type, Tuple

from ..core.LoopIR import LoopIR, LoopIR_Do
from ..core.memory import BarrierMechanism, BarrierMechanismTraits
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
    barrier_mechanism: Optional[Type[BarrierMechanism]]

    decl_stmt: LoopIR.stmt  # barrier alloc, or Fence

    # Information for Arrive/Await statements, split by usage.
    # Fence() stmts are decomposed as an Arrive + Await
    Arrive: Optional[SyncInfo]
    Await: Optional[SyncInfo]

    def __init__(self, s):
        self.decl_stmt = s
        self.Arrive = None
        self.Await = None
        if isinstance(s, LoopIR.SyncStmt):
            sync_type = s.sync_type
            assert not sync_type.is_split()
            self.barrier_mechanism = None
            self._init_Fence_impl(s)
        else:
            assert isinstance(s, LoopIR.Alloc)
            self.barrier_mechanism = s.mem
            assert issubclass(s.mem, BarrierMechanism)

    def get_srcinfo(self):
        return self.decl_stmt.srcinfo

    def is_fence(self):
        return self.barrier_mechanism is None

    def get_arrive(self) -> Optional[SyncInfo]:
        return self.Arrive

    def get_await(self) -> Optional[SyncInfo]:
        return self.Await

    def visit_Arrive(self, s: LoopIR.SyncStmt):
        mem = self.barrier_mechanism
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
        mem = self.barrier_mechanism
        assert mem
        sync_type = s.sync_type
        sync_tl = sync_type.second_sync_tl
        N = sync_type.N
        assert sync_type.is_await()

        # Enforce no multicast for any Await
        s.forbid_multicast("multicast is for Arrive, not Await")

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
        if traits.negative_await_N:  # TODO remove
            if N >= 0:
                kvetch_invalid(f"{mem.name()} requires N < 0")
        elif not traits.zero_await_N:
            if N < 0:
                kvetch_invalid(f"{mem.name()} requires N >= 0")
        else:
            if N != 0:
                kvetch_invalid(f"{mem.name()} requires N = 0")

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


class BarrierUsageAnalysis(LoopIR_Do):
    __slots__ = ["proc", "uses"]
    proc: LoopIR.proc
    uses: Dict[Sym, BarrierUsage]

    def __init__(self, proc):
        self.proc = proc
        self.uses = {}
        self.do_stmts(proc.body)

    def do_stmts(self, stmts):
        barriers_here = []
        for i, s in enumerate(stmts):
            barrier_mechanism = self.do_s(s)
            if barrier_mechanism is not None:
                barriers_here.append((s.name, barrier_mechanism, i))

        # Check barriers declared in this scope now that the full
        # scope has been scanned. Ignore Fence(s) which are trivially
        # correct for our purposes here (only later in CUDA code
        # lowering can we meaningfully inspect sync-tl, CollTiling).
        for name, barrier_mechanism, i in barriers_here:
            self.check_split_barrier(name, barrier_mechanism, stmts, i)

    def do_s(self, s):
        if isinstance(s, LoopIR.Alloc):
            if s.type.is_barrier():
                mem = s.mem
                assert mem and issubclass(mem, BarrierMechanism)
                assert s.name not in self.uses
                self.uses[s.name] = BarrierUsage(s)
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

    def check_split_barrier(
        self,
        name: Sym,
        barrier_mechanism: Type[BarrierMechanism],
        in_stmts: List[LoopIR.stmt],
        alloc_idx: int,
    ):
        usage: BarrierUsage = self.uses[name]
        assert not usage.is_fence()
        alloc = in_stmts[alloc_idx]
        assert isinstance(alloc, LoopIR.Alloc)
        assert alloc.name == name
        traits: BarrierMechanismTraits = barrier_mechanism.traits()

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
