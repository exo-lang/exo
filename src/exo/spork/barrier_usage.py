from dataclasses import dataclass
from typing import Optional, List, Dict, Type, Tuple
from ..core.LoopIR import LoopIR, LoopIR_Rewrite
from ..core.memory import BarrierType
from .actor_kinds import ActorKind
from .sync_types import SyncType


@dataclass(slots=True)
class SyncInfo:
    actor_kind: ActorKind
    stmts: List[LoopIR.stmt]


@dataclass(slots=True)
class BarrierUsage:
    # None iff this barrier is a Fence()
    barrier_type: Optional[Type[BarrierType]] = None

    # Information for up to 4 types of SyncStmt.
    # Fence() stmts are decomposed as an Arrive+Await
    Arrive: Optional[SyncInfo] = None
    Await: Optional[SyncInfo] = None
    ReverseArrive: Optional[SyncInfo] = None
    ReverseAwait: Optional[SyncInfo] = None

    def is_fence(self):
        return self.barrier_type is None


class BarrierUsageAnalysis(LoopIR_Rewrite):
    __slots__ = ["uses"]

    def __init__(self):
        self.uses = {}

    def map_s(self, s):
        if isinstance(s, LoopIR.Alloc):
            if s.type.is_barrier():
                mem = s.mem
                assert mem and issubclass(mem, BarrierType)
                assert s.name not in self.uses
                self.uses[s.name] = BarrierUsage(mem)
        elif isinstance(s, LoopIR.SyncStmt):
            sync_type: SyncType = s.sync_type
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
                    setattr(usage, attr, SyncInfo(actor_kind, [s]))
                else:
                    if sync_info.actor_kind != actor_kind:
                        sus = sync_info.stmts[0]
                        raise ValueError(
                            f"{s.srcinfo}: {s} mismatches actor kind of {sus} at {sus.srcinfo}"
                        )
                    sync_info.stmts.append(s)
            # Fence, but we ignore any with stmt.lowered set as a
            # debug backdoor for now.
            elif s.lowered is not None:
                assert s.name not in self.uses
                usage = BarrierUsage(None)
                usage.Arrive = SyncInfo(sync_type.first_actor_kind, [s])
                usage.Await = SyncInfo(sync_type.second_actor_kind, [s])
                self.uses[s.name] = usage
                assert usage.is_fence()

        super().map_s(s)

    def map_e(self, e):
        return None

    def run(self, proc):
        proc = super().apply_proc(proc)
        return proc
