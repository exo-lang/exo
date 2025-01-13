from typing import Optional
from .actor_kinds import ActorKind
from . import actor_kinds


class SyncType(object):
    """Enconding for a specific type of synchronization statement.

    Non-split barriers are defined by their first actor kind and
    second actor kind.

    Split barriers (arrive and await) define only a first actor kind
    or second actor kind, respectively; the other actor kind
    is set to _null_actor; note, the reference to the barrier variable
    is stored separately in LoopIR.

    """

    __slots__ = ["first_actor_kind", "second_actor_kind"]

    first_actor_kind: ActorKind
    second_actor_kind: ActorKind

    def __init__(self, first_actor_kind: ActorKind, second_actor_kind: ActorKind):
        assert isinstance(first_actor_kind, ActorKind)
        assert isinstance(second_actor_kind, ActorKind)
        assert first_actor_kind or second_actor_kind
        self.first_actor_kind = first_actor_kind
        self.second_actor_kind = second_actor_kind

    def __eq__(self, other):
        if not isinstance(other, SyncType):
            return False
        return (
            self.first_actor_kind == other.first_actor_kind
            and self.second_actor_kind == other.second_actor_kind
        )

    def __repr__(self):
        return f"exo.spork.sync_types.SyncType({self.first_actor_kind}, {self.second_actor_kind})"

    def __str__(self):
        if self.is_arrive():
            return f"arrive_type({self.first_actor_kind})"
        if self.is_await():
            return f"await_type({self.second_actor_kind})"
        return f"{self.first_actor_kind}->{self.second_actor_kind}"

    def is_split(self):
        return not self.first_actor_kind or not self.second_actor_kind

    def is_arrive(self):
        return not self.second_actor_kind

    def is_await(self):
        return not self.first_actor_kind

    def format_stmt(self, bar):
        if self.is_arrive():
            return f"Arrive({self.first_actor_kind}, {bar})"
        elif self.is_await():
            return f"Await({bar}, {self.second_actor_kind})"
        else:
            return f"Fence({self.first_actor_kind}, {self.second_actor_kind})"


def arrive_type(first_actor_kind: ActorKind):
    assert isinstance(first_actor_kind, ActorKind)
    assert first_actor_kind
    return SyncType(first_actor_kind, actor_kinds._null_actor)


def await_type(second_actor_kind: ActorKind):
    assert isinstance(second_actor_kind, ActorKind)
    assert second_actor_kind
    return SyncType(actor_kinds._null_actor, second_actor_kind)


cuda_syncthreads = SyncType(actor_kinds.cuda_sync, actor_kinds.cuda_generic)
wgmma_fence = SyncType(actor_kinds.wgmma_rmem, actor_kinds.wgmma_async_rmem)
fence_proxy_wgmma = SyncType(actor_kinds.cuda_sync, actor_kinds.wgmma_async_smem)
fence_proxy_tma = SyncType(actor_kinds.cuda_sync, actor_kinds.tma_to_gmem_async)
cuda_stream_synchronize = SyncType(actor_kinds.cuda_all, actor_kinds.cpu)
