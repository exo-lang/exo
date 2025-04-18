from typing import Optional
from .actor_kinds import ActorKind
from . import actor_kinds


class SyncType(object):
    """Enconding for a specific type of synchronization statement.

    Non-split barriers are defined by their first actor kind and
    second actor kind.

    Split barriers (arrive and await) define only a first actor kind
    or second actor kind, respectively; the other actor kind
    is set to None; note, the reference to the barrier variable
    is stored separately in LoopIR.

    is_reversed = True converts Arrive/Await to ReverseArrive/ReverseAwait

    N: used to parameterize split barriers
      * Await with N >= 0: wait for all but the last N-many Arrives
      * Await with N < 0: wait for "matched" arrive; skip first ~N arrives
    TODO behavior for arrive
    """

    __slots__ = ["first_actor_kind", "second_actor_kind", "is_reversed", "N"]

    first_actor_kind: Optional[ActorKind]
    second_actor_kind: Optional[ActorKind]
    is_reversed: bool
    N: int

    def __init__(
        self,
        first_actor_kind: ActorKind,
        second_actor_kind: ActorKind,
        is_reversed: bool,
        N: int,
    ):
        assert first_actor_kind is None or isinstance(first_actor_kind, ActorKind)
        assert second_actor_kind is None or isinstance(second_actor_kind, ActorKind)
        assert first_actor_kind is not None or second_actor_kind is not None
        self.first_actor_kind = first_actor_kind
        self.second_actor_kind = second_actor_kind
        self.is_reversed = is_reversed
        self.N = N
        if self.is_split():
            assert isinstance(N, int)
        else:
            assert N == 0

    def __eq__(self, other):
        if not isinstance(other, SyncType):
            return False
        return (
            self.first_actor_kind == other.first_actor_kind
            and self.second_actor_kind == other.second_actor_kind
        )

    def __repr__(self):
        return f"exo.spork.sync_types.SyncType({self.first_actor_kind}, {self.second_actor_kind}, {self.is_reversed}, {self.N_str()})"

    def __str__(self):
        if self.is_arrive():
            return f"arrive_type({self.is_reversed}, {self.first_actor_kind})"
        if self.is_await():
            return f"await_type({self.is_reversed}, {self.second_actor_kind}, {self.N_str()})"
        return f"fence_type({self.first_actor_kind}, {self.second_actor_kind})"

    def N_str(self):
        N = self.N
        return str(N) if N >= 0 else f"~{~N}"

    def is_split(self):
        return self.first_actor_kind is None or self.second_actor_kind is None

    def is_arrive(self):
        return self.second_actor_kind is None

    def is_await(self):
        return self.first_actor_kind is None

    def format_stmt(self, bar, lowered=None):
        lowered_suffix = "" if lowered is None else f", lowered={lowered!r}"
        r = "Reverse" if self.is_reversed else ""
        if self.is_arrive():
            return f"{r}Arrive({self.first_actor_kind}, {bar}, {self.N_str()}{lowered_suffix})"
        elif self.is_await():
            return f"{r}Await({bar}, {self.second_actor_kind}, {self.N_str()}{lowered_suffix})"
        else:
            assert not self.is_reversed
            return f"Fence({self.first_actor_kind}, {self.second_actor_kind}{lowered_suffix})"


def fence_type(first_actor_kind: ActorKind, second_actor_kind: ActorKind):
    assert isinstance(first_actor_kind, ActorKind)
    assert isinstance(second_actor_kind, ActorKind)
    return SyncType(first_actor_kind, second_actor_kind, False, 0)


def arrive_type(is_reversed, first_actor_kind: ActorKind, N):
    assert isinstance(first_actor_kind, ActorKind)
    assert first_actor_kind
    return SyncType(first_actor_kind, None, is_reversed, N)


def await_type(is_reversed, second_actor_kind: ActorKind, N: int):
    assert isinstance(second_actor_kind, ActorKind)
    assert second_actor_kind
    return SyncType(None, second_actor_kind, is_reversed, N)


cuda_syncthreads = fence_type(actor_kinds.cuda_classic, actor_kinds.Sm80_generic)
wgmma_fence = fence_type(actor_kinds.wgmma_fence_1, actor_kinds.wgmma_fence_2)
fence_proxy_wgmma = fence_type(actor_kinds.cuda_classic, actor_kinds.wgmma_async_smem)
fence_proxy_tma = fence_type(actor_kinds.cuda_classic, actor_kinds.tma_to_gmem_async)
cuda_stream_synchronize = fence_type(actor_kinds.cuda_api, actor_kinds.cpu)
