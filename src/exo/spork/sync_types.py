from typing import Optional
from .timelines import Instr_tl, Sync_tl


class SyncType(object):
    """Enconding for a specific type of synchronization statement.

    Non-split barriers (Fence) are defined by their first sync-tl
    and second sync-tl (synchronization timeline).

    Split barriers (arrive and await) define only a first sync-tl
    or second sync-tl, respectively; the other sync-tl
    is set to None; note, the reference to the barrier variable
    is stored separately in LoopIR.

    N: used to parameterize split barriers
      * Await with N >= 0: wait for all but the last N-many Arrives
      * Await with N < 0: wait for "matched" arrive; skip first ~N arrives
      * Arrive expects N=1 always for now
    """

    __slots__ = ["first_sync_tl", "second_sync_tl", "N"]

    first_sync_tl: Optional[Sync_tl]
    second_sync_tl: Optional[Sync_tl]
    N: int

    # Often we will need to check that all Arrives on the same queue barrier
    # array are compatible on some way; same for all Awaits.
    # We will store info for such checking as a list of 4 elements,
    # one element each for the 4 combinations of (Arrive/Await) and (front/back)
    front_arrive_idx = 0  # Arrive on front queue barrier array (+name)
    front_await_idx = 1  # Await on front queue barrier array (+name)
    back_arrive_idx = 2  # Arrive on back queue barrier array (-name)
    back_await_idx = 3  # Await on back queue barrier array (-name)
    n_info_idx = 4

    def __init__(
        self,
        first_sync_tl: Sync_tl,
        second_sync_tl: Sync_tl,
        N: int,
    ):
        assert first_sync_tl is None or isinstance(first_sync_tl, Sync_tl)
        assert second_sync_tl is None or isinstance(second_sync_tl, Sync_tl)
        assert first_sync_tl is not None or second_sync_tl is not None
        self.first_sync_tl = first_sync_tl
        self.second_sync_tl = second_sync_tl
        self.N = N
        assert isinstance(N, int)

    def __eq__(self, other):
        if not isinstance(other, SyncType):
            return False
        return (
            self.first_sync_tl == other.first_sync_tl
            and self.second_sync_tl == other.second_sync_tl
            and self.N == other.N
        )

    def __repr__(self):
        return f"exo.spork.sync_types.SyncType({self.first_sync_tl}, {self.second_sync_tl}, {self.N_str()})"

    def __str__(self):
        if self.is_arrive():
            return f"arrive_type({self.first_sync_tl})"
        if self.is_await():
            return f"await_type({self.second_sync_tl}, {self.N_str()})"
        return f"fence_type({self.first_sync_tl}, {self.second_sync_tl})"

    def info_idx(self, back: bool, swap=False):
        assert self.is_split()
        return (swap ^ self.is_await()) + 2 * bool(back)

    def copy(self):
        return SyncType(self.first_sync_tl, self.second_sync_tl, self.N)

    def N_str(self):
        N = self.N
        return str(N) if N >= 0 else f"~{~N}"

    def is_split(self):
        return self.first_sync_tl is None or self.second_sync_tl is None

    def is_arrive(self):
        return self.second_sync_tl is None

    def is_await(self):
        return self.first_sync_tl is None

    def fname(self):
        """ "function" name used in Exo object code"""
        if self.is_split():
            return "Await" if self.is_await() else "Arrive"
        else:
            return "Fence"

    def format_stmt(self, barrier_exprs):
        if self.is_arrive():
            fragments = [f"Arrive({self.first_sync_tl}, {self.N_str()})"] + [
                f" >> {bar}" for bar in barrier_exprs
            ]
            return "".join(fragments)
        elif self.is_await():
            assert len(barrier_exprs) == 1
            bar = barrier_exprs[0]
            return f"Await({bar}, {self.second_sync_tl}, {self.N_str()})"
        else:
            return f"Fence({self.first_sync_tl}, {self.second_sync_tl})"


def fence_type(first_sync_tl: Sync_tl, second_sync_tl: Sync_tl):
    assert isinstance(first_sync_tl, Sync_tl)
    assert isinstance(second_sync_tl, Sync_tl)
    return SyncType(first_sync_tl, second_sync_tl, 0)


def arrive_type(first_sync_tl: Sync_tl, N):
    assert isinstance(first_sync_tl, Sync_tl)
    assert first_sync_tl
    return SyncType(first_sync_tl, None, N)


def await_type(second_sync_tl: Sync_tl, N: int):
    assert isinstance(second_sync_tl, Sync_tl)
    assert second_sync_tl
    return SyncType(None, second_sync_tl, N)
