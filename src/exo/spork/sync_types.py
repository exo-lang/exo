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

    is_reversed = True converts Arrive/Await to ReverseArrive/ReverseAwait

    N: used to parameterize split barriers
      * Await with N >= 0: wait for all but the last N-many Arrives
      * Await with N < 0: wait for "matched" arrive; skip first ~N arrives
    TODO behavior for arrive
    """

    __slots__ = ["first_sync_tl", "second_sync_tl", "is_reversed", "N"]

    first_sync_tl: Optional[Sync_tl]
    second_sync_tl: Optional[Sync_tl]
    is_reversed: bool
    N: int

    def __init__(
        self,
        first_sync_tl: Sync_tl,
        second_sync_tl: Sync_tl,
        is_reversed: bool,
        N: int,
    ):
        assert first_sync_tl is None or isinstance(first_sync_tl, Sync_tl)
        assert second_sync_tl is None or isinstance(second_sync_tl, Sync_tl)
        assert first_sync_tl is not None or second_sync_tl is not None
        self.first_sync_tl = first_sync_tl
        self.second_sync_tl = second_sync_tl
        self.is_reversed = is_reversed
        self.N = N
        assert isinstance(N, int)

    def __eq__(self, other):
        if not isinstance(other, SyncType):
            return False
        return (
            self.first_sync_tl == other.first_sync_tl
            and self.second_sync_tl == other.second_sync_tl
        )

    def __repr__(self):
        return f"exo.spork.sync_types.SyncType({self.first_sync_tl}, {self.second_sync_tl}, {self.is_reversed}, {self.N_str()})"

    def __str__(self):
        if self.is_arrive():
            return f"arrive_type({self.is_reversed}, {self.first_sync_tl})"
        if self.is_await():
            return (
                f"await_type({self.is_reversed}, {self.second_sync_tl}, {self.N_str()})"
            )
        return f"fence_type({self.first_sync_tl}, {self.second_sync_tl})"

    def copy(self):
        return SyncType(
            self.first_sync_tl, self.second_sync_tl, self.is_reversed, self.N
        )

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
            r = "Reverse" if self.is_reversed else ""
            return r + ("Await" if self.is_await() else "Arrive")
        else:
            return "Fence"

    def paired_fname(self, has_reverse):
        """For pairing requirement

        Arrive <-> Await if no usage of ReverseArrive & ReverseAwait
        Otherwise, Arrive <-> ReverseAwait; ReverseArrive <-> Await

        """
        assert self.is_split()
        r = "Reverse" if has_reverse and not self.is_reversed else ""
        return r + ("Arrive" if self.is_await() else "Await")

    def format_stmt(self, bar, lowered=None):
        lowered_suffix = "" if lowered is None else f", lowered={lowered!r}"
        r = "Reverse" if self.is_reversed else ""
        if self.is_arrive():
            return f"{r}Arrive({self.first_sync_tl}, {bar}, {self.N_str()}{lowered_suffix})"
        elif self.is_await():
            return f"{r}Await({bar}, {self.second_sync_tl}, {self.N_str()}{lowered_suffix})"
        else:
            assert not self.is_reversed
            return f"Fence({self.first_sync_tl}, {self.second_sync_tl}{lowered_suffix})"


def fence_type(first_sync_tl: Sync_tl, second_sync_tl: Sync_tl):
    assert isinstance(first_sync_tl, Sync_tl)
    assert isinstance(second_sync_tl, Sync_tl)
    return SyncType(first_sync_tl, second_sync_tl, False, 0)


def arrive_type(is_reversed, first_sync_tl: Sync_tl, N):
    assert isinstance(first_sync_tl, Sync_tl)
    assert first_sync_tl
    return SyncType(first_sync_tl, None, is_reversed, N)


def await_type(is_reversed, second_sync_tl: Sync_tl, N: int):
    assert isinstance(second_sync_tl, Sync_tl)
    assert second_sync_tl
    return SyncType(None, second_sync_tl, is_reversed, N)
