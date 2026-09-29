from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Set
from functools import reduce
import operator


class Instr_tl(object):
    """The instruction timeline (instr-tl) is a property of each Exo @instr.

    This controls:
      * Whether the instruction is allowed in a given scope (see DeviceScope)
      * Qual_tl used for the instr parameters (function of Instr_tl x MemWin)

    This is not a critical concept: Qual_tl and DeviceScope together are the
    core of the timeline system. But in practice it's convenient for
    explanation to have a per-instr property; I may reconsider later.

    """

    __slots__ = ["_name"]
    _name: str

    def __init__(self, name: str):
        assert name.endswith("_instr"), "naming convention"
        self._name = name

    def __repr__(self):
        return self._name

    def is_cuda_async(self):
        return self in cuda_async_instr_tl

    def as_instr_tl(self):
        return self

    # Use default hash and equality (id-equality) from object.


"""Ordinary host CPU instructions"""
cpu_in_order_instr = Instr_tl("cpu_in_order_instr")

"""CPU calls CUDA API function that is stream ordered"""
cpu_cuda_stream_instr = Instr_tl("cpu_cuda_stream_instr")

"""Classic CUDA instructions that operate on the generic proxy
and follow the typical per-thread in-order execution abstraction."""
cuda_in_order_instr = Instr_tl("cuda_in_order_instr")

"""Ampere cp.async instructions"""
Sm80_cp_async_instr = Instr_tl("Sm80_cp_async_instr")

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_smem_async_instr = Instr_tl("tma_to_smem_async_instr")

"""cp{.reduce}.bulk.async instructions with global memory as destination"""
tma_to_gmem_async_instr = Instr_tl("tma_to_gmem_async_instr")

"""wgmma.mma_async instructions"""
wgmma_async_instr = Instr_tl("wgmma_async_instr")

"""tcgen05.mma instructions"""
tcgen05_mma_instr = Instr_tl("tcgen05_mma_instr")

"""tcgen05.cp instructions"""
tcgen05_cp_instr = Instr_tl("tcgen05_cp_instr")

"""tcgen05.cp instructions"""
tcgen05_shift_instr = Instr_tl("tcgen05_shift_instr")

"""tcgen05.ld instructions"""
tcgen05_ld_instr = Instr_tl("tcgen05_ld_instr")

"""tcgen05.st instructions"""
tcgen05_st_instr = Instr_tl("tcgen05_st_instr")

cuda_async_instr_tl = [
    Sm80_cp_async_instr,
    tma_to_smem_async_instr,
    tma_to_gmem_async_instr,
    wgmma_async_instr,
    tcgen05_mma_instr,
    tcgen05_cp_instr,
    tcgen05_shift_instr,
    tcgen05_ld_instr,
    tcgen05_st_instr,
]

cuda_basic_instr_tl = [cuda_in_order_instr] + cuda_async_instr_tl


class DeviceScope(object):
    """The DeviceScope is a static property of each Exo stmt in a proc.

    For now, all code is either CPU (cpu_basic_device) or CUDA (cuda_basic_device).
    The DeviceScope controls:

      * instructions allowed (based on the set of allowed instr-tl)
      * whether non-instr ("procedure") calls are allowed (cpu_basic_device only)
      * whether MemWin (memory type) allocation, read, write is allowed.
      * Qual_tl for non-instr memory access (indirectly, by default_instr_tl).

    """

    __slots__ = ["_name", "_default_instr_tl", "_instr_tl_set"]

    _name: str
    _default_instr_tl: Instr_tl
    _instr_tl_set: Set[Instr_tl]

    def __init__(self, name, default_instr_tl, instr_tl_set):
        self._name = name
        self._default_instr_tl = default_instr_tl
        self._instr_tl_set = set(instr_tl_set)

    def __repr__(self):
        return self._name

    def allows_instr_tl(self, instr_tl: Instr_tl):
        found = instr_tl in self._instr_tl_set
        assert found or isinstance(instr_tl, Instr_tl)
        return found

    def get_default_instr_tl(self):
        return self._default_instr_tl

    # Use default hash and equality (id-equality) from object.


cpu_basic_device = DeviceScope(
    "cpu_basic_device", cpu_in_order_instr, (cpu_in_order_instr, cpu_cuda_stream_instr)
)
cuda_basic_device = DeviceScope(
    "cuda_basic_device", cuda_in_order_instr, cuda_basic_instr_tl
)


class Qual_tl(object):
    """Property a specific access (read/mutate) on a memory location.

    This is not a property of a memory type or allocation as a whole;
    for example, SMEM could be written with tma_to_smem_async_qual
    and then read with wgmma_async_smem_qual.

    Each Qual_tl has an implementation class (impl_class)

      * "in_order": no restrictions.
      * "out_of_order": initial qual-tl of out-of-order accesses.
        Never in a precondition timeline set; may appear in a first sync-tl.
        Only VisRecords created by these are eligible for the
        out-of-order non-convergent abstract machine optimization.
      * "atomic": never an initial qual-tl, never in any sync-tl.
        Only in the precondition timeline set of atomic accesses,
        and added to new VisRecords of atomic accesses for all threads.

    and a proxy class (proxy_class), used for proxy fence codegen

      * "generic_ram": non-register memory accessed through the generic proxy.
      * "async_ram": non-register memory accessed through the async proxy.
      * "neither": registers, mbarriers, CPU, and stream-ordered accesses.
        NB cuda_mbarrier_qual must be "neither", and must never be in the
        qual-tl mask of data RAM, so that sync-check can't transitively
        carry generic -> async ordering without a codegen'd proxy fence.

    """

    __slots__ = [
        "_bit_index",
        "_bit",
        "_name",
        "_default_convergent_access",
        "_impl_class",
        "_proxy_class",
    ]
    _bit_index: int
    _bit: int
    _name: str
    _default_convergent_access: bool
    _impl_class: str
    _proxy_class: str

    _from_bit_index = []

    impl_classes = ("in_order", "out_of_order", "atomic")
    proxy_classes = ("generic_ram", "async_ram", "neither")

    def __init__(self, name, default_convergent_access, impl_class, proxy_class):
        assert name.endswith("_qual"), "naming convention"
        assert isinstance(default_convergent_access, bool)
        assert impl_class in self.impl_classes, impl_class
        assert proxy_class in self.proxy_classes, proxy_class
        self._bit_index = len(self._from_bit_index)
        assert self._bit_index <= 31, "camspork::qual_bits_t would overflow"
        self._bit = 1 << self._bit_index
        self._from_bit_index.append(self)
        self._name = name
        self._default_convergent_access = default_convergent_access
        self._impl_class = impl_class
        self._proxy_class = proxy_class

    def __repr__(self):
        return self._name

    def get_default_convergent_access(self):
        return self._default_convergent_access

    def as_bit(self):
        return self._bit

    def as_bit_index(self):
        return self._bit_index

    def impl_class(self) -> str:
        return self._impl_class

    def proxy_class(self) -> str:
        return self._proxy_class

    def out_of_order(self) -> bool:
        return self._impl_class == "out_of_order"

    def is_atomic(self) -> bool:
        return self._impl_class == "atomic"

    @staticmethod
    def make_bits(q) -> int:
        if isinstance(q, Qual_tl):
            return q.as_bit()
        else:
            bits = 0
            for qual_tl in q:
                bits |= qual_tl.as_bit()
            return bits

    @classmethod
    def get_all(cls) -> List[Qual_tl]:
        return cls._from_bit_index

    @classmethod
    def get_impl_class_bits(cls, impl_class: str) -> int:
        assert impl_class in cls.impl_classes, impl_class
        return cls.make_bits(
            q for q in cls._from_bit_index if q._impl_class == impl_class
        )

    @classmethod
    def get_proxy_class_bits(cls, proxy_class: str) -> int:
        assert proxy_class in cls.proxy_classes, proxy_class
        return cls.make_bits(
            q for q in cls._from_bit_index if q._proxy_class == proxy_class
        )

    @classmethod
    def get_all_out_of_order_bits(cls) -> int:
        return cls.get_impl_class_bits("out_of_order")

    @classmethod
    def get_all_atomic_bits(cls) -> int:
        return cls.get_impl_class_bits("atomic")

    # Use default hash and equality (id-equality) from object.


if True:
    # fmt: off
    cpu_in_order_qual = Qual_tl("cpu_in_order_qual", True, "in_order", "neither")
    cpu_cuda_stream_qual = Qual_tl("cpu_cuda_stream_qual", True, "in_order", "neither")
    cuda_in_order_rmem_qual = Qual_tl("cuda_in_order_rmem_qual", True, "in_order", "neither")
    cuda_in_order_ram_qual = Qual_tl("cuda_in_order_ram_qual", False, "in_order", "generic_ram")
    cuda_generic_atomic_qual = Qual_tl("cuda_generic_atomic_qual", False, "atomic", "generic_ram")
    cuda_mbarrier_qual = Qual_tl("cuda_mbarrier_qual", False, "in_order", "neither")
    # Non-bulk cp.async is generic proxy in sm_90+ terminology
    Sm80_cp_async_qual = Qual_tl("Sm80_cp_async_qual", False, "out_of_order", "generic_ram")
    tma_to_smem_async_qual = Qual_tl("tma_to_smem_async_qual", False, "out_of_order", "async_ram")
    tma_to_gmem_async_qual = Qual_tl("tma_to_gmem_async_qual", False, "out_of_order", "async_ram")
    tma_to_gmem_atomic_qual = Qual_tl("tma_to_gmem_atomic_qual", False, "atomic", "async_ram")
    wgmma_async_rmem_a_qual = Qual_tl("wgmma_async_rmem_a_qual", True, "out_of_order", "neither")
    # in-order relative to other wgmma, not to the thread's other instructions
    wgmma_async_rmem_d_qual = Qual_tl("wgmma_async_rmem_d_qual", True, "in_order", "neither")
    wgmma_rmem_fenced_qual = Qual_tl("wgmma_rmem_fenced_qual", True, "in_order", "neither")
    wgmma_async_smem_qual = Qual_tl("wgmma_async_smem_qual", False, "out_of_order", "async_ram")
    cuda_async_proxy_retired_qual = Qual_tl("cuda_async_proxy_retired_qual", False, "in_order", "async_ram")
    tcgen05_smem_qual = Qual_tl("tcgen05_smem_qual", False, "out_of_order", "async_ram")
    tcgen05_mma_tmem_qual = Qual_tl("tcgen05_mma_tmem_qual", True, "in_order", "neither")
    tcgen05_cp_tmem_qual = Qual_tl("tcgen05_cp_tmem_qual", True, "in_order", "neither")
    tcgen05_shift_qual = Qual_tl("tcgen05_shift_qual", True, "in_order", "neither")
    tcgen05_ld_qual = Qual_tl("tcgen05_ld_qual", True, "in_order", "neither")
    tcgen05_st_qual = Qual_tl("tcgen05_st_qual", True, "in_order", "neither")
    # fmt: on


@dataclass(slots=True)
class InstrQuals:
    """Value type of MemWin.qual_tl_dict

    initial: qual-tl of the VisRecord created by the access.
    precondition: precondition timeline set; the access is legal only if
    prior accesses' VisRecords are visible to the accessing thread
    with any of these qual-tl. Defaults to {initial}.

    Atomic qual-tl are configured per-instr (AtomicityInfo), not here.
    """

    initial: Qual_tl
    initial_bit: int
    precondition: Set[Qual_tl]  # really frozenset
    precondition_bits: int
    _all: Set[Qual_tl]

    def __init__(
        self, initial: Qual_tl, *, precondition: Qual_tl | List[Qual_tl] = None
    ):
        assert isinstance(initial, Qual_tl)
        assert not initial.is_atomic(), initial
        if precondition is None:
            precondition = frozenset((initial,))
        elif isinstance(precondition, Qual_tl):
            precondition = frozenset((precondition,))
        else:
            precondition = frozenset(precondition)

        assert precondition, "Empty precondition"
        for q in precondition:
            assert isinstance(q, Qual_tl), q
            assert not q.out_of_order(), q
            assert not q.is_atomic(), q

        self.initial = initial
        self.initial_bit = initial.as_bit()
        self.precondition = precondition = frozenset(precondition)
        self.precondition_bits = Qual_tl.make_bits(precondition)
        self._all = precondition | {initial}

    def __iter__(self):
        # Feeds into Qual_tl.make_bits() for autogenerating qual_tl_mask.
        return iter(self._all)


cuda_rmem_qual_tl_dict = {
    cuda_in_order_instr: InstrQuals(cuda_in_order_rmem_qual),
    Sm80_cp_async_instr: InstrQuals(cuda_in_order_rmem_qual),
    tma_to_smem_async_instr: InstrQuals(cuda_in_order_rmem_qual),
    tma_to_gmem_async_instr: InstrQuals(cuda_in_order_rmem_qual),
    # wgmma a/d has to be handled specially.
    #
    # cuda_in_order_rmem_qual is the precondition timeline set for tcgen05 access
    # so that we can wait for a prior read/write to a register to finish normally.
    tcgen05_ld_instr: InstrQuals(tcgen05_ld_qual, precondition=cuda_in_order_rmem_qual),
    tcgen05_st_instr: InstrQuals(tcgen05_st_qual, precondition=cuda_in_order_rmem_qual),
}

cuda_ram_qual_tl_dict = {
    cpu_in_order_instr: InstrQuals(cpu_in_order_qual),
    cpu_cuda_stream_instr: InstrQuals(cpu_cuda_stream_qual),
    cuda_in_order_instr: InstrQuals(cuda_in_order_ram_qual),
    Sm80_cp_async_instr: InstrQuals(
        Sm80_cp_async_qual, precondition=cuda_in_order_ram_qual
    ),
    tma_to_smem_async_instr: InstrQuals(
        tma_to_smem_async_qual, precondition=cuda_async_proxy_retired_qual
    ),
    tma_to_gmem_async_instr: InstrQuals(
        tma_to_gmem_async_qual, precondition=cuda_async_proxy_retired_qual
    ),
    wgmma_async_instr: InstrQuals(
        wgmma_async_smem_qual, precondition=cuda_async_proxy_retired_qual
    ),
    tcgen05_mma_instr: InstrQuals(
        tcgen05_smem_qual, precondition=cuda_async_proxy_retired_qual
    ),
    tcgen05_cp_instr: InstrQuals(
        tcgen05_smem_qual, precondition=cuda_async_proxy_retired_qual
    ),
}

# Somewhat broken qual-tl dict for tcgen05 TMEM (tensor memory).
# We use the precondition timeline set to model implicit pipelining.
# Precondition member q of an access means "prior access with initial
# qual-tl q -> this access" needs no synchronization.
cuda_tmem_qual_tl_dict = {
    tcgen05_mma_instr: InstrQuals(
        tcgen05_mma_tmem_qual,
        precondition=[
            tcgen05_mma_tmem_qual,  # tcgen05.mma -> tcgen05.mma
            tcgen05_cp_tmem_qual,  #  tcgen05.cp -> tcgen05.mma
            tcgen05_shift_qual,  #    tcgen05.shift -> tcgen05.mma
        ],
    ),
    tcgen05_cp_instr: InstrQuals(
        tcgen05_cp_tmem_qual,
        precondition=[
            tcgen05_mma_tmem_qual,  # tcgen05.mma -> tcgen05.cp
        ],
    ),
    tcgen05_shift_instr: InstrQuals(
        tcgen05_shift_qual,
        precondition=[
            tcgen05_mma_tmem_qual,  # tcgen05.mma -> tcgen05.shift
        ],
    ),
}

cuda_mbarrier_qual_tl_dict = {
    cuda_in_order_instr: InstrQuals(cuda_mbarrier_qual),
    Sm80_cp_async_instr: InstrQuals(cuda_mbarrier_qual),
    tma_to_smem_async_instr: InstrQuals(cuda_mbarrier_qual),
}


_cuda_in_order_quals = [
    cuda_in_order_rmem_qual,
    cuda_in_order_ram_qual,
    cuda_mbarrier_qual,
]
_Sm80_cp_async_quals = [
    Sm80_cp_async_qual,
    cuda_mbarrier_qual,
]
_tma_to_smem_async_quals = [tma_to_smem_async_qual]
_tma_to_gmem_async_quals = [tma_to_gmem_async_qual]
_wgmma_async_quals = [
    wgmma_async_rmem_a_qual,
    wgmma_async_rmem_d_qual,
    wgmma_async_smem_qual,
]
_tcgen05_async_quals = [
    tcgen05_mma_tmem_qual,
    tcgen05_cp_tmem_qual,
    tcgen05_shift_qual,
    tcgen05_ld_qual,
    tcgen05_st_qual,
]

_tcgen05_commit_quals = [
    tcgen05_mma_tmem_qual,
    tcgen05_cp_tmem_qual,
    tcgen05_shift_qual,
]

_cuda_device_quals = (
    _cuda_in_order_quals
    + [cpu_cuda_stream_qual, cuda_async_proxy_retired_qual]
    + _Sm80_cp_async_quals
    + _tma_to_smem_async_quals
    + _tma_to_gmem_async_quals
    + _wgmma_async_quals
    + _tcgen05_async_quals
)

_cuda_stream_quals = [
    cpu_cuda_stream_qual,
    cuda_in_order_ram_qual,
    Sm80_cp_async_qual,
    tma_to_smem_async_qual,
    tma_to_gmem_async_qual,
]


class Sync_tl(object):
    __slots__ = [
        "_name",
        "_timeline_set",
        "_timeline_set_bits",
        "_as_instr_tl",
    ]
    _name: str
    _timeline_set: Set[Qual_tl]
    _timeline_set_bits: int
    _as_instr_tl: Optional[Instr_tl]

    def __init__(
        self,
        name: str,
        timeline_set: List[Qual_tl],
        *,
        for_instr_tl: Optional[Instr_tl] = None,
    ):
        self._name = str(name)
        for tl in timeline_set:
            assert isinstance(tl, Qual_tl), tl
            # Atomic qual-tl must never be witnessed (first sync-tl) or
            # augmented (second sync-tl); camspork memoization relies on this.
            assert not tl.is_atomic(), tl
        self._timeline_set = frozenset(timeline_set)
        self._timeline_set_bits = Qual_tl.make_bits(self._timeline_set)
        self._as_instr_tl = for_instr_tl
        assert for_instr_tl is None or isinstance(for_instr_tl, Instr_tl)

    def __repr__(self):
        return f"<exo.spork.timelines.Sync_tl {self._name}>"

    def __str__(self):
        return self._name

    def as_instr_tl(self):
        instr_tl = self._as_instr_tl
        if instr_tl is None:
            raise TypeError(f"{self} is not an instr-tl")
        return self._as_instr_tl

    def get_timeline_set(self) -> Set[Qual_tl]:
        return self._timeline_set

    def get_timeline_set_bits(self) -> int:
        return self._timeline_set_bits

    def implements_first(self, other):
        """Is other "less-or-equally-featureful" than self as a first sync-tl?

        Return whether the `other` sync-tl is "implementable" with the
        `self` sync-tl, i.e. that a hardware barrier implementing
        Fence(self, L2) can be used to implement Fence(other, L2).

        """
        assert isinstance(other, Sync_tl)
        return self.implements_second(other)

    def implements_second(self, other):
        """Is other "less-or-equally-featureful" than self as a second sync-tl?

        Return whether the `other` sync-tl is "implementable" with the
        `self` sync-tl, i.e. that a hardware barrier implementing
        Fence(L1, self) can be used to implement Fence(L1, other).

        """
        assert isinstance(other, Sync_tl)
        self_bits = self._timeline_set_bits
        other_bits = other._timeline_set_bits
        return (self_bits & other_bits) == other_bits

    def disjoint_timeline_set(self, other):
        assert isinstance(other, Sync_tl)
        return 0 == (self._timeline_set_bits & other._timeline_set_bits)


def needs_proxy_fence(L1: Sync_tl, L2: Sync_tl) -> bool:
    """Whether fence.proxy.async must follow a sync with the given sync-tls

    True iff the first sync-tl contains any generic proxy RAM qual-tl and
    the second sync-tl contains any async proxy RAM qual-tl.
    See Qual_tl proxy_class.

    """
    generic_bits = Qual_tl.get_proxy_class_bits("generic_ram")
    async_bits = Qual_tl.get_proxy_class_bits("async_ram")
    return bool(
        (L1.get_timeline_set_bits() & generic_bits)
        and (L2.get_timeline_set_bits() & async_bits)
    )


empty_sync_tl = Sync_tl("empty_sync_tl", [])

"""Host in-order CPU instructions"""
cpu_in_order = Sync_tl(
    "cpu_in_order",
    [cpu_in_order_qual],
    for_instr_tl=cpu_in_order_instr,
)

"""First sync-tl of a cudaStreamSynchronize"""
cuda_stream_sync = Sync_tl("cuda_stream_sync", _cuda_device_quals)

"""Classic CUDA instructions that operate on the generic proxy
and follow the typical per-thread in-order execution abstraction."""
cuda_in_order = Sync_tl(
    "cuda_in_order",
    _cuda_in_order_quals,
    for_instr_tl=cuda_in_order_instr,
)

"""Ampere cp.async instructions"""
Sm80_cp_async = Sync_tl(
    "Sm80_cp_async", _Sm80_cp_async_quals, for_instr_tl=Sm80_cp_async_instr
)

"""CUDA in-order instructions + sm_80 cp.async

These are operations that sm_90a+ retroactively term the generic proxy"""
Sm80_generic = Sync_tl(
    "Sm80_generic",
    _cuda_in_order_quals + _Sm80_cp_async_quals,
)

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_smem_async = Sync_tl(
    "tma_to_smem_async",
    _tma_to_smem_async_quals,
    for_instr_tl=tma_to_smem_async_instr,
)

"""cp{.reduce}.bulk.async instructions with global memory as destination"""
tma_to_gmem_async = Sync_tl(
    "tma_to_gmem_async",
    _tma_to_gmem_async_quals,
    for_instr_tl=tma_to_gmem_async_instr,
)

"""wgmma instructions' actions on shared memory"""
wgmma_async_smem = Sync_tl("wgmma_async_smem", [wgmma_async_smem_qual])

"""actions on wgmma matrix tile registers, either by wgmma.async
instructions or by ordinary cuda synchronous instructions;
this is the first sync-tl of wgmma.fence"""
wgmma_fence_1 = Sync_tl(
    "wgmma_fence_1",
    [
        cuda_in_order_rmem_qual,
        wgmma_async_rmem_a_qual,
        wgmma_async_rmem_d_qual,
        wgmma_rmem_fenced_qual,
    ],
)

"""wgmma instructions' actions on registers;
this is the second sync-tl of wgmma.fence

NB wgmma_rmem_fenced_qual must not be in any other second sync-tl,
otherwise, some other sync could stand in for a missing wgmma.fence"""
wgmma_fence_2 = Sync_tl("wgmma_fence_2", [wgmma_rmem_fenced_qual])

"""wgmma instructions"""
wgmma_async = Sync_tl("wgmma_async", _wgmma_async_quals, for_instr_tl=wgmma_async_instr)

"""tcgen05.commit"""
tcgen05_commit = Sync_tl("tcgen05_commit", _tcgen05_commit_quals)

"""tcgen05.wait::ld"""
tcgen05_ld = Sync_tl("tcgen05_ld", [tcgen05_ld_qual], for_instr_tl=tcgen05_ld_instr)

"""tcgen05.wait::st"""
tcgen05_st = Sync_tl("tcgen05_st", [tcgen05_st_qual], for_instr_tl=tcgen05_st_instr)

"""mbarrier actions only

Typical usage: Arrive(cuda_mbarrier_only) for the "data ready" Arrive
after TMA loads with trailing barriers; the loaded data is synchronized
by the pending awaits, not by being witnessed by the Arrive."""
cuda_mbarrier_only = Sync_tl(
    "cuda_mbarrier_only",
    [cuda_mbarrier_qual],
)

"""mbarrier actions + memory accesses visible to the async proxy, only

Typical usage: first and second sync-tl for the "buffer free"
Arrive/Await before TMA overwrites SMEM previously read by
the async proxy (e.g. by wgmma)."""
cuda_async_proxy_retired = Sync_tl(
    "cuda_async_proxy_retired",
    [cuda_mbarrier_qual, cuda_async_proxy_retired_qual],
)

"""CUDA generic proxy + async proxy"""
cuda_generic_and_async_proxy = Sync_tl(
    "cuda_generic_and_async_proxy",
    _cuda_in_order_quals + [cuda_async_proxy_retired_qual],
)


def generate_latex_table(table_file, key_file):
    """Generate the SyncTL table (table_file) and QualTL key (key_file).

    These go into spork_b/SyncTLTable.tex and spork_b/QualTL.tex.
    tcgen05 is intentionally excluded for now.
    """
    # Row groups (by name prefix) are separated by horizontal lines.
    sync_tl_groups = [
        [empty_sync_tl],
        [cpu_in_order],
        [
            cuda_stream_sync,
            cuda_in_order,
            cuda_mbarrier_only,
            cuda_async_proxy_retired,
            cuda_generic_and_async_proxy,
        ],
        [Sm80_cp_async, Sm80_generic],
        [tma_to_smem_async, tma_to_gmem_async],
        [wgmma_async_smem, wgmma_fence_1, wgmma_fence_2, wgmma_async],
    ]
    # Column groups are separated by vertical lines in the table.
    # Atomic qual-tl are never in any sync-tl, so they are in the key only,
    # with no abbreviation.
    # fmt: off
    qual_tl_groups = [
        [
            (cpu_in_order_qual, "cpu", r"accessed by non-explicitly-async CPU instruction"),
            (cpu_cuda_stream_qual, "strm", r"accessed by stream-ordered CUDA API call (e.g. \lighttt{cudaMemcpyAsync})"),
        ],
        [
            (cuda_in_order_rmem_qual, "cuda1", r"register accessed by non-explicitly-async CUDA instruction"),
            (cuda_in_order_ram_qual, "cuda2", r"non-register accessed by non-explicitly-async CUDA instruction"),
            (cuda_mbarrier_qual, "mbar", r"usage of mbarriers"),
        ],
        [
            (Sm80_cp_async_qual, "Sm80", r"accessed by \lighttt{cp.async} instruction (non-bulk, i.e. not TMA)"),
            (tma_to_smem_async_qual, "g2s", r"accessed by \lighttt{cp.async.bulk} ``load'' instruction (GMEM$\to$SMEM)"),
            (tma_to_gmem_async_qual, "s2g", r"accessed by \lighttt{cp.async.bulk} ``store'' instruction (SMEM$\to$GMEM)"),
        ],
        [
            (wgmma_async_rmem_a_qual, "wgA", r"$A$ parameter in registers accessed by \lighttt{wgmma.mma\_async}"),
            (wgmma_async_rmem_d_qual, "wgD", r"$D$ parameter in registers accessed by \lighttt{wgmma.mma\_async}"),
            (wgmma_rmem_fenced_qual, "wgF", r"wgmma register operand ordered by \lighttt{wgmma.fence}"),
            (wgmma_async_smem_qual, "wgS", r"$A$ or $B$ parameter in SMEM accessed by \lighttt{wgmma.mma\_async}"),
        ],
        [
            (cuda_async_proxy_retired_qual, "async", r"retired memory access made visible to the async proxy"),
        ],
    ]
    atomic_qual_info = [
        (cuda_generic_atomic_qual, r"generic proxy atomic reduction (e.g. \lighttt{atomicAdd})"),
        (tma_to_gmem_atomic_qual, r"\lighttt{cp.reduce.async.bulk} atomic reduction (SMEM$\to$GMEM)"),
    ]
    # fmt: on

    def tex_name(x):
        return str(x).replace("_", "\\_")

    qual_tl_info = [info for group in qual_tl_groups for info in group]
    for q, abbrev, text in qual_tl_info:
        key_file.write(r"\texttt{%s} (%s): %s\\" % (tex_name(q), abbrev, text))
        key_file.write("\n")
    for q, text in atomic_qual_info:
        key_file.write(r"\texttt{%s}: %s\\" % (tex_name(q), text))
        key_file.write("\n")

    col_spec = "|".join("c" * len(group) for group in qual_tl_groups)
    table_file.write(r"\begin{tabular}{|r|%s|}" % col_spec)
    table_file.write("\n\\hline\n$\\tau_s$")
    for _, abbrev, _ in qual_tl_info:
        table_file.write(f" & {abbrev}")
    table_file.write(" \\\\\n")
    for group in sync_tl_groups:
        table_file.write("\\hline\n")
        for tau_s in group:
            table_file.write(r"\texttt{%s}" % tex_name(tau_s))
            for q, _, _ in qual_tl_info:
                if q.as_bit() & tau_s.get_timeline_set_bits():
                    table_file.write(r" & $\bullet$")
                else:
                    table_file.write(" & ")
            table_file.write("\\\\\n")
    table_file.write("\\hline\n\\end{tabular}\n")
