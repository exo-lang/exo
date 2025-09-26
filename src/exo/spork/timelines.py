from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, List, Set


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
and follow the typical per-thread in-order execution abstraction.

Barriers awaiting with sync-tl cuda_in_order also carry
temporal-only dependencies (protecting against write-after-read
hazards)

"""
cuda_in_order_instr = Instr_tl("cuda_in_order_instr")

"""Ampere cp.async instructions"""
Sm80_cp_async_instr = Instr_tl("Sm80_cp_async_instr")

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_smem_async_instr = Instr_tl("tma_to_smem_async_instr")

"""cp{.reduce}.bulk.async instructions with global memory as destination"""
tma_to_gmem_async_instr = Instr_tl("tma_to_gmem_async_instr")

"""wgmma.mma_async instructions"""
wgmma_async_instr = Instr_tl("wgmma_async_instr")

"""Sets scale-d = 0 for the next wgmma.mma_async instr"""
wgmma_zero_instr = Instr_tl("wgmma_zero_instr")

"""tcgen05 instructions (TODO)"""
tcgen05_async_instr = Instr_tl("tcgen05_asyc_instr")

cuda_async_instr_tl = [
    Sm80_cp_async_instr,
    tma_to_smem_async_instr,
    tma_to_gmem_async_instr,
    wgmma_async_instr,
    tcgen05_async_instr,
]

cuda_basic_instr_tl = [cuda_in_order_instr, wgmma_zero_instr] + cuda_async_instr_tl


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

    """

    __slots__ = ["_bit_index", "_bit", "_name"]
    _bit_index: int
    _bit: int
    _name: str

    _from_bit_index = []

    def __init__(self, name):
        assert name.endswith("_qual"), "naming convention"
        self._bit_index = len(self._from_bit_index)
        assert self._bit_index <= 31, "camspork::qual_bits_t would overflow"
        self._bit = 1 << self._bit_index
        self._from_bit_index.append(self)
        self._name = name

    def __repr__(self):
        return self._name

    def as_bit(self):
        return self._bit

    def as_bit_index(self):
        return self._bit_index

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

    # Use default hash and equality (id-equality) from object.


cpu_in_order_qual = Qual_tl("cpu_in_order_qual")
cpu_cuda_stream_qual = Qual_tl("cpu_cuda_stream_qual")
cuda_in_order_rmem_qual = Qual_tl("cuda_in_order_rmem_qual")
cuda_in_order_ram_qual = Qual_tl("cuda_in_order_ram_qual")
Sm80_cp_async_qual = Qual_tl("Sm80_cp_async_qual")
tma_to_smem_async_qual = Qual_tl("tma_to_smem_async_qual")
tma_to_gmem_async_qual = Qual_tl("tma_to_gmem_async_qual")
wgmma_async_rmem_a_qual = Qual_tl("wgmma_async_rmem_a_qual")
wgmma_async_rmem_d_qual = Qual_tl("wgmma_async_rmem_d_qual")
wgmma_async_smem_qual = Qual_tl("wgmma_async_smem_qual")
wgmma_zero_qual = Qual_tl("wgmma_zero_qual")
tcgen05_TODO_qual = Qual_tl("tcgen05_TODO_qual")


cuda_rmem_qual_tl_dict = {
    cuda_in_order_instr: cuda_in_order_rmem_qual,
    Sm80_cp_async_instr: cuda_in_order_rmem_qual,
    tma_to_smem_async_instr: cuda_in_order_rmem_qual,
    tma_to_gmem_async_instr: cuda_in_order_rmem_qual,
    wgmma_zero_instr: wgmma_zero_qual,  # wgmma a/d has to be handled specially.
}

cuda_ram_qual_tl_dict = {
    cpu_in_order_instr: cpu_in_order_qual,
    cpu_cuda_stream_instr: cpu_cuda_stream_qual,
    cuda_in_order_instr: cuda_in_order_ram_qual,
    Sm80_cp_async_instr: Sm80_cp_async_qual,
    tma_to_smem_async_instr: tma_to_smem_async_qual,
    tma_to_gmem_async_instr: tma_to_gmem_async_qual,
    wgmma_async_instr: wgmma_async_smem_qual,
}


_cuda_in_order_quals = [
    cuda_in_order_rmem_qual,
    cuda_in_order_ram_qual,
]
_Sm80_cp_async_quals = [Sm80_cp_async_qual]
_tma_to_smem_async_quals = [tma_to_smem_async_qual]
_tma_to_gmem_async_quals = [tma_to_gmem_async_qual]
_wgmma_async_quals = [
    wgmma_async_rmem_a_qual,
    wgmma_async_rmem_d_qual,
    wgmma_async_smem_qual,
]
_tcgen05_async_quals = [
    tcgen05_TODO_qual,  # Placeholder, for future tcgen05 work.
]

# Intentionally excludes wgmma_zero_qual
_cuda_device_quals = (
    _cuda_in_order_quals
    + [cpu_cuda_stream_qual]
    + _Sm80_cp_async_quals
    + _tma_to_smem_async_quals
    + _tma_to_gmem_async_quals
    + _wgmma_async_quals
    + _tcgen05_async_quals
)
_wgmma_rmem_quals = [
    wgmma_async_rmem_a_qual,
    wgmma_async_rmem_d_qual,
]
_cuda_async_proxy_quals = [
    tma_to_smem_async_qual,
    tma_to_gmem_async_qual,
    wgmma_async_smem_qual,
    tcgen05_TODO_qual,
]


class Sync_tl(object):
    __slots__ = [
        "_name",
        "_V1_transitive",
        "_full_timeline_set_bits",
        "_temporal_timeline_set_bits",
        "_as_instr_tl",
    ]
    _name: str
    _V1_transitive: bool
    _full_timeline_set_bits: int
    _temporal_timeline_set_bits: int
    _as_instr_tl: Optional[Instr_tl]

    def __init__(
        self,
        name: str,
        V1_transitive: bool,
        full_timeline_set: List[qual_tl],
        additional_temporal_timeline_set: List[qual_tl] = [],
        *,
        for_instr_tl: Optional[Instr_tl] = None,
    ):
        self._name = str(name)
        self._V1_transitive = bool(V1_transitive)

        tmp_bits = 0
        for tl in full_timeline_set:
            tmp_bits |= tl.as_bit()
        self._full_timeline_set_bits = tmp_bits
        for tl in additional_temporal_timeline_set:
            tmp_bits |= tl.as_bit()
        self._temporal_timeline_set_bits = tmp_bits
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

    def is_V1_transitive(self):
        return self._V1_transitive

    def get_full_timeline_set_bits(self):
        return self._full_timeline_set_bits

    def get_temporal_timeline_set_bits(self):
        return self._temporal_timeline_set_bits

    def implements_first(self, other):
        """Is other "less-or-equally-featureful" than self as a first sync-tl?

        Return whether the `other` sync-tl is "implementable" with the
        `self` sync-tl, i.e. that a hardware barrier implementing
        Fence(self, L2) can be used to implement Fence(other, L2).

        NB in the current model, temporal qual-tl does not really
        have an effect on V1, but we check anyway for future-proofing.

        """
        assert isinstance(other, Sync_tl)
        return self.implements_second(other) and (
            self._V1_transitive or not other._V1_transitive
        )

    def implements_second(self, other):
        """Is other "less-or-equally-featureful" than self as a second sync-tl?

        Return whether the `other` sync-tl is "implementable" with the
        `self` sync-tl, i.e. that a hardware barrier implementing
        Fence(L1, self) can be used to implement Fence(L1, other).

        """
        assert isinstance(other, Sync_tl)
        self_LF = self._full_timeline_set_bits
        other_LF = other._full_timeline_set_bits
        self_TF = self._temporal_timeline_set_bits
        other_TF = other._temporal_timeline_set_bits

        # L^F of other must be a subset of L^F of self
        # L^T of other must be a subset of L^T of self
        return (self_LF & other_LF) == other_LF and (self_TF & other_TF) == other_TF

    def disjoint_full_timeline_set(self, other):
        assert isinstance(other, Sync_tl)
        return 0 == (self._full_timeline_set_bits & other._full_timeline_set_bits)


empty_sync_tl = Sync_tl("empty_sync_tl", False, [])

"""Host in-order CPU instructions"""
cpu_in_order = Sync_tl(
    "cpu_in_order",
    True,
    [cpu_in_order_qual],
    for_instr_tl=cpu_in_order_instr,
)

"""First sync-tl of a cudaStreamSynchronize"""
cuda_stream_sync = Sync_tl("cuda_stream_sync", True, _cuda_device_quals)

"""Classic CUDA instructions that operate on the generic proxy
and follow the typical per-thread in-order execution abstraction.

Barriers awaiting with sync-tl cuda_in_order also carry
temporal-only dependencies (protecting against write-after-read
hazards)

"""
cuda_in_order = Sync_tl(
    "cuda_in_order",
    True,
    _cuda_in_order_quals,
    _cuda_device_quals,  # Temporal-only
    for_instr_tl=cuda_in_order_instr,
)

"""Temporal-only CUDA device actions"""
cuda_temporal = Sync_tl("cuda_temporal", False, [], _cuda_device_quals)

"""Ampere cp.async instructions"""
Sm80_cp_async = Sync_tl(
    "Sm80_cp_async", False, _Sm80_cp_async_quals, for_instr_tl=Sm80_cp_async_instr
)

"""CUDA classic + sm_80 cp.async

These are operations that sm_90a+ retroactively term the generic proxy"""
Sm80_generic = Sync_tl(
    "Sm80_generic",
    False,
    _cuda_in_order_quals + _Sm80_cp_async_quals,
    _cuda_device_quals,  # Temporal-only
)

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_smem_async = Sync_tl(
    "tma_to_smem_async",
    False,
    _tma_to_smem_async_quals,
    for_instr_tl=tma_to_smem_async_instr,
)

"""cp{.reduce}.bulk.async instructions with global memory as destination"""
tma_to_gmem_async = Sync_tl(
    "tma_to_gmem_async",
    False,
    _tma_to_gmem_async_quals,
    for_instr_tl=tma_to_gmem_async_instr,
)

"""wgmma instructions' actions on shared memory"""
wgmma_async_smem = Sync_tl("wgmma_async_smem", False, [wgmma_async_smem_qual])

"""actions on wgmma matrix tile registers, either by wgmma.async
instructions or by ordinary cuda synchronous instructions;
this is the first sync-tl of wgmma.fence"""
wgmma_fence_1 = Sync_tl(
    "wgmma_fence_1",
    False,
    [cuda_in_order_rmem_qual] + _wgmma_rmem_quals,
)

"""wgmma instructions' actions on registers;
this is the second sync-tl of wgmma.fence"""
wgmma_fence_2 = Sync_tl("wgmma_fence_2", False, _wgmma_rmem_quals)

"""wgmma instructions"""
wgmma_async = Sync_tl(
    "wgmma_async", False, _wgmma_async_quals, for_instr_tl=wgmma_async_instr
)

"""CUDA async proxy (TMA and wgmma, excluding register access)"""
cuda_async_proxy = Sync_tl("cuda_async_proxy", False, _cuda_async_proxy_quals)

"""CUDA async proxy + wgmma register access"""
cuda_async_proxy_wgmma = Sync_tl(
    "cuda_async_proxy_wgmma", False, _cuda_async_proxy_quals + _wgmma_rmem_quals
)

"""CUDA generic proxy + async proxy; temporal dependencies carried"""
cuda_generic_and_async_proxy = Sync_tl(
    "cuda_generic_and_async_proxy",
    False,
    _cuda_in_order_quals + _Sm80_cp_async_quals + _cuda_async_proxy_quals,
    _cuda_device_quals,  # Temporal-only
)
