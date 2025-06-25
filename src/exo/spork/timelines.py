from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, List


class Instr_tl(object):
    __slots__ = ["_name"]
    _name: str

    def __init__(self, name: str):
        self._name = name

    def __repr__(self):
        return f"<exo.spork.timelines.Instr_tl {self._name}>"

    def __str__(self):
        return self._name

    # Use default hash and equality (id-equality) from object.


"""Host CPU instructions"""
cpu_in_order_instr = Instr_tl("cpu_in_order_instr")

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

"""wgmma instructions"""
wgmma_async_instr = Instr_tl("wgmma_async_instr")

cuda_async_instr_tl = [
    Sm80_cp_async_instr,
    tma_to_smem_async_instr,
    tma_to_gmem_async_instr,
    wgmma_async_instr,
]


class Usage_tl(object):
    __slots__ = ["_name"]
    _name: str

    def __init__(self, name: str):
        self._name = name

    def __repr__(self):
        return f"<exo.spork.timelines.Usage_tl {self._name}>"

    def __str__(self):
        return self._name

    # Use default hash and equality (id-equality) from object.


cpu_usage = Usage_tl("cpu_usage")
cuda_ram_usage = Usage_tl("cuda_ram_usage")
cuda_sync_rmem_usage = Usage_tl("cuda_sync_rmem_usage")
cuda_async_a_rmem_usage = Usage_tl("cuda_async_a_rmem_usage")
cuda_async_d_rmem_usage = Usage_tl("cuda_async_d_rmem_usage")


class Qual_tl(object):
    __slots__ = ["instr_tl", "usage_tl", "_bit_index", "_bit"]
    instr_tl: Instr_tl
    usage_tl: Usage_tl
    _bit_index: int
    _bit: int

    _from_bit_index = []
    _from_tuple = {}

    @classmethod
    def _alloc_bit(cls, _instr_tl: Instr_tl, _usage_tl: Usage_tl):
        assert isinstance(_instr_tl, Instr_tl)
        assert isinstance(_usage_tl, Usage_tl)
        tup = (_instr_tl, _usage_tl)
        assert tup not in cls._from_tuple

        bit_index = len(cls._from_tuple)
        _qual_tl = object.__new__(Qual_tl)
        _qual_tl.instr_tl = _instr_tl
        _qual_tl.usage_tl = _usage_tl
        _qual_tl._bit_index = bit_index
        _qual_tl._bit = 1 << bit_index
        cls._from_bit_index.append(_qual_tl)
        cls._from_tuple[tup] = _qual_tl

        assert len(cls._from_bit_index) == len(cls._from_tuple)

    def __new__(cls, _instr_tl: Instr_tl, _usage_tl: Usage_tl):
        try:
            return cls._from_tuple[(_instr_tl, _usage_tl)]
        except KeyError:
            assert isinstance(_instr_tl, Instr_tl)
            assert isinstance(_usage_tl, Usage_tl)
            # Implementors: register with qual_tl._alloc_bit if needed.
            raise ValueError(f"Unsupported: Qual_tl({_instr_tl}, {_usage_tl})")

    def __repr__(self):
        return f"Qual_tl({self.instr_tl}, {self.usage_tl})"

    def as_bit(self):
        return self._bit

    def as_bit_index(self):
        return self._bit_index


Qual_tl._alloc_bit(cpu_in_order_instr, cpu_usage)
Qual_tl._alloc_bit(cuda_in_order_instr, cuda_sync_rmem_usage)
Qual_tl._alloc_bit(cuda_in_order_instr, cuda_ram_usage)
Qual_tl._alloc_bit(Sm80_cp_async_instr, cuda_ram_usage)
Qual_tl._alloc_bit(tma_to_smem_async_instr, cuda_ram_usage)
Qual_tl._alloc_bit(tma_to_gmem_async_instr, cuda_ram_usage)
Qual_tl._alloc_bit(wgmma_async_instr, cuda_async_a_rmem_usage)
Qual_tl._alloc_bit(wgmma_async_instr, cuda_async_d_rmem_usage)
Qual_tl._alloc_bit(wgmma_async_instr, cuda_ram_usage)


_cuda_in_order_qual = [
    Qual_tl(cuda_in_order_instr, cuda_ram_usage),
    Qual_tl(cuda_in_order_instr, cuda_sync_rmem_usage),
]
_Sm80_cp_async_qual = [Qual_tl(Sm80_cp_async_instr, cuda_ram_usage)]
_tma_to_smem_async_qual = [Qual_tl(tma_to_smem_async_instr, cuda_ram_usage)]
_tma_to_gmem_async_qual = [Qual_tl(tma_to_gmem_async_instr, cuda_ram_usage)]
_wgmma_async_qual = [
    Qual_tl(wgmma_async_instr, cuda_async_a_rmem_usage),
    Qual_tl(wgmma_async_instr, cuda_async_d_rmem_usage),
    Qual_tl(wgmma_async_instr, cuda_ram_usage),
]
_cuda_device_qual = (
    _cuda_in_order_qual
    + _Sm80_cp_async_qual
    + _tma_to_smem_async_qual
    + _tma_to_gmem_async_qual
    + _wgmma_async_qual
)
_wgmma_rmem_qual = [
    Qual_tl(wgmma_async_instr, cuda_async_a_rmem_usage),
    Qual_tl(wgmma_async_instr, cuda_async_d_rmem_usage),
]
_cuda_async_proxy_qual = [
    Qual_tl(itl, cuda_ram_usage)
    for itl in (tma_to_smem_async_instr, tma_to_gmem_async_instr, wgmma_async_instr)
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

"""Host CPU instructions"""
cpu_in_order = Sync_tl(
    "cpu_in_order",
    True,
    [Qual_tl(cpu_in_order_instr, cpu_usage)],
    for_instr_tl=cpu_in_order_instr,
)

"""Classic CUDA instructions that operate on the generic proxy
and follow the typical per-thread in-order execution abstraction.

Barriers awaiting with sync-tl cuda_in_order also carry
temporal-only dependencies (protecting against write-after-read
hazards)

"""
cuda_in_order = Sync_tl(
    "cuda_in_order",
    True,
    _cuda_in_order_qual,
    _cuda_device_qual,  # Temporal-only
    for_instr_tl=cuda_in_order_instr,
)

"""Temporal-only CUDA device actions"""
cuda_temporal = Sync_tl("cuda_temporal", False, [], _cuda_device_qual)

"""Ampere cp.async instructions"""
Sm80_cp_async = Sync_tl(
    "Sm80_cp_async", False, _Sm80_cp_async_qual, for_instr_tl=Sm80_cp_async_instr
)

"""CUDA classic + sm_80 cp.async

These are operations that sm_90a+ retroactively term the generic proxy"""
Sm80_generic = Sync_tl(
    "Sm80_generic",
    False,
    _cuda_in_order_qual + _Sm80_cp_async_qual,
    _cuda_device_qual,  # Temporal-only
)

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_smem_async = Sync_tl(
    "tma_to_smem_async",
    False,
    _tma_to_smem_async_qual,
    for_instr_tl=tma_to_smem_async_instr,
)

"""cp{.reduce}.bulk.async instructions with global memory as destination"""
tma_to_gmem_async = Sync_tl(
    "tma_to_gmem_async",
    False,
    _tma_to_gmem_async_qual,
    for_instr_tl=tma_to_gmem_async_instr,
)

"""wgmma instructions' actions on shared memory"""
wgmma_async_smem = Sync_tl(
    "wgmma_async_smem", False, [Qual_tl(wgmma_async_instr, cuda_ram_usage)]
)

"""actions on wgmma matrix tile registers, either by wgmma.async
instructions or by ordinary cuda synchronous instructions;
this is the first sync-tl of wgmma.fence"""
wgmma_fence_1 = Sync_tl(
    "wgmma_fence_1",
    False,
    [Qual_tl(cuda_in_order_instr, cuda_sync_rmem_usage)] + _wgmma_rmem_qual,
)

"""wgmma instructions' actions on registers;
this is the second sync-tl of wgmma.fence"""
wgmma_fence_2 = Sync_tl("wgmma_fence_2", False, _wgmma_rmem_qual)

"""wgmma instructions"""
wgmma_async = Sync_tl(
    "wgmma_async", False, _wgmma_async_qual, for_instr_tl=wgmma_async_instr
)

"""CUDA async proxy (TMA and wgmma, excluding register access)"""
cuda_async_proxy = Sync_tl("cuda_async_proxy", False, _cuda_async_proxy_qual)

"""CUDA async proxy + wgmma register access"""
cuda_async_proxy_wgmma = Sync_tl(
    "cuda_async_proxy_wgmma", False, _cuda_async_proxy_qual + _wgmma_rmem_qual
)

"""CUDA generic proxy + async proxy; temporal dependencies carried"""
cuda_generic_and_async_proxy = Sync_tl(
    "cuda_generic_and_async_proxy",
    False,
    _cuda_in_order_qual + _Sm80_cp_async_qual + _cuda_async_proxy_qual,
    _cuda_device_qual,  # Temporal-only
)
