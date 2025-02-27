from __future__ import annotations

from enum import Enum
from typing import Optional, Set


class ActorSignature(object):
    __slots__ = ["name"]
    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"<exo.spork.ActorSignature {self.name}>"

    # Use default hash and equality (id-equality) from object.


sig_cpu = ActorSignature("sig_cpu")
sig_cuda_sync = ActorSignature("sig_cuda_sync")
sig_non_bulk_cp_async = ActorSignature("sig_non_bulk_cp_async")
sig_tma_to_smem = ActorSignature("sig_tma_to_smem")
sig_tma_to_gmem = ActorSignature("sig_tma_to_gmem")
sig_wgmma_rmem_a = ActorSignature("sig_wgmma_rmem_a")
sig_wgmma_rmem_d = ActorSignature("sig_wgmma_rmem_d")
sig_wgmma_smem = ActorSignature("sig_wgmma_smem")


class ActorKind(object):
    name: str
    # A barrier is transitive iff its first actor kind is V1_transitive
    V1_transitive: bool
    signatures: Set[ActorSignature]

    __slots__ = ["name", "V1_transitive", "signatures"]

    def __init__(self, name, V1_transitive, signatures: Set[ActorSignature]):
        self.name = name
        self.V1_transitive = V1_transitive
        self.signatures = signatures
        assert isinstance(signatures, set)
        assert all(isinstance(s, ActorSignature) for s in signatures)

    def implements(self, other):
        """True when other is "less-featureful" than self (or equal)

        Return whether the `other` actor kind is "implementable" with the
        `self` actor kind, i.e. that a hardware barrier implementing
        Fence(A1, A2) can be used to implement Fence(A1', A2')
        where A1 and A2 implements A1' and A2' respectively.

        This is the case when A1' has a set of actor signatures that is a
        subset of that of A1, and A1' does not require a V1-transitivity
        promise A1 does not implement.
        """
        assert isinstance(other, ActorKind)
        return other.signatures.issubset(self.signatures) and (
            self.V1_transitive or not other.V1_transitive
        )

    def __repr__(self):
        return f"<exo.spork.actor_kind.ActorKind {self.name}>"

    def __str__(self):
        return self.name

    def __bool__(self):
        return bool(self.signatures)


"""No instructions; internal use only"""
_null_actor = ActorKind("_null_actor", False, set())

"""Host CPU instructions"""
cpu = ActorKind("cpu", True, {sig_cpu})

"""All actions on the CUDA device"""
cuda_all = ActorKind(
    "cuda_all",
    True,
    {
        sig_cuda_sync,
        sig_non_bulk_cp_async,
        sig_tma_to_smem,
        sig_tma_to_gmem,
        sig_wgmma_rmem_a,
        sig_wgmma_rmem_d,
        sig_wgmma_smem,
    },
)

"""All actions on both the CPU and the CUDA device"""
cpu_cuda_all = ActorKind("cpu_cuda_all", True, cuda_all.signatures | {sig_cpu})

"""Typical CUDA instructions that operate on the generic proxy
and follow the typical per-thread in-order execution abstraction"""
cuda_sync = ActorKind("cuda_sync", True, {sig_cuda_sync})

"""Ampere cp.async instructions"""
non_bulk_cp_async = ActorKind("non_bulk_cp_async", False, {sig_non_bulk_cp_async})

"""CUDA generic proxy (sync and async instructions)"""
cuda_generic = ActorKind("cuda_generic", False, {sig_cuda_sync, sig_non_bulk_cp_async})

"""CUDA async proxy (TMA and wgmma, excluding register access)"""
cuda_async_proxy = ActorKind(
    "cuda_async_proxy",
    False,
    {sig_tma_to_smem, sig_tma_to_gmem, sig_wgmma_smem},
)

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_smem_async = ActorKind("tma_to_smem_async", False, {sig_tma_to_smem})

"""cp{.reduce}.bulk.async instructions with global memory as destination"""
tma_to_gmem_async = ActorKind("tma_to_gmem_async", False, {sig_tma_to_gmem})

"""wgmma instructions"""
wgmma_async = ActorKind(
    "wgmma_async",
    False,
    {sig_wgmma_rmem_a, sig_wgmma_rmem_d, sig_wgmma_smem},
)

"""wgmma instructions' actions on shared memory"""
wgmma_async_smem = ActorKind("wgmma_async_smem", False, {sig_wgmma_smem})

"""actions on wgmma matrix tile registers, either by wgmma.async
instructions or by ordinary cuda synchronous instructions;
this is the first actor kind of wgmma.fence"""
wgmma_fence_1 = ActorKind(
    "wgmma_fence_1",
    False,
    {sig_cuda_sync, sig_wgmma_rmem_a, sig_wgmma_rmem_d},
)

"""wgmma instructions' actions on registers;
this is the second actor kind of wgmma.fence"""
wgmma_fence_2 = ActorKind(
    "wgmma_fence_2",
    False,
    {sig_wgmma_rmem_a, sig_wgmma_rmem_d},
)

# _null_actor excluded
actor_kind_dict = {
    actor_kind.name: actor_kind
    for actor_kind in [
        cpu,
        cuda_all,
        cpu_cuda_all,
        cuda_sync,
        non_bulk_cp_async,
        cuda_generic,
        tma_to_smem_async,
        tma_to_gmem_async,
        wgmma_async,
        wgmma_async_smem,
        wgmma_fence_1,
        wgmma_fence_2,
    ]
}


class AnyActorKind(object):
    def __repr__(self):
        return "any_actor_kind"

    def __contains__(self, actor_kind):
        return isinstance(actor_kind, ActorKind)


any_actor_kind = AnyActorKind()
