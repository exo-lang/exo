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

    def __str__(self):
        return self.name

    # Use default hash and equality (id-equality) from object.


sig_cpu = ActorSignature("sig_cpu")
sig_cuda_api = ActorSignature("sig_cuda_api")
sig_cuda_classic = ActorSignature("sig_cuda_classic")
sig_Sm80_cp_async = ActorSignature("sig_Sm80_cp_async")
sig_tma_to_smem = ActorSignature("sig_tma_to_smem")
sig_tma_to_gmem = ActorSignature("sig_tma_to_gmem")
sig_wgmma_rmem_a = ActorSignature("sig_wgmma_rmem_a")
sig_wgmma_rmem_d = ActorSignature("sig_wgmma_rmem_d")
sig_wgmma_smem = ActorSignature("sig_wgmma_smem")

cuda_device_signatures = {
    sig_cuda_classic,
    sig_Sm80_cp_async,
    sig_tma_to_smem,
    sig_tma_to_gmem,
    sig_wgmma_rmem_a,
    sig_wgmma_rmem_d,
    sig_wgmma_smem,
}


class ActorKind(object):
    name: str
    # A barrier is transitive iff its first actor kind is V1_transitive
    V1_transitive: bool
    full_signatures: Set[ActorSignature]
    temporal_signatures: Set[ActorSignature]

    __slots__ = ["name", "V1_transitive", "full_signatures", "temporal_signatures"]

    def __init__(
        self,
        name,
        V1_transitive,
        full_signatures: Set[ActorSignature],
        additional_temporal_signatures: Optional[Set[ActorSignature]] = None,
    ):
        assert isinstance(full_signatures, set)
        assert all(isinstance(s, ActorSignature) for s in full_signatures)
        self.name = name
        self.V1_transitive = V1_transitive
        self.full_signatures = full_signatures
        self.temporal_signatures = full_signatures
        if additional_temporal_signatures:
            assert isinstance(additional_temporal_signatures, set)
            assert all(
                isinstance(s, ActorSignature) for s in additional_temporal_signatures
            )
            self.temporal_signatures = full_signatures | additional_temporal_signatures

    def implements_first(self, other):
        """Is other "less-or-equally-featureful" than self as a first actor kind?

        Return whether the `other` actor kind is "implementable" with the
        `self` actor kind, i.e. that a hardware barrier implementing
        Fence(A1, A2) can be used to implement Fence(A1', A2)
        where A1 implements A1'.

        This is the case when A1' has a set of actor signatures that is a
        subset of that of A1, and A1' does not require a V1-transitivity
        promise A1 does not implement.

        NB in the current model, temporal_signatures does not really
        have an effect on V1, but we check anyway for future-proofing.

        """
        assert isinstance(other, ActorKind)
        return self.implements_second(other) and (
            self.V1_transitive or not other.V1_transitive
        )

    def implements_second(self, other):
        """Is other "less-or-equally-featureful" than self as a second actor kind?

        Return whether the `other` actor kind is "implementable" with the
        `self` actor kind, i.e. that a hardware barrier implementing
        Fence(A1, A2) can be used to implement Fence(A1, A2')
        where A2 implements A2'.
        """
        assert isinstance(other, ActorKind)
        return other.temporal_signatures.issubset(
            self.temporal_signatures
        ) and other.full_signatures.issubset(self.full_signatures)

    def __repr__(self):
        return f"<exo.spork.actor_kind.ActorKind {self.name}>"

    def __str__(self):
        return self.name

    def __bool__(self):
        return bool(self.temporal_signatures)


empty_actor_kind = ActorKind("empty_actor_kind", False, set())

"""Host CPU instructions"""
cpu = ActorKind("cpu", True, {sig_cpu})

"""All actions on the CUDA device

So-named because everything in CUDA participates in "API synchronization",
or the implicit ordering between actions done by different API calls
(cudaMemcpyAsync, kernel launch, etc.) on the same stream.
"""
cuda_api = ActorKind("cuda_api", True, {sig_cuda_api})

"""All actions on both the CPU and the CUDA device"""
cpu_cuda_api = ActorKind("cpu_cuda_api", True, {sig_cuda_api, sig_cpu})

"""Temporal-only CUDA device actions"""
cuda_temporal = ActorKind("cuda_temporal", False, set(), cuda_device_signatures)

"""Classic CUDA instructions that operate on the generic proxy
and follow the typical per-thread in-order execution abstraction.

Barriers awaiting with actor kind cuda_classic also carry
temporal-only dependencies (protecting against write-after-read
hazards)

"""
cuda_classic = ActorKind(
    "cuda_classic", True, {sig_cuda_classic}, cuda_device_signatures
)

"""Ampere cp.async instructions"""
Sm80_cp_async = ActorKind("Sm80_cp_async", False, {sig_Sm80_cp_async})

"""CUDA classic + sm_80 cp.async

These are operations that sm_90a+ retroactively term the generic proxy"""
Sm80_generic = ActorKind(
    "Sm80_generic", False, {sig_cuda_classic, sig_Sm80_cp_async}, cuda_device_signatures
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
    {sig_cuda_classic, sig_wgmma_rmem_a, sig_wgmma_rmem_d},
)

"""wgmma instructions' actions on registers;
this is the second actor kind of wgmma.fence"""
wgmma_fence_2 = ActorKind(
    "wgmma_fence_2",
    False,
    {sig_wgmma_rmem_a, sig_wgmma_rmem_d},
)

"""CUDA async proxy (TMA and wgmma, excluding register access)"""
cuda_async_proxy = ActorKind(
    "cuda_async_proxy",
    False,
    {sig_tma_to_smem, sig_tma_to_gmem, sig_wgmma_smem},
)

"""CUDA async proxy + wgmma register access"""
cuda_async_proxy_wgmma = ActorKind(
    "cuda_async_proxy_wgmma",
    False,
    {sig_tma_to_smem, sig_tma_to_gmem} | wgmma_async.full_signatures,
)

"""CUDA generic proxy + async proxy"""
cuda_generic_and_async_proxy = ActorKind(
    "cuda_generic_and_async_proxy",
    False,
    Sm80_generic.full_signatures | cuda_async_proxy.full_signatures,
    cuda_device_signatures,  # Temporal dependencies carried.
)


# Valid actor kinds for CudaAsync block
cuda_async_actor_kinds = (
    Sm80_cp_async,
    tma_to_smem_async,
    tma_to_gmem_async,
    wgmma_async,
)


class AnyActorKind(object):
    def __repr__(self):
        return "any_actor_kind"

    def __contains__(self, actor_kind):
        return isinstance(actor_kind, ActorKind)


any_actor_kind = AnyActorKind()
