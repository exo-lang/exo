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


class ActorKindCategory(Enum):
    SYNTHETIC = 0
    SYNC = 1
    ASYNC = 2


class ActorKind(object):
    name: str
    category: ActorKindCategory
    signatures: Set[ActorSignature]

    __slots__ = ["name", "category", "signatures"]

    def __init__(self, name, category, signatures: Set[ActorSignature]):
        self.name = name
        self.category = category
        self.signatures = signatures
        assert isinstance(signatures, set)
        assert all(isinstance(s, ActorSignature) for s in signatures)
        assert (category == ActorKindCategory.ASYNC) == name.endswith(
            "_async"
        ), "naming convention: names of async actor kinds must end with _async"

    def is_synthetic(self):
        return self.category == ActorKindCategory.SYNTHETIC

    def is_async(self):
        assert self.category != ActorKindCategory.SYNTHETIC
        return self.category == ActorKindCategory.ASYNC

    def subset_of(self, other):
        return self.signatures.issubset(other.signatures)

    def __repr__(self):
        return f"<exo.spork.actor_kind.ActorKind {self.name}>"

    def __str__(self):
        return self.name

    def __bool__(self):
        return bool(self.signatures)


"""No instructions; internal use only"""
_null_actor = ActorKind("_null_actor", ActorKindCategory.SYNTHETIC, set())

"""Host CPU instructions"""
cpu = ActorKind("cpu", ActorKindCategory.SYNC, {sig_cpu})

"""All actions on the CUDA device"""
cuda_all = ActorKind(
    "cuda_all",
    ActorKindCategory.SYNTHETIC,
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

"""Typical CUDA instructions that operate on the generic proxy
and follow the typical per-thread in-order execution abstraction"""
cuda_sync = ActorKind("cuda_sync", ActorKindCategory.SYNC, {sig_cuda_sync})

"""Ampere cp.async instructions"""
non_bulk_cp_async = ActorKind(
    "non_bulk_cp_async", ActorKindCategory.ASYNC, {sig_non_bulk_cp_async}
)

"""CUDA generic proxy (sync and async instructions)"""
cuda_generic = ActorKind(
    "cuda_generic", ActorKindCategory.SYNTHETIC, {sig_cuda_sync, sig_non_bulk_cp_async}
)

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_smem_async = ActorKind(
    "tma_to_smem_async", ActorKindCategory.ASYNC, {sig_tma_to_smem}
)

"""cp{.reduce}.bulk.async instructions with global memory as destination"""
tma_to_gmem_async = ActorKind(
    "tma_to_gmem_async", ActorKindCategory.ASYNC, {sig_tma_to_gmem}
)

"""wgmma instructions"""
wgmma_async = ActorKind(
    "wgmma_async",
    ActorKindCategory.ASYNC,
    {sig_wgmma_rmem_a, sig_wgmma_rmem_d, sig_wgmma_smem},
)

"""wgmma instructions' actions on registers"""
wgmma_async_rmem = ActorKind(
    "wgmma_async_rmem",
    ActorKindCategory.SYNTHETIC,
    {sig_wgmma_rmem_a, sig_wgmma_rmem_d},
)

"""wgmma instructions' actions on shared memory"""
wgmma_async_smem = ActorKind(
    "wgmma_async_smem", ActorKindCategory.SYNTHETIC, {sig_wgmma_smem}
)

"""actions on wgmma matrix tile registers, either by wgmma.async
instructions or by ordinary cuda synchronous instructions"""
wgmma_rmem = ActorKind(
    "wgmma_rmem",
    ActorKindCategory.SYNTHETIC,
    {sig_cuda_sync, sig_wgmma_rmem_a, sig_wgmma_rmem_d},
)


# _null_actor excluded
actor_kind_dict = {
    actor_kind.name: actor_kind
    for actor_kind in [
        cpu,
        cuda_all,
        cuda_sync,
        non_bulk_cp_async,
        cuda_generic,
        tma_to_smem_async,
        tma_to_gmem_async,
        wgmma_async,
        wgmma_async_rmem,
        wgmma_async_smem,
        wgmma_rmem,
    ]
}


class AnyActorKind(object):
    def __repr__(self):
        return "any_actor_kind"

    def __contains__(self, actor_kind):
        return isinstance(actor_kind, ActorKind)


any_actor_kind = AnyActorKind()
