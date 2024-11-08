from __future__ import annotations

from enum import Enum
from typing import Optional, Set


"""Actor signatures

Unusually for Python, we just use a bitfield for this.
This is to match with my planned underlying dynamic checker.
"""
sig_cpu = 1
sig_cuda_sync = 2
sig_non_bulk_cp_async = 4
sig_tma_to_shared = 8
sig_tma_to_global = 16
sig_wgmma_reg_a = 32
sig_wgmma_reg_d = 64
sig_wgmma_mem = 128


class ActorKindCategory(Enum):
    SYNTHETIC = 0
    SYNC = 1
    ASYNC = 2


class ActorKind(object):
    name: str
    category: ActorKindCategory
    sigbits: int
    allowed_parent: Optional[ActorKind]

    def __init__(self, name, category, sigbits, allowed_parent=None):
        self.name = name
        self.category = category
        self.sigbits = sigbits
        self.allowed_parent = allowed_parent
        assert (category == ActorKindCategory.ASYNC) == name.endswith(
            "_async"
        ), "naming convention: names of async actor kinds must end with _async"
        assert category != ActorKindCategory.SYNTHETIC or allowed_parent is None

    def is_synthetic(self):
        return self.category == ActorKindCategory.SYNTHETIC

    def is_async(self):
        assert self.category != ActorKindCategory.SYNTHETIC
        return self.category == ActorKindCategory.ASYNC

    def allows_parent(self, parent: ActorKind):
        return parent is self or parent is self.allowed_parent

    def __repr__(self):
        return f"<exo.spork.actor_kind.ActorKind {self.name}>"

    def __str__(self):
        return self.name

    def __bool__(self):
        return self.sigbits != 0


"""No instructions; internal use only"""
_null_actor = ActorKind("_null_actor", ActorKindCategory.SYNTHETIC, 0)

"""Host CPU instructions"""
cpu = ActorKind("cpu", ActorKindCategory.SYNC, sig_cpu)

"""All actions on the CUDA device"""
cuda_all = ActorKind(
    "cuda_all",
    ActorKindCategory.SYNTHETIC,
    sig_cuda_sync
    | sig_non_bulk_cp_async
    | sig_tma_to_shared
    | sig_tma_to_global
    | sig_wgmma_reg_a
    | sig_wgmma_reg_d
    | sig_wgmma_mem,
)

"""Typical CUDA instructions that operate on the generic proxy
and follow the typical per-thread in-order execution abstraction"""
cuda_sync = ActorKind("cuda_sync", ActorKindCategory.SYNC, sig_cuda_sync, cpu)

"""Ampere cp.async instructions"""
non_bulk_cp_async = ActorKind(
    "non_bulk_cp_async", ActorKindCategory.ASYNC, sig_non_bulk_cp_async, cuda_sync
)

"""CUDA generic proxy (sync and async instructions)"""
cuda_generic = ActorKind(
    "cuda_generic", ActorKindCategory.SYNTHETIC, sig_cuda_sync | sig_non_bulk_cp_async
)

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_shared_async = ActorKind(
    "tma_to_shared_async", ActorKindCategory.ASYNC, sig_tma_to_shared, cuda_sync
)

"""cp{.reduce}.bulk.async instructions with global memory as destination"""
tma_to_global_async = ActorKind(
    "tma_to_global_async", ActorKindCategory.ASYNC, sig_tma_to_global, cuda_sync
)

"""wgmma instructions"""
wgmma_async = ActorKind(
    "wgmma_async",
    ActorKindCategory.ASYNC,
    sig_wgmma_reg_a | sig_wgmma_reg_d | sig_wgmma_mem,
    cuda_sync,
)

"""wgmma instructions' actions on registers"""
wgmma_async_reg = ActorKind(
    "wgmma_async_reg", ActorKindCategory.SYNTHETIC, sig_wgmma_reg_a | sig_wgmma_reg_d
)

"""wgmma instructions' actions on shared memory"""
wgmma_async_mem = ActorKind(
    "wgmma_async_mem", ActorKindCategory.SYNTHETIC, sig_wgmma_mem
)

"""actions on wgmma matrix tile registers, either by wgmma.async
instructions or by ordinary cuda synchronous instructions"""
wgmma_reg = ActorKind(
    "wgmma_reg",
    ActorKindCategory.SYNTHETIC,
    sig_cuda_sync | sig_wgmma_reg_a | sig_wgmma_reg_d,
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
        tma_to_shared_async,
        tma_to_global_async,
        wgmma_async,
        wgmma_async_reg,
        wgmma_async_mem,
        wgmma_reg,
    ]
}
