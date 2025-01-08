from __future__ import annotations

from ..core.prelude import Sym
from enum import Enum
from typing import Optional, Set


class ActorKindCategory(Enum):
    SYNTHETIC = 0
    SYNC = 1
    ASYNC = 2


class ActorKind(object):
    name: str
    category: ActorKindCategory
    subkinds: Set[ActorKind]
    allowed_parent: Optional[ActorKind]
    sym: Sym  # Questionable

    def __init__(self, name, category, subkinds, allowed_parent=None):
        self.name = name
        self.category = category
        self.subkinds = subkinds
        self.allowed_parent = allowed_parent
        self.sym = Sym(name)
        assert (category == ActorKindCategory.ASYNC) == name.endswith(
            "_async"
        ), "naming convention: names of async actor kinds must end with _async"
        assert category != ActorKindCategory.SYNTHETIC or allowed_parent is None

    def __contains__(self, other):
        return other in self.subkinds

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


"""No instructions; internal use only"""
_null_actor = ActorKind("_null_actor", ActorKindCategory.SYNTHETIC, set())

"""Host CPU instructions"""
cpu = ActorKind("cpu", ActorKindCategory.SYNC, set())

"""Typical CUDA instructions that operate on the generic proxy
and follow the typical per-thread in-order execution abstraction"""
cuda_sync = ActorKind("cuda_sync", ActorKindCategory.SYNC, set(), cpu)

"""Ampere cp.async instructions"""
non_bulk_cp_async = ActorKind(
    "non_bulk_cp_async", ActorKindCategory.ASYNC, set(), cuda_sync
)

"""CUDA generic proxy (sync and async instructions)"""
cuda_generic = ActorKind(
    "cuda_generic", ActorKindCategory.SYNTHETIC, {cuda_sync, non_bulk_cp_async}
)

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_shared_async = ActorKind(
    "tma_to_shared_async", ActorKindCategory.ASYNC, set(), cuda_sync
)

"""cp{.reduce}.bulk.async instructions with global memory as destination"""
tma_to_global_async = ActorKind(
    "tma_to_global_async", ActorKindCategory.ASYNC, set(), cuda_sync
)

"""wgmma instructions' actions on registers"""
wgmma_async_reg = ActorKind("wgmma_async_reg", ActorKindCategory.SYNTHETIC, set())

"""wgmma instructions' actions on shared memory"""
wgmma_async_mem = ActorKind("wgmma_async_mem", ActorKindCategory.SYNTHETIC, set())

"""wgmma instructions"""
wgmma_async = ActorKind(
    "wgmma_async",
    ActorKindCategory.ASYNC,
    {wgmma_async_reg, wgmma_async_mem},
    cuda_sync,
)

"""actions on wgmma matrix tile registers, either by wgmma.async
instructions or by ordinary cuda synchronous instructions"""
wgmma_reg = ActorKind(
    "wgmma_reg", ActorKindCategory.SYNTHETIC, {cuda_sync, wgmma_async_reg}
)

"""All actions on the CUDA device"""
cuda_all = ActorKind(
    "cuda_all",
    ActorKindCategory.SYNTHETIC,
    {
        cuda_sync,
        non_bulk_cp_async,
        cuda_generic,
        tma_to_shared_async,
        tma_to_global_async,
        wgmma_async,
        wgmma_async_reg,
        wgmma_async_mem,
        wgmma_reg,
    },
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
