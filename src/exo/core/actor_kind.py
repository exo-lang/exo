from __future__ import annotations

from typing import Optional


class ActorKind(object):
    name: str
    is_cuda: bool
    is_cuda_async: bool
    allowed_parent: Optional[ActorKind]

    def __init__(self, name, is_cuda, is_cuda_async, allowed_parent=None):
        self.name = name
        self.is_cuda = is_cuda
        self.is_cuda_async = is_cuda_async
        self.allowed_parent = allowed_parent

    def allows_parent(self, parent: ActorKind):
        return parent is self or parent is self.allowed_parent

    def __repr__(self):
        return f"<exo.core.actor_kind.ActorKind {self.name}>"

    def __str__(self):
        return self.name


"""Host CPU instructions"""
cpu = ActorKind("cpu", False, False)

"""Typical CUDA instructions that operate on the generic proxy
and follow the typical in-order execution abstraction"""
cuda_generic = ActorKind("cuda_generic", True, False, cpu)

"""Ampere cp.async instructions"""
non_bulk_cp_async = ActorKind("non_bulk_cp_async", True, True, cuda_generic)

"""cp.async.bulk instructions with cluster/block shared memory as destination"""
tma_to_shared = ActorKind("tma_to_shared", True, True, cuda_generic)

"""cp.async.bulk instructions with global memory as destination"""
tma_to_global = ActorKind("tma_to_global", True, True, cuda_generic)

"""wgmma instructions"""
wgmma = ActorKind("wgmma", True, True, cuda_generic)

"""Placeholder, represents no instructions"""
null_actor = ActorKind("null_actor", False, False)
