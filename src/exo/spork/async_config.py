from typing import Optional
from .actor_kinds import ActorKind
from . import actor_kinds


class BaseAsyncConfig(object):
    """Base class for a configuration of an async block.

    For example, the derived CudaDeviceFunction configures a block of code to be interpreted as code lowered to a CUDA device function.

    At a minimum, the derived class must specify the ActorKind that the child statements execute with, the name of the device (compiler backend), and parent_async_type.
    """

    __slots__ = []

    def actor_kind(self):
        raise NotImplementedError()

    def device_name(self):
        raise NotImplementedError()

    def parent_async_type(self) -> Optional[type]:
        """Dictates allowed nesting of async blocks in other async blocks.

        If None, we require that the async block is not nested inside another (i.e. must be the child of top-level CPU code).
        Otherwise, parent_async_type must return a type object, and we require the parent async block have a configuration of that type.
        """
        raise NotImplementedError()


class CudaDeviceFunction(BaseAsyncConfig):
    __slots__ = ["block_size", "cluster_size"]

    def __init__(self, block_size : int, cluster_size : int = 1):
        assert isinstance(block_size, int) and block_size > 0
        assert isinstance(cluster_size, int) and cluster_size > 0
        self.block_size = block_size
        self.cluster_size = cluster_size

    def actor_kind(self):
        return actor_kinds.cuda_sync  # Synchronous (non-async) CUDA instr

    def device_name(self):
        return "cuda"

    def parent_async_type(self):
        return None

    def __repr__(self):
        return f"CudaDeviceFunction({self.block_size}, {self.cluster_size})"

    def __eq__(self, other):
        return type(other) == CudaDeviceFunction and self.block_size == other.block_size and self.cluster_size == other.cluster_size


class CudaAsync(BaseAsyncConfig):
    __slots__ = ["_actor_kind"]

    def __init__(self, actor_kind):
        assert actor_kind in [actor_kinds.non_bulk_cp_async, actor_kinds.tma_to_shared_async, actor_kinds.tma_to_global_async, actor_kinds.wgmma_async]
        self._actor_kind = actor_kind

    def actor_kind(self):
        return self._actor_kind

    def device_name(self):
        return "cuda"

    def parent_async_type(self):
        return CudaDeviceFunction

    def __repr__(self):
        return f"CudaAsync({self._actor_kind})"
    
    def __eq__(self, other):
        return type(other) == CudaAsync and self._actor_kind == other._actor_kind

