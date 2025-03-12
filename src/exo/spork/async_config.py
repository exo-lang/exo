from typing import Optional
from .actor_kinds import ActorKind
from . import actor_kinds
from .base_with_context import BaseWithContext


class BaseAsyncConfig(BaseWithContext):
    """Base class for a configuration of an async block.

    For example, the derived CudaDeviceFunction configures a block of
    code to be interpreted as code lowered to a CUDA device function.

    At a minimum, the derived class must specify the ActorKind that
    the child statements execute with, the name of the device
    (compiler backend), and parent_actor_kind.

    """

    __slots__ = []

    def get_actor_kind(self):
        raise NotImplementedError()

    def get_device_name(self):
        raise NotImplementedError()

    def parent_actor_kind(self) -> ActorKind:
        """Controls allowed nesting of async blocks in other async blocks.

        The async block must appear in a code block with the given actor kind
        """
        raise NotImplementedError()


class CudaDeviceFunction(BaseAsyncConfig):
    __slots__ = ["blockDim", "clusterDim", "blocks_per_sm"]

    def __init__(self, blockDim: int, clusterDim: int = 1, blocks_per_sm: int = 1):
        assert isinstance(blockDim, int) and blockDim > 0
        assert isinstance(clusterDim, int) and clusterDim > 0
        self.blockDim = blockDim
        self.clusterDim = clusterDim
        self.blocks_per_sm = blocks_per_sm

    def get_actor_kind(self):
        return actor_kinds.cuda_sync  # Synchronous (non-async) CUDA instr

    def get_device_name(self):
        return "cuda"

    def parent_actor_kind(self):
        return actor_kinds.cpu

    def __repr__(self):
        return f"CudaDeviceFunction({self.blockDim}, {self.clusterDim}, {self.blocks_per_sm})"

    def __eq__(self, other):
        return (
            type(other) == CudaDeviceFunction
            and self.blockDim == other.blockDim
            and self.clusterDim == other.clusterDim
            and self.blocks_per_sm == other.blocks_per_sm
        )


class CudaAsync(BaseAsyncConfig):
    __slots__ = ["_actor_kind"]

    def __init__(self, actor_kind):
        assert actor_kind in [
            actor_kinds.Sm80_cp_async,
            actor_kinds.tma_to_smem_async,
            actor_kinds.tma_to_gmem_async,
            actor_kinds.wgmma_async,
        ]
        self._actor_kind = actor_kind

    def get_actor_kind(self):
        return self._actor_kind

    def get_device_name(self):
        return "cuda"

    def parent_actor_kind(self):
        return actor_kinds.cuda_classic

    def __repr__(self):
        return f"CudaAsync({self._actor_kind})"

    def __eq__(self, other):
        return type(other) == CudaAsync and self._actor_kind == other._actor_kind
