from typing import Optional

from ..core.prelude import SrcInfo
from . import actor_kind
from .actor_kind import ActorKind


class LoopMode(object):
    # None if the loop mode doesn't correspond to cuda.
    # Otherwise, this is the "index" of how nested this loop type
    # should be in the cuda programming model.
    # (e.g. thread's cuda_nesting > block's cuda_nesting)
    cuda_nesting: Optional[int]

    def loop_mode_name(self):
        raise NotImplemented

    def new_actor_kind(self, old_actor_kind: ActorKind):
        raise NotImplemented

    def cuda_can_nest_in(self, other):
        assert self.cuda_nesting is not None
        return other.cuda_nesting is not None and self.cuda_nesting > other.cuda_nesting

    def _unpack_positive_int(self, value, name):
        if hasattr(value, "val"):
            value = value.val
        int_value = int(value)
        if int_value != value or int_value <= 0:
            raise ValueError(f"{name} must be positive integer")
        return int_value


class Seq(LoopMode):
    cuda_nesting = None

    def __init__(self):
        self.cuda_nesting = None

    def loop_mode_name(self):
        return "seq"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        # Sequential for loop does not change actor kind
        return old_actor_kind


seq = Seq()


class Par(LoopMode):
    cuda_nesting = None

    def __init__(self):
        self.cuda_nesting = None

    def loop_mode_name(self):
        return "par"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return actor_kind.cpu


par = Par()


class CudaClusters(LoopMode):
    cuda_nesting = 2
    blocks: int

    def __init__(self, blocks):
        self.blocks = self._unpack_positive_int(blocks, "block count")

    def loop_mode_name(self):
        return "cuda_clusters"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return actor_kind.cuda_generic


class CudaBlocks(LoopMode):
    cuda_nesting = 3
    warps: int

    def __init__(self, warps=1):
        self.warps = self._unpack_positive_int(warps, "warp count")

    def loop_mode_name(self):
        return "cuda_blocks"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return actor_kind.cuda_generic


cuda_blocks = CudaBlocks()


class CudaWarpgroups(LoopMode):
    cuda_nesting = 4

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warpgroups"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return actor_kind.cuda_generic


cuda_warpgroups = CudaWarpgroups()


class CudaWarps(LoopMode):
    cuda_nesting = 5

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warps"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return actor_kind.cuda_generic


cuda_warps = CudaWarps()


class CudaThreads(LoopMode):
    cuda_nesting = 6

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_threads"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return actor_kind.cuda_generic


cuda_threads = CudaThreads()


class CudaAsync(LoopMode):
    actor_kind: ActorKind

    def __init__(self, actor_kind):
        self.cuda_nesting = None
        self.actor_kind = actor_kind
        if not isinstance(actor_kind, ActorKind):
            raise TypeError("Must paramaterize cuda_async loop with ActorKind")
        if not actor_kind.is_cuda_async:
            raise TypeError("ActorKind must be cuda async")

    def loop_mode_name(self):
        return "cuda_async"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return self.actor_kind


loop_mode_dict = {
    "seq": Seq,
    "par": Par,
    "cuda_clusters": CudaClusters,
    "cuda_blocks": CudaBlocks,
    "cuda_warpgroups": CudaWarpgroups,
    "cuda_warps": CudaWarps,
    "cuda_threads": CudaThreads,
    "cuda_async": CudaAsync,
}


def format_loop_cond(lo_str: str, hi_str: str, loop_mode: LoopMode):
    strings = [loop_mode.loop_mode_name(), "(", lo_str, ",", hi_str]
    for attr in loop_mode.__dict__:
        if attr != "cuda_nesting":
            strings.append(f",{attr}={getattr(loop_mode, attr)}")
    strings.append(")")
    return "".join(strings)
