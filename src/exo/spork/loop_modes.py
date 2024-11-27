from typing import Optional

from ..core.prelude import SrcInfo
from .actor_kinds import ActorKind, actor_kind_dict, cpu, cuda_sync
from . import lane_units


class LoopMode(object):
    is_par = False
    is_async = False

    def loop_mode_name(self):
        raise NotImplementedError()

    def new_actor_kind(self, old_actor_kind: ActorKind):
        raise NotImplementedError()

    def lane_unit(self):
        raise NotImplementedError()

    def _unpack_positive_int(self, value, name):
        if hasattr(value, "val"):
            value = value.val
        int_value = int(value)
        if int_value != value or int_value <= 0:
            raise ValueError(f"{name} must be positive integer")
        return int_value


class Seq(LoopMode):
    is_par = False
    is_async = False

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "seq"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        # Sequential for loop does not change actor kind
        return old_actor_kind


seq = Seq()


class Par(LoopMode):
    is_par = True
    is_async = False

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "par"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cpu

    def lane_unit(self):
        return lane_units.cpu_thread


par = Par()


class CudaClusters(LoopMode):
    is_par = True
    is_async = False

    blocks: int
    blockDim: int

    def __init__(self, blocks, blockDim):
        self.blocks = self._unpack_positive_int(blocks, "block count")
        self.blockDim = self._unpack_positive_int(blockDim, "blockDim")

    def loop_mode_name(self):
        return "cuda_clusters"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_cluster


class CudaBlocks(LoopMode):
    is_par = True
    is_async = False

    # Must be None if the for-blocks loops is a child of a for-clusters loop.
    # Otherwise this must be an integer.
    blockDim: Optional[int]

    def __init__(self, blockDim=None):
        if blockDim is None:
            self.blockDim = None
        else:
            self.blockDim = self._unpack_positive_int(blockDim, "blockDim")

    def loop_mode_name(self):
        return "cuda_blocks"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_block


cuda_blocks = CudaBlocks()


class CudaWarpgroups(LoopMode):
    is_par = True
    is_async = False

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warpgroups"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_warpgroup


cuda_warpgroups = CudaWarpgroups()


class CudaWarps(LoopMode):
    is_par = True
    is_async = False

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warps"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_warp


cuda_warps = CudaWarps()


class CudaThreads(LoopMode):
    is_par = True
    is_async = False

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_threads"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_thread


cuda_threads = CudaThreads()


def loop_mode_for_async_actor_kind(actor_kind):
    assert actor_kind.is_async()

    class _AsyncLoopMode(LoopMode):
        is_par = False
        is_async = True

        def __init__(self):
            pass

        def loop_mode_name(self):
            return actor_kind.name

        def new_actor_kind(self, old_actor_kind: ActorKind):
            return actor_kind

        def __repr__(self):
            return f"loop_mode_for_async_actor_kind({actor_kind})()"

    return _AsyncLoopMode


def make_loop_mode_dict():
    loop_mode_dict = {
        "seq": Seq,
        "par": Par,
        "cuda_clusters": CudaClusters,
        "cuda_blocks": CudaBlocks,
        "cuda_warpgroups": CudaWarpgroups,
        "cuda_warps": CudaWarps,
        "cuda_threads": CudaThreads,
    }

    # Allow use of names of async actor kinds as loop modes
    for name, actor_kind in actor_kind_dict.items():
        assert name == actor_kind.name
        if not actor_kind.is_synthetic() and actor_kind.is_async():
            loop_mode_dict[name] = loop_mode_for_async_actor_kind(actor_kind)
    return loop_mode_dict


loop_mode_dict = make_loop_mode_dict()


def format_loop_cond(lo_str: str, hi_str: str, loop_mode: LoopMode):
    strings = [loop_mode.loop_mode_name(), "(", lo_str, ",", hi_str]
    for attr in loop_mode.__dict__:
        strings.append(f",{attr}={getattr(loop_mode, attr)}")
    strings.append(")")
    return "".join(strings)
