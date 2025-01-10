from typing import Optional, Set

from ..core.prelude import SrcInfo
from . import collectives
from . import actor_kinds


class LoopMode(object):
    is_par = False
    is_async = False
    allowed_actor_kinds = set()

    def loop_mode_name(self):
        raise NotImplementedError()

    def collective_unit(self):
        raise NotImplementedError()


class Seq(LoopMode):
    is_par = False
    is_async = False
    allowed_actor_kinds = actor_kinds.any_actor_kind

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "seq"


seq = Seq()


class Par(LoopMode):
    is_par = True
    is_async = False
    allowed_actor_kinds = {actor_kinds.cpu}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "par"

    def collective_unit(self):
        return collectives.cpu_thread


par = Par()


class CudaClusters(LoopMode):
    is_par = True
    is_async = False
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_clusters"

    def collective_unit(self):
        return collectives.cuda_cluster


cuda_clusters = CudaClusters()


class CudaBlocks(LoopMode):
    is_par = True
    is_async = False
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_blocks"

    def collective_unit(self):
        return collectives.cuda_block


cuda_blocks = CudaBlocks()


class CudaWarpgroups(LoopMode):
    is_par = True
    is_async = False
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warpgroups"

    def collective_unit(self):
        return collectives.cuda_warpgroup


cuda_warpgroups = CudaWarpgroups()


class CudaWarps(LoopMode):
    is_par = True
    is_async = False
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warps"

    def collective_unit(self):
        return collectives.cuda_warp


cuda_warps = CudaWarps()


class CudaThreads(LoopMode):
    is_par = True
    is_async = False
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_threads"

    def collective_unit(self):
        return collectives.cuda_thread


cuda_threads = CudaThreads()


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
    return loop_mode_dict


loop_mode_dict = make_loop_mode_dict()


# NOTE, the kwargs are not really used right now 2025-01-09
def format_loop_cond(lo_str: str, hi_str: str, loop_mode: LoopMode):
    strings = [loop_mode.loop_mode_name(), "(", lo_str, ",", hi_str]
    for attr in loop_mode.__dict__:
        strings.append(f",{attr}={getattr(loop_mode, attr)}")
    strings.append(")")
    return "".join(strings)
