from typing import Optional, Set

from ..core.prelude import SrcInfo
from . import collectives
from . import actor_kinds


class LoopMode(object):
    is_par = False
    allowed_actor_kinds = set()

    def loop_mode_name(self):
        raise NotImplementedError()

    def collective_unit(self):
        raise NotImplementedError()


class Seq(LoopMode):
    is_par = False
    allowed_actor_kinds = actor_kinds.any_actor_kind

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "seq"


seq = Seq()


class Par(LoopMode):
    is_par = True
    allowed_actor_kinds = {actor_kinds.cpu}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "par"

    def collective_unit(self):
        # TODO remove
        return collectives.cpu_thread


par = Par()


class _CodegenPar(LoopMode):
    """Internal use loop mode for use in code generation of parallel loops

    Contains a C string for the "index expression" (e.g. "threadIdx.x / 32")
    and optional bounds"""

    is_par = True

    def __init__(self, c_index, static_bounds):
        # Compiled C string giving index of parallel loop "iteration"
        self.c_index = c_index
        assert isinstance(c_index, str)

        # Pair of optional ints, giving [lo, hi) for c_index to test against.
        # None means no test needed.
        # This is intentionally separate from the lo, hi of the loop itself
        # since this may be used for underhanded purposes in codegen.
        self.static_bounds = static_bounds
        assert len(static_bounds) == 2
        lo, hi = static_bounds
        assert lo is None or isinstance(lo, int)
        assert hi is None or isinstance(hi, int)

    def loop_mode_name(self):
        return "_codegen_par"


class CudaTasks(LoopMode):
    is_par = True
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_tasks"


# TODO remove
class CudaClusters(LoopMode):
    is_par = True
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_clusters"

    def collective_unit(self):
        return collectives.cuda_cluster


cuda_clusters = CudaClusters()


# TODO remove
class CudaBlocks(LoopMode):
    is_par = True
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_blocks"

    def collective_unit(self):
        # TODO remove
        return collectives.cuda_block


cuda_blocks = CudaBlocks()


# TODO remove
class CudaWarpgroups(LoopMode):
    is_par = True
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warpgroups"

    def collective_unit(self):
        # TODO remove
        return collectives.cuda_warpgroup


cuda_warpgroups = CudaWarpgroups()


# TODO remove
class CudaWarps(LoopMode):
    is_par = True
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warps"

    def collective_unit(self):
        # TODO remove
        return collectives.cuda_warp


cuda_warps = CudaWarps()


class CudaThreads(LoopMode):
    is_par = True
    allowed_actor_kinds = {actor_kinds.cuda_sync}

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_threads"

    def collective_unit(self):
        # TODO remove
        return collectives.cuda_thread


cuda_threads = CudaThreads()


def make_loop_mode_dict():
    loop_mode_dict = {
        "seq": Seq,
        "par": Par,
        "_codegen_par": _CodegenPar,
        "cuda_tasks": CudaTasks,
        "cuda_threads": CudaThreads,
        # TODO remove these
        "cuda_clusters": CudaClusters,
        "cuda_blocks": CudaBlocks,
        "cuda_warpgroups": CudaWarpgroups,
        "cuda_warps": CudaWarps,
    }
    return loop_mode_dict


loop_mode_dict = make_loop_mode_dict()


def format_loop_cond(lo_str: str, hi_str: str, loop_mode: LoopMode):
    strings = [loop_mode.loop_mode_name(), "(", lo_str, ",", hi_str]
    for attr in loop_mode.__dict__:
        strings.append(f",{attr}={getattr(loop_mode, attr)!r}")
    strings.append(")")
    return "".join(strings)
