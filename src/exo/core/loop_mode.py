class LoopMode(object):
    def loop_mode_name(self):
        raise NotImplemented


class Seq(LoopMode):
    def __init__(self):
        pass

    def loop_mode_name(self):
        return "seq"


seq = Seq()


class Par(LoopMode):
    def __init__(self):
        pass

    def loop_mode_name(self):
        return "par"


par = Par()


class CudaClusters(LoopMode):
    blocks: int

    def __init__(self, blocks):
        self.blocks = blocks

    def loop_mode_name(self):
        return "cuda_clusters"


class CudaBlocks(LoopMode):
    warps: int

    def __init__(self, warps=0):
        self.warps = warps

    def loop_mode_name(self):
        return "cuda_blocks"


cuda_blocks = CudaBlocks()


class CudaWarpgroups(LoopMode):
    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warpgroups"


cuda_warpgroups = CudaWarpgroups()


class CudaWarps(LoopMode):
    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warps"


cuda_warps = CudaWarps()


class CudaThreads(LoopMode):
    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_threads"


cuda_threads = CudaThreads()

loop_mode_dict = {
    "seq": Seq,
    "par": Par,
    "cuda_blocks": CudaBlocks,
    "cuda_warpgroups": CudaWarpgroups,
    "cuda_warps": CudaWarps,
    "cuda_threads": CudaThreads,
}


def format_loop_cond(lo_str: str, hi_str: str, loop_mode: LoopMode):
    strings = [loop_mode.loop_mode_name(), "(", lo_str, ",", hi_str]
    for attr in loop_mode.__dict__:
        strings.append(f",{attr}={getattr(loop_mode, attr)}")
    strings.append(")")
    return "".join(strings)
