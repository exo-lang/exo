from typing import Optional, Set, Tuple
from .coll_algebra import CollUnit, cuda_thread


class LoopMode(object):
    def loop_mode_name(self):
        raise NotImplementedError()


class Seq(LoopMode):
    __slots__ = ["pragma_unroll"]

    def __init__(self, pragma_unroll=None):
        self.pragma_unroll = pragma_unroll
        assert pragma_unroll is None or isinstance(pragma_unroll, int)

    def loop_mode_name(self):
        return "seq"


seq = Seq()


class Par(LoopMode):
    __slots__ = []

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "par"


par = Par()


class _CodegenPar(LoopMode):
    """Internal use loop mode for use in code generation of parallel loops

    Contains a C string for the "index expression" (e.g. "threadIdx.x / 32")
    and optional bounds"""

    __slots__ = ["c_index", "static_bounds", "warp_name_filter"]

    c_index: str
    static_bounds: Optional[Tuple[int, int]]
    warp_name_filter: Optional[str]

    def __init__(self, c_index, static_bounds, warp_name_filter=None):
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

        self.warp_name_filter = warp_name_filter

    def loop_mode_name(self):
        return "_codegen_par"


class CudaTasks(LoopMode):
    __slots__ = []

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_tasks"


cuda_tasks = CudaTasks()


class CudaThreads(LoopMode):
    __slots__ = ["unit"]

    def __init__(self, unit=cuda_thread):
        assert isinstance(unit, CollUnit)
        self.unit = unit

    def loop_mode_name(self):
        return "cuda_threads"


cuda_threads = CudaThreads()


def make_loop_mode_dict():
    loop_mode_dict = {
        "seq": Seq,
        "par": Par,
        "_codegen_par": _CodegenPar,
        "cuda_tasks": CudaTasks,
        "cuda_threads": CudaThreads,
    }
    return loop_mode_dict


loop_mode_dict = make_loop_mode_dict()


def format_loop_cond(lo_str: str, hi_str: str, loop_mode: LoopMode):
    """loop_mode(lo, hi, kwarg1=value1, kwarg2=value2,...)"""
    strings = [loop_mode.loop_mode_name(), "(", lo_str, ",", hi_str]
    for attr in loop_mode.__slots__:
        value = getattr(loop_mode, attr)
        if value is None and isinstance(loop_mode, Seq):
            # Avoid adding pragma_unroll=None for every seq(...)
            pass
        else:
            strings.append(f",{attr}={value!r}")
    strings.append(")")
    return "".join(strings)
