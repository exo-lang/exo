from dataclasses import dataclass
from typing import Optional, Set, Tuple

from .coll_algebra import CollUnit, cuda_thread


loop_mode_dict = {}


class LoopMode(object):
    __slots__ = []

    @classmethod
    def loop_mode_name(cls):
        raise NotImplementedError

    @classmethod
    def is_par(cls):
        raise NotImplementedError

    def format_loop_cond(self, lo, hi):
        return format_loop_cond(lo, hi, self)


def loop_mode_class(_loop_mode_name, _is_par):
    assert _loop_mode_name
    _loop_mode_name = str(_loop_mode_name)
    assert isinstance(_is_par, bool)

    @classmethod
    def loop_mode_name(cls):
        return _loop_mode_name

    @classmethod
    def is_par(cls):
        return _is_par

    def decorator(cls):
        # Frozen, add __slots__, loop_mode_name(), is_par() functions,
        # LoopMode base class, and register in loop_mode_dict.
        cls_dict = dict(loop_mode_name=loop_mode_name, is_par=is_par, **cls.__dict__)
        cls = type(cls.__name__, (LoopMode,), cls_dict)
        cls = dataclass(frozen=True, slots=True)(cls)
        assert _loop_mode_name not in loop_mode_dict
        loop_mode_dict[_loop_mode_name] = cls
        return cls

    return decorator


@loop_mode_class("seq", False)
class Seq:
    pragma_unroll: Optional[int] = None

    def __post_init__(self, pragma_unroll=None):
        assert self.pragma_unroll is None or isinstance(self.pragma_unroll, int)


seq = Seq()


@loop_mode_class("par", True)
class Par:
    pass


par = Par()


@loop_mode_class("_codegen_par", True)
class _CodegenPar:
    """Internal use loop mode for use in code generation of parallel loops

    Contains a C string for the "index expression" (e.g. "threadIdx.x / 32")
    and optional bounds"""

    c_index: str
    comment: Optional[str]
    static_bounds: Optional[Tuple[int, int]]
    warp_name_filter: Optional[str] = None

    def __post_init__(self):
        # Compiled C string giving index of parallel loop "iteration"
        assert isinstance(self.c_index, str)

        # Pair of optional ints, giving [lo, hi) for c_index to test against.
        # None means no test needed.
        # This is intentionally separate from the lo, hi of the loop itself
        # since this may be used for underhanded purposes in codegen.
        lo, hi = self.static_bounds
        assert lo is None or isinstance(lo, int)
        assert hi is None or isinstance(hi, int)


@loop_mode_class("cuda_tasks", True)
class CudaTasks:
    pass


cuda_tasks = CudaTasks()


@loop_mode_class("cuda_threads", True)
class CudaThreads(LoopMode):
    unit: CollUnit = cuda_thread

    def __post_init__(self, unit=cuda_thread):
        assert isinstance(self.unit, CollUnit)


cuda_threads = CudaThreads()


def format_loop_cond(lo, hi, loop_mode: LoopMode):
    """loop_mode(lo, hi, kwarg1=value1, kwarg2=value2,...)"""
    strings = [loop_mode.loop_mode_name(), "(", str(lo), ", ", str(hi)]
    for attr in loop_mode.__slots__:
        value = getattr(loop_mode, attr)
        if value is None and isinstance(loop_mode, Seq):
            # Avoid adding pragma_unroll=None for every seq(...)
            pass
        else:
            strings.append(f", {attr}={value!r}")
    strings.append(")")
    return "".join(strings)
