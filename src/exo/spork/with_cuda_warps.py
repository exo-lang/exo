from .base_with_context import BaseWithContext


class CudaWarps(BaseWithContext):
    __slots__ = ["lo", "hi"]

    def __init__(self, lo: int, hi: int):
        assert isinstance(lo, int)
        assert isinstance(hi, int)
        assert 0 <= lo < hi
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return f"CudaWarps({self.lo}, {self.hi})"
