from .base_with_context import BaseWithContext
from typing import Optional


class CudaWarps(BaseWithContext):
    __slots__ = ["lo", "hi", "name"]

    def __init__(
        self,
        lo: Optional[int] = None,
        hi: Optional[int] = None,
        *,
        name: Optional[str] = None,
    ):
        self.lo = lo
        self.hi = hi
        self.name = name
        if name is None:
            assert (
                lo is not None
            ), "CudaWarps.lo must be given if CudaWarps.name is not given"
            assert (
                hi is not None
            ), "CudaWarps.hi must be given if CudaWarps.name is not given"
        if lo is not None:
            assert hi is None or lo < hi
            assert lo >= 0
        if hi is not None:
            assert lo is None or lo < hi
            assert hi >= 1

    def __repr__(self):
        args = []
        if self.lo is not None or self.hi is not None:
            args.append(str(self.lo))
            args.append(str(self.hi))
        if self.name is not None:
            args.append(f"name={self.name!r}")
        return "CudaWarps(" + ", ".join(args) + ")"
