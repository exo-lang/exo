from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class CudaWarpConfig:
    # For the user to configure CudaDeviceFunction
    name: str
    count: int
    setmaxnreg_dec: Optional[int] = None
    setmaxnreg_inc: Optional[int] = None

    def __post_init__(self):
        assert self.count > 0
        assert self.setmaxnreg_dec is None or self.setmaxnreg_inc is None


@dataclass(slots=True)
class WarpLayoutInfo:
    # Internal use.
    offset: int  # active if threadIdx.x >= offset * 32 &&
    count: int  # threadIdx.x < (offset + count) * 32
    cname: str  # suffix for codegen
    setmaxnreg: int  # if not 0, setmaxnreg to this value before execution
