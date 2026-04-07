from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Dict, Optional
from ..backend.compiler_fwd import SyncCodegenCtx
from .barrier_usage import BarrierUsage
from .cuda_device_setup_builder import CudaDeviceSetupBuilder
from .distributed_memory import DistributedAllocState, ThreadIter
from ..core.LoopIR import LoopIR
from ..core.prelude import Sym


class LoweredBarrierType(Enum):
    garden_variety_fence = auto()
    cluster_sync = auto()
    wgmma_fence = auto()
    mbarrier = auto()
    Sm80_commit_group = auto()
    tma_to_gmem_commit_group = auto()
    wgmma_commit_group = auto()


@dataclass(slots=True)
class LoweredBarrier:
    # If set, two barrier objects of the same type_enum (in Exo code)
    # cannot be live at the same time.
    solitary: bool

    # More specific than the BarrierMechanism (specialized by sync-tl).
    # Also applies to Fence(...), which has no associated barrier object.
    type_enum: LoweredBarrierType

    # Lower SyncStmt, Alloc, Free to lines of C++ code (List[str])
    # (you may assume the statement uses this lowered barrier)
    codegen_sync_stmt: Callable[[LoopIR.SyncStmt, SyncCodegenCtx], List[str]] = None
    codegen_alloc: Callable[[LoopIR.Alloc], List[str]] = lambda a: [f"// {a}"]
    codegen_free: Callable[[LoopIR.Free], List[str]] = lambda a: [f"// {a}"]

    # Special case for TMA mbarriers
    codegen_cta_mask: Callable[[LoopIR.BarrierExpr, SyncCodegenCtx], str] = None
    codegen_barrier_arg: Callable[[LoopIR.BarrierExpr, SyncCodegenCtx], str] = None

    def __repr__(self):
        return f"LoweredBarrier({self.solitary}, {self.type_enum})"


@dataclass(slots=True)
class AddBarrierCtx:
    name: Sym
    get_usage: Callable[[Sym], BarrierUsage]
    coll_tilings: DistributedAllocState
    thread_iters: Dict[Sym, ThreadIter]
    device_setup_builder: CudaDeviceSetupBuilder

    # Shape of the non-distributed part of the allocation,
    # and the index in that shape of the managed ring buffer dimension.
    const_shape: List[int]
    managed_ring_buffer_dim_idx: Optional[int]
