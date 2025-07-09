from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List
from ..core.LoopIR import LoopIR


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

    # More specific than the BarrierType (specialized by sync-tl).
    # Also applies to Fence(...), which has no associated barrier object.
    type_enum: LoweredBarrierType

    # Lower SyncStmt, Alloc, Free to lines of C++ code (List[str])
    # (you may assume the statement uses this lowered barrier)
    codegen_sync_stmt: Callable[[LoopIR.SyncStmt], List[str]] = None
    codegen_alloc: Callable[[LoopIR.Alloc], List[str]] = lambda a: [f"// {a}"]
    codegen_free: Callable[[LoopIR.Free], List[str]] = lambda a: []

    # Special case for TMA mbarriers
    codegen_cta_mask: Callable[[LoopIR.BarrierExpr], str] = None
    codegen_barrier_arg: Callable[[LoopIR.BarrierExpr], str] = None
