from dataclasses import dataclass
from typing import Dict, List, Type, Optional
from .memory import DRAM, MemWin, AllocableMemWin, Memory, SpecialWindow, BarrierType
from ..spork.coll_algebra import CollUnit
from ..spork.timelines import Instr_tl, Usage_tl


@dataclass(slots=True)
class AccessInfo:
    mem: Type[MemWin] = DRAM
    usage_tl: Usage_tl = None
    ext_instr_tl: List[Instr_tl] = tuple()
    ext_usage_tl: List[Usage_tl] = tuple()
    out_of_order: bool = None
    access_by_owner_only: Optional[bool] = None
    const: bool = False

    # For warp shuffles and TMA: identifies that the first
    # len(distributed_coll_units) dimensions of the window parameter
    # are expected to be distributed dimensions, as if accessed with
    #
    # def my_instr(..., param : [T][sz0, sz1, ...]):
    #     for i0 in cuda_threads(0, sz0, unit=distributed_coll_units[0]):
    #         for i1 in cuda_threads(0, sz1, unit=distributed_coll_units[1]):
    #             # ...
    #             param[i0, i1, ... ]
    distributed_coll_units: List[CollUnit] = ()


@dataclass(init=False, slots=True)
class InstrInfo:
    instr_format: Optional[List[str]]  # Split by lines
    c_utils: List[str]
    c_includes: List[str]
    cu_utils: List[str]
    cu_includes: List[str]
    coll_unit: CollUnit
    instr_tl: Instr_tl
    access_info: Dict[str, AccessInfo]

    # The instr expects a trailing barrier expr iff barrier_type is not None.
    # barrier_coll_units is akin to AccessInfo.distributed_coll_units.
    # The barrier must be allocated in barrier_type and have dim len(barrier_coll_units).
    barrier_type: Optional[Type[BarrierType]]
    barrier_coll_units: List[CollUnit]

    # For internal use
    _tparam_dict: dict
    _formatted_tparam_kwargs: str
