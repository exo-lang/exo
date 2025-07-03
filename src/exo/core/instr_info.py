from dataclasses import dataclass
from typing import Dict, List, Type
from .memory import DRAM, MemWin, AllocableMemWin, Memory, SpecialWindow
from ..spork.coll_algebra import CollUnit
from ..spork.timelines import Instr_tl, Usage_tl


@dataclass(slots=True)
class AccessInfo:
    mem: Type[MemWin] = DRAM
    usage_tl: Usage_tl = None
    ext_instr_tl: List[Instr_tl] = tuple()
    ext_usage_tl: List[Usage_tl] = tuple()
    out_of_order: bool = None
    access_by_owner_only: bool = False


@dataclass(init=False, slots=True)
class InstrInfo:
    instr_format: List[str]  # Split by lines
    c_utils: List[str]
    c_includes: List[str]
    cu_utils: List[str]
    cu_includes: List[str]
    coll_unit: CollUnit
    instr_tl: Instr_tl
    access_info: Dict[str, AccessInfo]

    # For internal use
    _formatted_tparam_kwargs: str
