from __future__ import annotations

from ..API import instr

from . import actor_kinds
from .coll_algebra import cuda_warpgroup


@instr
class unsafe_setmaxnreg:
    def behavior(imm_reg_count: size, is_inc: size):
        pass

    def instance(self, imm_reg_count, is_inc):
        inc_dec = "inc" if is_inc else "dec"
        self.actor_kind = actor_kinds.cuda_classic
        self.instr_format = (
            f'asm("setmaxnreg.{inc_dec}.sync.aligned.u32 {imm_reg_count};");'
        )
        self.coll_unit = cuda_warpgroup
