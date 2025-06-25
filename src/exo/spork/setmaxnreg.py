from __future__ import annotations

from ..API import instr

from .coll_algebra import cuda_warpgroup
from .timelines import cuda_in_order_instr


@instr
class unsafe_setmaxnreg:
    def behavior(imm_reg_count: size, is_inc: size):
        pass

    def instance(self, imm_reg_count, is_inc):
        inc_dec = "inc" if is_inc else "dec"
        self.instr_tl = cuda_in_order_instr
        self.instr_format = [
            (f'asm("setmaxnreg.{inc_dec}.sync.aligned.u32 {imm_reg_count};");')
        ]
        self.coll_unit = cuda_warpgroup
