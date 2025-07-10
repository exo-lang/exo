from __future__ import annotations
from .cuda_fwd import *
from math import prod

__all__ = []


class cuda_warp_shfl_impl(InstrInfo):
    __slots__ = ["mask"]

    def instance_impl(self, *shape):
        ntid = prod(shape)
        assert ntid in (1, 2, 4, 8, 16, 32), f"Cannot shuffle in {ntid} threads"

        self.coll_unit = ntid * cuda_thread
        self.instr_tl = cuda_in_order_instr
        distributed_coll_units = [
            prod(shape[i:]) * cuda_thread for i in range(1, len(shape) + 1)
        ]
        for access_info in (self.access_info["inputs"], self.access_info["outputs"]):
            access_info.mem = CudaRmem
            access_info.access_by_owner_only = True
            access_info.distributed_coll_units = distributed_coll_units

        if ntid == 32:
            self.mask = "0xFFFFFFFF"
        else:
            assert 0, "Implement me"


@instr
class cuda_warp_broadcast_sync_1f32(cuda_warp_shfl_impl):
    def behavior(size0: size, outputs: [f32][size0], inputs: [f32][size0], i0: index):
        assert i0 >= 0
        assert i0 < size0
        for dst0 in seq(0, size0):
            outputs[dst0] = inputs[i0]

    def instance(self, size0):
        self.instance_impl(size0)

    def codegen(self, args):
        inp = args.inputs.index()
        out = args.outputs.index()
        return [f"{out} = __shfl_sync({self.mask}, {inp}, {args.i0});"]


__all__.append("cuda_warp_broadcast_sync_1f32")
