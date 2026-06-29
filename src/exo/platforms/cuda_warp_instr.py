from __future__ import annotations
from .cuda_fwd import *
from math import prod

from ..scalars import ScalarInfo, f32

__all__ = []


class cuda_warp_broadcast_impl(InstrInfo):
    __slots__ = ["mask"]

    valid_num_types = ScalarInfo.same()

    def instance_impl(self, *shape):
        ntid = prod(shape)
        assert ntid in (1, 2, 4, 8, 16, 32), f"Cannot shuffle in {ntid} threads"

        self.coll_unit = ntid * cuda_thread
        self.instr_tl = cuda_in_order_instr
        distributed_coll_units = [
            prod(shape[i:]) * cuda_thread for i in range(1, len(shape) + 1)
        ]
        for access_info in (self.access_info["src"], self.access_info["dst"]):
            access_info.mem = CudaRmem
            access_info.distributed_coll_units = distributed_coll_units

        if ntid == 32:
            self.mask = "0xFFFFFFFF"
        else:
            assert 0, "Implement me"


@instr
class cuda_warp_broadcast_sync_1d(cuda_warp_broadcast_impl):
    def behavior(size0: size, dst: [R][size0], src: [R][size0], i0: index):
        assert i0 >= 0
        assert i0 < size0
        for dst0 in seq(0, size0):
            dst[dst0] = src[i0]

    def instance(self, size0):
        self.instance_impl(size0)

    def codegen(self, args):
        inp = args.src.index()
        out = args.dst.index()
        return [f"{out} = __shfl_sync({self.mask}, {inp}, {args.i0});"]


cuda_warp_broadcast_sync_1f32 = cuda_warp_broadcast_sync_1d.partial(dst=f32, src=f32)
__all__.append("cuda_warp_broadcast_sync_1d")
__all__.append("cuda_warp_broadcast_sync_1f32")


@instr
class cuda_warp_broadcast_sync_2d(cuda_warp_broadcast_impl):
    def behavior(
        size0: size,
        size1: size,
        dst: [R][size0, size1],
        src: [R][size0, size1],
        i0: index,
        i1: index,
    ):
        assert i0 >= 0
        assert i0 < size0
        assert i1 >= 0
        assert i1 < size1
        for dst0 in seq(0, size0):
            for dst1 in seq(0, size1):
                dst[dst0, dst1] = src[i0, i1]

    def instance(self, size0, size1):
        self.instance_impl(size0, size1)

    def codegen(self, args):
        inp = args.src.index()
        out = args.dst.index()
        idx = f"{args.i1} + {args.i0} * {args.size1}"
        return [f"{out} = __shfl_sync({self.mask}, {inp}, {idx});"]


cuda_warp_broadcast_sync_2f32 = cuda_warp_broadcast_sync_2d.partial(dst=f32, src=f32)
__all__.append("cuda_warp_broadcast_sync_2d")
__all__.append("cuda_warp_broadcast_sync_2f32")


class cuda_shfl_xor_sync_impl(InstrInfo):
    __slots__ = ["mask"]

    valid_num_types = ScalarInfo.same()

    def instance(self, *, laneMask):
        assert laneMask in (1, 2, 4, 8, 16), "Only support one bit for laneMask"
        self.coll_unit = 2 * cuda_threads_strided(1, laneMask)
        self.instr_tl = cuda_in_order_instr
        distributed_coll_units = [cuda_thread]
        for access_info in self.access_info.values():
            access_info.mem = CudaRmem
            access_info.distributed_coll_units = distributed_coll_units
        # Generate mask that includes just the two threads exchanging values.
        mask_base = 1 << laneMask | 1
        threadIdx_mask = 31 & ~laneMask
        mask_shift = f"threadIdx.x & {threadIdx_mask}"
        self.mask = f"{mask_base} << ({mask_shift})"


@instr
class cuda_shfl_xor_sync_1d(cuda_shfl_xor_sync_impl):
    def behavior(dst: [R][2], src: [R][2]):
        for i in seq(0, 2):
            dst[i] = src[1 - i]

    def codegen(self, args):
        inp = args.src.index()
        out = args.dst.index()
        return [f"{out} = __shfl_xor_sync({self.mask}, {inp}, {args.laneMask});"]


cuda_shfl_xor_sync_1f32 = cuda_shfl_xor_sync_1d.partial(dst=f32, src=f32)
__all__.append("cuda_shfl_xor_sync_1d")
__all__.append("cuda_shfl_xor_sync_1f32")


@instr
class cuda_shfl_xor_sync_sum_1d(cuda_shfl_xor_sync_impl):
    def behavior(dst: [R][2], src: [R][2]):
        for i in seq(0, 2):
            dst[i] = src[0] + src[1]

    def codegen(self, args):
        inp = args.src.index()
        out = args.dst.index()
        return [
            f"{out} = {inp} + __shfl_xor_sync({self.mask}, {inp}, {args.laneMask});"
        ]


cuda_shfl_xor_sync_sum_1f32 = cuda_shfl_xor_sync_sum_1d.partial(dst=f32, src=f32)
__all__.append("cuda_shfl_xor_sync_sum_1d")
__all__.append("cuda_shfl_xor_sync_sum_1f32")
