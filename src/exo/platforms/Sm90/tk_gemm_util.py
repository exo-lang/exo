from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Union

from exo import *
from exo.stdlib.scheduling import *

from exo.platforms.cuda import *  #      Foundational exo cuda features, e.g. cuda_warp
from exo.platforms.Sm90 import *  #      H100 (sm_90) TMA, wgmma instructions & memories
from exo.platforms.cuda_tk import *  #   Wrappers for ThunderKittens register tile primitives

from exo.scalars import ScalarInfo, f16, bf16, f32


gemm_type = Union[str, ScalarInfo]


def perfect_div(numerator, denominator):
    div, mod = divmod(numerator, denominator)
    assert mod == 0
    return div


default_warp_config = [
    CudaWarpConfig("producer", 4, setmaxnreg_dec=40),
    CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),
]


@dataclass(slots=True)
class GemmConfig:
    # Number of CTAs per cluster in M and N dimensions
    ncta_M: int = 1
    ncta_N: int = 1
    # Tile size for a single CTA
    cta_M: int = 256
    cta_N: int = 256
    # SMEM ring depth & swizzling
    ring_depth: int = 4
    swizzle: int = 128
    # Precision and majorness of A/B/C, row or col
    A_type: gemm_type = bf16
    A_major: str = "row"
    B_type: gemm_type = bf16
    B_major: str = "row"
    C_type: gemm_type = f32
    C_major: str = "row"
    # Epilogue control
    enable_split_k: bool = False
    ping_pong: bool = False

    def __post_init__(self):
        assert self.swizzle == 128, f"{self.swizzle} not supported"
        assert self.A_major == "row", f"{self.A_major} not supported"
        assert self.C_major == "row", f"{self.C_major} not supported"

    def make_proc_name(self) -> str:
        suffix = ""
        if self.swizzle != 128:
            suffix += f"_SW{self.swizzle}"
        suffix += "_pingpong" if self.ping_pong else "_coop"
        if self.enable_split_k:
            suffix += "_splitK"
        return (
            f"Sm90_tk_gemm_r{self.ring_depth}"
            f"_m{self.ncta_M}n{self.ncta_N}_m{self.cta_M}n{self.cta_N}{suffix}"
        )

    def make_smem_K(self):
        A_info = ScalarInfo(self.A_type)
        B_info = ScalarInfo(self.B_type)
        smem_K = self.swizzle * 8 // A_info.bits
        assert A_info.bits == B_info.bits, f"{A_info}, {B_info}"
        return smem_K

    def make_smem_box_A(self):
        # batch dim: 1
        # M dim: cta_M for cooperative, cta_M / 2 for ping-pong
        #        with each CTA responsible for 1/ncta_N of it.
        # split-K dim: 1
        # clusterK dim: smem_K
        assert self.A_major == "row", self.A_major
        M = (
            perfect_div(self.cta_M, 2 * self.ncta_N)
            if self.ping_pong
            else perfect_div(self.cta_M, self.ncta_N)
        )
        return (1, M, 1, self.make_smem_K())

    def make_smem_box_B(self):
        assert self.B_major in ("row", "col"), self.B_major
        if self.B_major == "row":
            B_info = ScalarInfo(self.B_type)
            assert B_info.bits == 16, f"{B_info} cannot be transposed (row major B)"
            assert self.swizzle == 128, "Not supported"
            # batch dim: 1
            # split-K dim: 1
            # K dim: smem_K (each CTA responsible for 1/ncta_M of it).
            # N dim: 64
            K = perfect_div(self.make_smem_K(), self.ncta_M)
            return (1, 1, K, 64)
        else:
            N = perfect_div(self.cta_N, self.ncta_M)
            return (1, N, 1, self.make_smem_K())

    def make_smem_box_C(self):
        # batch dim: 1
        # M dim: cta_M
        # N dim: swizzle / sizeof(C_type)
        #
        # Note the TMA will be repeated on the N dimension
        # as needed to iterate to cta_N.
        #
        # This is an unfortunate Exo-GPU limitation.
        # ThunderKittens engineers a TensorMap with one extra dimension
        # to handle this iteration internally.
        C_info = ScalarInfo(self.C_type)
        return (1, self.cta_M, perfect_div(self.swizzle * 8, C_info.bits))


def sched_inline_stuff(p):
    # task_n should be the inner-most task loop.
    # Its body in the "device task"
    loops = p.find_all("for task_n in _:_")
    assert len(loops) == 1
    loop_c = loops[0]

    # Inline calls to non-instr functions.
    for c in loop_c.body():
        if isinstance(c, ForCursor):
            assert not isinstance(c.loop_mode(), CudaTasks)
        if isinstance(c, CallCursor) and not c.subproc().is_instr():
            p = inline(p, c)

    # Inline windows.
    # But, crucially, we don't inline the TensorMap created outside the device task.
    loop_c = p.forward(loop_c)
    for c in loop_c.body():
        if isinstance(c, WindowStmtCursor):
            p = inline_window(p, c)

    return p


def sched_add_annoying_barriers(p, config: GemmConfig):
    loops = p.find_all("for iter_k in _:_")
    assert len(loops) == 1
    loop_c = loops[0]

    # Cut the loop at 1, and set the cursor to be pointing to
    # the second loop (iter_k >= 1).
    p = cut_loop(p, loop_c, 1)
    loop_c = p.forward(loop_c).next()
    assert str(loop_c.name()) == "iter_k"

    # Insert a no-op at the end of the second iter_k loop's body,
    # which will soon be wrapped with loops.
    ring_depth = config.ring_depth
    body_c = loop_c.body()
    num_stmts_original = len(body_c)
    original_last_stmt_c = body_c[num_stmts_original - 1]
    p = insert_pass(p, original_last_stmt_c.after())
    pass_c = p.forward(original_last_stmt_c).next()

    # Set up loop structure
    # for cta_m
    #   for cta_n
    #     for wg_m
    #       Await(wgmma_cg[cta_m, cta_n, wg_m], cuda_in_order, 1)
    #     # Unblock the producer +ring_depth iterations in the future.
    #     Arrive(cuda_in_order) >> war[cta_m, :, iter_k + ring_depth] >> war[:, cta_n, iter_k + ring_depth]
    p = add_loop(p, pass_c, "cta_m", config.ncta_M)
    p = add_loop(p, pass_c, "cta_n", config.ncta_N)
    p = add_loop(p, pass_c, "wg_m", 2)
    pass_c = p.forward(pass_c)
    wg_m_c = pass_c.parent()
    p = set_loop_mode(p, wg_m_c, CudaThreads(unit=cuda_warpgroup))
    cta_n_c = wg_m_c.parent()
    p = set_loop_mode(p, cta_n_c, CudaThreads(unit=cuda_cta_in_cluster))
    cta_m_c = cta_n_c.parent()
    p = set_loop_mode(p, cta_m_c, CudaThreads(unit=config.ncta_N * cuda_cta_in_cluster))
    pass_c = p.forward(pass_c)

    # Add the Arrive/Await at correct locations
    p = insert_arrive(
        p,
        wg_m_c.after(),
        cuda_in_order,
        (
            f"war[cta_m, :, iter_k + {ring_depth - 1}]",
            f"war[:, cta_n, iter_k + {ring_depth - 1}]",
        ),
    )
    p = insert_await(
        p, pass_c.after(), "wgmma_cg[cta_m, cta_n, wg_m]", cuda_in_order, 1
    )

    # Only consumer executes this.
    p = wrap_with_context(p, cta_m_c, CudaWarps(name="consumer"))

    return p


def sched_final_changes(gemm: Procedure, config: GemmConfig):
    name = config.make_proc_name()
    gemm = sched_inline_stuff(gemm)
    gemm = simplify(gemm)
    gemm = rename(gemm, name)

    if True:
        L = 2
        K_split = 2 if config.enable_split_k else 1
        M = 900
        N = 700
        cluster_K = 224
        start = time.time()
        gemm.sync_check(L=L, M=M, N=N, K_split=K_split, cluster_K=cluster_K)
        dt = time.time() - start
        print(f"sync_check %s: %.3f s" % (name, dt))
    else:
        print(f"NO SYNC CHECK %s" % (name,))

    return gemm


def handwrite_row_col_coop_main_loop(config: GemmConfig):
    A_type = ScalarInfo(config.A_type)
    B_type = ScalarInfo(config.B_type)
    D_type = ScalarInfo(config.C_type)
    ncta_M = config.ncta_M
    ncta_N = config.ncta_N
    cta_M = config.cta_M
    cta_N = config.cta_N
    cluster_M = ncta_M * cta_M
    cluster_N = ncta_N * cta_N
    ring_depth = config.ring_depth
    swizzle = config.swizzle

    M_wg_tiles = perfect_div(cta_M, 128)
    smem_K = config.make_smem_K()
    smem_box_A = config.make_smem_box_A()
    smem_box_B = config.make_smem_box_B()

    assert config.A_major == "row"
    assert config.B_major == "col"

    # fmt: off
    @proc
    def main_loop(
        cluster_K: size,
        D_rmem: [D_type][ncta_M, ncta_N, 2, 4, M_wg_tiles, 16, cta_N],
        A_win: [A_type][cluster_M, cluster_K],
        B_win: [B_type][cluster_N, cluster_K],
    ):
        assert stride(A_win, 1) == 1
        assert stride(B_win, 1) == 1
        war: barrier[
            ncta_M,
            ncta_N,
            ((cluster_K + smem_K - 1) / smem_K + ring_depth) @ ring_buffer_by(ring_depth),
        ] @ CudaMbarrierPreArrive(ring_depth)

        raw: barrier[
            ncta_M,
            ncta_N,
            ((cluster_K + smem_K - 1) / smem_K + 0) @ ring_buffer_by(ring_depth),
        ] @ CudaMbarrierPreArrive(0)

        A_smem: A_type[ncta_M, ncta_N, ring_depth, cta_M, smem_K] @ Sm90_SmemSwizzled(swizzle)
        B_smem: B_type[ncta_M, ncta_N, ring_depth, cta_N, smem_K] @ Sm90_SmemSwizzled(swizzle)
        wgmma_cg: barrier[ncta_M, ncta_N, 2] @ Sm90_WgmmaCommitGroup

        for iter_k in seq(0, ((cluster_K + smem_K - 1) / smem_K)):
            with CudaWarps(0, 1, name="producer"):
                # Each CTA waits for its respective write-after-read protection barrier.
                for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                    for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                        Await(war[cta_m, cta_n, iter_k], cuda_temporal, 0)
                # CTAs cooperate along the N dimension to multicast the needed tiles of A.
                for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                    Sm90_tma_load_multicast_2d(
                        A_smem[cta_m, :, iter_k % ring_depth, :, :],
                        A_win[
                            cta_m * cta_M :
                            cta_m * cta_M + cta_M,
                            iter_k * smem_K :
                            iter_k * smem_K + smem_K,
                        ],
                        ncta=ncta_N, cta_stride=1, size0=cta_M, size1=smem_K,
                        smem_box=smem_box_A, dst=A_type, src=A_type,
                    ) >> raw[cta_m, :, iter_k]
                # CTAs cooperate along the M dimension to multicast the needed tiles of B.
                for cta_n in cuda_threads(0, ncta_N, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N)):
                    Sm90_tma_load_multicast_2d(
                        B_smem[:, cta_n, iter_k % ring_depth, :, :],
                        B_win[
                            cta_n * cta_N :
                            cta_n * cta_N + cta_N,
                            iter_k * smem_K :
                            iter_k * smem_K + smem_K,
                        ],
                        ncta=ncta_M, cta_stride=ncta_N, size0=cta_N, size1=smem_K,
                        smem_box=smem_box_B, dst=B_type, src=B_type,
                    ) >> raw[:, cta_n, iter_k]
                # Each CTA arrives on read-after-write protection barriers.
                # Multicast to any CTA with the same cta_m OR the same cta_n.
                for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                    for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                        Arrive(cuda_temporal) >> raw[cta_m, :, iter_k] >> raw[:, cta_n, iter_k]
            with CudaWarps(name="consumer"):
                for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                    for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                        # Each CTA waits for its respective read-after-write protection barrier.
                        Await(raw[cta_m, cta_n, iter_k], cuda_generic_and_async_proxy, 0)
                        # Each warpgroup does its own WGMMAs, then arrives on its own commit group.
                        for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                            Fence(wgmma_fence_1, wgmma_fence_2)
                            for ms in seq(0, M_wg_tiles, pragma_unroll=0):
                                Sm90_tk_mma_row_col(
                                    D_rmem[cta_m, cta_n, wg_m, :, ms, :, :],
                                    A_smem[cta_m, cta_n, iter_k % ring_depth,
                                           (wg_m * M_wg_tiles + ms) * 64 :
                                           (wg_m * M_wg_tiles + ms) * 64 + 64, :],
                                    B_smem[cta_m, cta_n, iter_k % ring_depth, :, :],
                                    D=D_type, A=A_type, B=B_type, N=cta_N, K=smem_K,
                                )
                            Arrive(wgmma_async) >> wgmma_cg[cta_m, cta_n, wg_m]
                        # This is where it gets annoying.
                        # Each iteration except iter_k = 0 has to Await for its commit group
                        # then Arrive on the war (write-after-read) mbarriers
                        # multicasted like done in the producer.
                        #
                        # We use scheduling for this (despite "handwrite") because too annoying.
                        # We have to cut the iter_k loop and insert the barriers only on
                        # the second loop.
        # End iter_k loop

        # Consumer has to wait for all wgmma to retire and do the
        # final arrive on the war mbarriers. Note: for the persistent
        # kernel to work, this Arrive needs to be done, so mbarriers are left in
        # a consistent state. The sync_check enforces this: each allocated mbarrier
        # must have been Arrived on exactly once (one_shot_arrive=True).
        with CudaWarps(name="consumer"):
            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                        Await(wgmma_cg[cta_m, cta_n, wg_m], cuda_in_order, 0)
                    Arrive(cuda_in_order) >> war[
                        cta_m, :, ((cluster_K + smem_K - 1) / smem_K + ring_depth - 1)] >> war[
                        :, cta_n, ((cluster_K + smem_K - 1) / smem_K + ring_depth - 1)]

    main_loop = sched_add_annoying_barriers(main_loop, config)
    return simplify(main_loop)


def handwrite_coop_epilogue(config: GemmConfig):
    C_type = ScalarInfo(config.C_type)
    D_type = C_type
    ncta_M = config.ncta_M
    ncta_N = config.ncta_N
    cta_M = config.cta_M
    cta_N = config.cta_N
    cluster_M = ncta_M * cta_M
    cluster_N = ncta_N * cta_N
    ring_depth = config.ring_depth
    swizzle = config.swizzle

    M_wg_tiles = perfect_div(cta_M, 128)
    smem_box_C = config.make_smem_box_C()

    assert config.C_major == "row", "not supported"
    assert not config.enable_split_k
    assert not config.ping_pong

    # Helper for staging RMEM into SMEM.
    # The [cta_M, cta_N] logical RMEM tile has its cta_n dimension split
    # to [cta_M, outer_N, inner_N] then re-ordered
    # to [outer_N, cta_M, inner_N]. This is dictated by the TMA.
    # The returned advice.instr magically does the copy from RMEM to this SMEM layout.
    advice: CudaTkRsInstrAdvice = cuda_tk_store_rs_advice(
        16, cta_N, dst=C_type, src=D_type, swizzle=swizzle
    )
    store_smem = advice.instr
    outer_N = advice.outer_cols
    inner_N = advice.inner_cols

    # fmt: off
    @proc
    def epilogue(
        C_win: [C_type][cluster_M, cluster_N],
        D_rmem: [D_type][ncta_M, ncta_N, 2, 4, M_wg_tiles, 16, cta_N],
    ):
        assert stride(C_win, 1) == 1
        C_smem: C_type[ncta_M, ncta_N, outer_N, cta_M, inner_N] @ Sm90_SmemSwizzled(swizzle)
        for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
            for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                with CudaWarps(name="consumer"):
                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                        for w in cuda_threads(0, 4, unit=cuda_warp):
                            # Each warp writes a (16, cta_N) tile to SMEM.
                            # This is repeated if cta_M > 128.
                            for ms in seq(0, M_wg_tiles, pragma_unroll=0):
                                store_smem(
                                    C_smem[cta_m, cta_n, :,
                                        (wg_m * (64 * M_wg_tiles)) + ms * 64 + w * 16:
                                        (wg_m * (64 * M_wg_tiles)) + ms * 64 + w * 16 + 16,
                                        :],
                                    D_rmem[cta_m, cta_n, wg_m, w, ms, :, :],
                                )
                # Each CTA waits for its own RMEM->SMEM
                # then does a proxy fence (cuda_generic_and_async_proxy)
                # and issues the TMA.
                Fence(cuda_in_order, cuda_generic_and_async_proxy)
                with CudaWarps(3, 4, name="producer"):
                    for ns in seq(0, outer_N):
                        Sm90_tma_store_2d(
                            C_win[
                                cta_M * cta_m :
                                cta_M * cta_m + cta_M,
                                cta_N * cta_n + ns * inner_N :
                                cta_N * cta_n + ns * inner_N + inner_N,
                            ],
                            C_smem[cta_m, cta_n, ns, :, :],
                            dst=C_type, src=D_type, size0=cta_M, size1=inner_N,
                            smem_box=smem_box_C, swizzle=128,
                        )
                        tma_cg: barrier @ Sm90_TmaCommitGroup
                        Arrive(tma_to_gmem_async) >> tma_cg
                        Await(tma_cg, cuda_in_order, 0)

    return simplify(epilogue)


def handwrite_gemm(config: GemmConfig):
    A_type = ScalarInfo(config.A_type)
    B_type = ScalarInfo(config.B_type)
    C_type = ScalarInfo(config.C_type)
    D_type = ScalarInfo(config.C_type)
    ncta_M = config.ncta_M
    ncta_N = config.ncta_N
    cta_M = config.cta_M
    cta_N = config.cta_N
    cluster_M = ncta_M * cta_M
    cluster_N = ncta_N * cta_N
    swizzle = config.swizzle

    M_wg_tiles = perfect_div(cta_M, 128)

    smem_box_A = config.make_smem_box_A()
    smem_box_B = config.make_smem_box_B()
    smem_box_C = config.make_smem_box_C()

    # Each warp stores (16 x cta_N) tiles.
    D_tile_mem = Sm90_TkRmemTileD(cta_N)

    main_loop = handwrite_row_col_coop_main_loop(config)
    epilogue = handwrite_coop_epilogue(config)

    # fmt: off
    @proc
    def gemm(
        L: size,
        M: size,
        N: size,
        K_split: size,
        cluster_K: size,
        # [batch, m, task_k, ks]
        # Note, k = task_k * cluster_K + ks.
        # We have to split the dim due to affine indexing restrictions.
        # This consumes an extra dimension of the tensor map.
        A: A_type[L, M, K_split, cluster_K] @ CudaGmemLinear,
        B: B_type[L, N, K_split, cluster_K] @ CudaGmemLinear,
        C: C_type[L, M, N] @ CudaGmemLinear,
    ):
        assert stride(A, 3) == 1
        assert stride(B, 3) == 1
        assert stride(C, 2) == 1
        assert cluster_K > 0
        assert cluster_K % 4 == 0

        A_tensorMap = A[:,:,:,:] @ Sm90_tensorMap(swizzle, *smem_box_A)
        B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(swizzle, *smem_box_B)
        C_tensorMap = C[:,:,:] @ Sm90_tensorMap(swizzle, *smem_box_C)

        with CudaDeviceFunction(
            clusterDim=ncta_M * ncta_N,
            warp_config=default_warp_config,
            blocks_per_sm=1,
            unsafe_no_shutdown_cluster_sync=True,
        ):
            for batch in cuda_tasks(0, L):
                for task_k in cuda_tasks(0, K_split):
                    for task_m in cuda_tasks(0, (M + cluster_M - 1) / cluster_M):
                        for task_n in cuda_tasks(0, (N + cluster_N - 1) / cluster_N):
                            # D_rmem[cta_m, cta_n, wg_m, w, ms, mt, nt] where
                            #
                            # cta_m strides by ncta_N * cuda_cta_in_cluster
                            # cta_n strides by cuda_cta_in_cluster
                            # wg_m strides by cuda_warpgroup
                            # w strides by cuda_warp
                            #
                            # Each warp holds M_wg_tile-many (16, cta_N) tiles.
                            #
                            # m = wg_m * (M_wg_tiles * 64) + ms * 64 + w * 16 + mt
                            D_rmem: D_type[ncta_M, ncta_N, 2, 4, M_wg_tiles, 16, cta_N] @ D_tile_mem
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    with CudaWarps(name="consumer"):
                                        for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                            for ms in seq(0, M_wg_tiles, pragma_unroll=0):
                                                Sm90_tk_zero_scale_d(
                                                    D_rmem[cta_m, cta_n, wg_m, :, ms, :, :],
                                                    D=D_type, N=cta_N,
                                                )

                            main_loop(
                                cluster_K,
                                D_rmem[:, :, :, :, :, :, :],
                                A_tensorMap[
                                    batch,
                                    task_m * cluster_M :
                                    task_m * cluster_M + cluster_M,
                                    task_k,
                                    :
                                ],
                                B_tensorMap[
                                    batch,
                                    task_n * cluster_N :
                                    task_n * cluster_N + cluster_N,
                                    task_k,
                                    :,
                                ],
                            )

                            # This cluster-wide sync is required for the main_loop
                            # and the epilogue to safely alias SMEM.
                            Fence(cuda_in_order, cuda_in_order)

                            epilogue(
                                C_tensorMap[
                                    batch,
                                    task_m * cluster_M :
                                    task_m * cluster_M + cluster_M,
                                    task_n * cluster_N :
                                    task_n * cluster_N + cluster_N,
                                ],
                                D_rmem[:, :, :, :, :, :, :],
                            )

                            # This cluster-wide sync is required for the epilogue and
                            # the main_loop (of the next task mapped to the same
                            # SM/cluster) to safely alias SMEM.
                            Fence(cuda_in_order, cuda_in_order)

    return sched_final_changes(gemm, config)


tmp_config = GemmConfig(ncta_M=2, ncta_N=4, B_major="col")
gemm = handwrite_gemm(tmp_config)
print(gemm)
