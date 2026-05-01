from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import time
from typing import Union

from exo import *
from exo.stdlib.scheduling import *

from exo.platforms.cuda import *  #      Foundational exo cuda features, e.g. cuda_warp
from exo.platforms.Sm90 import *  #      H100 (sm_90) TMA, wgmma instructions & memories
from exo.platforms.cuda_tk import *  #   Wrappers for ThunderKittens register tile primitives

from exo.scalars import ScalarInfo, f16, bf16, f32

__all__ = [
    "GemmConfig",
    "GemmTestBug",
    "handwrite_gemm",
    "schedule_gemm",
]


gemm_type = Union[str, ScalarInfo]


def perfect_div(numerator, denominator):
    div, mod = divmod(numerator, denominator)
    assert mod == 0
    return div


default_warp_config = [
    CudaWarpConfig("producer", 4, setmaxnreg_dec=40),
    CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),
]

L_divisor = 1
M_divisor = 16
N_divisor = 16
K_cluster_divisor = 16


class GemmTestBug(Enum):
    none = auto()
    wrong_wgmma_cg = auto()
    coop_missing_cluster_sync_before_epilogue = auto()
    coop_missing_cluster_sync_after_epilogue = auto()
    missing_await_raw = auto()
    coop_missing_arrive_war_threads = auto()
    coop_missing_final_arrive_war = auto()
    coop_wrong_tma_to_gmem_timeline = auto()
    ping_pong_reorder_await_war = auto()
    ping_pong_reorder_arrive_raw = auto()


@dataclass(slots=True)
class GemmConfig:
    # Note not all state combinations are supported.
    # We try to assert but could have missed some possibilities.

    # Number of CTAs per cluster in M and N dimensions
    ncta_M: int = 1
    ncta_N: int = 1
    # Tile size for a single CTA
    cta_M: int = 128
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
    # Mostly for debug; force A to be staged in RMEM
    A_in_rmem: bool = False
    # Epilogue control
    enable_split_k: bool = False
    ping_pong: bool = False
    # For sync_check testing, create wrong gemms
    bug: GemmTestBug = GemmTestBug.none

    def __post_init__(self):
        assert self.swizzle == 128, f"{self.swizzle} not supported"
        assert self.A_major == "row", f"{self.A_major} not supported"
        assert self.C_major == "row", f"{self.C_major} not supported"

    def make_sporkbench_case(self) -> dict:
        return dict(
            algorithm="gemm",
            proc=self.make_proc_name(),
            args=["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
            A_major=self.A_major,
            B_major=self.B_major,
            C_major=self.C_major,
            L_divisor=L_divisor,
            M_divisor=M_divisor,
            N_divisor=N_divisor,
            K_cluster_divisor=K_cluster_divisor,
            K_split_max=256 if self.enable_split_k else 1,
            A_type=str(self.A_type),
            B_type=str(self.B_type),
            C_type=str(self.C_type),
        )

    def make_proc_name(self) -> str:
        suffix = ""
        if self.swizzle != 128:
            suffix += f"_SW{self.swizzle}"
        suffix += "_ping_pong" if self.ping_pong else "_coop"
        if self.enable_split_k:
            suffix += "_splitK"
        if self.A_in_rmem:
            suffix += "_Armem"
        A_type = self.A_type
        B_type = self.B_type
        C_type = self.C_type

        majors = f"C{self.C_major[0]}A{self.A_major[0]}B{self.B_major[0]}"

        return (
            f"Sm90_tk_gemm_{majors}_{C_type}_{A_type}{B_type}_r{self.ring_depth}"
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
        # M dim: cta_M (coop) or cta_M / 2 (ping-pong)
        # N dim: swizzle / sizeof(C_type)
        #
        # Note the TMA will be repeated on the N dimension
        # as needed to iterate to cta_N.
        #
        # This is an unfortunate Exo-GPU limitation.
        # ThunderKittens engineers a TensorMap with one extra dimension
        # to handle this iteration internally.
        C_info = ScalarInfo(self.C_type)
        M = perfect_div(self.cta_M, 2) if self.ping_pong else self.cta_M
        return (1, M, perfect_div(self.swizzle * 8, C_info.bits))


def sched_inline_stuff(p):
    # task_n should be the inner-most task loop.
    # Its body is the "device task"
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


def sched_cut_sync_iter_k(p, config: GemmConfig):
    """Cut the iter_k loop at 1, and add synchronization in the iter_k >= 1 loop.

    The synchronization waits for the wgmma of the previous iteration to retire,
    then arrives on the `war` mbarriers, unblocking the producer warps to
    overwrite the ring buffer slot the wgmma have read from.

    """
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
    # For cooperative:
    #
    # for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
    #   for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
    #     for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
    #       Await(wgmma_cg[cta_m, cta_n, wg_m], cuda_in_order, 1)
    #     # Unblock the producer +ring_depth iterations in the future.
    #     Arrive(cuda_in_order) >> war[cta_m, :, iter_k + ring_depth] >> war[:, cta_n, iter_k + ring_depth]
    #
    # For ping-pong, same, but the war[...] arrive is moved into the wg_m (warpgroup)
    # loop and additionally indexed by wg_m. There are two producer warps, each of
    # which serve only their respective consumer warpgroup.
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
    if config.ping_pong:
        arrive_gap_c = pass_c.after()
        p = insert_arrive(
            p,
            arrive_gap_c,
            cuda_in_order,
            (
                f"war[cta_m, :, wg_m, iter_k + {ring_depth - 1}]",
                f"war[:, cta_n, wg_m, iter_k + {ring_depth - 1}]",
            ),
        )
    else:
        arrive_gap_c = wg_m_c.after()
        p = insert_arrive(
            p,
            arrive_gap_c,
            cuda_in_order,
            (
                f"war[cta_m, :, iter_k + {ring_depth - 1}]",
                f"war[:, cta_n, iter_k + {ring_depth - 1}]",
            ),
        )
    arrive_c = p.forward(arrive_gap_c).anchor().next()
    should_be_1 = 2 if config.bug == GemmTestBug.wrong_wgmma_cg else 1
    p = insert_await(
        p,
        pass_c.after(),
        "wgmma_cg[cta_m, cta_n, wg_m]",
        cuda_in_order,
        should_be_1,
    )

    # Only consumer executes this.
    p = wrap_with_context(p, cta_m_c, CudaWarps(name="consumer"))

    # (Ignore) For testing, if bug enabled, delete an Await in the iter_k >= 1 loop.
    if config.bug == GemmTestBug.missing_await_raw:
        loop_c = p.forward(loop_c)
        print(loop_c)
        await_c = loop_c.body()[1].body()[0].body()[0].body()[0]
        assert isinstance(await_c, SyncCursor)
        assert await_c.first_sync_tl() == None
        assert await_c.second_sync_tl() == cuda_generic_and_async_proxy
        assert await_c.name() == "raw"
        p = add_if(p, await_c, "False", unsafe_disable_check=True)

    # (Ignore) For testing, if bug enabled, deactivate some Arrive threads
    if config.bug == GemmTestBug.coop_missing_arrive_war_threads:
        p = wrap_with_context(p, arrive_c, CudaWarps(0, 4))

    return p


def sched_final_changes(gemm: Procedure, config: GemmConfig):
    name = config.make_proc_name()
    gemm = sched_inline_stuff(gemm)
    gemm = simplify(gemm)
    gemm = rename(gemm, name)

    if config.bug == GemmTestBug.none:
        L = 2
        K_split = 2 if config.enable_split_k else 1
        M = 900
        N = 700
        K_cluster = config.make_smem_K() * config.ring_depth * 2 + 16
        start = time.time()
        gemm.sync_check(L=L, M=M, N=N, K_split=K_split, K_cluster=K_cluster)
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

    assert not config.A_in_rmem, "not supported"
    assert config.A_major == "row"
    assert config.B_major == "col"
    assert not config.ping_pong

    # fmt: off
    @proc
    def main_loop(
        K_cluster: size,
        D_rmem: [D_type][ncta_M, ncta_N, 2, 4, M_wg_tiles, 16, cta_N],
        A_win: [A_type][cluster_M, K_cluster],
        B_win: [B_type][cluster_N, K_cluster],
    ):
        assert stride(A_win, 1) == 1
        assert stride(B_win, 1) == 1
        war: barrier[
            ncta_M,
            ncta_N,
            ((K_cluster + smem_K - 1) / smem_K + ring_depth) @ ring_buffer_by(ring_depth),
        ] @ CudaMbarrierPreArrive(ring_depth)

        raw: barrier[
            ncta_M,
            ncta_N,
            ((K_cluster + smem_K - 1) / smem_K + 0) @ ring_buffer_by(ring_depth),
        ] @ CudaMbarrierPreArrive(0)

        A_smem: A_type[ncta_M, ncta_N, ring_depth, cta_M, smem_K] @ Sm90_SmemSwizzled(swizzle)
        B_smem: B_type[ncta_M, ncta_N, ring_depth, cta_N, smem_K] @ Sm90_SmemSwizzled(swizzle)
        wgmma_cg: barrier[ncta_M, ncta_N, 2] @ Sm90_WgmmaCommitGroup

        for iter_k in seq(0, ((K_cluster + smem_K - 1) / smem_K)):
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
                    Arrive(cuda_in_order
                        ) >> war[cta_m, :, ((K_cluster + smem_K - 1) / smem_K + ring_depth - 1)
                        ] >> war[:, cta_n, ((K_cluster + smem_K - 1) / smem_K + ring_depth - 1)]

    # Does nothing by default (this is just for the Exo test suite)
    main_loop = inject_coop_main_loop_final_war_arrive_bug(main_loop, config)

    main_loop = sched_cut_sync_iter_k(main_loop, config)
    return simplify(main_loop)


def inject_coop_main_loop_final_war_arrive_bug(main_loop, config):
    # Really hard-wired cursor movement since PAST doesn't search for Exo-GPU features.
    loops = main_loop.find_all("for iter_k in _:_")
    assert len(loops) == 1
    loop_c = loops[0]
    sync_cta_n = loop_c.next().only_child(2)
    assert sync_cta_n.name() == "cta_n"
    arrive_cursor = sync_cta_n.body()[1]
    assert isinstance(arrive_cursor, SyncCursor)
    assert arrive_cursor.name() == "war"
    if config.bug == GemmTestBug.coop_missing_arrive_war_threads:
        # If only one consumer warpgroup arrives on the war mbarrier, there
        # should be a WAR hazard detected due to potential overlap between
        # the producer and the other consumer warpgroup.
        main_loop = wrap_with_context(main_loop, arrive_cursor, CudaWarps(0, 4))
    if config.bug == GemmTestBug.coop_missing_final_arrive_war:
        main_loop = add_if(main_loop, arrive_cursor, "False", True)
    return main_loop


def handwrite_row_row_coop_main_loop(config: GemmConfig):
    A_type = ScalarInfo(config.A_type)
    B_type = ScalarInfo(config.B_type)
    D_type = ScalarInfo(config.C_type)
    ncta_M = config.ncta_M
    ncta_N = config.ncta_N
    cta_M = config.cta_M
    cta_N = config.cta_N
    wg_M = perfect_div(config.cta_M, 2)
    cluster_M = ncta_M * cta_M
    cluster_N = ncta_N * cta_N
    ring_depth = config.ring_depth
    swizzle = config.swizzle

    M_wg_tiles = perfect_div(cta_M, 128)
    smem_K = config.make_smem_K()
    smem_box_A = config.make_smem_box_A()
    smem_box_B = config.make_smem_box_B()
    A_in_rmem = config.A_in_rmem

    assert config.A_major == "row"
    assert config.B_major == "row"
    assert not config.ping_pong
    assert config.bug == GemmTestBug.none

    # fmt: off
    @proc
    def main_loop(
        K_cluster: size,
        D_rmem: [D_type][ncta_M, ncta_N, 2, 4, M_wg_tiles, 16, cta_N],
        A_win: [A_type][cluster_M, K_cluster],
        B_win: [B_type][K_cluster, cluster_N],
    ):
        assert stride(A_win, 1) == 1
        assert stride(B_win, 1) == 1
        war: barrier[
            ncta_M,
            ncta_N,
            ((K_cluster + smem_K - 1) / smem_K + ring_depth) @ ring_buffer_by(ring_depth),
        ] @ CudaMbarrierPreArrive(ring_depth)

        raw: barrier[
            ncta_M,
            ncta_N,
            ((K_cluster + smem_K - 1) / smem_K + 0) @ ring_buffer_by(ring_depth),
        ] @ CudaMbarrierPreArrive(0)

        A_smem: A_type[ncta_M, ncta_N, ring_depth, cta_M, smem_K] @ Sm90_SmemSwizzled(swizzle)
        B_smem: B_type[ncta_M, ncta_N, ring_depth, cta_N / 64, smem_K, 64] @ Sm90_SmemSwizzled(swizzle)
        wgmma_cg: barrier[ncta_M, ncta_N, 2] @ Sm90_WgmmaCommitGroup

        for iter_k in seq(0, ((K_cluster + smem_K - 1) / smem_K)):
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
                # This is where Exo-GPU TMA has weakness.
                # Loop order is n_outer, k, n_inner
                # And only the last 2 for loops are subsumed by the TMA instr.
                # ThunderKittens uses clever striding to automate all 3 dimensions.
                for cta_n in cuda_threads(0, ncta_N, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N)):
                    for n_outer in seq(0, cta_N / 64):
                        Sm90_tma_load_multicast_2d(
                            B_smem[:, cta_n, iter_k % ring_depth, n_outer, :, :],
                            B_win[
                                cta_n * cta_N + n_outer * 64:
                                cta_n * cta_N + n_outer * 64 + 64,
                                iter_k * smem_K :
                                iter_k * smem_K + smem_K,
                            ],
                            ncta=ncta_M, cta_stride=ncta_N, size0=smem_K, size1=64,
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
                            if A_in_rmem:
                                # Load A from SMEM -> RMEM, B from SMEM.
                                # This exercises the wgmma instruction's support for accepting A from RMEM
                                # instead of SMEM. For gemm, this feature is pointless (this is just for testing).
                                A_rmem: A_type[4, wg_M/64, 16, smem_K] @ Sm90_TkRmemTileA(smem_K)
                                for ms in seq(0, M_wg_tiles, pragma_unroll=0):
                                    for w in cuda_threads(0, 4, unit=cuda_warp):
                                        cuda_tk_load_rs_inner_cols_64(
                                            A_rmem[w, ms, :, :],
                                            A_smem[
                                                cta_m, cta_n,
                                                iter_k % ring_depth:
                                                iter_k % ring_depth + 1,
                                                wg_m * wg_M + ms * 64 + w * 16 :
                                                wg_m * wg_M + ms * 64 + w * 16 + 16,
                                                :],
                                            dst=A_type, src=A_type, rows=16, outer_cols=1,
                                        )
                                Fence(wgmma_fence_1, wgmma_fence_2)
                                for ms in seq(0, M_wg_tiles, pragma_unroll=0):
                                    Sm90_tk_mma_rmem_row(
                                        D_rmem[cta_m, cta_n, wg_m, :, ms, :, :],
                                        A_rmem[:, ms, :, :],
                                        B_smem[cta_m, cta_n, iter_k % ring_depth, :, :, :],
                                        D=D_type, A=A_type, B=B_type, N64=cta_N // 64, K=smem_K,
                                    )
                                Arrive(wgmma_async) >> wgmma_cg[cta_m, cta_n, wg_m]
                                Await(wgmma_cg[cta_m, cta_n, wg_m], cuda_in_order, 0)
                            else:
                                # Normal path. Load A and B from SMEM.
                                Fence(wgmma_fence_1, wgmma_fence_2)
                                for ms in seq(0, M_wg_tiles, pragma_unroll=0):
                                    Sm90_tk_mma_row_row(
                                        D_rmem[cta_m, cta_n, wg_m, :, ms, :, :],
                                        A_smem[cta_m, cta_n, iter_k % ring_depth,
                                               (wg_m * M_wg_tiles + ms) * 64 :
                                               (wg_m * M_wg_tiles + ms) * 64 + 64, :],
                                        B_smem[cta_m, cta_n, iter_k % ring_depth, :, :, :],
                                        D=D_type, A=A_type, B=B_type, N64=cta_N // 64, K=smem_K,
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
                        #
                        # NB this synchronization is suboptimal for the A_in_rmem path,
                        # since it already waits for the wgmma. But A_in_rmem is just a quick test.
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
                    Arrive(cuda_in_order
                        ) >> war[cta_m, :, ((K_cluster + smem_K - 1) / smem_K + ring_depth - 1)
                        ] >> war[:, cta_n, ((K_cluster + smem_K - 1) / smem_K + ring_depth - 1)]

    main_loop = sched_cut_sync_iter_k(main_loop, config)
    return simplify(main_loop)


def handwrite_row_col_ping_pong_main_loop(config: GemmConfig):
    A_type = ScalarInfo(config.A_type)
    B_type = ScalarInfo(config.B_type)
    D_type = ScalarInfo(config.C_type)
    ncta_M = config.ncta_M
    ncta_N = config.ncta_N
    cta_M = config.cta_M
    cta_N = config.cta_N
    wg_M = perfect_div(config.cta_M, 2)
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
    assert config.ping_pong

    # fmt: off
    @proc
    def main_loop(
        K_cluster: size,
        D_rmem: [D_type][ncta_M, ncta_N, 2, 4, M_wg_tiles, 16, cta_N],
        A_win: [A_type][cluster_M, K_cluster],
        B_win: [B_type][cluster_N, K_cluster],
    ):
        assert stride(A_win, 1) == 1
        assert stride(B_win, 1) == 1
        war: barrier[
            ncta_M,
            ncta_N,
            2,
            ((K_cluster + smem_K - 1) / smem_K + ring_depth) @ ring_buffer_by(ring_depth),
        ] @ CudaMbarrierPreArrive(ring_depth)

        raw: barrier[
            ncta_M,
            ncta_N,
            2,
            ((K_cluster + smem_K - 1) / smem_K + 0) @ ring_buffer_by(ring_depth),
        ] @ CudaMbarrierPreArrive(0)

        A_smem: A_type[
            ncta_M,
            ncta_N,
            2,
            ((K_cluster + smem_K - 1) / smem_K + 0) @ ring_buffer_by(ring_depth),
            wg_M,
            smem_K,
        ].ring_guarded_by(war) @ Sm90_SmemSwizzled(swizzle)
        B_smem: B_type[
            ncta_M,
            ncta_N,
            2,
            ((K_cluster + smem_K - 1) / smem_K + 0) @ ring_buffer_by(ring_depth),
            cta_N,
            smem_K,
        ].ring_guarded_by(war) @ Sm90_SmemSwizzled(swizzle)
        wgmma_cg: barrier[ncta_M, ncta_N, 2] @ Sm90_WgmmaCommitGroup

        for iter_k in seq(0, ((K_cluster + smem_K - 1) / smem_K)):
            with CudaWarps(name="producer"):
                for ping in cuda_threads(0, 2, unit=cuda_warp):
                    # Each half-CTA waits for its respective write-after-read protection barrier.
                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            Await(war[cta_m, cta_n, ping, iter_k], cuda_temporal, 0)
                    # CTAs cooperate along the N dimension to multicast the needed tiles of A.
                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        Sm90_tma_load_multicast_2d(
                            A_smem[cta_m, :, ping, iter_k, :, :],
                            A_win[
                                cta_m * cta_M + wg_M * ping :
                                cta_m * cta_M + wg_M * ping + wg_M,
                                iter_k * smem_K :
                                iter_k * smem_K + smem_K,
                            ],
                            ncta=ncta_N, cta_stride=1, size0=wg_M, size1=smem_K,
                            smem_box=smem_box_A, dst=A_type, src=A_type,
                        ) >> raw[cta_m, :, ping, iter_k]
                    # CTAs cooperate along the M dimension to multicast the needed tiles of B.
                    for cta_n in cuda_threads(0, ncta_N, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N)):
                        Sm90_tma_load_multicast_2d(
                            B_smem[:, cta_n, ping, iter_k, :, :],
                            B_win[
                                cta_n * cta_N :
                                cta_n * cta_N + cta_N,
                                iter_k * smem_K :
                                iter_k * smem_K + smem_K,
                            ],
                            ncta=ncta_M, cta_stride=ncta_N, size0=cta_N, size1=smem_K,
                            smem_box=smem_box_B, dst=B_type, src=B_type,
                        ) >> raw[:, cta_n, ping, iter_k]
                    # Each half-CTA arrives on read-after-write protection barriers.
                    # Multicast to any CTA with the same cta_m OR the same cta_n.
                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            Arrive(cuda_temporal) >> raw[cta_m, :, ping, iter_k] >> raw[:, cta_n, ping, iter_k]
            with CudaWarps(name="consumer"):
                for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                    for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                        for ping in cuda_threads(0, 2, unit=cuda_warpgroup):
                            # Each half-CTA waits for its respective read-after-write protection barrier.
                            Await(raw[cta_m, cta_n, ping, iter_k], cuda_generic_and_async_proxy, 0)
                            # Each warpgroup does its own WGMMAs, then arrives on its own commit group.
                            Fence(wgmma_fence_1, wgmma_fence_2)
                            for ms in seq(0, M_wg_tiles, pragma_unroll=0):
                                Sm90_tk_mma_row_col(
                                    D_rmem[cta_m, cta_n, ping, :, ms, :, :],
                                    A_smem[cta_m, cta_n, ping, iter_k,
                                           ms * 64 :
                                           ms * 64 + 64, :],
                                    B_smem[cta_m, cta_n, ping, iter_k, :, :],
                                    D=D_type, A=A_type, B=B_type, N=cta_N, K=smem_K,
                                )
                            Arrive(wgmma_async) >> wgmma_cg[cta_m, cta_n, ping]
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
                        Arrive(cuda_in_order
                            ) >> war[cta_m, :, wg_m, ((K_cluster + smem_K - 1) / smem_K + ring_depth - 1)
                            ] >> war[:, cta_n, wg_m, ((K_cluster + smem_K - 1) / smem_K + ring_depth - 1)]

    # Does nothing by default (this is just for the Exo test suite)
    main_loop = inject_ping_pong_main_loop_bug(main_loop, config)

    main_loop = sched_cut_sync_iter_k(main_loop, config)
    return simplify(main_loop)


def inject_ping_pong_main_loop_bug(main_loop, config):
    # Really hard-wired cursor movement since PAST doesn't search for Exo-GPU features.
    loops = main_loop.find_all("for iter_k in _:_")
    assert len(loops) == 1
    loop_c = loops[0]
    iter_k_body = loop_c.body()
    assert len(iter_k_body) == 2, "expected producer and consumer block"
    producer_ping_loop = iter_k_body[0].only_child()
    assert producer_ping_loop.name() == "ping"
    producer_body = producer_ping_loop.body()

    # There should be an Await loop, 2 TMA, an Arrive.
    assert len(producer_body) == 4
    await_stmt_idx = 0
    await_loops_c = producer_body[await_stmt_idx]
    await_c = await_loops_c.only_child(2)
    assert await_c.name() == "war"
    assert await_c.first_sync_tl() == None
    assert await_c.second_sync_tl() == cuda_temporal
    arrive_stmt_idx = 3
    arrive_loops_c = producer_body[arrive_stmt_idx]
    arrive_c = arrive_loops_c.only_child(2)
    assert arrive_c.name() == "raw"

    if config.bug == GemmTestBug.ping_pong_reorder_await_war:
        main_loop = reorder_stmts(
            main_loop, producer_body[await_stmt_idx : await_stmt_idx + 2]
        )
    if config.bug == GemmTestBug.ping_pong_reorder_arrive_raw:
        main_loop = reorder_stmts(
            main_loop, producer_body[arrive_stmt_idx - 1 : arrive_stmt_idx + 1]
        )

    return main_loop


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
    enable_split_k = bool(config.enable_split_k)

    assert config.C_major == "row", "not supported"
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

    # (Ignore me) artificial bug injection for sync_check testing.
    _cuda_generic_and_async_proxy = cuda_generic_and_async_proxy
    if config.bug == GemmTestBug.coop_wrong_tma_to_gmem_timeline:
        _cuda_generic_and_async_proxy = cuda_in_order

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
                Fence(cuda_in_order, _cuda_generic_and_async_proxy)
                with CudaWarps(3, 4, name="producer"):
                    for ns in seq(0, outer_N):
                        if enable_split_k:
                            Sm90_tma_reduce_add_2d(
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
                        else:
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


def handwrite_ping_pong_epilogue(config: GemmConfig):
    C_type = ScalarInfo(config.C_type)
    D_type = C_type
    ncta_M = config.ncta_M
    ncta_N = config.ncta_N
    cta_M = config.cta_M
    cta_N = config.cta_N
    wg_M = perfect_div(cta_M, 2)
    cluster_M = ncta_M * cta_M
    cluster_N = ncta_N * cta_N
    ring_depth = config.ring_depth
    swizzle = config.swizzle

    M_wg_tiles = perfect_div(cta_M, 128)
    smem_box_C = config.make_smem_box_C()
    enable_split_k = bool(config.enable_split_k)

    assert config.C_major == "row", "not supported"
    assert config.ping_pong

    # Helper for staging RMEM into SMEM.
    # The [wg_M, cta_N] logical RMEM tile has its cta_n dimension split
    # to [wg_M, outer_N, inner_N] then re-ordered
    # to [outer_N, wg_M, inner_N]. This is dictated by the TMA.
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

        C_barrier: barrier[ncta_M, ncta_N, 3 @ ring_buffer_by(2)] @ CudaMbarrierPreArrive(1)
        tmp_barrier: barrier[ncta_M, ncta_N, 2 @ ring_buffer_by(2)] @ CudaMbarrierPreArrive(0)

        C_smem: C_type[
            ncta_M,
            ncta_N,
            2 @ ring_buffer_by(1),
            outer_N,
            wg_M,
            inner_N,
        ].ring_guarded_by(C_barrier) @ Sm90_SmemSwizzled(swizzle)

        for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
            for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                with CudaWarps(name="consumer"):
                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                        # Wait for "ring buffer" slot.
                        Await(C_barrier[cta_m, cta_n, wg_m], cuda_temporal, 0)

                        for w in cuda_threads(0, 4, unit=cuda_warp):
                            # Each warp writes a (16, cta_N) tile to SMEM.
                            # This is repeated if cta_M > 128.
                            for ms in seq(0, M_wg_tiles, pragma_unroll=0):
                                store_smem(
                                    C_smem[cta_m, cta_n, wg_m, :,
                                        ms * 64 + w * 16:
                                        ms * 64 + w * 16 + 16,
                                        :],
                                    D_rmem[cta_m, cta_n, wg_m, w, ms, :, :],
                                )

                        # TODO this should be a fence.
                        Arrive(cuda_in_order) >> tmp_barrier[cta_m, cta_n, wg_m]
                        Await(tmp_barrier[cta_m, cta_n, wg_m], cuda_generic_and_async_proxy, 0)

                        with CudaWarps(0, 1):
                            for ns in seq(0, outer_N):
                                if enable_split_k:
                                    Sm90_tma_reduce_add_2d(
                                        C_win[
                                            cta_M * cta_m + wg_M * wg_m :
                                            cta_M * cta_m + wg_M * wg_m + wg_M,
                                            cta_N * cta_n + ns * inner_N :
                                            cta_N * cta_n + ns * inner_N + inner_N,
                                        ],
                                        C_smem[cta_m, cta_n, wg_m, ns, :, :],
                                        dst=C_type, src=D_type, size0=wg_M, size1=inner_N,
                                        smem_box=smem_box_C, swizzle=128,
                                    )
                                else:
                                    Sm90_tma_store_2d(
                                        C_win[
                                            cta_M * cta_m + wg_M * wg_m :
                                            cta_M * cta_m + wg_M * wg_m + wg_M,
                                            cta_N * cta_n + ns * inner_N :
                                            cta_N * cta_n + ns * inner_N + inner_N,
                                        ],
                                        C_smem[cta_m, cta_n, wg_m, ns, :, :],
                                        dst=C_type, src=D_type, size0=wg_M, size1=inner_N,
                                        smem_box=smem_box_C, swizzle=128,
                                    )
                                tma_cg: barrier @ Sm90_TmaCommitGroup
                                Arrive(tma_to_gmem_async) >> tma_cg
                                Await(tma_cg, cuda_in_order, 0)

                            # Free "ring buffer" slot.
                            Arrive(cuda_in_order) >> C_barrier[cta_m, cta_n, wg_m + 1]

    return simplify(epilogue)


def handwrite_row_col_gemm(config: GemmConfig):
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
    enable_split_k = bool(config.enable_split_k)

    # Each warp stores (16 x cta_N) tiles.
    D_tile_mem = Sm90_TkRmemTileD(cta_N)

    coop = not config.ping_pong
    if coop:
        main_loop = handwrite_row_col_coop_main_loop(config)
        epilogue = handwrite_coop_epilogue(config)
    else:
        main_loop = handwrite_row_col_ping_pong_main_loop(config)
        epilogue = handwrite_ping_pong_epilogue(config)

    sync_before_epilogue = (
        coop and config.bug != GemmTestBug.coop_missing_cluster_sync_before_epilogue
    )
    sync_after_epilogue = (
        coop and config.bug != GemmTestBug.coop_missing_cluster_sync_after_epilogue
    )

    # fmt: off
    @proc
    def gemm(
        L: size,
        M: size,
        N: size,
        K_split: size,
        K_cluster: size,
        # [batch, m, task_k, ks]
        # Note, k = task_k * K_cluster + ks.
        # We have to split the dim due to affine indexing restrictions.
        # This consumes an extra dimension of the tensor map.
        A: A_type[L, M, K_split, K_cluster] @ CudaGmemLinear,
        B: B_type[L, N, K_split, K_cluster] @ CudaGmemLinear,
        C: C_type[L, M, N] @ CudaGmemLinear,
    ):
        assert stride(A, 3) == 1
        assert stride(B, 3) == 1
        assert stride(C, 2) == 1
        assert L % L_divisor == 0
        assert M % M_divisor == 0
        assert N % N_divisor == 0
        assert K_cluster > 0
        assert K_cluster % K_cluster_divisor == 0

        A_tensorMap = A[:,:,:,:] @ Sm90_tensorMap(swizzle, *smem_box_A)
        B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(swizzle, *smem_box_B)
        C_tensorMap = C[:,:,:] @ Sm90_tensorMap(swizzle, *smem_box_C)

        if enable_split_k:
            cudaMemsetAsync0_3d(L, M, N, C[:, :, :], dst=C_type)

        with CudaDeviceFunction(
            clusterDim=ncta_M * ncta_N,
            warp_config=default_warp_config,
            blocks_per_sm=1,
            unsafe_no_shutdown_cluster_sync=coop,  # Co-op schedule already does cluster-wide sync
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
                                K_cluster,
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
                            # This aliasing only occurs if not using the ping-pong schedule.
                            if sync_before_epilogue:
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
                            if sync_after_epilogue:
                                Fence(cuda_in_order, cuda_in_order)

    gemm = sched_final_changes(gemm, config)
    return gemm


def handwrite_row_row_gemm(config: GemmConfig):
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
    enable_split_k = bool(config.enable_split_k)

    # Each warp stores (16 x cta_N) tiles.
    D_tile_mem = Sm90_TkRmemTileD(cta_N)

    coop = not config.ping_pong
    if coop:
        main_loop = handwrite_row_row_coop_main_loop(config)
        epilogue = handwrite_coop_epilogue(config)
    else:
        assert 0, "ping-pong not supported"

    sync_before_epilogue = (
        coop and config.bug != GemmTestBug.coop_missing_cluster_sync_before_epilogue
    )
    sync_after_epilogue = (
        coop and config.bug != GemmTestBug.coop_missing_cluster_sync_after_epilogue
    )

    # fmt: off
    @proc
    def gemm(
        L: size,
        M: size,
        N: size,
        K_split: size,
        K_cluster: size,
        # [batch, m, task_k, ks]
        # Note, k = task_k * K_cluster + ks.
        # We have to split the dim due to affine indexing restrictions.
        # This consumes an extra dimension of the tensor map.
        A: A_type[L, M, K_split, K_cluster] @ CudaGmemLinear,
        B: B_type[L, K_split, K_cluster, N] @ CudaGmemLinear,
        C: C_type[L, M, N] @ CudaGmemLinear,
    ):
        assert stride(A, 3) == 1
        assert stride(B, 3) == 1
        assert stride(C, 2) == 1
        assert L % L_divisor == 0
        assert M % M_divisor == 0
        assert N % N_divisor == 0
        assert K_cluster > 0
        assert K_cluster % K_cluster_divisor == 0

        A_tensorMap = A[:,:,:,:] @ Sm90_tensorMap(swizzle, *smem_box_A)
        B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(swizzle, *smem_box_B)
        C_tensorMap = C[:,:,:] @ Sm90_tensorMap(swizzle, *smem_box_C)

        if enable_split_k:
            cudaMemsetAsync0_3d(L, M, N, C[:, :, :], dst=C_type)

        with CudaDeviceFunction(
            clusterDim=ncta_M * ncta_N,
            warp_config=default_warp_config,
            blocks_per_sm=1,
            unsafe_no_shutdown_cluster_sync=coop,  # Co-op schedule already does cluster-wide sync
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
                                K_cluster,
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
                                    task_k,
                                    :,
                                    task_n * cluster_N :
                                    task_n * cluster_N + cluster_N,
                                ],
                            )

                            # This cluster-wide sync is required for the main_loop
                            # and the epilogue to safely alias SMEM.
                            # This aliasing only occurs if not using the ping-pong schedule.
                            if sync_before_epilogue:
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
                            if sync_after_epilogue:
                                Fence(cuda_in_order, cuda_in_order)

    gemm = sched_final_changes(gemm, config)
    return gemm


def handwrite_gemm(config: GemmConfig, sporkbench_cases: Optional[list] = None):
    assert config.A_major == "row", "not supported"
    assert config.C_major == "row", "not supported"

    if config.A_in_rmem:
        assert config.A_major == "row" and config.B_major == "row", "not supported"

    if config.B_major == "col":
        gemm = handwrite_row_col_gemm(config)
    else:
        gemm = handwrite_row_row_gemm(config)

    if sporkbench_cases is not None:
        sporkbench_cases.append(config.make_sporkbench_case())
    return gemm


def find_parent_loop(p: Procedure, cursor, iter_name):
    cursor = p.forward(cursor)
    while not (isinstance(cursor, ForCursor) and cursor.name() == iter_name):
        cursor = cursor.parent()
    return cursor


def schedule_gemm(config: GemmConfig, cases=None):
    # fmt: off
    assert config.A_major == "row", "not supported"
    assert config.B_major == "col", "not supported"
    assert config.C_major == "row", "not supported"
    assert not config.ping_pong, "not supported"
    assert config.bug == GemmTestBug.none, "not supported"
    assert not config.A_in_rmem, "not supported"

    coop = not config.ping_pong

    A_type = ScalarInfo(config.A_type)
    B_type = ScalarInfo(config.B_type)
    C_type = ScalarInfo(config.C_type)
    D_type = ScalarInfo(config.C_type)
    swizzle = config.swizzle
    assert swizzle == 128, "not supported"

    unsafe = False
    enable_split_k = config.enable_split_k
    smem_M = config.cta_M
    smem_N = config.cta_N
    smem_K = config.make_smem_K()
    ring_depth = config.ring_depth
    ncta_M = config.ncta_M
    ncta_N = config.ncta_N

    # Derived constants
    wg_M = smem_M // 2
    wg_N = smem_N
    cluster_M = smem_M * ncta_M
    cluster_N = smem_N * ncta_N

    # CudaDeviceFunction context
    my_warp_config = [
        CudaWarpConfig("producer", 4, setmaxnreg_dec=40),  # 4 producer warps (some unused)
        CudaWarpConfig("consumer", 8, setmaxnreg_inc=232), # 2 consumer warpgroups (8 warps)
    ]
    cuda_device_function_ctx = CudaDeviceFunction(
        clusterDim=ncta_M * ncta_N,
        warp_config=my_warp_config,
        blocks_per_sm=1,
        unsafe_no_shutdown_cluster_sync=coop,  # Co-op schedule already does cluster-wide sync
    )

    smem_box_A = config.make_smem_box_A()
    smem_box_B = config.make_smem_box_B()
    smem_box_C = config.make_smem_box_C()

    @proc
    def gemm(
            L: size, M: size, N: size, K_split: size, K_cluster: size,
            A: A_type[L, M, K_split, K_cluster] @ CudaGmemLinear,  # Row-major (K-major)
            B: B_type[L, N, K_split, K_cluster] @ CudaGmemLinear,  # Column-major (K-major)
            C: C_type[L, M, N] @ CudaGmemLinear,  # Row-major
    ):
        assert L > 0
        assert M > 0
        assert N > 0
        assert K_cluster > 0
        assert L % L_divisor == 0
        assert M % M_divisor == 0
        assert N % N_divisor == 0
        assert K_cluster % K_cluster_divisor == 0
        if enable_split_k:
            for memset_batch in seq(0, L):
                for memset_m in seq(0, M):
                    for memset_n in seq(0, N):
                        C[memset_batch, memset_m, memset_n] = 0

        A_tensorMap = A[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_A)
        B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_B)
        C_tensorMap = C[:,:,:] @ Sm90_tensorMap(128, *smem_box_C)

        for batch in seq(0, L):
            for task_k in seq(0, K_split):
                for m in seq(0, M):
                    for n in seq(0, N):
                        D_rmem: f32
                        D_rmem = 0
                        for k in seq(0, K_cluster):
                            D_rmem += A_tensorMap[batch, m, task_k, k] * B_tensorMap[batch, n, task_k, k]
                        if enable_split_k:
                            C_tensorMap[batch, m, n] += D_rmem
                        else:
                            C[batch, m, n] = D_rmem

    gemm = simplify(gemm)  # Get rid of enable_split_k if stmts.

    # Extract cursors to initial proc.
    batch_loop = gemm.find_loop("batch")
    task_k_loop = gemm.find_loop("task_k")
    m_loop = gemm.find_loop("m")
    n_loop = gemm.find_loop("n")
    k_loop = gemm.find_loop("k")
    D_rmem = gemm.find_alloc_or_arg("D_rmem")
    D_zero = gemm.find("D_rmem = 0")
    if enable_split_k:
        C_assign = gemm.find("_ += D_rmem")
    else:
        C_assign = gemm.find("_ = D_rmem")
    gap_before_main = k_loop.before()
    gap_after_main = k_loop.after()

    # Set up cuda_tasks loops and CudaDeviceFunction.
    gemm = set_loop_mode(gemm, batch_loop, CudaTasks)
    gemm = set_loop_mode(gemm, task_k_loop, CudaTasks)
    gemm = divide_loop(gemm, m_loop, cluster_M, ("task_m", "sub_task_m"), tail="guard")
    task_m_loop = gemm.forward(m_loop)
    sub_task_m_loop = task_m_loop.only_child()
    gemm = set_loop_mode(gemm, task_m_loop, CudaTasks)
    gemm = divide_loop(gemm, n_loop, cluster_N, ("task_n", "sub_task_n"), tail="guard")
    task_n_loop = gemm.forward(n_loop)
    sub_task_n_loop = task_n_loop.only_child()
    gemm = set_loop_mode(gemm, task_n_loop, CudaTasks)
    gemm = wrap_with_context(gemm, batch_loop, cuda_device_function_ctx)

    # Move task_n loop outside, under batch, task_k loops.
    # TODO is there a smart way to do this?
    gemm = lift_scope(gemm, task_n_loop)
    gemm = lift_scope(gemm, task_n_loop)
    gemm = lift_scope(gemm, task_n_loop)
    inner_task_loop = task_m_loop

    # Generate CTA loops.
    # These are supposed to be perfect, because the inner loop from the
    # task_m/task_n have constant bounds cluster_M, cluster_N.
    gemm = divide_loop(gemm, sub_task_m_loop, smem_M, ("cta_m", "sub_cta_m"), perfect=True)
    cta_m_loop = gemm.forward(sub_task_m_loop)
    pre_fission_sub_cta_m_loop = cta_m_loop.only_child()
    gemm = set_loop_mode(gemm, cta_m_loop, CudaThreads(unit=ncta_N * cuda_cta_in_cluster))
    gemm = divide_loop(gemm, sub_task_n_loop, smem_N, ("cta_n", "sub_cta_n"), perfect=True)
    n_cta_loop = gemm.forward(sub_task_n_loop)
    gemm = set_loop_mode(gemm, n_cta_loop, CudaThreads(unit=cuda_cta_in_cluster))

    # Move cta_n loop outside for/if to be just under cta_m loop.
    gemm = lift_scope(gemm, n_cta_loop)
    gemm = lift_scope(gemm, n_cta_loop)

    # Divide the sub_cta_m loop into warpgroup loops.
    gemm = divide_loop(gemm, pre_fission_sub_cta_m_loop, wg_M, ("wg_m", "sub_wg_m"), perfect=True)
    gemm = set_loop_mode(gemm, pre_fission_sub_cta_m_loop, CudaThreads(unit=cuda_warpgroup))

    # Set up M loop structure within each warpgroup.
    # We will have
    #   * (ms) Outer loop: strides by 64
    #   * (mw) Middle loop: strides by 16 (warp loop)
    #   * (mi) Inner loop: strides by 1
    # We do not actually update the loop mode of the mw-loop, since
    # all mw loops will eventually be replaced by wgmma instructions.
    sub_wg_m = gemm.forward(pre_fission_sub_cta_m_loop).only_child()
    assert sub_wg_m.name() == "sub_wg_m"
    gemm = divide_loop(gemm, sub_wg_m, 16, ("ms_mw", "mi"), perfect=True)
    gemm = divide_loop(gemm, sub_wg_m, 4, ("ms", "mw"), perfect=True)
    ms_loop = gemm.forward(sub_wg_m)
    assert ms_loop.name() == "ms"
    gemm = set_loop_mode(gemm, ms_loop, Seq(pragma_unroll=2))

    # expand dim of D_rmem so each iteration uses its own D_rmem.
    # This enables future parallelization.
    # Set the memory type for wgmma accumulators.
    #
    # We have to order the mw (warp) dimension to the left
    # since it's a distributed dimension. This violates the typical
    # slow-to-fast dimension ordering, but is required by
    # Exo-GPU's design (questionable in hindsight).
    gemm = expand_dim(gemm, D_rmem, smem_N, "sub_cta_n")  # Rightmost dimension
    gemm = expand_dim(gemm, D_rmem, 16, "mi")
    gemm = expand_dim(gemm, D_rmem, wg_M // 64, "ms")
    gemm = expand_dim(gemm, D_rmem, 4, "mw")
    gemm = expand_dim(gemm, D_rmem, 2, "wg_m")
    gemm = expand_dim(gemm, D_rmem, ncta_N, "cta_n")
    gemm = expand_dim(gemm, D_rmem, ncta_M, "cta_m")  # Leftmost dimension
    gemm = set_memory(gemm, D_rmem, Sm90_TkRmemTileD(smem_N))
    D_rmem = gemm.forward(D_rmem)

    # Set up the main loop.
    # First we have to lift D_rmem out, then fission out the
    # zero prologue and GMEM-write epilogue.
    # Divide K loop to yield main loop (iter_k)
    # Move iter_k loop to be just under the tasks loops.
    #
    # Non-perfect K loop:
    # This is harder than for M/N, since we have to think about how
    # zero padding makes the extra K loads safe (D += 0 is no-op).
    gemm = divide_loop(gemm, k_loop, smem_K, ("iter_k", "sub_iter_k"), tail="guard")
    iter_k_loop = gemm.forward(k_loop)
    sub_iter_k_loop = iter_k_loop.only_child()
    k_lifts = 0
    parent = iter_k_loop.parent()
    while True:
        if isinstance(parent, ForCursor):
            if isinstance(parent.loop_mode(), CudaTasks):
                break
        k_lifts += 1
        parent = parent.parent()
    gemm = lift_alloc(gemm, D_rmem, n_lifts=k_lifts)
    gemm = fission(gemm, gap_before_main, n_lifts=k_lifts, unsafe_disable_checks=unsafe)
    gemm = fission(gemm, gap_after_main, n_lifts=k_lifts, unsafe_disable_checks=unsafe)
    for i in range(k_lifts):
        gemm = lift_scope(gemm, iter_k_loop)
    gap_before_main = gemm.forward(gap_before_main)
    gap_after_main = gemm.forward(gap_after_main)
    D_rmem = gemm.forward(D_rmem)
    iter_k_loop = gemm.forward(iter_k_loop)

    # Set up barrier objects for later use.
    raw_dims = [ncta_M, ncta_N, f"((K_cluster + {smem_K - 1}) / {smem_K})"]
    war_dims = [ncta_M, ncta_N, f"((K_cluster + {smem_K - 1}) / {smem_K}) + {ring_depth}"]
    cg_dims = [ncta_M, ncta_N, 2]
    war_mech = CudaMbarrierPreArrive(ring_depth)
    raw_mech = CudaMbarrierPreArrive(0)
    gemm = insert_barrier_alloc(gemm, iter_k_loop.before(), "raw", raw_dims, raw_mech, 2, ring_depth)
    gemm = insert_barrier_alloc(gemm, iter_k_loop.before(), "war", war_dims, war_mech, 2, ring_depth)
    gemm = insert_barrier_alloc(gemm, iter_k_loop.before(), "wgmma_cg", cg_dims, Sm90_WgmmaCommitGroup, None, None)

    # Stage A_smem, B_smem tiles above wg_m loop in swizzled memory.
    # TODO how to get a more stable reference to the input cursor.
    # We can't rely on the old cursor since it's forwarded to
    # the wrong loop after fission.
    # TODO sucky that we have to use f-strings here; PAST can't get local variables???
    wg_m_main_loop = iter_k_loop.only_child(3)
    assert wg_m_main_loop.name() == "wg_m"
    assert wg_m_main_loop.loop_mode().unit == cuda_warpgroup
    gemm = stage_mem(gemm, wg_m_main_loop,
        f"A_tensorMap[batch, "
        f"(task_m * {ncta_M} + cta_m) * {smem_M} : (task_m * {ncta_M} + cta_m + 1) * {smem_M}, "
        f"task_k, iter_k * {smem_K}: (iter_k + 1) * {smem_K}]",
        "A_smem",
        False,
        Sm90_SmemSwizzled(128),
    )
    gemm = stage_mem(gemm, wg_m_main_loop,
        f"B_tensorMap[batch, "
        f"(task_n * {ncta_N} + cta_n) * {smem_N} : (task_n * {ncta_N} + cta_n + 1) * {smem_N}, "
        f"task_k, iter_k * {smem_K}: (iter_k + 1) * {smem_K}]",
        "B_smem",
        False,
        Sm90_SmemSwizzled(128),
    )
    A_smem = gemm.find_alloc_or_arg("A_smem")
    B_smem = gemm.find_alloc_or_arg("B_smem")

    # Lift SMEM tiles to cluster scope, with ring buffering (based on iter_k).
    # Generate one shard per CTA in cluster.
    gemm = simplify(gemm)  # stage_mem generates exprs too hard for Exo to understand...
    for smem_cursor in (A_smem, B_smem):
        gemm = expand_dim(gemm, smem_cursor, ring_depth, f"iter_k % {ring_depth}")
        gemm = expand_dim(gemm, smem_cursor, ncta_N, "cta_n")
        gemm = expand_dim(gemm, smem_cursor, ncta_M, "cta_m")
        gemm = lift_alloc(gemm, smem_cursor, n_lifts=3)

    # Fission CTA loops before and after B_smem load.
    # TODO how to get these cursors more elegantly?
    wg_m_main_loop = gemm.forward(wg_m_main_loop)
    B_smem_loop = wg_m_main_loop.prev()
    A_smem_loop = B_smem_loop.prev()
    gemm = fission(gemm, B_smem_loop.before(), n_lifts=2, unsafe_disable_checks=unsafe)
    gemm = fission(gemm, B_smem_loop.after(), n_lifts=2, unsafe_disable_checks=unsafe)

    # We have to reorder the loops for the B_smem load to be cta_n, cta_m.
    # This is needed for multicasting.
    # CTAs with the same cta_m value will multicast the same tile of B @ GMEM.
    # Therefore, to substitute the instruction later, we need to have cta_m inner.
    # Unlike CPU Exo, transposing these parallel loops requires us to rewrite
    # the unit, to still have the same assignment of loop iters to CTAs (manual for now).
    A_smem_loop = gemm.forward(A_smem_loop)
    B_smem_loop = gemm.forward(B_smem_loop)
    B_smem_cta_n_loop = B_smem_loop.parent()
    B_smem_cta_m_loop = B_smem_cta_n_loop.parent()
    gemm = reorder_loops(gemm, B_smem_cta_m_loop)
    gemm = update_loop_mode(gemm, B_smem_cta_n_loop, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N))
    gemm = update_loop_mode(gemm, B_smem_cta_m_loop, unit=cuda_cta_in_cluster)

    # Insert arrive/await around SMEM load code.
    # These are per-CTA statements.
    # We then need to fission them from the SMEM load code,
    # because the (inner) CTA loop for the SMEM load is part of the TMA instr.
    gemm = insert_await(gemm, A_smem_loop.before(), "war[cta_m, cta_n, iter_k]", cuda_temporal, 0)
    gemm = fission(gemm, A_smem_loop.before(), n_lifts=1)
    gemm = insert_arrive(gemm, B_smem_loop.after(), cuda_temporal, ("raw[cta_m, :, iter_k]", "raw[:, cta_n, iter_k]"))
    gemm = fission(gemm, B_smem_loop.after(), n_lifts=1)

    # Substitute multicast TMA.
    A_smem_loop = gemm.forward(A_smem_loop)
    A_tma = A_smem_loop.parent()  # CTA loop (for multicast) is parent
    B_smem_loop = gemm.forward(B_smem_loop)
    B_tma = B_smem_loop.parent()  # CTA loop (for multicast) is parent
    gemm = unsafe_remove_if(gemm, A_smem_loop, True)
    A_tma = gemm.forward(A_tma)
    gemm = replace(
        gemm,
        A_tma,
        Sm90_tma_load_multicast_2d.partial(
            cta_stride=1,
            swizzle=swizzle,
            smem_box=smem_box_A,
        )
    )
    gemm = set_trailing_barrier_expr(gemm, A_tma, "raw[cta_m, :, iter_k]")
    gemm = unsafe_remove_if(gemm, B_smem_loop, True)
    gemm = replace(
        gemm,
        B_tma,
        Sm90_tma_load_multicast_2d.partial(
            cta_stride=ncta_N,
            swizzle=swizzle,
            smem_box=smem_box_B,
        )
    )
    gemm = set_trailing_barrier_expr(gemm, B_tma, "raw[:, cta_n, iter_k]")

    # wgmma M-warpgroup loop (children should be replaced with wgmma later)
    # Place mbarrier await now.
    # Arrive on mbarrier shall be inserted when we cut the iter_k loop.
    wg_m_main_loop = gemm.forward(wg_m_main_loop)
    gemm = insert_await(gemm, wg_m_main_loop.before(), "raw[cta_m, cta_n, iter_k]", cuda_generic_and_async_proxy, 0)

    # Wrap these future wgmma instrs with wgmma.fence before, cg arrive after
    # Note, cg Await will be handled when we cut the iter_k loop due to
    # we don't Await on iteration 0.
    #
    # Loop structure
    # for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
    #   Fence(wgmma_fence_1, wgmma_fence_2)
    #   for ms:
    #     for mw:  <-- replace with wgmma
    #       ...
    #         for sub_iter_k:
    # Arrive(...) >> ...
    wgmma_ms_cursor = gemm.forward(wg_m_main_loop).only_child(1)
    assert wgmma_ms_cursor.name() == "ms"
    wgmma_mw_cursor = wgmma_ms_cursor.only_child(1)
    assert wgmma_mw_cursor.name() == "mw"
    gemm = unsafe_remove_if(gemm, wgmma_ms_cursor, True)
    gemm = replace(gemm, wgmma_mw_cursor, Sm90_tk_mma_row_col)
    gemm = insert_fence(gemm, wgmma_ms_cursor.before(), wgmma_fence_1, wgmma_fence_2)
    # Arrive(wgmma_async, 1) >> wgmma_cg[cta_m, cta_n, wg_m]
    gemm = insert_arrive(gemm, wgmma_ms_cursor.after(), wgmma_async, "wgmma_cg[cta_m, cta_n, wg_m]")

    # Main loop warp specialization.
    # I think there's 3 statements at this level right now.
    # First two should be the A/B smem load, last one should be accum.
    iter_k_loop = gemm.forward(iter_k_loop)
    assert len(iter_k_loop.body()) == 3
    gemm = wrap_with_context(gemm, iter_k_loop.body()[:2], CudaWarps(0, 1, name="producer"))
    gemm = wrap_with_context(gemm, iter_k_loop.body()[2], CudaWarps(name="consumer"))

    # Finalize zero prologue.
    # We have to unsafely remove the if-guards to substitute the zero instr.
    D_zero = gemm.forward(D_zero)
    zero_m_loop = find_parent_loop(gemm, D_zero, "wg_m")
    gemm = unsafe_remove_if(gemm, zero_m_loop, True)
    gemm = wrap_with_context(gemm, zero_m_loop, CudaWarps(name="consumer"))
    zero_m_loop = gemm.forward(zero_m_loop)
    gemm = replace(gemm, zero_m_loop.find("for mw in _:_"), Sm90_tk_zero_scale_d)

    # Finalize write-to-C epilogue.
    # Need to wait for wgmma beforehand, and do a final arrive to war barrier.
    # This wait loop is fissioned out from the epilogue.
    cta_m_epilogue = find_parent_loop(gemm, C_assign, "cta_m")
    assert cta_m_epilogue.loop_mode().unit == ncta_N * cuda_cta_in_cluster
    wg_m_sync_loop = cta_m_epilogue.only_child(2)
    assert wg_m_sync_loop.loop_mode().unit == cuda_warpgroup

    # Await for the commit group inside the consumer wg_m loop (wargroup scope)
    # then Arrive on the mbarrier war's final ring buffer entry,
    # just outside the wg_m loop (at CTA scope).
    #
    # This is where we do the fissioning, to split off the
    # rest of the epilogue to be after the mbarrier arrive.
    gemm = insert_await(gemm, wg_m_sync_loop.body().before(), "wgmma_cg[cta_m, cta_n, wg_m]", cuda_in_order, 0)
    wg_m_sync_loop = gemm.forward(wg_m_sync_loop)
    gemm = fission(gemm, wg_m_sync_loop.body()[0].after(), n_lifts=3)
    # NB cursor is forwarded to the first loop ... arbitrary ahh decision
    # Fissioned gave us (sync loop, epilogue loop)
    # with forwarding by default to the sync_loop, so we adjust cta_m_epilogue.
    cta_m_epilogue = gemm.forward(cta_m_epilogue).next()
    gemm = insert_arrive(gemm, wg_m_sync_loop.after(), cuda_in_order, [
        f"war[cta_m, :, ((K_cluster + {smem_K - 1}) / {smem_K} + {ring_depth - 1})]",
        f"war[:, cta_n, ((K_cluster + {smem_K - 1}) / {smem_K} + {ring_depth - 1})]",
    ])
    cta_m_sync_loop = find_parent_loop(gemm, wg_m_sync_loop, "cta_m")
    gemm = wrap_with_context(gemm, cta_m_sync_loop, CudaWarps(name="consumer"))

    # Got cursor to D -> C epilogue cta_m loop (from before).
    # We now remove the if-guarding here in preparation
    # for using TMA.
    assert cta_m_epilogue.loop_mode().unit == ncta_N * cuda_cta_in_cluster
    gemm = unsafe_remove_if(gemm, cta_m_epilogue, True)

    # Replace C_tensorMap {+=|=} D_rmem with
    # C_smem = D_rmem; C_tensorMap {+=|=} C_smem
    # NB temporarily, the SMEM is at warpgroup scope.
    cta_m_epilogue = gemm.forward(cta_m_epilogue)
    epilogue_ms_loop = cta_m_epilogue.find("for ms in _:_")
    gemm = stage_mem(gemm, epilogue_ms_loop,
        f"D_rmem[cta_m, cta_n, wg_m, 0:4, 0:{wg_M // 64}, 0:16, 0:{smem_N}]",
        "C_smem",
        False,  # accum
        Sm90_SmemSwizzled(swizzle),
    )
    # This is not great, should have stage_mem give back cursors.
    C_smem = gemm.find_alloc_or_arg("C_smem")
    C_smem_init = C_smem.next()

    # Expand and lift C_smem from warpgroup to cluster scope.
    C_smem = gemm.forward(C_smem)
    gemm = expand_dim(gemm, C_smem, 2, "wg_m")
    gemm = expand_dim(gemm, C_smem, ncta_N, "cta_n")
    gemm = expand_dim(gemm, C_smem, ncta_M, "cta_m")
    gemm = simplify(gemm)
    gemm = lift_alloc(gemm, C_smem, n_lifts=3)

    # Helper for staging RMEM into SMEM.
    # The [cta_M, cta_N] logical RMEM tile has its cta_n dimension split
    # to [cta_M, outer_N, inner_N] then re-ordered
    # to [outer_N, cta_M, inner_N]. This is dictated by the TMA.
    # The returned advice.instr magically does the copy from RMEM to this SMEM layout.
    advice: CudaTkRsInstrAdvice = cuda_tk_store_rs_advice(
        16, smem_N, dst=C_type, src=D_type, swizzle=swizzle
    )
    store_smem = advice.instr
    outer_N = advice.outer_cols
    inner_N = advice.inner_cols

    # Need to fuse and reorder stuff to put C_smem in the correct layout.
    #   Before: [cta_m, cta_n, wg_m, mw, ms, mi, n]
    #   Middle: [cta_m, cta_n, outer_N, wg_m, ms, mw, mi, inner_N]
    #   After: [cta_m, cta_n, outer_N, m, inner_N]
    gemm = divide_dim(gemm, C_smem, 6, inner_N)
    gemm = rearrange_dim(gemm, C_smem, [0, 1, 6, 2, 4, 3, 5, 7])
    for m_dim in range(0, 3):
        gemm = mult_dim(gemm, C_smem, 3, 4)

    # Each warp writes registers to shared memory.
    # Autogenerated loop structure is
    #   warp loop (i0)
    #     ms loop (i1)
    #       mi loop (i2)
    #         n loop (i3)
    # We have to update the warp/ms loop modes and split the N loop.
    # Also wrap it with CudaWarps(name="consumer")
    C_smem_init = gemm.forward(C_smem_init)
    gemm = set_loop_mode(gemm, C_smem_init, CudaThreads(unit=cuda_warp))
    gemm = set_loop_mode(gemm, C_smem_init.only_child(1), Seq(pragma_unroll=2))
    gemm = divide_loop(gemm, C_smem_init.only_child(3), inner_N, ("no", "ni"), perfect=True)
    gemm = simplify(gemm)
    gemm = replace(gemm, C_smem_init.only_child(2), store_smem)

    # The original C_tensorMap {+=|=} ... loop must be replaced with TMA.
    # This requires fusing away the M loops and dividing the N loop.
    epilogue_ms_loop = find_parent_loop(gemm, C_assign, "ms")
    gemm = fission(gemm, epilogue_ms_loop.before(), n_lifts=1)
    epilogue_ms_loop = gemm.forward(epilogue_ms_loop)
    epilogue_wg_m_loop = epilogue_ms_loop.parent()
    epilogue_sub_cta_n_loop = epilogue_ms_loop.only_child(3)
    assert epilogue_sub_cta_n_loop.name() == "sub_cta_n"
    gemm = divide_loop(gemm, epilogue_sub_cta_n_loop, inner_N, ("tma_no", "tma_ni"), perfect=True)
    assert epilogue_wg_m_loop.loop_mode().unit == cuda_warpgroup
    assert epilogue_wg_m_loop.only_child(3).name() == "mi"
    for m_dim in range(2, -1, -1):
        # XXX if we don't do the simplify each time, the scheduling
        # doesn't work. Same with going in reverse order.
        gemm = mult_loops(gemm, epilogue_wg_m_loop.only_child(m_dim), "tma_m")
        gemm = simplify(gemm)
        epilogue_wg_m_loop = gemm.forward(epilogue_wg_m_loop)
    tma_to_gmem = gemm.forward(epilogue_wg_m_loop)
    assert tma_to_gmem.only_child(1).name() == "tma_no"

    # This is where Exo-GPU TMA has weakness.
    # Loop order is tma_m, tma_no, tma_ni.
    # Reorder to tma_no, tma_m, tma_ni.
    # And only the last 2 for loops are subsumed by the TMA instr.
    # ThunderKittens uses clever striding to automate all 3 dimensions.
    gemm = reorder_loops(gemm, tma_to_gmem)
    tma_to_gmem = gemm.forward(tma_to_gmem)
    tma_no_loop = tma_to_gmem.parent()
    assert tma_no_loop.name() == "tma_no"
    tma_instr = Sm90_tma_reduce_add_2d if enable_split_k else Sm90_tma_store_2d
    gemm = replace(gemm, tma_to_gmem, tma_instr.partial(swizzle=swizzle, smem_box=smem_box_C))

    # Each CTA assigns the producer warp to do the TMA, and syncs before and after it.
    # The Fence has to be outside the with CudaWarps, hence the fissioning.
    # NB this is where we rely on the fact that (with = if) under the hood.
    gemm = insert_barrier_alloc(gemm, tma_no_loop.before(), "tma_cg", [], Sm90_TmaCommitGroup, None)
    gemm = insert_fence(gemm, tma_no_loop.before(), cuda_in_order, cuda_generic_and_async_proxy)
    gemm = wrap_with_context(gemm, tma_no_loop, CudaWarps(3, 4, name="producer"))
    tma_no_loop = gemm.forward(tma_no_loop)
    gemm = insert_await(gemm, tma_no_loop.after(), "tma_cg", cuda_in_order, 0)
    gemm = insert_arrive(gemm, tma_no_loop.after(), tma_to_gmem_async, "tma_cg")

    # Also at this point set the RMEM -> SMEM writes to be by the consumer warps.
    gemm = wrap_with_context(
        gemm,
        find_parent_loop(gemm, C_smem_init, "wg_m"),
        CudaWarps(name="consumer"),
    )

    # We need cluster syncs before and after the C_smem usage to protect the
    # allocation aliasing with prior SMEM allocs.
    gemm = insert_fence(gemm, C_smem.before(), cuda_in_order, cuda_in_order)
    inner_task_loop = gemm.forward(inner_task_loop)
    gemm = insert_fence(gemm, inner_task_loop.body().after(), cuda_in_order, cuda_in_order)

    # Substitute cuda memset for 0-init.
    if enable_split_k:
        gemm = replace(gemm, gemm.find_loop("memset_batch"), cudaMemsetAsync0_3f32)

    # Specialize initial iteration of iter_k loop
    # and add the wgmma -> war synchronization for iter_k >= 1 loop.
    # NB this doubles the size of the proc ... you can eliminate this
    # temporarily to make things easier to read.
    if True:
        gemm = sched_cut_sync_iter_k(gemm, config)

    gemm = simplify(gemm)
    proc_name = config.make_proc_name() + "_sched"
    gemm = rename(gemm, proc_name)
    K_cluster = config.make_smem_K() * config.ring_depth * 2 + 16
    gemm.sync_check(L=2, M=600, N=800, K_cluster=K_cluster, K_split=2 if enable_split_k else 1)

    # sporkbench cases
    if cases is not None:
        j_case = {
            "algorithm": "gemm",
            "A_type": str(A_type),
            "B_type": str(B_type),
            "C_type": str(C_type),
            "proc": proc_name,
            "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
            "A_major": config.A_major, "B_major": config.B_major, "C_major": config.C_major,
        }
        if not enable_split_k:
            j_case["K_split_max"] = 1
        cases.append(j_case)

    # TODO test cursors

    return gemm
