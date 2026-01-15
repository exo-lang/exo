# fmt: off
"""

DO NOT EDIT if you are in the Exo repository

Edit if you are in sporkbench

For now I'm planning to copy/paste the sporkbench dir into Exo.

"""

from __future__ import annotations

import time

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

from .Sm90a_gemm_pre_config import Sm90aGemmConfig

def make_Sm90a_gemm(config: Sm90aGemmConfig, ncta_M: int, ncta_N: int):
    assert isinstance(config.smem_M, int), "Need to import Sm90a_gemm_pre_config first and set config variables"
    assert isinstance(config.smem_N, int), "Need to import Sm90a_gemm_pre_config first and set config variables"
    assert isinstance(config.tma_to_gmem, int), "Need to import Sm90a_gemm_pre_config first and set config variables"
    assert isinstance(config.enable_split_k, int), "Need to import Sm90a_gemm_pre_config first and set config variables"
    my_warp_config = [
        CudaWarpConfig("producer", 1, setmaxnreg_dec=40),
        CudaWarpConfig("unused", 3, setmaxnreg_dec=40),
        CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),
    ]

    tma_to_gmem = bool(config.tma_to_gmem)
    enable_split_k = bool(config.enable_split_k)
    smem_M = config.smem_M
    smem_N = config.smem_N
    smem_K = 32
    assert smem_M % 128 == 0
    wg_M = smem_M // 2
    assert smem_N % 8 == 0
    assert 8 <= smem_N <= 256
    wg_N = smem_N
    RING = config.RING
    cluster_M = smem_M * ncta_M
    cluster_N = smem_N * ncta_N

    if enable_split_k:
        assert tma_to_gmem

    # (batch dim, MN smem, k_task, K smem)
    smem_box_A = (1, smem_M // ncta_N, 1, smem_K)
    smem_box_B = (1, smem_N // ncta_M, 1, smem_K)  # ncta_M is not a typo
    # (batch dim, N smem, M smem)
    smem_box_C = (1, smem_N, smem_M)

    # K dimension of tensor is K_split * cluster_K
    # i.e. each task (cluster) is responsible for cluster_M * cluster_N * cluster_K
    # We divide the K dim into [K_split, cluster_K] as a workaround for Exo
    # quasi-affine indexing restrictions.
    # Unfortunately, the caller has to manually pass cluster_K = K // K_split.
    @proc
    def xgemm_Sm90_wgmma(
        L: size, M: size, N: size, K_split: size, cluster_K: size,
        A: f32[L,M,K_split,cluster_K] @ CudaGmemLinear,
        B: f32[L,N,K_split,cluster_K] @ CudaGmemLinear,
        C: f32[L,N,M] @ CudaGmemLinear
    ):
        assert L > 0
        assert M > 0
        assert N > 0
        assert cluster_K > 0
        assert cluster_K % 4 == 0

        A_tensorMap = A[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_A)
        B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_B)
        C_tensorMap = C[:,:,:] @ Sm90_tensorMap(0, *smem_box_C)  # Only for tma_to_gmem=True

        if enable_split_k:
            cudaMemsetAsync0_3f32(L, N, M, C[:,:,:])

        with CudaDeviceFunction(clusterDim=ncta_M * ncta_N, warp_config=my_warp_config, blocks_per_sm=1):
          for batch in cuda_tasks(0, L):
            for task_k in cuda_tasks(0, K_split):
              for task_n in cuda_tasks(0, (N + cluster_N - 1) / cluster_N):
                for task_m in cuda_tasks(0, (M + cluster_M - 1) / cluster_M):
                    D_rmem : f32[ncta_M, ncta_N, 2, wg_M, wg_N] @ Sm90_RmemMatrixD(wg_M, wg_N)

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Sm90_zero_scale_d_f32(D_rmem[cta_m,cta_n,wg_m,:,:], M=wg_M, N=wg_N)

                    raw : barrier[ncta_M, ncta_N] @ CudaMbarrier
                    war : barrier(raw)[ncta_M, ncta_N] @ CudaMbarrier
                    cg : barrier[ncta_M, ncta_N, 2] @ CudaCommitGroup

                    A_smem : f32[ncta_M, ncta_N, RING, smem_M, smem_K] @ Sm90_SmemSwizzled(128)
                    B_smem : f32[ncta_M, ncta_N, RING, smem_N, smem_K] @ Sm90_SmemSwizzled(128)

                    # This loop should be cut at 1
                    for iter_k in seq(0, (cluster_K + smem_K - 1) / smem_K):
                        with CudaWarps(name="producer"):
                            # TMA producer warp
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    Await(war[cta_m,cta_n], cuda_temporal, ~(RING-1))
                                Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
                                    A_smem[cta_m,:,iter_k % RING,:,:],
                                    A_tensorMap[
                                        batch,
                                        (ncta_M*task_m + cta_m) * smem_M:
                                        ((ncta_M*task_m + cta_m)+1) * smem_M,
                                        task_k,
                                        iter_k * smem_K:
                                        iter_k * smem_K + smem_K],
                                    ncta=ncta_N, cta_stride=1, size0=smem_M, size1=smem_K, smem_box=smem_box_A
                                ) >> raw[cta_m,:]
                            for cta_n in cuda_threads(0, ncta_N, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N)):
                                Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
                                    B_smem[:,cta_n,iter_k % RING,:,:],
                                    B_tensorMap[
                                        batch,
                                        (ncta_N*task_n+cta_n) * smem_N:
                                        (ncta_N*task_n+cta_n+1) * smem_N,
                                        task_k,
                                        iter_k * smem_K:
                                        iter_k * smem_K + smem_K],
                                    ncta=ncta_M, cta_stride=ncta_N, size0=smem_N, size1=smem_K, smem_box=smem_box_B
                                ) >> raw[:,cta_n]
                                for cta_m in cuda_threads(0, ncta_M, unit=cuda_cta_in_cluster):
                                    Arrive(cuda_temporal) >> raw[cta_m,:] >> raw[:,cta_n]

                        with CudaWarps(name="consumer"):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    # Producer warpgroups
                                    Await(raw[cta_m,cta_n], cuda_generic_and_async_proxy, ~0)
                                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                        Fence(wgmma_fence_1, wgmma_fence_2)
                                        for mma_k in seq(0, smem_K / 8):
                                            Sm90_mma_async_tf32(D_rmem[cta_m,cta_n,wg_m,:,:],
                                                A_smem[cta_m,cta_n,iter_k % RING,(wg_m*wg_M): ((wg_m+1)*wg_M),mma_k*8:mma_k*8+8],
                                                B_smem[cta_m,cta_n,iter_k % RING,:,mma_k*8:mma_k*8+8], M=wg_M, N=wg_N)
                                        Arrive(wgmma_async) >> cg[cta_m,cta_n,wg_m]
                                        if iter_k >= 1:
                                            Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 1)
                                    Arrive(cuda_in_order) >> war[cta_m,:] >> war[:,cta_n]

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 0)

                    if tma_to_gmem:
                        Fence(cuda_in_order, cuda_in_order)
                        C_smem: f32[ncta_M, ncta_N, smem_N, smem_M] @ CudaSmemLinear
                        for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                            for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                with CudaWarps(name="consumer"):
                                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                        Sm90_mma_store_d_col_major_tf32(
                                            wg_M, wg_N, C_smem[cta_m, cta_n, :,wg_m * wg_M: wg_m * wg_M + wg_M],
                                            D_rmem[cta_m,cta_n,wg_m,:,:], M=wg_M, N=wg_N)
                                Fence(cuda_in_order, cuda_generic_and_async_proxy)
                                with CudaWarps(name="producer"):
                                    if enable_split_k:
                                        Sm90_reduce_tensor_to_gmem_linear_2f32(
                                            C_tensorMap[
                                                batch,
                                                (ncta_N*task_n + cta_n) * smem_N: (ncta_N*task_n + cta_n) * smem_N + smem_N,
                                                (ncta_M*task_m + cta_m) * smem_M: (ncta_M*task_m + cta_m) * smem_M + smem_M],
                                            C_smem[cta_m, cta_n, :, :],
                                            size0=smem_N, size1=smem_M, smem_box=smem_box_C,
                                        )
                                    else:
                                        Sm90_copy_tensor_to_gmem_linear_2f32(
                                            C_tensorMap[
                                                batch,
                                                (ncta_N*task_n + cta_n) * smem_N: (ncta_N*task_n + cta_n) * smem_N + smem_N,
                                                (ncta_M*task_m + cta_m) * smem_M: (ncta_M*task_m + cta_m) * smem_M + smem_M],
                                            C_smem[cta_m, cta_n, :, :],
                                            size0=smem_N, size1=smem_M, smem_box=smem_box_C,
                                        )
                                    tma_cg: barrier @ CudaCommitGroup
                                    Arrive(tma_to_gmem_async) >> tma_cg
                                    Await(tma_cg, cuda_in_order, 0)
                        Fence(cuda_in_order, cuda_in_order)  # cluster scope
                    else:
                        # Await(cg, cuda_in_order, 0) and write D_rmem -> C steps are fissioned.
                        # We must not arrive on the epilogue cluster sync until all wgmma retire.
                        cluster_sync: barrier @ CudaClusterSync
                        Arrive(cuda_in_order) >> cluster_sync

                        for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                            for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                with CudaWarps(name="consumer"):
                                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                        Sm90_mma_store_d_col_major_tf32(
                                            M - ((ncta_M*task_m + cta_m) * smem_M + wg_m * wg_M),
                                            N - (ncta_N*task_n + cta_n) * smem_N,
                                            C[batch,
                                              (ncta_N*task_n + cta_n) * smem_N
                                            : ((ncta_N*task_n + cta_n)+1) * smem_N,
                                              (ncta_M*task_m + cta_m) * smem_M + wg_m * wg_M
                                            : (ncta_M*task_m + cta_m) * smem_M + wg_m * wg_M + wg_M],
                                            D_rmem[cta_m,cta_n,wg_m,:,:], M=wg_M, N=wg_N)

                        Await(cluster_sync, cuda_in_order, 0)


    if not tma_to_gmem:
        xgemm_Sm90_wgmma = inline_window(xgemm_Sm90_wgmma, "C_tensorMap = _")
    xgemm_Sm90_wgmma = rename(xgemm_Sm90_wgmma, config.make_proc_name(ncta_M, ncta_N))
    xgemm_Sm90_wgmma = cut_loop(xgemm_Sm90_wgmma, xgemm_Sm90_wgmma.find_loop("iter_k"), 1)
    xgemm_Sm90_wgmma = simplify(xgemm_Sm90_wgmma)
    K_split = 2 if enable_split_k else 1
    t = time.time()
    xgemm_Sm90_wgmma.sync_check(L=2, M=500, N=800, cluster_K=240, K_split=K_split)
    dt = time.time() - t
    print("%.3f s, %s" % (dt, xgemm_Sm90_wgmma.name()))
    return xgemm_Sm90_wgmma
