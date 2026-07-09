# fmt: off

"""

DO NOT EDIT if you are in the Exo repository.

Edit if you are in sporkbench

For now I'm planning to copy/paste the sporkbench dir into Exo.

"""

from __future__ import annotations
from exo import *
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import auto_stage_mem
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *


from .Sm90a_gemm_pre_config import Sm90aGemmConfig

def schedule_Sm90a_gemm(config: Sm90aGemmConfig, ncta_M, ncta_N):
    unsafe = False
    enable_split_k = config.enable_split_k
    smem_M = config.smem_M
    smem_N = config.smem_N
    smem_K = 32
    RING = config.RING

    # Derived constants
    wg_M = smem_M // 2
    wg_N = smem_N
    cluster_M = smem_M * ncta_M
    cluster_N = smem_N * ncta_N

    # CudaDeviceFunction context
    my_warp_config = [
        CudaWarpConfig("producer", 1, setmaxnreg_dec=40),  # 1 producer warp
        CudaWarpConfig("unused", 3, setmaxnreg_dec=40),    # 3 unused warps
        CudaWarpConfig("consumer", 8, setmaxnreg_inc=232), # 2 consumer warpgroups (8 warps)
    ]
    cuda_device_function_ctx = CudaDeviceFunction(
        clusterDim=ncta_M * ncta_N,
        warp_config=my_warp_config
    )

    # TMA SMEM box sizes.
    # (batch dim, MN smem, k_task, K smem)
    smem_box_A = (1, smem_M // ncta_N, 1, smem_K)
    smem_box_B = (1, smem_N // ncta_M, 1, smem_K)  # ncta_M is not a typo
    # (batch dim, N smem, M smem)
    smem_box_C = (1, smem_N, smem_M)

    @proc
    def gemm(
            L: size, M: size, N: size, K_split: size, cluster_K: size,
            A: f32[L, M, K_split, cluster_K] @ CudaGmemLinear,  # Row-major (K-major)
            B: f32[L, N, K_split, cluster_K] @ CudaGmemLinear,  # Column-major (K-major)
            C: f32[L, N, M] @ CudaGmemLinear,  # Column-major
    ):
        assert L > 0
        assert M > 0
        assert N > 0
        assert cluster_K > 0
        assert cluster_K % 4 == 0
        if enable_split_k:
            for memset_batch in seq(0, L):
                for memset_n in seq(0, N):
                    for memset_m in seq(0, M):
                        C[memset_batch, memset_n, memset_m] = 0

        A_tensorMap = A[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_A)
        B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_B)
        C_tensorMap = C[:,:,:] @ Sm90_tensorMap(0, *smem_box_C)  # Only for enable_split_k=True

        for batch in seq(0, L):
            for task_k in seq(0, K_split):
                for m in seq(0, M):
                    for n in seq(0, N):
                        D_rmem: f32
                        D_rmem = 0
                        for k in seq(0, cluster_K):
                            D_rmem += A_tensorMap[batch, m, task_k, k] * B_tensorMap[batch, n, task_k, k]
                        if enable_split_k:
                            C_tensorMap[batch, n, m] += D_rmem
                        else:
                            C[batch, n, m] = D_rmem

    gemm = simplify(gemm)  # Get rid of enable_split_k if stmts.
    if not enable_split_k:
        gemm = inline_window(gemm, "C_tensorMap = _")

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

    # expand dim of D_rmem so each iteration uses its own D_rmem.
    # This enables future parallelization.
    # Set the memory type for wgmma accumulators.
    gemm = expand_dim(gemm, D_rmem, smem_N, "sub_cta_n")
    gemm = expand_dim(gemm, D_rmem, wg_M, "sub_wg_m")
    gemm = expand_dim(gemm, D_rmem, 2, "wg_m")
    gemm = expand_dim(gemm, D_rmem, ncta_N, "cta_n")
    gemm = expand_dim(gemm, D_rmem, ncta_M, "cta_m")
    gemm = set_memory(gemm, D_rmem, Sm90_RmemMatrixD(wg_M, smem_N))
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
    gemm = insert_barrier_alloc(gemm, iter_k_loop.before(), "raw", None, [ncta_M, ncta_N], CudaMbarrier)
    gemm = insert_barrier_alloc(gemm, iter_k_loop.before(), "war", "raw", [ncta_M, ncta_N], CudaMbarrier)
    gemm = insert_barrier_alloc(gemm, iter_k_loop.before(), "cg", None, [ncta_M, ncta_N, 2], CudaCommitGroup)

    # Stage A_smem, B_smem tiles above wg_m loop in swizzled memory.
    # TODO how to get a more stable reference to the input cursor.
    # We can't rely on the old cursor since it's forwarded to
    # the wrong loop after fission.
    # TODO sucky that we have to use f-strings here; PAST can't get local variables???
    wg_m_main_loop = iter_k_loop.only_child(3)
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
        gemm = expand_dim(gemm, smem_cursor, RING, f"iter_k % {RING}")
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
    gemm = insert_await(gemm, A_smem_loop.before(), "war[cta_m, cta_n]", cuda_temporal, ~(RING-1))
    gemm = fission(gemm, A_smem_loop.before(), n_lifts=1)
    gemm = insert_arrive(gemm, B_smem_loop.after(), cuda_temporal, ("raw[cta_m, :]", "raw[:, cta_n]"))
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
        # Most explicit way to use an instruction template.
        # We give explicit values for all template parameters.
        Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
            ncta=ncta_N,
            cta_stride=1,
            size0=smem_M,
            size1=smem_K,
            smem_box=smem_box_A,
        )
    )
    gemm = set_trailing_barrier_expr(gemm, A_tma, "raw[cta_m, :]")
    gemm = unsafe_remove_if(gemm, B_smem_loop, True)
    gemm = replace(
        gemm,
        B_tma,
        # replace should be able to deduce the commented-out arguments.
        # We use partial evaluation to provide the minimum information.
        Sm90_multicast_copy_tensor_to_smem_swizzled_2f32.partial(
            # ncta=ncta_M,
            cta_stride=ncta_N,
            # size0=smem_N,
            # size1=smem_K,
            smem_box=smem_box_B,
        )
    )
    gemm = set_trailing_barrier_expr(gemm, B_tma, "raw[:, cta_n]")

    # wgmma M-warpgroup loop (children should be replaced with wgmma later)
    # Surround with mbarrier await/arrive.
    wg_m_main_loop = gemm.forward(wg_m_main_loop)
    gemm = insert_await(gemm, wg_m_main_loop.before(), "raw[cta_m, cta_n]", cuda_generic_and_async_proxy, ~0)
    gemm = insert_arrive(gemm, wg_m_main_loop.after(), cuda_in_order, ("war[cta_m, :]", "war[:, cta_n]"))

    # wgmma K loop needs to be tiled by 8.
    # Wrap these future wgmma instrs with wgmma.fence before, cg arrive after
    gemm = divide_loop(gemm, sub_iter_k_loop, 8, ("mma_k", "sub_mma_k"), perfect=True)
    mma_k_loop = gemm.forward(sub_iter_k_loop)
    sub_mma_k_loop = mma_k_loop.only_child()
    # Loop structure
    # Fence(wgmma_fence_1, wgmma_fence_2)
    # for mma_k_loop in seq(0, 4):  # Lifted out below
    #     for sub_wg_m: ...  <-- replace with wgmma
    #         ...
    #             for sub_mma_k:
    # Arrive(...) >> ...
    for i in range(4):  # TODO again inelegant
        gemm = lift_scope(gemm, mma_k_loop)
    mma_k_loop = gemm.forward(mma_k_loop)
    wgmma_cursor = mma_k_loop.only_child()
    gemm = unsafe_remove_if(gemm, wgmma_cursor, True)
    gemm = replace(gemm, wgmma_cursor, Sm90_mma_async_tf32)
    gemm = insert_fence(gemm, mma_k_loop.before(), wgmma_fence_1, wgmma_fence_2)
    # Arrive(wgmma_async, 1) >> cg[cta_m, cta_n, wg_m]
    # if k_iter >= 1:
    #     Await(cg[cta_m, cta_n, wg_m], cuda_in_order, 1)
    gemm = insert_await(gemm, mma_k_loop.after(), "cg[cta_m, cta_n, wg_m]", cuda_in_order, 1)
    mma_k_loop = gemm.forward(mma_k_loop)
    gemm = add_if(gemm, mma_k_loop.next(), "iter_k >= 1", True)
    gemm = insert_arrive(gemm, mma_k_loop.after(), wgmma_async, "cg[cta_m, cta_n, wg_m]")

    # Main loop warp specialization.
    # I think there's 3 statements at this level right now.
    # First two should be the A/B smem load, last one should be accum.
    iter_k_loop = gemm.forward(iter_k_loop)
    assert len(iter_k_loop.body()) == 3
    gemm = wrap_with_context(gemm, iter_k_loop.body()[:2], CudaWarps(name="producer"))
    gemm = wrap_with_context(gemm, iter_k_loop.body()[2], CudaWarps(name="consumer"))

    # Finalize zero prologue.
    # We have to unsafely remove the if-guards to substitute the zero instr.
    D_zero = gemm.forward(D_zero)
    zero_m_loop = D_zero.parent().parent().parent().parent().parent()  # TODO better way?
    gemm = wrap_with_context(gemm, zero_m_loop, CudaWarps(name="consumer"))
    sub_wg_m_loop = gemm.forward(zero_m_loop).only_child()
    gemm = unsafe_remove_if(gemm, sub_wg_m_loop, True)
    gemm = replace(gemm, sub_wg_m_loop, Sm90_zero_scale_d_f32)

    # Finalize write-to-C epilogue.
    # Need to wait for wgmma beforehand.
    # This wait loop is fissioned out from the main epilogue.
    C_assign = gemm.forward(C_assign)
    assign_sub_wg_m_loop = C_assign.parent().parent().parent().parent()  # TODO better way?
    assign_wg_m_loop = assign_sub_wg_m_loop.parent()
    gemm = insert_await(gemm, assign_wg_m_loop.body().before(), "cg[cta_m, cta_n, wg_m]", cuda_in_order, 0)
    gemm = wrap_with_context(gemm, assign_wg_m_loop, CudaWarps(name="consumer"))
    assign_wg_m_loop = gemm.forward(assign_wg_m_loop)
    epilogue_gap = assign_wg_m_loop.body()[0].after()
    # TODO: is there a better way to set n_lifts?
    gemm = fission(gemm, epilogue_gap, n_lifts=4)
    inner_task_loop = gemm.forward(inner_task_loop)
    assign_sub_wg_m_loop = gemm.forward(assign_sub_wg_m_loop)
    if enable_split_k:
        # NB we are not yet ready to schedule enable_split_k=True
        # Here I'm trying to stage the tile in C_smem, which will be accumulated into
        # C_tensorMap. Problem is this causes C_smem to be zero'd, then +='d.
        # We need to reason that this can be fused to a simple assignment.
        assert 0
        consumer_epilogue_assign = assign_sub_wg_m_loop.parent().parent()
        print(consumer_epilogue_assign)
        gemm = stage_mem(gemm, consumer_epilogue_assign,
            f"C_tensorMap[batch, "
            f"(task_n * {ncta_N} + cta_n) * {smem_N} : (task_n * {ncta_N} + cta_n + 1) * {smem_N}, "
            f"(task_m * {ncta_M} + cta_m) * {smem_M} : (task_m * {ncta_M} + cta_m + 1) * {smem_M}]",
            "C_smem",
            True,  # accum
            CudaSmemLinear,
        )
        C_smem = gemm.find_alloc_or_arg("C_smem")
        gemm = expand_dim(gemm, C_smem, ncta_N, "cta_n")
        gemm = expand_dim(gemm, C_smem, ncta_M, "cta_m")
        gemm = simplify(gemm)
        gemm = lift_alloc(gemm, C_smem, n_lifts=2)
        gemm = insert_fence(gemm, C_smem.before(), cuda_in_order, cuda_in_order)
        gemm = insert_fence(gemm, inner_task_loop.body().after(), cuda_in_order, cuda_in_order)
    else:
        gemm = replace(gemm, assign_sub_wg_m_loop, Sm90_mma_store_d_col_major_tf32)
        # Surround store_d loop with cluster arrive/await
        inner_task_loop = gemm.forward(inner_task_loop)
        cluster_arrive_here = inner_task_loop.body().after().anchor().before()
        gemm = insert_barrier_alloc(gemm, cluster_arrive_here, "cluster_sync", None, [], CudaClusterSync)
        cluster_arrive_here = gemm.forward(cluster_arrive_here)
        gemm = insert_arrive(gemm, cluster_arrive_here, cuda_in_order, "cluster_sync")
        gemm = insert_await(gemm, inner_task_loop.body().after(), "cluster_sync", cuda_in_order, 0)

    # Substitute cuda memset for 0-init.
    if enable_split_k:
        gemm = replace(gemm, gemm.find_loop("memset_batch"), cudaMemsetAsync0_3f32)

    # Specialize initial iteration of iter_k loop.
    # NB this doubles the size of the proc ... you can eliminate this
    # temporarily to make things easier to read.
    if True:
        gemm = cut_loop(gemm, iter_k_loop, 1)

    gemm = simplify(gemm)
    print(gemm)

    # TODO test cursors

    return gemm
