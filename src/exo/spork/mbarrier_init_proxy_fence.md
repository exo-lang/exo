# Why the proxy fence after `mbarrier.init` is folklore

Research notes, 2026-09-23, for `cuda_sync_state.py` and `cuda_device_setup_builder.py`.
No code has been changed yet.

## Summary

TMA (`cp.async.bulk{.tensor}...mbarrier::complete_tx::bytes`) moves its *data* through the async proxy.
Its accesses to the *mbarrier* go through the **generic** proxy.
Every operation on an mbarrier is therefore generic-proxy:

* init
* arrive
* expect_tx
* try_wait / test_wait
* TMA's complete-tx

Ordinary causality orders these operations, so:

* no `fence.proxy.async` is needed after `mbarrier.init`, and
* none is needed in the main loop for the mbarrier object itself.

This explains why GEMM main loops interleave generic mbarrier ops with TMA complete-tx and never fence.
The fence people emit once after init is not required.

`fence.proxy.async` is still needed for **data**.
For example, generic SMEM writes must be fenced before a TMA store reads them.

## NVIDIA primary sources

PTX ISA 9.4, https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

* **`cp.async.bulk` (§9.7.10.28.4.1) and `cp.async.bulk.tensor` (§9.7.10.28.5.3)**, on the `mbar` operand:
  "This instruction accesses its mbarrier operand using generic-proxy."
  * The same sentence appears on `st.async`, `red.async`, `clusterlaunchcontrol.try_cancel` and `tcgen05.commit`.
  * The complete-tx has `.release` semantics at `.cluster` scope.
* **§9.7.10.28.2:** only the data movement is async-proxy.
  "The cp{.reduce}.async.bulk operations are performed in the asynchronous proxy".
* **§8.9.5 (proxy-preserved base causality order):** ordering is preserved for operations "performed to the same address, using the generic proxy".
* **`fence.mbarrier_init`:** "only applies to the prior mbarrier.init operations".
  * It is a restricted `.release.cluster` fence, meaning cross-CTA visibility of the init.
  * It is not a proxy fence.
  * The spec's example pattern is `mbarrier.init; fence.mbarrier_init.release.cluster; barrier.cluster.arrive.relaxed;`.
* **History:** the "mbarrier operand using generic-proxy" sentence first appears in the PTX 9.3/9.4 docs.
  Archived docs through CUDA 13.2 (PTX 9.2) say nothing either way.
  Older sm_90 behaviour was never formally specified.

CUDA C++ Programming Guide:

* **Origin of the folklore:** the CUDA 12.4 guide had `init(&bar, ...); cde::fence_proxy_async_shared_cta();`
  with the comment "Make initialized barrier visible in async proxy."
* **Current guide (async-copies):** the fence is gone.
  * It is now `init(&bar, blockDim.x); __syncthreads();`.
  * One example even calls `ptx::mbarrier_init(&bar, 1)` and immediately issues `cp_async_bulk_tensor(..., &bar)` from the same thread, with no fence.
  * `fence_proxy_async` remains only where it belongs: before TMA reads SMEM that was written by the generic proxy.

## CUTLASS (local checkout `cutlass/`, commit 3f5bafb3)

* **Init fence:** `include/cutlass/arch/barrier.h:711-722`, `fence_barrier_init()`, emits `fence.mbarrier_init.release.cluster`.
  * Its comment says it must be composed with `__syncthreads()` or `cluster_arrive()` + `cluster_wait()`.
  * `fence.proxy.async` is never used after init.
* **Proxy fence for data only:** `fence_view_async_shared()` (`barrier.h:727-732`, `fence.proxy.async.shared::cta`).
  * Example: `epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp:736`, "ensure smem writes are visible to TMA".
* **Barrier setup:** `include/cutlass/pipeline/sm90_pipeline.hpp:305-323`, `PipelineTmaAsync::init_barriers`, calls `fence_barrier_init()` after init.
* **Synchronization after init:** `sm90_gemm_tma_warpspecialized_pingpong.hpp:511-520`.
  * Cluster > 1: `cute::cluster_arrive_relaxed()`, then `cluster_wait()`.
    The relaxed arrive has no release, so `fence.mbarrier_init` supplies it.
  * Cluster = 1: `__syncthreads()`.
* **Tutorial:** `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu:158-165` does init, then `cluster_sync()`, with no fence at all.

## Secondary sources (low value)

* NVIDIA forum threads [316226](https://forums.developer.nvidia.com/t/when-to-use-fence-proxy-async-shared-cta-in-gemm-scenarios/316226)
  and [357574](https://forums.developer.nvidia.com/t/why-arent-there-explicit-async-proxy-generic-proxy-fences-in-the-cuda-guide-tma-prefetching-example/357574).
  Both have community answers only, with no NVIDIA staff resolution.
* The widely cited Colfax TMA tutorial and various blogs repeat the old (12.4) guide's post-init fence.

## What is actually required after init

Ordinary thread synchronization, at a scope that covers every thread or CTA that will touch the mbarrier:

* **CTA-only:** `__syncthreads()` / `barrier.cta.sync`.
* **Cluster:** a release at cluster scope, then `barrier.cluster.wait`. Either form works:
  * a full `barrier.cluster.arrive` (release by default), or
  * `fence.mbarrier_init.release.cluster` + `barrier.cluster.arrive.relaxed` (cheaper).

## Implications for Exo-GPU

* **Existing init sequence is already sufficient.** `cuda_device_setup_builder.py` emits, after the `mbarrier.init` loop:
  * `barrier.cta.sync 0` if clusterDim == 1;
  * full `barrier.cluster.arrive.aligned` + `barrier.cluster.wait.aligned` otherwise.
* **The post-init `fence.proxy.async` is unnecessary.** It is controlled by `require_proxy_fence()`.
  * `add_barrier` in `cuda_sync_state.py` triggers it when any Arrive/Await sync-tl intersects `internal_cuda_async_proxy_detection`.
  * It is harmless and costs one instruction per CTA, so it can be deleted.
  * Deleting it also makes an earlier "bug" moot: kernels whose TMA-load mbarriers use only `cuda_temporal` / `cuda_in_order` never trigger the fence.
    That fence is not needed anyway.
* **Test expectation to update when removing it:** `tests/cuda/test_cuda_sync.py` `MbarrierQualConfig.have_init_proxy_fence`.
  It is True for the `*_to_wgmma` configs, with the comment "we still need the proxy fence at startup".
* **The per-Await `fence.proxy.async` in `add_mbarrier_ring` / `generate_await` is a different thing.**
  It fences *data* (generic writes -> async-proxy reads), so it stays.

## Confidence

* **Spec reading:** high.
* **"Harmless but unnecessary on sm_90":** high-medium.
  The spec only made the generic-proxy mbarrier operand explicit in PTX 9.3+.
* **Formal model coverage:** PTX 9.4 §8.9.5 still does not formally integrate async-proxy fences, only alias fences.
  So the formal model of the async proxy remains incomplete.
