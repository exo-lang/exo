#pragma once

#include <cassert>
#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "../util/require.hpp"
#include "syncv_types.hpp"
#include "tl_sig.hpp"

namespace camspork {

struct SyncvTable;

// We record pending barrier awaits as (barrier index, counter) pairs.
// barrier_index_bits many bits are used for the index,
// where this is used for a lookup in an internal table.
// (32 - barrier_index_bits) bits are used for the counter.
//
// This limits the number of live barriers and the number of times
// a barrier can be used. The latter is capped in a real CUDA program
// by the number of times an mbarrier can be used: pow(2, 20).
constexpr uint32_t barrier_index_bits = 10;
constexpr uint32_t max_live_barriers = 1u << barrier_index_bits;
using pending_await_t = uint32_t;

inline uint32_t pending_await_barrier_index(pending_await_t id) {
  return id & ((1u << barrier_index_bits) - 1u);
}

inline int32_t pending_await_arrive_count(pending_await_t id) {
  return int32_t(id >> barrier_index_bits);
}

inline pending_await_t pack_pending_await(
    uint32_t barrier_index, int32_t arrive_count) {
  const uint32_t id = barrier_index | uint32_t(arrive_count)
                                          << barrier_index_bits;
  CAMSPORK_REQUIRE_CMP(pending_await_barrier_index(id), ==, barrier_index,
      "implementation limit: barrier_index overflow");
  CAMSPORK_REQUIRE_CMP(pending_await_arrive_count(id), ==, arrive_count,
      "implementation limit: arrive_count overflow");
  return id;
}

// Use signed values for arrive_count.
inline pending_await_t pack_pending_await(
    uint32_t barrier_index, uint32_t arrive_count) = delete;

struct VisRecordDebugData {
  std::vector<TlSigInterval> visibility_set;
  std::vector<pending_await_t> pending_await_list;
};

// Window into a multidimensional array of assignment records.
// The outer-extents array gives the size of the base array.
// The offset and inner extents gives the location and size of the window.
struct AssignmentRecordWindow {
  assignment_record_id *base;
  const uint32_t *begin_outer_extent;
  const uint32_t *end_outer_extent;
  const uint32_t *begin_offset;
  const uint32_t *end_offset;
  const uint32_t *begin_inner_extent;
  const uint32_t *end_inner_extent;
};

struct SyncvAccessInfo {
  // Init VisRecord to vis_level_unordered if is_ooo, otherwise
  // vis_level_full_ordered
  bool is_ooo;

  // If true: only one thread must have visibility, and we init with one
  // VisRecord for all threads. If false: all threads must have visibility, and
  // we init with a new VisRecord per thread.
  bool is_convergent;

  bool is_write_only;

  qual_bits_t initial_qual_bit;
  qual_bits_t extended_qual_bits;
  qual_bits_t atomic_qual_bits;

  // thread_access_granularity is for out-of-order non-convergent abstract
  // machine optimization.
  //
  // Only valid for is_ooo=true cases.
  // By the official semantics, in the is_convergent=false case, we need to add
  // a VisRecord for each thread in the ThreadCuboid. If is_ooo=false, the
  // VisRecords only have timeline signatures with vis_flag_issue, and
  // vis_flag_issue is only used for synchronizes_with for Fence and Arrive (and
  // technically JoinThreads as well).
  //
  // The effect of thread_access_granularity > 1 is to replace the per-thread
  // VisRecord with VisRecords for aligned groups of
  // thread_access_granularity-many threads (rounding down tid_lo and rounding
  // up tid_hi), e.g. threads 2, 3, 4, 5, 6, 7, 8 wit
  // thread_access_granularity=4 gives VisRecords for [0, 3], [4, 7], [8, 11].
  //
  // If we know that we always have thread_access_granularity dividing the
  // tid_lo and tid_hi of all thread intervals executing a Fence or Arrive with
  // the access's initial qual-tl in the sync's first sync-tl, then the
  // modification is harmless, as it's immaterial which threads in each aligned
  // group of thread_access_granularity-many threads is considered active.
  //
  // For JoinThreads, this should be safe since we in practice only use that to
  // join tasks (CUDA clusters) and the full CUDA grid, so just make sure
  // thread_access_granularity divides the cluster thread count.
  //
  // This is a critical optimization for keeping the size of the VisRecord
  // memoization table managable; however, it's only applicable for is_ooo=true,
  // as otherwise this modification would also affect vis_flag_full, which isn't
  // at all legitimate to do. Note this really is a pure performance hack, not
  // actually meaningful semantics in any way.
  uint32_t thread_access_granularity;
  uint32_t barrier_count;
  const barrier_id *trailing_barriers;
};

struct SyncvDebugValidateInput {
  size_t size;
  const assignment_record_id *p_records;
};

struct SyncvLogRequest;

struct SyncvFence {
  qual_bits_t L1_qual_bits;
  qual_bits_t L2_full_qual_bits;
  qual_bits_t L2_temporal_qual_bits;
};

struct SyncvArrive {
  barrier_id home_barrier;
  uint32_t barrier_count;
  const barrier_id *all_barriers;
  qual_bits_t L1_qual_bits;
};

struct SyncvAwait {
  barrier_id bar;
  int32_t N;
  qual_bits_t L2_full_qual_bits;
  qual_bits_t L2_temporal_qual_bits;
};

struct SyncvJoinThreads {};

// *** Primary Implemented Interface ***
SyncvTable *new_syncv_table(const syncv_init_t &init);
SyncvTable *copy_syncv_table(const SyncvTable *table);
void delete_syncv_table(SyncvTable *table);

void on_r(SyncvTable *, assignment_record_id *, const ThreadCuboid &,
    SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_r(SyncvTable *, assignment_record_id *, const ThreadCuboid &,
    SyncvAccessInfo, const SyncvLogRequest &);
void on_r(SyncvTable *, AssignmentRecordWindow, const ThreadCuboid &,
    SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_r(SyncvTable *, AssignmentRecordWindow, const ThreadCuboid &,
    SyncvAccessInfo, const SyncvLogRequest &);
void on_rw(SyncvTable *, assignment_record_id *, const ThreadCuboid &,
    SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_rw(SyncvTable *, assignment_record_id *, const ThreadCuboid &,
    SyncvAccessInfo, const SyncvLogRequest &);
void on_rw(SyncvTable *, AssignmentRecordWindow, const ThreadCuboid &,
    SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_rw(SyncvTable *, AssignmentRecordWindow, const ThreadCuboid &,
    SyncvAccessInfo, const SyncvLogRequest &);
void on_check_free(SyncvTable *, AssignmentRecordWindow, const ThreadCuboid &,
    SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_check_free(SyncvTable *, AssignmentRecordWindow, const ThreadCuboid &,
    SyncvAccessInfo, const SyncvLogRequest &);

void clear_visibility(SyncvTable *table, size_t N, assignment_record_id *array);
void alloc_barriers(SyncvTable *table, size_t N, barrier_id *barriers);
void free_barriers(
    SyncvTable *table, size_t N, barrier_id *barriers, bool check_arrive_await);
void on_fence(SyncvTable *table, const ThreadCuboid &cuboid,
    const SyncvFence &fence, const SyncvLogRequest &);
void on_fence(SyncvTable *table, const ThreadCuboid &cuboid,
    const SyncvFence &fence, decltype(nullptr) = nullptr);
void on_arrive(SyncvTable *table, const ThreadCuboid &cuboid,
    const SyncvArrive &arrive, const SyncvLogRequest &);
void on_arrive(SyncvTable *table, const ThreadCuboid &cuboid,
    const SyncvArrive &arrive, decltype(nullptr) = nullptr);
void on_await(SyncvTable *table, const ThreadCuboid &cuboid,
    const SyncvAwait &await, const SyncvLogRequest &);
void on_await(SyncvTable *table, const ThreadCuboid &cuboid,
    const SyncvAwait &await, decltype(nullptr) = nullptr);
void on_join_threads(SyncvTable *table, const ThreadCuboid &cuboid,
    const SyncvJoinThreads &, const SyncvLogRequest &);
void on_join_threads(SyncvTable *table, const ThreadCuboid &cuboid,
    const SyncvJoinThreads &, decltype(nullptr) = nullptr);
void begin_no_checking(SyncvTable *table);
void end_no_checking(SyncvTable *table);

// *** Debug Inspection Interface ***
void debug_get_vis_record_data(
    const SyncvTable *table, uint32_t id, VisRecordDebugData *out);
void debug_validate_state(SyncvTable *table, size_t input_count,
    const SyncvDebugValidateInput *p_inputs);

}  // namespace camspork
