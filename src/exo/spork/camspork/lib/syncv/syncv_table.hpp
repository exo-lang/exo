#pragma once

#include <cassert>
#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "syncv_types.hpp"
#include "tl_sig.hpp"
#include "../util/require.hpp"

namespace camspork
{

struct SyncvTable;

// NOTE: pending awaits don't seem to be represented here the same way as in the thesis.
// The thesis describes TrArrive events as having infinite outbound PendingAwaits "up to infinity"
// i.e. {PendingAwait(x, z^*, z_{await} \mid z_{await} \ge arrive_count)}
// meanwhile TrAwait has a single inbound PendingAwait.
// The actual software's PendingAwait object (here) seems to represent that entire infinite range.
// Hence, the await implementation is responsible for matching any software PendingAwait with
// sufficiently low arrive_count, not just an exact arrive_count.
struct PendingAwait
{
    barrier_id hamster_barrier_id;
    int32_t arrive_count;

    bool operator== (PendingAwait other) const
    {
        return hamster_barrier_id == other.hamster_barrier_id && arrive_count == other.arrive_count;
    }

    bool operator!= (PendingAwait other) const
    {
        return !(*this == other);
    }
};

inline bool operator< (PendingAwait lhs, barrier_id rhs) { return lhs.hamster_barrier_id < rhs; }
inline bool operator< (barrier_id lhs, PendingAwait rhs) { return lhs < rhs.hamster_barrier_id; }
inline bool operator< (PendingAwait lhs, PendingAwait rhs) { return lhs.hamster_barrier_id < rhs.hamster_barrier_id; }
inline bool operator== (barrier_id lhs, PendingAwait rhs) { return lhs == rhs.hamster_barrier_id; }
inline bool operator== (PendingAwait lhs, barrier_id rhs) { return lhs.hamster_barrier_id == rhs; }
inline bool operator!= (barrier_id lhs, PendingAwait rhs) { return lhs != rhs.hamster_barrier_id; }
inline bool operator!= (PendingAwait lhs, barrier_id rhs) { return lhs.hamster_barrier_id != rhs; }

// Adapters for old code using pre-Hamster barriers.
using pending_await_t = PendingAwait;

inline uint32_t pending_await_barrier_index(PendingAwait id)
{
    return id.hamster_barrier_id.data;
}

inline int32_t pending_await_arrive_count(PendingAwait id)
{
    return id.arrive_count;
}

inline PendingAwait pack_pending_await(uint32_t barrier_index, int32_t arrive_count)
{
    return PendingAwait{barrier_id{barrier_index}, arrive_count};
}

struct VisRecordDebugData
{
    std::vector<TlSigInterval> visibility_set;
    std::vector<PendingAwait> pending_await_list;
    barrier_id free_on_arrive;
    qual_bits_t qual_tl_mask;
    uint64_t flags;  // vis_record_before_alloc_flag, vis_record_freed_flag (u64: keeps sizeof static_assert padding-free)
};

struct AssignmentRecordDebugData
{
    // VisRecord IDs (for debug_get_vis_record_data); may contain duplicates.
    std::vector<uint32_t> mutate_vis_record_ids;
    std::vector<uint32_t> read_vis_record_ids;
};

struct BarrierDebugData
{
    int32_t arrive_count;
    int32_t await_count;
    int32_t expected_arrive_count;  // 0 = unchecked
    uint32_t flags;  // one_shot_await_flag, arrive_per_thread_flag
};

// Window into a multidimensional array of assignment records.
// The outer-extents array gives the size of the base array.
// The offset and inner extents gives the location and size of the window.
struct AssignmentRecordWindow
{
    assignment_record_id* base;
    const uint32_t* begin_outer_extent;
    const uint32_t* end_outer_extent;
    const uint32_t* begin_offset;
    const uint32_t* end_offset;
    const uint32_t* begin_inner_extent;
    const uint32_t* end_inner_extent;
};

struct SyncvAccessInfo
{
    // If true: only one thread must have visibility, and we init with one VisRecord for all threads.
    // If false: all threads must have visibility, and we init with a new VisRecord per thread.
    bool is_convergent;

    // New VisRecords get initial_qual_bit (plus atomic_qual_bits for all threads, if atomic).
    // Checks against prior VisRecords require a qual-tl in precondition_qual_bits (includes atomic_qual_bits).
    qual_bits_t initial_qual_bit;
    qual_bits_t precondition_qual_bits;
    qual_bits_t atomic_qual_bits;
    qual_bits_t qual_tl_mask;

    // thread_access_granularity is for out-of-order non-convergent abstract machine optimization.
    //
    // Only valid if the initial qual-tl is out-of-order (in syncv_init_t::ooo_qual_mask).
    // By the official semantics, in the is_convergent=false case, we need to add a VisRecord for each thread
    // in the ThreadCuboid. If the initial qual-tl is out-of-order, then it's never in any precondition set,
    // so it's only used for synchronizes_with for Fence and Arrive (and technically JoinThreads as well).
    //
    // The effect of thread_access_granularity > 1 is to replace the per-thread VisRecord with
    // VisRecords for aligned groups of thread_access_granularity-many threads
    // (rounding down tid_lo and rounding up tid_hi), e.g. threads 2, 3, 4, 5, 6, 7, 8 with thread_access_granularity=4
    // gives VisRecords for [0, 3], [4, 7], [8, 11].
    //
    // If we know that we always have thread_access_granularity dividing the tid_lo and tid_hi
    // of all thread intervals executing a Fence or Arrive with the access's initial qual-tl in the sync's first sync-tl,
    // then the modification is harmless, as it's immaterial which threads in each aligned group
    // of thread_access_granularity-many threads is considered active.
    //
    // For JoinThreads, this should be safe since we in practice only use that to join tasks (CUDA clusters) and
    // the full CUDA grid, so just make sure thread_access_granularity divides the cluster thread count.
    //
    // This is a critical optimization for keeping the size of the VisRecord memoization table managable; however,
    // it's only applicable for out-of-order initial qual-tl, as otherwise this modification would also affect
    // checks against precondition sets, which isn't at all legitimate to do.
    // Note this really is a pure performance hack, not actually meaningful semantics in any way.
    uint32_t thread_access_granularity;
    uint32_t barrier_count;
    const barrier_id* trailing_barriers;
};

struct SyncvDebugValidateInput
{
    // For memory leak checking, we need to know what are the root objects from outside the syncv_table.
    // The type of array passed here is based on which pointer is not null.
    // Sloppy, replace if we have 3 or more types.
    size_t size;
    const assignment_record_id* p_records;
    const barrier_id* p_barriers;
};

struct SyncvLogRequest;

struct SyncvFence
{
    qual_bits_t L1_qual_bits;
    qual_bits_t L2_qual_bits;
};

struct SyncvArrive
{
    // Every barrier the Arrive targets, without duplicates.
    // Each one's own arrive count is read and incremented (no "home barrier").
    uint32_t barrier_count;
    const barrier_id* all_barriers;
    qual_bits_t L1_qual_bits;
    uint32_t flags;  // arrive_pre_arrive_flag
};

struct SyncvAwait
{
    barrier_id bar;
    int32_t N;
    qual_bits_t L2_qual_bits;
};

struct SyncvJoinThreads
{
};


// *** Primary Implemented Interface ***
SyncvTable* new_syncv_table(const syncv_init_t& init);
SyncvTable* copy_syncv_table(const SyncvTable* table);
void delete_syncv_table(SyncvTable* table);

void on_r(SyncvTable*, assignment_record_id*, const ThreadCuboid&, SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_r(SyncvTable*, assignment_record_id*, const ThreadCuboid&, SyncvAccessInfo, const SyncvLogRequest&);
void on_r(SyncvTable*, AssignmentRecordWindow, const ThreadCuboid&, SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_r(SyncvTable*, AssignmentRecordWindow, const ThreadCuboid&, SyncvAccessInfo, const SyncvLogRequest&);
void on_rw(SyncvTable*, assignment_record_id*, const ThreadCuboid&, SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_rw(SyncvTable*, assignment_record_id*, const ThreadCuboid&, SyncvAccessInfo, const SyncvLogRequest&);
void on_rw(SyncvTable*, AssignmentRecordWindow, const ThreadCuboid&, SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_rw(SyncvTable*, AssignmentRecordWindow, const ThreadCuboid&, SyncvAccessInfo, const SyncvLogRequest&);
void on_check_free(SyncvTable*, AssignmentRecordWindow, const ThreadCuboid&, SyncvAccessInfo, decltype(nullptr) = nullptr);
void on_check_free(SyncvTable*, AssignmentRecordWindow, const ThreadCuboid&, SyncvAccessInfo, const SyncvLogRequest&);
void set_managed_ring_buffer_barriers(
        SyncvTable*,
        qual_bits_t,
        AssignmentRecordWindow,
        uint32_t alloc_on_await_count,
        const barrier_id* alloc_on_await_barriers,
        barrier_id free_on_arrive,
        decltype(nullptr) = nullptr);
void set_managed_ring_buffer_barriers(
        SyncvTable*,
        qual_bits_t,
        AssignmentRecordWindow,
        uint32_t alloc_on_await_count,
        const barrier_id* alloc_on_await_barriers,
        barrier_id free_on_arrive,
        const SyncvLogRequest&);
void on_check_free_on_arrive(SyncvTable*, AssignmentRecordWindow, decltype(nullptr) = nullptr);
void on_check_free_on_arrive(SyncvTable*, AssignmentRecordWindow, const SyncvLogRequest&);

void clear_visibility(SyncvTable* table, size_t N, assignment_record_id* array);
void alloc_barriers(SyncvTable* table, size_t N, barrier_id* barriers, uint32_t flags, int32_t expected_arrive_count);
void free_barriers(SyncvTable* table, size_t N, barrier_id* barriers, bool check_arrive_await);
void on_fence(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvFence& fence, const SyncvLogRequest&);
void on_fence(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvFence& fence, decltype(nullptr) = nullptr);
void on_arrive(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvArrive& arrive, const SyncvLogRequest&);
void on_arrive(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvArrive& arrive, decltype(nullptr) = nullptr);
void on_await(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvAwait& await, const SyncvLogRequest&);
void on_await(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvAwait& await, decltype(nullptr) = nullptr);
void on_join_threads(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvJoinThreads&, const SyncvLogRequest&);
void on_join_threads(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvJoinThreads&, decltype(nullptr) = nullptr);
void begin_no_checking(SyncvTable* table);
void end_no_checking(SyncvTable* table);



// *** Debug Inspection Interface ***
void debug_get_vis_record_data(const SyncvTable* table, uint32_t id, VisRecordDebugData* out);
// id = 0 (no AssignmentRecord) gives empty lists.
void debug_get_assignment_record_data(const SyncvTable* table, assignment_record_id id, AssignmentRecordDebugData* out);
void debug_get_barrier_data(const SyncvTable* table, barrier_id id, BarrierDebugData* out);
void debug_validate_state(SyncvTable* table, size_t input_count, const SyncvDebugValidateInput* p_inputs);

}
