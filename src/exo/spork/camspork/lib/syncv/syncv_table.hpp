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
    // Init VisRecord to vis_level_unordered if is_ooo, otherwise vis_level_full_ordered
    bool is_ooo;

    // If true: only one thread must have visibility, and we init with one VisRecord for all threads.
    // If false: all threads must have visibility, and we init with a new VisRecord per thread.
    bool is_convergent;

    bool is_write_only;

    qual_bits_t initial_qual_bit;
    qual_bits_t extended_qual_bits;
    qual_bits_t atomic_qual_bits;

    // thread_access_granularity is for out-of-order non-convergent abstract machine optimization.
    //
    // Only valid for is_ooo=true cases.
    // By the official semantics, in the is_convergent=false case, we need to add a VisRecord for each thread
    // in the ThreadCuboid. If is_ooo=false, the VisRecords only have timeline signatures with vis_flag_issue,
    // and vis_flag_issue is only used for synchronizes_with for Fence and Arrive (and technically JoinThreads as well).
    //
    // The effect of thread_access_granularity > 1 is to replace the per-thread VisRecord with
    // VisRecords for aligned groups of thread_access_granularity-many threads
    // (rounding down tid_lo and rounding up tid_hi), e.g. threads 2, 3, 4, 5, 6, 7, 8 wit thread_access_granularity=4
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
    // it's only applicable for is_ooo=true, as otherwise this modification would also affect vis_flag_full,
    // which isn't at all legitimate to do.
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
    qual_bits_t L2_full_qual_bits;
    qual_bits_t L2_temporal_qual_bits;
};

struct SyncvArrive
{
    barrier_id home_barrier;
    uint32_t barrier_count;
    const barrier_id* all_barriers;
    qual_bits_t L1_qual_bits;
};

struct SyncvAwait
{
    barrier_id bar;
    int32_t N;
    qual_bits_t L2_full_qual_bits;
    qual_bits_t L2_temporal_qual_bits;
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
        AssignmentRecordWindow,
        uint32_t alloc_on_await_count,
        const barrier_id* alloc_on_await_barriers,
        barrier_id free_on_arrive,
        decltype(nullptr) = nullptr);
void set_managed_ring_buffer_barriers(
        SyncvTable*,
        AssignmentRecordWindow,
        uint32_t alloc_on_await_count,
        const barrier_id* alloc_on_await_barriers,
        barrier_id free_on_arrive,
        const SyncvLogRequest&);

void clear_visibility(SyncvTable* table, size_t N, assignment_record_id* array);
void alloc_barriers(SyncvTable* table, size_t N, barrier_id* barriers, uint32_t flags);
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
void debug_validate_state(SyncvTable* table, size_t input_count, const SyncvDebugValidateInput* p_inputs);

}
