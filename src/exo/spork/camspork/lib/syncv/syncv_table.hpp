#pragma once

#include <cassert>
#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "syncv_types.hpp"
#include "tl_sig.hpp"

namespace camspork
{

struct SyncvTable;

// We record pending barrier awaits as (barrier ID, counter) pairs.
// barrier_id_bits many bits are used for the barrier ID.
// (32 - barrier_id_bits) bits are used for the counter.
//
// This limits the number of live barriers and the number of times
// a barrier can be used. The latter is capped in a real CUDA program
// by the number of times an mbarrier can be used: pow(2, 20).
constexpr uint32_t barrier_id_bits = 10;
constexpr uint32_t max_live_barriers = 1u << barrier_id_bits;
using pending_await_t = uint32_t;

inline uint32_t pending_await_barrier_id(pending_await_t id)
{
    return id & ((1u << barrier_id_bits) - 1u);
}

inline uint32_t pending_await_counter(pending_await_t id)
{
    return id >> barrier_id_bits;
}

inline pending_await_t pack_pending_await(uint32_t barrier_id, uint32_t counter)
{
    const uint32_t id = barrier_id | counter << barrier_id_bits;
    assert(pending_await_barrier_id(id) == barrier_id);
    assert(pending_await_counter(id) == counter);
    return id;
}

struct VisRecordDebugData
{
    uint8_t original_qual_tl;
    std::vector<TlSigInterval> visibility_set;
    std::vector<pending_await_t> pending_await_list;
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

struct SyncvDebugValidateInput
{
    size_t size;
    const assignment_record_id* p_records;
};



// *** Primary Implemented Interface ***
SyncvTable* new_syncv_table(const syncv_init_t& init);
SyncvTable* copy_syncv_table(const SyncvTable* table);
void delete_syncv_table(SyncvTable* table);
void on_r(SyncvTable* table, assignment_record_id* p_record, const ThreadCuboid& cuboid, uint32_t bitfield);
void on_rw(SyncvTable* table, assignment_record_id* p_record, const ThreadCuboid& cuboid, uint32_t bitfield);
void on_r(SyncvTable* table, AssignmentRecordWindow window, const ThreadCuboid& cuboid, uint32_t bitfield);
void on_rw(SyncvTable* table, AssignmentRecordWindow window, const ThreadCuboid& cuboid, uint32_t bitfield);
void on_check_free(SyncvTable* table, AssignmentRecordWindow window, const ThreadCuboid& cuboid, uint32_t bitfield);
void clear_visibility(SyncvTable* table, size_t N, assignment_record_id* array);
void alloc_barriers(SyncvTable* table, size_t N, barrier_id* barriers);
void free_barriers(SyncvTable* table, size_t N, barrier_id* barriers);
void on_fence(SyncvTable* table, bool transitive, const ThreadCuboid& cuboid,
        uint32_t L1_bitfield, uint32_t L2_full_bitfield, uint32_t L2_temporal_bitfield);
void on_arrive(SyncvTable* table, barrier_id* bar, TlSigInterval V1, bool transitive);
void on_await(SyncvTable* table, barrier_id* bar, TlSigInterval V2_full, TlSigInterval V2_temporal);
void begin_no_checking(SyncvTable* table);
void end_no_checking(SyncvTable* table);



// *** Debug Inspection Interface ***
void debug_get_read_vis_record_data(const SyncvTable* table, uint32_t id, VisRecordDebugData* out);
void debug_get_mutate_vis_record_data(const SyncvTable* table, uint32_t id, VisRecordDebugData* out);
void debug_validate_state(SyncvTable* table, size_t input_count, const SyncvDebugValidateInput* p_inputs);
void debug_pre_delete_check(SyncvTable* table);

}
