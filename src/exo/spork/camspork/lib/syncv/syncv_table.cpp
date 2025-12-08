#include "syncv_table.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <functional>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "tl_sig.hpp"
#include "vis_record_history_log.hpp"
#include "../program/log_request.hpp"
#include "../util/bit_util.hpp"
#include "../util/cuboid_util.hpp"
#include "../util/node_pool.hpp"
#include "../util/require.hpp"

// Maybe replace later
#include <map>
#include <unordered_map>
#include <unordered_set>
template <typename K, typename V> using BinaryTree = std::map<K, V>;
template <typename K, typename V> using Map = std::unordered_map<K, V>;
template <typename V> using Set = std::unordered_set<V>;
template <typename V> using MultiSet = std::unordered_multiset<V>;

namespace camspork
{

namespace
{

using refcnt_t = uint32_t;

// We attach a linked list of pending awaits to a non-forwarded VisRecord.
// These used to be reference-counted, but are now uniquely owned (fix outdated comments if found).
struct PendingAwaitNode
{
    nodepool::id<PendingAwaitNode> camspork_next_id;
    pending_await_t await_id;

    refcnt_t get_refcnt() const
    {
        return 1;
    }
};

// We encode a visibility set as a list of sorted, minimal
// tl-sig intervals. The intervals are sorted in that
// a.tid_hi <= b.tid_lo for a before b in the list, and the list
// is minimal in that no more intervals are used than needed
// (mostly by merging adjacent intervals with the same qual_bits_by_vis).
struct TlSigIntervalListNode
{
    TlSigInterval data;
    nodepool::id<TlSigIntervalListNode> camspork_next_id;

    refcnt_t get_refcnt() const
    {
        return 1;  // Replace if refcnt member added
    }
};

static_assert(sizeof(TlSigIntervalListNode) == 28, "Check that you meant to change this perf-critical struct");

struct VisRecord
{
    // Owning reference to singly-linked list.
    nodepool::id<TlSigIntervalListNode> visibility_set;

    // Owning reference to tree node.
    nodepool::id<PendingAwaitNode> pending_awaits;

    static constexpr uint32_t num_memoize_hash_bits = 62;

    // TODO move these to VisRecordListNode.

    uint64_t forwarded_flag: 1;

    // This has nothing to do with the main purpose of the struct; only needed for assignment_record_remove_duplicates.
    // This should be in AssignmentRecordVisNode conceptually, but that would waste 4 bytes.
    uint64_t tmp_is_duplicate: 1;

    // Calculated by memoize_or_forward.
    uint64_t memoize_hash_bits: num_memoize_hash_bits;
};

enum class VisRecordKind
{
    Default,
};

// Note, VisRecordKind currently serves no purpose.
// It used to separately encode VisRecord for Reads and Mutates,
// but these cases were unified due to the advent of vis_flag_temporal.
template <VisRecordKind K>
struct VisRecordListNode
{
    static constexpr VisRecordKind vis_record_kind = K;

    // Count of owning references.
    // AssignmentRecord references (and AssignmentRecordVisNode) are owning.
    // Forwarding references are owning.
    // Memoization table references are non-owning.
    refcnt_t refcnt;

    // If in base state, this should be 0.
    // If in the forwarding state, this is an owning reference to the forwarded-to visibility record.
    nodepool::id<VisRecordListNode<K>> camspork_next_id;

    // If the visibility record is in the base state, this is the valid data.
    // If the visibility record is in the forwarding state, the data is that of the record at get(camspork_next_id).
    VisRecord base_data;

    bool is_forwarded() const
    {
        return base_data.forwarded_flag;
    }

    refcnt_t get_refcnt() const
    {
        return refcnt;
    }
};

using DefaultVisRecordListNode = VisRecordListNode<VisRecordKind::Default>;

static_assert(sizeof(DefaultVisRecordListNode) == 24, "Check that you meant to change this perf-critical struct");

template <VisRecordKind K>
struct AssignmentRecordVisNode
{
    static constexpr VisRecordKind vis_record_kind = K;

    // Linked list of read/mutate vis records for an assignment record.
    // Don't use the camspork_next_id in the VisRecord itself.
    nodepool::id<VisRecordListNode<K>> vis_record_id;
    nodepool::id<AssignmentRecordVisNode<K>> camspork_next_id;

    refcnt_t get_refcnt() const
    {
        return 1;  // Replace if refcnt member added
    }
};

using DefaultAssignmentRecordVisNode = AssignmentRecordVisNode<VisRecordKind::Default>;

// Assignment record: collection of mutate visibility records + collection of read visibility records.
// This is associated for each position (scalar, or value in a tensor)
// of the program undergoing synchronization validation.
//
// Multiple positions may reference the same assignment record.
// We implement a copy-on-write strategy (exception: may modify in-place if no one will hold
// a reference to the old assignment record anymore).
struct AssignmentRecord
{
    nodepool::id<AssignmentRecord> camspork_next_id{0};
    refcnt_t refcnt = 0;

    // Zero or more mutate visibility records.
    // I think multiple mutate visibility records are needed only for atomics.
    nodepool::id<DefaultAssignmentRecordVisNode> mutate_vis_records_head_id{0};

    // Zero or more read visibility records.
    nodepool::id<DefaultAssignmentRecordVisNode> read_vis_records_head_id{0};

    // See assignment_record_remove_duplicates.
    // This can become a bitfield if we need the space for something else, but I had issues with -Wconversion.
    uint16_t lazy_last_augment_counter_bits;

    refcnt_t get_refcnt() const
    {
        return refcnt;
    }
};


struct BarrierArriveState
{
    // Linked lists of owning references to VisRecord.
    // A base-state VisRecord is in the list N-many times iff the VisRecord has
    // (parent, arrive_count) N-many times in its pending_awaits.
    // Forwarding-state VisRecords may be in the lists as well ... ignore them if found.
    //
    // Re-use of "assignment record" struct is just pragmatic (maybe confusing).
    nodepool::id<DefaultAssignmentRecordVisNode> vis_records_head_id{0};
};


struct BarrierState
{
    int32_t arrive_count;
    int32_t await_count;

    // Sorted by arrive_count.
    // Entries removed from the list upon matched await.
    BinaryTree<int32_t, BarrierArriveState> arrive_states;
};


struct AssignmentRecordCensusEntry
{
    uint32_t count = 0;
    nodepool::id<AssignmentRecord> new_node_id{};
    size_t linear_index_in_input = 0;
};

using CensusMap = Map<nodepool::id<AssignmentRecord>, AssignmentRecordCensusEntry>;

struct history_log_vis_record_id
{
    VisRecordHistoryLog::vis_record_id_t data;

    template <VisRecordKind K>
    history_log_vis_record_id(nodepool::id<VisRecordListNode<K>> node_id)
    {
        // We currently pass through the ID directly.
        // If multiple VisRecordKind are added, then we have to add some bits to disambiguate.
        static_assert(sizeof(data) == 4);
        data = node_id.id_bits;
        // This used to be: data = node_id.id_bits << 1 | IsMutate;
        // CAMSPORK_REQUIRE_CMP(node_id.id_bits, <=, 0x7FFF'FFFF, "too many node IDs");
    }

    operator VisRecordHistoryLog::vis_record_id_t() const
    {
        return data;
    }
};

struct SyncvTrivialLogger
{
    template <typename Input, typename VisRecordList>
    void excut_log_assignment_records(
            const SyncvTable&, Input, const VisRecordList&, bool)
    {
    }

    void excut_update_assignment_record_ids(const CensusMap&)
    {
    }

    void excut_update_assignment_record_ids(nodepool::id<AssignmentRecord>)
    {
    }

    void history_set_sync_stmt_info(const SyncvFence&)
    {
    }

    void history_set_sync_stmt_info(const SyncvArrive&, const BarrierState&, uint32_t)
    {
    }

    void history_set_sync_stmt_info(const SyncvAwait&, const BarrierState&, uint32_t, uint32_t)
    {
    }

    template <VisRecordKind K>
    void history_new_vis_record(SyncvTable&, nodepool::id<VisRecordListNode<K>>)
    {
    }

    template <VisRecordKind K>
    void history_vis_record_change(
            SyncvTable&, nodepool::id<VisRecordListNode<K>>, nodepool::id<VisRecordListNode<K>>, bool)
    {
    }

    template <VisRecordKind K>
    void history_vis_record_checked(nodepool::id<VisRecordListNode<K>>, bool)
    {
    }

    template <VisRecordKind K>
    void history_vis_record_error(nodepool::id<VisRecordListNode<K>>, TlSig)
    {
    }
};

struct VisRecordChunkMaxHash
{
    uint64_t max_hash;  // Max of memoize_hash_bits of VisRecords in chunk.

    bool operator< (const VisRecordChunkMaxHash& other)
    {
        return max_hash < other.max_hash;
    }
};

template <VisRecordKind K>
struct VisRecordChunk : VisRecordChunkMaxHash
{
    static constexpr VisRecordKind vis_record_kind = K;
    std::vector<nodepool::id<VisRecordListNode<K> > > nodes;
};

struct VisRecordEntropy
{
    uint32_t x = 0x19980724;
    uint32_t y = 0x20010106;

    // Copied pseudo random number generation code.
    // http://www.jcgt.org/published/0009/03/02/
    // Hash Functions for GPU Rendering, Mark Jarzynski, Marc Olano, NVIDIA
    void pcg3d_z(uint32_t z)
    {
        x = x*1664525u + 1013904223u;
        y = y*1664525u + 1013904223u;
        z = z*1664525u + 1013904223u;

        x += y*z;
        y += z*x;
        z += x*y;

        x ^= x >> 16u;
        y ^= y >> 16u;
        z ^= z >> 16u;

        x += y*z;
        y += z*x;
        z += x*y;
    }

    void pcg3d_z(uint64_t) = delete;

    void operator() (TlSigInterval interval)
    {
        pcg3d_z(interval.tid_lo);
        pcg3d_z(interval.tid_hi);
        for (qual_bits_t q : interval.qual_bits_by_vis.array) {
            pcg3d_z(q);
        }
    }

    void operator() (pending_await_t await_id)
    {
        pcg3d_z(await_id);
    }

    uint32_t get() const
    {
        // Note, comment out below lines to test edge cases.
        // return 0;
        // return UINT32_MAX;

        return x * 137 + y * 19;
    }
};

}  // end namespace



// "Everything" struct that implements "backend state" for the synchronization and barrier environments.
// The environments consist of IDs that index into this table. The reason we have this is the synchronization
// enviroment defines many "global" operations that potentially modify every visibility record in existence,
// so we centralize their state here, and optimize these global operations by memoizing identical visibility records.
//
// NOTE: do NOT include "back pointers" to the SyncvTable as that will defeat copying.
// For the most part this is trivially copyable because of our use of node pools.
// A linked list can be copied by just copying the node pool -- the IDs (indexing into the pools)
// remain valid for the copy.
struct SyncvTable
{
    // Failure flag
    // This is to be set upon an exception being thrown through the non-private
    // interface. If this happens, the internal state may be inconsistent, but memory
    // shouldn't be formally leaked since we'll delete the memory pools later anyway.
    // Ignore all further SyncvTable commands if failed = true.
    bool failed = false;

    // Number of times begin_no_checking was called minus end_no_checking. (TODO remove???)
    uint32_t no_checking_counter = 0;

    // Counters for operations
    uint64_t augment_counter = 0;     // Number of Fence+Await+ThreadJoin

    auto get_augment_counter_bits() const
    {
        return static_cast<decltype(AssignmentRecord::lazy_last_augment_counter_bits)>(augment_counter);
    }

    // Memory pool state.
    uintptr_t original_memory_budget = 0;
    uintptr_t current_memory_budget = 0;
    std::tuple<
        nodepool::Pool<AssignmentRecord>,
        nodepool::Pool<TlSigIntervalListNode>,
        nodepool::Pool<PendingAwaitNode>,
        nodepool::Pool<DefaultVisRecordListNode>,
        nodepool::Pool<DefaultAssignmentRecordVisNode>> pool_tuple;

    // Barrier state.
    // The Nth bit is 1 if N is allocated as a barrier ID.
    uint64_t live_barrier_bits[max_live_barriers / 64] = {0};
    BarrierState barrier_states[max_live_barriers];

    // Memoization table state.
    //
    // All non-forwarding-state VisRecord must be in the memoization vis_record_table, except that
    // when we apply modifications to the VisRecord objects in the table, the modified ones are temporarily moved
    // into the modified_vis_records list.
    //
    // The VisRecord objects are sorted by hash key. This is split into "chunks" (to avoid expensive insertions),
    // with the full sorted table being the concatenation of all the chunks.
    //
    // The vis_record_table itself does not own VisRecord references.
    // Thus, VisRecord objects must be removed from the table upon deallocation.
    // The modified_vis_records table DOES own VisRecord references, to avoid unexpected deallocations
    // causing inconsistent memoization state while we are mapping over the memoization table.
    std::vector<VisRecordChunk<VisRecordKind::Default>> vis_record_table;
    std::vector<nodepool::id<DefaultVisRecordListNode>> modified_vis_records;


    // *** Memory Pool Allocators; Linked List Manipulation ***
    // See nodepool for more info.



    template <typename ListNode>
    ListNode& alloc_default_node(nodepool::id<ListNode>* out_id)
    {
        using TypedPool = nodepool::Pool<ListNode>;
        TypedPool& pool = std::get<TypedPool>(pool_tuple);
        return pool.alloc_default_node(&current_memory_budget, out_id);
    }

    template <typename ListNode>
    void extend_free_list(nodepool::id<ListNode> head_id)
    {
        using TypedPool = nodepool::Pool<ListNode>;
        TypedPool& pool = std::get<TypedPool>(pool_tuple);
        pool.extend_free_list(head_id);
    }

    template <typename ListNode>
    void insert_next_node(nodepool::id<ListNode>* p_insert_after, nodepool::id<ListNode> insert_me)
    {
        using TypedPool = nodepool::Pool<ListNode>;
        TypedPool& pool = std::get<TypedPool>(pool_tuple);
        return pool.insert_next_node(p_insert_after, insert_me);
    }

    // Given a pointer to the camspork_next_id member of a node in a list,
    // but don't add it to the free chain: the node is returned to the caller.
    template <typename ListNode>
    [[nodiscard]] nodepool::id<ListNode> remove_next_node(nodepool::id<ListNode>* p_id)
    {
        using TypedPool = nodepool::Pool<ListNode>;
        TypedPool& pool = std::get<TypedPool>(pool_tuple);
        return pool.remove_next_node(p_id);
    }

    // Given a pointer to the camspork_next_id member of a node in a list,
    // remove the next node of the list and add its memory to the free chain.
    // This shouldn't be used if the ListNode itself owns stuff.
    template <typename ListNode>
    void remove_and_free_next_node(nodepool::id<ListNode>* p_id)
    {
        nodepool::id<ListNode> victim_id = remove_next_node(p_id);
        CAMSPORK_REQUIRE(!get(victim_id).camspork_next_id, "Should have been removed from list");
        extend_free_list(victim_id);               // so this free only adds 1 node to free chain.
    }

    template <typename ListNode>
    ListNode& get(nodepool::id<ListNode> id)
    {
        using TypedPool = nodepool::Pool<ListNode>;
        return std::get<TypedPool>(pool_tuple).get(id);
    }

    template <typename ListNode>
    const ListNode& get(nodepool::id<ListNode> id) const
    {
        using TypedPool = nodepool::Pool<ListNode>;
        return std::get<TypedPool>(pool_tuple).get(id);
    }

    template <typename ListNode>
    uint32_t debug_node_pool_size() const
    {
        using TypedPool = nodepool::Pool<ListNode>;
        const uint32_t sz{std::get<TypedPool>(pool_tuple).size()};
        return sz;
    }

    template <typename ListNode>
    Set<nodepool::id<ListNode>> debug_free_node_ids() const
    {
        Set<nodepool::id<ListNode>> id_set;
        using TypedPool = nodepool::Pool<ListNode>;
        std::get<TypedPool>(pool_tuple).get_free_ids(&id_set);
        return id_set;
    }

    template <typename ListNode>
    const nodepool::Pool<ListNode>& debug_get_pool() const
    {
        using TypedPool = nodepool::Pool<ListNode>;
        return std::get<TypedPool>(pool_tuple);
    }

    // Increment reference count of assignment record
    void incref(nodepool::id<AssignmentRecord> id)
    {
        AssignmentRecord& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "should not have started with 0 refcnt");
        node.refcnt++;
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "reference count overflow");
    }

    // Decrement reference count of assignment record.
    void decref(nodepool::id<AssignmentRecord> id, uint32_t nref = 1)
    {
        CAMSPORK_REQUIRE(id, "decref(0)");
        AssignmentRecord& node = get(id);
        CAMSPORK_REQUIRE_CMP(nref, <=, node.refcnt, "tried to decref more than the refcnt");
        node.refcnt -= nref;
        if (0 == node.refcnt) {
            reset_assignment_record(&node);
            CAMSPORK_REQUIRE(!node.camspork_next_id, "Unexpected: AssignmentRecord next only used for free list");
            extend_free_list(id);
        }
    }

    // Increment reference count of visibility record.
    template <VisRecordKind K>
    void incref(nodepool::id<VisRecordListNode<K>> id, uint32_t added_refcnt = 1)
    {
        VisRecordListNode<K>& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "should not have started with 0 refcnt");
        node.refcnt += added_refcnt;
        CAMSPORK_REQUIRE_CMP(node.refcnt, >, added_refcnt, "reference count overflow");
    }

    // Decrement reference count of visibility record,
    // and handle necessary free-ing in case of 0 refcnt.
    template <VisRecordKind K>
    void decref(nodepool::id<VisRecordListNode<K>> id)
    {
        CAMSPORK_REQUIRE(id, "decref(0)");
        VisRecordListNode<K>& node = get(id);
        if (0 == --node.refcnt) {
            if (node.is_forwarded()) {
                // Add physical storage of victim visibility record to free chain,
                // then decref owning reference to forwarded-to visibility record,
                auto fwd_id = node.camspork_next_id;
                CAMSPORK_REQUIRE(fwd_id, "reporting forwarding state but not forwarded anywhere");
                free_single_vis_record(id);
                decref(fwd_id);  // Hope for tail call.
            }
            else {
                // Non-forwarded (base) visibility record must be removed from memoization first.
                auto memoized_id = remove_memoized(id);
                CAMSPORK_REQUIRE_CMP(id, ==, memoized_id, "should have been found in memoization table");
                CAMSPORK_REQUIRE(!get(memoized_id).camspork_next_id, "Should not have forwarding id set.");
                free_single_vis_record(memoized_id);
            }
        }
    }

    void reset_vis_record_data(VisRecord* p_data)
    {
        static_assert(sizeof(*p_data) == 16, "update me");
        extend_free_list(p_data->visibility_set);
        p_data->visibility_set = {};
        extend_free_list(p_data->pending_awaits);
        p_data->pending_awaits = {};
        p_data->memoize_hash_bits = 0;
    }

    template <VisRecordKind K>
    void free_single_vis_record(nodepool::id<VisRecordListNode<K>> id)
    {
        CAMSPORK_REQUIRE(id, "unexpected 0 id");
        VisRecordListNode<K>& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, ==, 0, "unexpected nonzero refcnt");
        reset_vis_record_data(&node.base_data);
        node.camspork_next_id = {};  // Avoid freeing entire list.
        extend_free_list(id);
    }

    template <VisRecordKind K>
    void assignment_record_remove_vis_records(nodepool::id<AssignmentRecordVisNode<K>>* p_head_id)
    {
        // Decref visibility records
        const auto head_id = *p_head_id;
        auto id = head_id;
        while (id) {
            AssignmentRecordVisNode<K>& node = get(id);
            decref(node.vis_record_id);
            id = node.camspork_next_id;
        }

        // Free physical storage of linked list
        extend_free_list(head_id);

        // 0 out input.
        *p_head_id = {};
    }

    void reset_assignment_record(AssignmentRecord* p_record)
    {
        assignment_record_remove_vis_records(&p_record->mutate_vis_records_head_id);
        assignment_record_remove_vis_records(&p_record->read_vis_records_head_id);
        p_record->lazy_last_augment_counter_bits = 0;
    }



    // *** Operations on Visibility Records ***
    // If the visibility record is modified, you need to be careful to update the memoization table.



    // Allocate a new visibility record.
    // This will later need to be added to the memoization table.
    template <VisRecordKind K, typename ThreadInit>
    VisRecordListNode<K>& alloc_vis_record(
            const ThreadInit& thread_init, SyncvAccessInfo access, nodepool::id<VisRecordListNode<K>>* out)
    {
        nodepool::id<VisRecordListNode<K>> vis_record_id;
        VisRecordListNode<K>& vis_record = alloc_default_node(&vis_record_id);
        vis_record.refcnt = 1;
        vis_record.base_data.forwarded_flag = 0;
        vis_record.base_data.visibility_set = {};
        vis_record.base_data.pending_awaits = {};

        // Initialize visibility set = linked list of intervals generated from the initial thread / thread cuboid.
        const qual_bits_t q = access.initial_qual_bit;
        static_assert(vis_flags_all == 15, "review if this needs changing");
        const auto vis_flags = access.is_ooo ? vis_flag_issue : vis_flags_all;
        const QualBitsByVis qual_bits_by_vis = qual_vis_product(q, vis_flags);
        nodepool::id<TlSigIntervalListNode>* p_node_id = &vis_record.base_data.visibility_set;
        thread_init.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
        {
            TlSigIntervalListNode& tl_sigs_node = alloc_default_node(p_node_id);
            tl_sigs_node.data = TlSigInterval{tid_lo, tid_hi, qual_bits_by_vis};
            p_node_id = &tl_sigs_node.camspork_next_id;
        });
        CAMSPORK_REQUIRE(vis_record.base_data.visibility_set, "buggy: empty visibility set");

        // Add pending awaits
        for (uint32_t i = 0; i < access.barrier_count; ++i) {
            const auto barrier_index = get_barrier_index(access.trailing_barriers[i]);
            BarrierState& state = barrier_states[barrier_index];
            const pending_await_t info = pack_pending_await(barrier_index, state.arrive_count);
            nodepool::id<PendingAwaitNode> new_id;
            PendingAwaitNode& node = alloc_default_node(&new_id);
            node.await_id = info;
            node.camspork_next_id = vis_record.base_data.pending_awaits;
            vis_record.base_data.pending_awaits = new_id;
            // This keeps a reference to the new VisRecord.
            // Memoization may cause the reference to change to a
            // forwarding-state VisRecord, which isn't a well-tested code path probably.
            extend_barrier_arrive_state(vis_record_id, info);
        }

        // Add "atomic-only" visibility across all possible threads, if applicable,
        // All this must be done before trying to memoize.
        if (access.atomic_qual_bits) {
            static_assert(sizeof(TlSigInterval::tid_hi) == 4);
            const QualBitsByVis qual_bits_by_vis = qual_vis_product(access.atomic_qual_bits, vis_flag_atomic_only);
            const TlSigInterval atomic_interval{0, UINT32_MAX, qual_bits_by_vis};
            union_tl_sig_interval(&vis_record.base_data, atomic_interval);
        }

        *out = vis_record_id;
        return vis_record;
    }

    // Union tl_sig interval into the visibility sets.
    // Caller will have to make changes to the memoization table afterwards.
    void union_tl_sig_interval(VisRecord* p, TlSigInterval input)
    {
        // Non-empty input check (cartesian product of non-empty thread interval and non-empty qual-tl set).
        CAMSPORK_REQUIRE_CMP(input.tid_hi, >, input.tid_lo, "non-empty input check");
        CAMSPORK_REQUIRE_CMP(0, !=, input.qual_bits_by_vis.array[0], "non-empty input check");  // why atomic-only?
        using node_id = nodepool::id<TlSigIntervalListNode>;

        // Modify and/or add intervals.
        // We can view each pointer-to-node_id as a "gap" between intervals (imagine an arrow between nodes = a gap).
        // We abuse language to consider there to be a "gap" before the leftmost and after the rightmost intervals.
        // This runs for N+1 iterations where N is the current interval count.
        uint32_t gap_tid_lo = 0;
        for (node_id* p_id = &p->visibility_set; 1;) {
            // Must remember the next iteration's node now, so we don't get confused by insertion.
            const node_id original_next_node_id = *p_id;

            uint32_t gap_tid_hi;
            if (original_next_node_id) {
                gap_tid_hi = get(original_next_node_id).data.tid_lo;
                CAMSPORK_REQUIRE_CMP(gap_tid_lo, <=, gap_tid_hi, "intervals were out of order.");
            }
            else {
                gap_tid_hi = input.tid_hi;  // For "gap" after the rightmost interval.
            }

            // If the gap is non-empty and overlaps the input interval, we need to insert a new interval.
            TlSigInterval new_interval{};
            new_interval.tid_lo = std::max(gap_tid_lo, input.tid_lo);
            new_interval.tid_hi = std::min(gap_tid_hi, input.tid_hi);
            new_interval.qual_bits_by_vis = input.qual_bits_by_vis;
            if (new_interval.tid_hi > new_interval.tid_lo) {
                node_id new_node_id{};
                TlSigIntervalListNode& new_node = alloc_default_node(&new_node_id);
                new_node.data = new_interval;
                insert_next_node(p_id, new_node_id);
            }

            if (!original_next_node_id) {
                break;
            }

            // This runs N times (leftmost interval is the "next node" of the imaginary before-left gap)
            // We will modify each original interval, which is the "next node" of the gap just processed.
            //
            // The interval may be subdivided into up to 3 intervals depending on overlap with input interval,
            // since only the overlapped portion should have its qual_bits_by_vis modified.
            //
            // 1st interval: keeps original bits, left of intersection.
            // 2nd interval: qual_bits_by_vis augmented, footprint of intersection.
            // 3rd interval: keeps original bits, right of intersection.
            TlSigIntervalListNode& next_node = get(original_next_node_id);
            const TlSigInterval original_data = next_node.data;
            const uint32_t intersect_tid_lo = std::max(original_data.tid_lo, input.tid_lo);
            const uint32_t intersect_tid_hi = std::min(original_data.tid_hi, input.tid_hi);
            const QualBitsByVis union_bits = input.qual_bits_by_vis | original_data.qual_bits_by_vis;
            const bool change_needed =
                    (union_bits != original_data.qual_bits_by_vis) && (intersect_tid_lo < intersect_tid_hi);

            // Possibly add 1st interval.
            if (change_needed && original_data.tid_lo < intersect_tid_lo) {
                node_id new_node_id{};
                TlSigIntervalListNode& new_node = alloc_default_node(&new_node_id);
                new_node.data.tid_lo = original_data.tid_lo;
                new_node.data.tid_hi = intersect_tid_lo;
                new_node.data.qual_bits_by_vis = original_data.qual_bits_by_vis;
                CAMSPORK_REQUIRE_CMP(*p_id, ==, original_next_node_id, "linked list corrupt");
                insert_next_node(p_id, new_node_id);
            }

            // Now update iteration state (we do this now so the following insertions work)
            p_id = &next_node.camspork_next_id;
            gap_tid_lo = next_node.data.tid_hi;

            if (change_needed) {
                // 2nd interval; we recycle the existing node since the 2nd interval is guaranteed non-empty
                next_node.data.tid_lo = intersect_tid_lo;
                next_node.data.tid_hi = intersect_tid_hi;
                next_node.data.qual_bits_by_vis = union_bits;

                // Possibly add 3rd interval, insert after 2nd interval.
                if (intersect_tid_hi < original_data.tid_hi) {
                    node_id new_node_id{};
                    TlSigIntervalListNode& new_node = alloc_default_node(&new_node_id);
                    new_node.data.tid_lo = intersect_tid_hi;
                    new_node.data.tid_hi = original_data.tid_hi;
                    new_node.data.qual_bits_by_vis = original_data.qual_bits_by_vis;
                    insert_next_node(p_id, new_node_id);
                    p_id = &new_node.camspork_next_id;  // Need to point to the real gap (after original_data.tid_hi)
                }
            }
        }

        // Merge redundant intervals
        // For each "current node", we try to merge it with the next node, if it exists.
        CAMSPORK_REQUIRE(p->visibility_set, "unexpected empty visibility set");
        TlSigIntervalListNode* p_current_node = &get(p->visibility_set);
        for (node_id next_id; (next_id = p_current_node->camspork_next_id); ) {
            TlSigIntervalListNode* p_next_node = &get(next_id);

            TlSigInterval& current = p_current_node->data;
            const TlSigInterval& next = p_next_node->data;
            CAMSPORK_REQUIRE_CMP(current.tid_lo, <, current.tid_hi, "invalid interval");
            CAMSPORK_REQUIRE_CMP(current.tid_hi, <=, next.tid_lo, "invalid interval overlap");
            CAMSPORK_REQUIRE_CMP(next.tid_lo, <, next.tid_hi, "invalid interval");

            if (current.tid_hi == next.tid_lo && current.qual_bits_by_vis == next.qual_bits_by_vis) {
                // Merge next node into current node, and remove next node from list.
                current.tid_hi = next.tid_hi;
                CAMSPORK_REQUIRE_CMP(p_current_node->camspork_next_id, ==, next_id, "corrupt linked list");
                remove_and_free_next_node(&p_current_node->camspork_next_id);
            }
            else {
                // If not merged, we need to process the next node in the next iteration.
                // If we did merge, we keep the same current node, since we may need to merge with the new next node.
                p_current_node = p_next_node;
            }
        }
    }

    bool valid_adjacent(TlSigInterval first, TlSigInterval second) const
    {
        // Check that the two tl_sig intervals can be adjacent in a valid visibility set encoding.
        return first.tid_lo < first.tid_hi && first.tid_hi <= second.tid_lo && second.tid_lo < second.tid_hi
          && (first.tid_hi < second.tid_lo || first.qual_bits_by_vis != second.qual_bits_by_vis);
    }

    // Check if visibility records are equal.
    bool equal(const VisRecord& a, const VisRecord& b) const
    {
        static_assert(sizeof(a) == 16, "Update me");

        // Check equal intervals. We rely on (and enforce) the non-redundant encoding requirement.
        {
            using node_id = nodepool::id<TlSigIntervalListNode>;
            node_id id_a = a.visibility_set;
            node_id id_b = b.visibility_set;

            while (id_a && id_b) {
                const TlSigIntervalListNode& current_a = get(id_a);
                const TlSigIntervalListNode& current_b = get(id_b);
                id_a = current_a.camspork_next_id;
                id_b = current_b.camspork_next_id;
                CAMSPORK_REQUIRE(!id_a || valid_adjacent(current_a.data, get(id_a).data), "invalid adjacent intervals");
                CAMSPORK_REQUIRE(!id_b || valid_adjacent(current_b.data, get(id_b).data), "invalid adjacent intervals");

                if (current_a.data != current_b.data) {
                    return false;
                }
            }

            if (id_a != id_b) {
                return false;  // Lists have different lengths.
            }
        }

        // Check equal pending awaits.
        {
            using node_id = nodepool::id<PendingAwaitNode>;
            node_id id_a = a.pending_awaits;
            node_id id_b = b.pending_awaits;

            while (id_a && id_b) {
                const PendingAwaitNode& current_a = get(id_a);
                const PendingAwaitNode& current_b = get(id_b);
                id_a = current_a.camspork_next_id;
                id_b = current_b.camspork_next_id;

                if (current_a.await_id != current_b.await_id) {
                    return false;
                }
            }

            if (id_a != id_b) {
                return false;  // Lists have different lengths.
            }
        }
        return true;
    }

    bool synchronizes_with(
            bool transitive, const VisRecord& vis_record, const ThreadCuboid& cuboid, qual_bits_t qual_bits)
    {
        // Check for any intersections between TlSigIntervals generated by ThreadCuboid + qual_bits + flags, and
        // those stored in the VisRecord, with flags including vis_flag_issue, and also vis_flag_full if transitive.
        bool intersects = false;
        nodepool::id<TlSigIntervalListNode> node_id = vis_record.visibility_set;
        const QualBitsByVis qv = qual_vis_product(qual_bits, vis_flag_issue | (transitive ? vis_flag_full : 0));
        cuboid.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
        {
            while (true) {
                if (!node_id) {
                    return;  // Exit lambda
                }
                const TlSigIntervalListNode& node = get(node_id);
                TlSigInterval vis_set_interval = node.data;
                TlSigInterval gen_interval{tid_lo, tid_hi, qv};
                intersects |= vis_set_interval.intersects(gen_interval);
                if (vis_set_interval.tid_hi > tid_hi) {
                    // Keep checking new vis_set_interval against gen_interval until the visibility set interval
                    // has a tid_hi above that of the generated interval (from the ThreadCuboid).
                    // NB on this code path node_id is not updated,
                    // so vis_set_interval will be checked against a new gen_interval.
                    return;
                }
                node_id = node.camspork_next_id;
            }
        });
        return intersects;
    }

    bool all_visible_to(
            const VisRecord& vis_record, const ThreadCuboid& cuboid,
            qual_bits_t want_qual_bits, int32_t vis_flag, TlSig* out_fail_tl_sig)
    {
        // IsConvergent = false case
        // Check that all threads generated by the ThreadCuboid have at least one qual-tl in common with
        // an overlapping interval in the ordered visibility set of the vis_record.
        const int vis_flag_index = get_low_bit_index(vis_flag);
        CAMSPORK_REQUIRE_CMP(vis_flag, ==, (1 << vis_flag_index), "not a vis flag");
        CAMSPORK_REQUIRE_CMP(vis_flag_index, <, num_vis_flags, "not a vis flag");
        bool all_visible = true;
        cuboid.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
        {
            nodepool::id<TlSigIntervalListNode> node_id = vis_record.visibility_set;
            // Interval of threads [missing_threads_lo, tid_hi) not yet found to overlap with
            // a valid TlSigInterval in the visibility set.
            uint32_t missing_threads_lo = tid_lo;

            while (node_id) {
                // Search TlSigIntervals in the VisRecord, skipping any that don't have
                // a qual-tl in common with one of the required visibility flags.
                const TlSigIntervalListNode& node = get(node_id);
                node_id = node.camspork_next_id;
                const TlSigInterval data = node.data;
                if (!(want_qual_bits & data.qual_bits_by_vis.array[vis_flag_index])) {
                    continue;
                }
                // If the interval is after the interval of missing threads, we have found a gap
                // [missing_threads_lo, data.tid_lo) that is not found. We fail at this point (break out of loop).
                if (data.tid_lo > missing_threads_lo) {
                    break;
                }
                missing_threads_lo = std::max(missing_threads_lo, data.tid_hi);
            }
            const bool local_visible = missing_threads_lo >= tid_hi;
            if (!local_visible && all_visible) {
                all_visible = false;
                // Record missing tl-sig the first time a missing one is found.
                out_fail_tl_sig->tid = missing_threads_lo;
                out_fail_tl_sig->qual_tl = get_low_bit_index(want_qual_bits);
                out_fail_tl_sig->vis_flag = vis_flag;
            }
        });
        return all_visible;
    }

    bool any_visible_to(
            const VisRecord& vis_record, const ThreadCuboid& cuboid,
            qual_bits_t want_qual_bits, int32_t vis_flag, TlSig* out_fail_tl_sig)
    {
        // IsConvergent = true case
        // Check that at least one thread generated by the ThreadCuboid has at least one one qual-tl in common with
        // an overlapping interval in the ordered visibility set of the vis_record.
        const int vis_flag_index = get_low_bit_index(vis_flag);
        CAMSPORK_REQUIRE_CMP(vis_flag, ==, (1 << vis_flag_index), "not a vis flag");
        CAMSPORK_REQUIRE_CMP(vis_flag_index, <, num_vis_flags, "not a vis flag");
        uint32_t fail_tid = UINT32_MAX;
        bool any_visible = false;
        cuboid.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
        {
            nodepool::id<TlSigIntervalListNode> node_id = vis_record.visibility_set;

            while (node_id) {
                // Search TlSigIntervals in the VisRecord, skipping any that don't have
                // a qual-tl in common in with one of the required visibility flags.
                const TlSigIntervalListNode& node = get(node_id);
                node_id = node.camspork_next_id;
                const TlSigInterval data = node.data;
                if (!(want_qual_bits & data.qual_bits_by_vis.array[vis_flag_index])) {
                    continue;
                }
                // tid overlap check (non-empty intersection).
                const auto intersect_tid_lo = std::max(tid_lo, data.tid_lo);
                const auto intersect_tid_hi = std::min(tid_hi, data.tid_hi);
                any_visible |= intersect_tid_lo < intersect_tid_hi;
            }
            fail_tid = std::min(fail_tid, tid_lo);
        });
        if (!any_visible) {
            out_fail_tl_sig->tid = fail_tid;
            out_fail_tl_sig->qual_tl = get_low_bit_index(want_qual_bits);
            out_fail_tl_sig->vis_flag = vis_flag;
        }
        return any_visible;
    }

    bool visible_to(
            const VisRecord& vis_record, const ThreadCuboid& cuboid,
            qual_bits_t want_qual_bits, int32_t vis_flag, bool is_convergent, TlSig* out_fail_tl_sig)
    {
        if (is_convergent) {
            return any_visible_to(vis_record, cuboid, want_qual_bits, vis_flag, out_fail_tl_sig);
        }
        else {
            return all_visible_to(vis_record, cuboid, want_qual_bits, vis_flag, out_fail_tl_sig);
        }
    }

    // *** Barrier ID Allocation ***
    // For now barrier_id::data only stores the barrier index number + 1, but this could change.



    uint32_t get_barrier_index(barrier_id bar) const
    {
        CAMSPORK_REQUIRE_CMP(bar.data, !=, 0, "null barrier");
        const auto index = (bar.data - 1);
        CAMSPORK_REQUIRE_CMP(index, <, max_live_barriers, "max_live_barriers limit exceeded");
        return uint32_t(index);
    }

    void set_barrier_index(barrier_id* bar, uint32_t index) const
    {
        CAMSPORK_REQUIRE_CMP(index, <, max_live_barriers, "max_live_barriers limit exceeded");
        bar->data = index + 1;
    }

    void alloc_barriers(size_t N, barrier_id* barriers)
    {
        uint32_t barrier_index = ~0u;
        size_t num_allocated = 0;

        if (N == 0) {
            return;
        }

        for (uint32_t word_index = 0; word_index < max_live_barriers / 64; ++word_index) {
            uint64_t negated_bits;
            while ((negated_bits = ~live_barrier_bits[word_index]) != 0) {
                CAMSPORK_REQUIRE_CMP(barriers[num_allocated].data, ==, 0, "allocated barrier without free");
                uint8_t bit_index = pop_low_bit_index(&negated_bits);
                barrier_index = word_index * 64 + bit_index;
                set_barrier_index(&barriers[num_allocated], barrier_index);
                live_barrier_bits[word_index] = ~negated_bits;
                barrier_states[barrier_index] = {};
                if (++num_allocated >= N) {
                    return;
                }
            }
        }

        CAMSPORK_REQUIRE(false, "Exceeded implementation limit (max number of barriers per program)");
    }

    void free_barriers(size_t N, barrier_id* barriers, bool check_arrive_await)
    {
        for (size_t i = 0; i < N; ++i) {
            if (!barriers[i]) {
                continue;
            }
            const auto barrier_index = get_barrier_index(barriers[i]);
            BarrierState& state = barrier_states[barrier_index];
            if (check_arrive_await) {
                if (state.arrive_count != state.await_count) {
                    std::string message =
                        "Arrive count (" + std::to_string(state.arrive_count) + ") != Await count ("
                        + std::to_string(state.await_count) + ")";
                    throw SyncvCheckFail{std::move(message), i};
                }
            }

            // Normally, retire_barrier_arrive augments VisRecords, but here we just pass no_op,
            // so the only intended effect is for us to free memory.
            for (auto& pair : state.arrive_states) {
                const auto arrive_count = pair.first;
                BarrierArriveState& arrive = pair.second;
                auto no_op = [] (auto, auto) {};
                const pending_await_t await_id = pack_pending_await(barrier_index, arrive_count);
                retire_barrier_arrive(&arrive, await_id, no_op, SyncvTrivialLogger{});
            }
            state.arrive_states.clear();

            uint64_t& word = live_barrier_bits[barrier_index / 64u];
            const uint64_t bit = uint64_t(1) << (barrier_index & 63u);
            CAMSPORK_REQUIRE_CMP((word & bit), !=, 0, "Barrier ID was not allocated");
            word &= ~bit;
            barriers[i].data = 0;
        }
        memoize_modified();
    }



    // *** Memoization ***

    uint64_t hash_vis_record(const VisRecord& vis_record) const
    {
        // 62-bit hash
        // (hi)
        // 6 bits: issue QualTL i.e. unique q such that (tid, q, vis_flag_issue) exists (num_qual_tl if no such q).
        // 32 bits: max tid of timeline signatures (tid, q, v) with v != atomic-only; 0 if no such tid.
        // 24 bits: entropy, TODO.
        // (lo)
        //
        static_assert(VisRecord::num_memoize_hash_bits == 62);
        static_assert(num_qual_tl == 32, "Need more than 6 bits to encode QualTL + sentinel");
        uint32_t non_atomic_max_tid = 0;
        qual_bits_t issue_qual_bits = 0;
        VisRecordEntropy entropy{};

        const nodepool::id<TlSigIntervalListNode>* p_id = &vis_record.visibility_set;

        while (*p_id) {
            const TlSigIntervalListNode& node = get(*p_id);
            const TlSigInterval tl_sigs = node.data;
            p_id = &node.camspork_next_id;

            const bool tl_sigs_atomic_only = tl_sigs.is_atomic_only();
            if (const auto q_tmp = tl_sigs.qual_bits_by_vis.array[vis_flag_index_issue]) {
                issue_qual_bits |= q_tmp;
            }
            // Convert exclusive tid_hi to inclusive non_atomic_max_tid.
            non_atomic_max_tid = std::max(non_atomic_max_tid, tl_sigs_atomic_only ? uint32_t(0) : tl_sigs.tid_hi - 1u);

            entropy(tl_sigs);
        }

        nodepool::id<PendingAwaitNode> await_node_id = vis_record.pending_awaits;
        while (await_node_id) {
            const PendingAwaitNode& node = get(await_node_id);
            entropy(node.await_id);
            await_node_id = node.camspork_next_id;
        }

        uint32_t qual_tl = num_qual_tl;
        if (issue_qual_bits != 0) {
            qual_tl = get_low_bit_index(issue_qual_bits);
            CAMSPORK_REQUIRE_CMP((issue_qual_bits >> qual_tl), ==, 1, "multiple issue QualTL");
        }

        uint64_t hash = entropy.get() & 0xFF'FFFF;
        hash |= uint64_t(non_atomic_max_tid) << 24;
        hash |= uint64_t(qual_tl) << (32 + 24);
        return hash;
    }

    static uint8_t issue_qual_tl_from_hash(uint64_t hash)
    {
        return uint8_t(hash >> (32 + 24));
    }

    static uint32_t max_non_atomic_tid_from_hash(uint64_t hash)
    {
        return uint32_t(hash >> 24);
    }

    // Get inclusive range of possible hash values of VisRecord that
    // could theoretically be modified when interpreting a non-transitive Arrive with the given QualTL
    // as the only QualTL in its SyncTL, and with the minimum tid_lo in the ThreadCuboid being min_tid_lo.
    //
    // For the transitive case, this has to be done for all qual_tl values, including qual-tl = num_qual_tl
    // indicating no issue QualTL.
    static std::pair<uint64_t, uint64_t> hash_bounds_for_arrive(uint8_t qual_tl, uint32_t min_tid_lo)
    {
        // max_tid must be at least min_tid_lo.
        // NB max_tid is inclusive, unlike exclusive tid_hi.
        CAMSPORK_REQUIRE_CMP(qual_tl, <=, num_qual_tl, "Out-of-range qual-tl");
        uint64_t qual_hash = uint64_t(qual_tl) << (32 + 24);
        uint64_t hash_lo = qual_hash | uint64_t(min_tid_lo) << 24;
        uint64_t hash_hi = qual_hash | ((uint64_t(1) << (32 + 24)) - 1u);
        return {hash_lo, hash_hi};
    }

    template <VisRecordKind K>
    void update_hash(VisRecordChunk<K>& chunk) const
    {
        CAMSPORK_REQUIRE(!chunk.nodes.empty(), "empty VisRecord chunk");
        chunk.max_hash = read_hash_helper(chunk.nodes.back());
    }

    enum class MemoizeAction
    {
        // Return id of matching VisRecord, or 0 if not found. Read-only operation.
        Find,

        // If matching VisRecord found, remove it from table and return its id; else return 0.
        Remove,

        // If matching VisRecord found, set the command VisRecord to the forwarding state pointing to the matching
        // VisRecord. Return the ID of the matching VisRecord in the memoization table.
        // Otherwise, add the command VisRecord to the memoization table.
        // Recall that the memoization table does not own its reference to the VisRecord.
        MemoizeOrForward,

        // Run callback for all memoized VisRecords with hash values in the inclusive hash_bounds range.
        // Any modified VisRecord are removed from the table and added to modified_vis_records.
        // Must call memoize_modified() afterwards.
        EditAll,

        // Like EditAll but only applies to the matching VisRecord.
        EditOne,
    };

    // Skeleton function for a variety of functions interacting with the memoization table.
    //
    // Command must have members / member functions
    //
    //     static constexpr MemoizeAction memoize_action;
    //
    //     // "matching VisRecord" is one that is equal(...) to this one.
    //     // The hash of the VisRecord must be in the inclusive bounds of hash_bounds.
    //     nodepool::id<VisRecordListNode<K>> node_id;
    //
    //     bool operator() (SyncvTable& env, nodepool::id<VisRecordListNode<K>> vis_record_id)
    //         // needed for EditOne and EditAll.
    //         // Returns `changed` flag.
    template <typename Command>
    nodepool::id<VisRecordListNode<VisRecordKind::Default>> for_vis_record_hash_bounds(
            std::pair<uint64_t, uint64_t> hash_bounds,
            Command&& command)
    {
        constexpr VisRecordKind K = VisRecordKind::Default;
        const uint64_t hash_lo = hash_bounds.first;
        const uint64_t hash_hi = hash_bounds.second;
        CAMSPORK_REQUIRE_CMP(hash_lo, <=, hash_hi, "Invalid hash bounds");

        constexpr MemoizeAction memoize_action = command.memoize_action;
        const size_t max_chunk_size = 2 + vis_record_table.size() / 2u;

        VisRecordListNode<K>* p_command_node = nullptr;
        if constexpr (memoize_action != MemoizeAction::EditAll) {
            p_command_node = &get(command.node_id);
            CAMSPORK_REQUIRE(!p_command_node->is_forwarded(), "command input node must not be forwarded");
            const uint64_t command_hash = p_command_node->base_data.memoize_hash_bits;
            CAMSPORK_REQUIRE_CMP(command_hash, >=, hash_lo, "Incorrect hash bounds");
            CAMSPORK_REQUIRE_CMP(command_hash, <=, hash_hi, "Incorrect hash bounds");
        }

        // Find the first chunk containing hashes at least hash_lo.
        const auto first_chunk_iter = std::lower_bound(
            vis_record_table.begin(),
            vis_record_table.end(),
            VisRecordChunkMaxHash{hash_lo});

        size_t intra_chunk_index = 0, chunk_index = vis_record_table.size();
        if (first_chunk_iter != vis_record_table.end()) {
            chunk_index = first_chunk_iter - vis_record_table.begin();
            // Find the index within the chunk of a VisRecord having a hash at least hash_lo.
            intra_chunk_index = static_cast<size_t>(std::lower_bound(
                first_chunk_iter->nodes.begin(),
                first_chunk_iter->nodes.end(),
                hash_lo,
                [this] (auto lhs, auto rhs)
                {
                    return this->read_hash_helper(lhs) < this->read_hash_helper(rhs);
                }
            ) - first_chunk_iter->nodes.begin());
        }

        // Inspect VisRecord objects.
        uint64_t debug_hash = hash_lo;
        for (; chunk_index < vis_record_table.size(); chunk_index++) {
            VisRecordChunk<K>& chunk = vis_record_table[chunk_index];
            nodepool::id<VisRecordListNode<K>> return_id{};

            for (; intra_chunk_index < chunk.nodes.size(); intra_chunk_index++) {
                nodepool::id<VisRecordListNode<K>> cur_id = chunk.nodes[intra_chunk_index];
                const VisRecordListNode<K>& cur_node = get(cur_id);
                CAMSPORK_REQUIRE(!cur_node.is_forwarded(), "Forwarded VisRecord should not be memoized?");

                CAMSPORK_REQUIRE_CMP(debug_hash, <=, cur_node.base_data.memoize_hash_bits, "VisRecord sorting bug");
                debug_hash = cur_node.base_data.memoize_hash_bits;

                // Exit when hash seen is above range given.
                // If memoizing, insert the new VisRecord here.
                if (cur_node.base_data.memoize_hash_bits > hash_hi) {
                    if constexpr (memoize_action == MemoizeAction::MemoizeOrForward) {
                        chunk.nodes.insert(chunk.nodes.begin() + intra_chunk_index, command.node_id);
                        if (chunk.nodes.size() > max_chunk_size) {
                            // Split the chunk in half if it's too big.
                            const size_t halfway = chunk.nodes.size() / 2;

                            if (false) {
                                fprintf(stderr, "Split chunk %lu [0, %lu, %lu]\n",
                                    chunk_index, halfway, chunk.nodes.size());
                            }

                            VisRecordChunk<K> new_chunk;
                            new_chunk.nodes = std::vector<nodepool::id<VisRecordListNode<K> > >(
                                    chunk.nodes.begin() + halfway, chunk.nodes.end());
                            chunk.nodes.resize(halfway);

                            this->update_hash(new_chunk);
                            this->update_hash(chunk);

                            // !!! Caution iterator invalidation !!!
                            // chunk is not usable after this.
                            vis_record_table.insert(vis_record_table.begin() + chunk_index + 1, std::move(new_chunk));
                        }
                        // Splitting special case, don't go to finalize_chunk_edit.
                        return command.node_id;
                    }
                    goto finalize_chunk_edit;
                }

                // Main work by cases.
                if constexpr (memoize_action == MemoizeAction::Find) {
                    if (equal(p_command_node->base_data, cur_node.base_data)) {
                        return cur_id;  // Return now; we didn't edit the table.
                    }
                }
                else if constexpr (memoize_action == MemoizeAction::Remove) {
                    if (equal(p_command_node->base_data, cur_node.base_data)) {
                        chunk.nodes.erase(chunk.nodes.begin() + intra_chunk_index);
                        return_id = cur_id;
                        goto finalize_chunk_edit;
                    }
                }
                else if constexpr (memoize_action == MemoizeAction::MemoizeOrForward) {
                    VisRecordListNode<K>& command_node = *p_command_node;
                    if (equal(command_node.base_data, cur_node.base_data)) {
                        // If equivalent memoized node found, forward input node to it.
                        const nodepool::id fwd_id = cur_id;
                        CAMSPORK_REQUIRE(fwd_id, "unexpected null");
                        CAMSPORK_REQUIRE_CMP(fwd_id, !=, command.node_id, "Trying to memoize something already in the memoization table.");

                        reset_vis_record_data(&command_node.base_data);
                        command_node.camspork_next_id = fwd_id;
                        command_node.base_data.forwarded_flag = 1;
                        CAMSPORK_REQUIRE(command_node.is_forwarded(), "should now be in forwarding state");
                        incref(fwd_id);  // Forwarding reference is owning.
                        return fwd_id;   // Return now; we didn't edit the table.
                    }
                }
                else {
                    // Editing cases.
                    bool should_edit = true;
                    if constexpr (memoize_action == MemoizeAction::EditOne) {
                        if (equal(p_command_node->base_data, cur_node.base_data)) {
                            CAMSPORK_REQUIRE_CMP(command.node_id, ==, cur_id, "Non-unique entry somehow?");
                        }
                        else {
                            should_edit = false;
                        }
                    }

                    if (should_edit) {
                        // Avoid unexpected deallocation during callback.
                        // Otherwise, this could cause the memoization table to change unexpecedly.
                        incref(cur_id);
                        const bool changed = command(*this, cur_id);

                        if (changed) {
                            // Remove from memoization and add to modified_vis_records.
                            // This steals the incref from before.
                            chunk.nodes.erase(chunk.nodes.begin() + intra_chunk_index);
                            modified_vis_records.push_back(cur_id);
                            intra_chunk_index--;
                        }
                        else {
                            decref(cur_id);
                        }

                        if constexpr (memoize_action == MemoizeAction::EditOne) {
                            return_id = command.node_id;
                            goto finalize_chunk_edit;
                        }
                    }
                }
            }

            // Inspect all chunks starting from index 0 except for the first one inspected
            // which the second std::lower_bound allowed us to skip some work in.
            intra_chunk_index = 0;

            if (memoize_action == MemoizeAction::EditAll) {
                goto finalize_chunk_edit;
            }
            continue;

            // EditAll always runs this code per chunk.
            // Other code paths go here only just before returning the result.
          finalize_chunk_edit:
            // Remove empty chunks
            if (memoize_action != MemoizeAction::Find && chunk.nodes.empty()) {
                // fprintf(stderr, "Removed empty chunk\n");
                vis_record_table.erase(vis_record_table.begin() + chunk_index);
                chunk_index--;
            }
            // Update hash of chunks
            else {
                if (memoize_action != MemoizeAction::Find) {
                    this->update_hash(chunk);
                }
            }
            if constexpr (memoize_action != MemoizeAction::EditAll) {
                return return_id;
            }
        }

        // If we are doing memoize-or-forward and we are still not returned here,
        // we have to add the unique VisRecord to the back of the table.
        if constexpr (memoize_action == MemoizeAction::MemoizeOrForward) {
            if (vis_record_table.empty()) {
                vis_record_table.push_back({});
            }
            VisRecordChunk<K>& chunk = vis_record_table.back();
            chunk.nodes.push_back(command.node_id);
            this->update_hash(chunk);
            return command.node_id;
        }

        return {};
    }

    uint64_t read_hash_helper(uint64_t hash) const
    {
        return hash;
    }

    template <VisRecordKind K>
    uint64_t read_hash_helper(nodepool::id<VisRecordListNode<K>> id) const
    {
        return get(id).base_data.memoize_hash_bits;
    }



    // "Remove forwarding"; replace ID of forwarded visibility record
    // with ID of base visibility record that the original record forwarded to.
    // Assumes the given ID is intended as an owning ID.
    // Return record data.
    template <VisRecordKind K>
    VisRecord& remove_forwarding(nodepool::id<VisRecordListNode<K>>* p_id)
    {
        const nodepool::id<VisRecordListNode<K>> old_id = *p_id;
        nodepool::id<VisRecordListNode<K>> id = old_id;
        CAMSPORK_REQUIRE(id, "null input to remove_forwarding");
        VisRecordListNode<K>* p_node = &get(id);
        CAMSPORK_REQUIRE_CMP(p_node->refcnt, !=, 0, "unexpected 0 refcnt");

        if (!p_node->is_forwarded()) {
            return p_node->base_data;  // No ID change
        }

        // Resolve the forwarding.
        do {
            id = p_node->camspork_next_id;
            CAMSPORK_REQUIRE(id, "node in forwarding state but null next_id");
            p_node = &get(id);
            CAMSPORK_REQUIRE_CMP(p_node->refcnt, !=, 0, "unexpected 0 refcnt");
        } while (p_node->is_forwarded());

        CAMSPORK_REQUIRE_CMP(id, !=, old_id, "forwarded to itself");
        incref(id);
        decref(old_id);  // Will take care of deallocating chain of forwarding if needed.
        CAMSPORK_REQUIRE_CMP(*p_id, ==, old_id, "unexpected modification of *p_id");
        *p_id = id;
        return p_node->base_data;
    }

    // Like remove_forwarding but non-destructive, i.e., don't actually replace the ID of a forwarding visibility
    // record with that of the forwarded-to base visibility record.
    template <VisRecordKind K>
    VisRecord const_resolve_forwarding(
            nodepool::id<VisRecordListNode<K>> id,
            nodepool::id<VisRecordListNode<K>>* p_out_id=nullptr) const
    {
        CAMSPORK_REQUIRE(id, "null input to const_resolve_forwarding");
        const VisRecordListNode<K>* p_node = &get(id);
        CAMSPORK_REQUIRE_CMP(p_node->refcnt, !=, 0, "unexpected 0 refcnt");

        while (p_node->is_forwarded()) {
            id = p_node->camspork_next_id;
            CAMSPORK_REQUIRE(id, "node in forwarding state but null next_id");
            p_node = &get(id);
            CAMSPORK_REQUIRE_CMP(p_node->refcnt, !=, 0, "unexpected 0 refcnt");
        }

        if (p_out_id) {
            *p_out_id = id;
        }
        return p_node->base_data;
    }


    // Add a new visibility record, or return existing memoized one, constructed from the given thread cuboid
    // + qual_bits_by_vis (in TlSigInterval format).
    // The returned ID is an owning reference (ownership count given by added_refcnt).
    template <VisRecordKind K, typename ThreadInit>
    [[nodiscard]] nodepool::id<VisRecordListNode<K>> memoize_new_vis_record(
            const ThreadInit& thread_init, SyncvAccessInfo access, uint32_t added_refcnt)
    {
        nodepool::id<VisRecordListNode<K>> new_vis_id;
        auto& new_vis = alloc_vis_record<K>(thread_init, access, &new_vis_id);
        if (!new_vis.base_data.pending_awaits) {
            CAMSPORK_REQUIRE_CMP(new_vis.refcnt, ==, 1, "expected 1 refcnt initially");
        }

        // Either insert into memoization, or forward to existing duplicate.
        // result_id gains added_refcnt-many references, while the originally created
        // VisRecord (which may be the same one, if not a duplicate) loses the 1 refcnt
        // that it was initially created with.
        //
        // In most cases, for a duplicate, the decref leads to the duplicate being deleted.
        // However, this is not the case if alloc_vis_record caused the VisRecord to gain
        // additional references (due to pending awaits).
        // It would be more efficient (but riskier) to defer adding those pending await
        // references until after we know this is not a duplicate, so we can free instantly.
        const auto result_id = memoize_or_forward(new_vis_id);
        incref(result_id, added_refcnt);
        decref(new_vis_id);
        return result_id;
    }

    template <VisRecordKind K>
    struct RemoveMemoizedCommand
    {
        nodepool::id<VisRecordListNode<K>> node_id;
        static constexpr MemoizeAction memoize_action = MemoizeAction::Remove;
    };

    // This removes the given node from the memoization table, but does not decrement the reference count or free it.
    // Recall that the memoization table does not own (reference count) the VisRecords contained.
    template <VisRecordKind K>
    [[nodiscard]] nodepool::id<VisRecordListNode<K>> remove_memoized(nodepool::id<VisRecordListNode<K>> id)
    {
        CAMSPORK_REQUIRE(id, "unexpected null");

        RemoveMemoizedCommand<K> command{id};
        const uint64_t hash = read_hash_helper(id);
        return for_vis_record_hash_bounds({hash, hash}, command);
    }

    template <VisRecordKind K>
    struct FindMemoizedCommand
    {
        nodepool::id<VisRecordListNode<K>> node_id;
        static constexpr MemoizeAction memoize_action = MemoizeAction::Find;
    };

    template <VisRecordKind K>
    nodepool::id<VisRecordListNode<K>> find_memoized(nodepool::id<VisRecordListNode<K>> id) const
    {
        CAMSPORK_REQUIRE(id, "unexpected null");

        FindMemoizedCommand<K> command{id};
        const uint64_t hash = read_hash_helper(id);
        return const_cast<SyncvTable*>(this)->for_vis_record_hash_bounds({hash, hash}, command);
    }

    template <VisRecordKind K>
    struct MemoizeOrForwardCommand
    {
        nodepool::id<VisRecordListNode<K>> node_id;
        static constexpr MemoizeAction memoize_action = MemoizeAction::MemoizeOrForward;
    };

    // Given an existing visibility record in the base state that's not in the memoization table, either
    //   * Add it to the memoization table, if it's unique. Return itself.
    //   * Put it in the forwarding state (discard existing state) and forward to equal already-memoized record.
    //     Return ID of memoized record.
    // This initializes VisRecord::memoize_hash_bits, as specified by the documentation for that data member.
    template <VisRecordKind K>
    nodepool::id<VisRecordListNode<K>> memoize_or_forward(nodepool::id<VisRecordListNode<K>> id)
    {
        CAMSPORK_REQUIRE(id, "unexpected null");
        VisRecordListNode<K>& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "should have nonzero refcnt");
        CAMSPORK_REQUIRE(!node.is_forwarded(), "should not already be forwarded");
        CAMSPORK_REQUIRE(!node.camspork_next_id, "shouldn't be in any linked list (forwarded?)");

        // Calculate hash here as promised in the comment for the memoize_hash_bits member variable.
        const auto hash = hash_vis_record(node.base_data);
        node.base_data.memoize_hash_bits = hash;  // No way to silence stupid bitfield truncation warning!

        const uint64_t real_hash = node.base_data.memoize_hash_bits;
        CAMSPORK_REQUIRE_CMP(hash, ==, real_hash, "hash bitfield truncation");

        MemoizeOrForwardCommand<K> command{id};
        return for_vis_record_hash_bounds({hash, hash}, command);
    }

    void memoize_modified()
    {
        for (nodepool::id<VisRecordListNode<VisRecordKind::Default>> id : modified_vis_records) {
            memoize_or_forward(id);
            decref(id);
        }
        modified_vis_records.clear();
    }

    struct AugmentVisRecordCallback
    {
        const ThreadCuboid* p_cuboid;
        qual_bits_t L2_full_qual_bits;
        qual_bits_t L2_temporal_qual_bits;

        template <VisRecordKind K>
        void operator() (SyncvTable& env, nodepool::id<VisRecordListNode<K>> vis_record_id)
        {
            auto& node = env.get(vis_record_id);
            CAMSPORK_REQUIRE(!node.is_forwarded(), "Unexpected modification of forwarding state VisRecord");
            p_cuboid->to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
                {
                    QualBitsByVis q_by_vis{};
                    const auto q_temporal = L2_temporal_qual_bits;
                    q_by_vis.array[vis_flag_index_atomic_only] = q_temporal;
                    q_by_vis.array[vis_flag_index_temporal] = q_temporal;
                    q_by_vis.array[vis_flag_index_full] = L2_full_qual_bits;
                    env.union_tl_sig_interval(&node.base_data, TlSigInterval{tid_lo, tid_hi, q_by_vis});
                }
            );
        }
    };

    template <typename Logger>
    struct FenceUpdateCommand
    {
        static constexpr MemoizeAction memoize_action = MemoizeAction::EditAll;

        const ThreadCuboid* p_cuboid;
        bool transitive;
        qual_bits_t L1_qual_bits, L2_full_qual_bits, L2_temporal_qual_bits;
        Logger& logger;

        static constexpr bool enable_debug_printf = false;

        template <VisRecordKind K>
        bool operator() (SyncvTable& env, nodepool::id<VisRecordListNode<K>> vis_record_id) const
        {
            auto& node = env.get(vis_record_id);
            CAMSPORK_REQUIRE(!node.is_forwarded(), "unexpected forwarding state");
            VisRecord* p_record = &node.base_data;

            AugmentVisRecordCallback augment{};
            augment.p_cuboid = p_cuboid;
            augment.L2_full_qual_bits = L2_full_qual_bits;
            augment.L2_temporal_qual_bits = L2_temporal_qual_bits;

            const bool syncs = env.synchronizes_with(transitive, *p_record, *p_cuboid, L1_qual_bits);
            if (syncs) {
                augment(env, vis_record_id);
            }
            return syncs;
        };
    };

    BarrierArriveState& get_barrier_arrive_state(pending_await_t info)
    {
        const auto barrier_index = pending_await_barrier_index(info);
        const auto arrive_count = pending_await_arrive_count(info);
        return barrier_states[barrier_index].arrive_states[arrive_count];
    }

    const BarrierArriveState* get_const_barrier_arrive_state(pending_await_t info) const
    {
        const auto barrier_index = pending_await_barrier_index(info);
        const auto arrive_count = pending_await_arrive_count(info);
        const auto& map = barrier_states[barrier_index].arrive_states;
        auto it = map.find(arrive_count);
        if (it == map.end()) {
            debug_print(stderr, info);
            return nullptr;
        }
        return &it->second;
    }

    template <typename Logger>
    struct ArriveUpdateCommand
    {
        static constexpr MemoizeAction memoize_action = MemoizeAction::EditAll;

        const ThreadCuboid* p_cuboid;
        bool transitive;
        qual_bits_t L1_qual_bits;
        std::vector<pending_await_t> pending_awaits;
        Logger& logger;

        static constexpr bool enable_debug_printf = false;

        ArriveUpdateCommand(ArriveUpdateCommand&&) = delete;

        template <VisRecordKind K>
        bool operator() (SyncvTable& env, nodepool::id<VisRecordListNode<K>> vis_record_id)
        {
            CAMSPORK_REQUIRE_CMP(L1_qual_bits, !=, 0, "should be if'd out in this case");
            auto& node = env.get(vis_record_id);
            CAMSPORK_REQUIRE(!node.is_forwarded(), "unexpected forwarding state");
            VisRecord* p_record = &node.base_data;

            const bool syncs = env.synchronizes_with(transitive, *p_record, *p_cuboid, L1_qual_bits);
            if (syncs) {
                // Extend pending_awaits list of the VisRecord.
                const nodepool::id<PendingAwaitNode> old_await_node = p_record->pending_awaits;
                nodepool::id<PendingAwaitNode> new_await_node = old_await_node;
                // Add nodes to the head of the list.
                for (auto iter = pending_awaits.rbegin(); iter != pending_awaits.rend(); ++iter) {
                    const auto tmp_id = new_await_node;
                    PendingAwaitNode& node = env.alloc_default_node(&new_await_node);
                    node.camspork_next_id = tmp_id;
                    node.await_id = *iter;
                    // fprintf(stderr, "ADDING await_id %u\n", node.await_id);
                }
                p_record->pending_awaits = new_await_node;

                // Extend BarrierArriveState to hold the new VisRecord.
                for (pending_await_t info : pending_awaits) {
                    env.extend_barrier_arrive_state(vis_record_id, info);
                }
            }
            return syncs;
        }
    };

    // Extend BarrierArriveState to hold the new VisRecord.
    template <VisRecordKind K>
    void extend_barrier_arrive_state(
        nodepool::id<VisRecordListNode<K>> vis_record_id,
        pending_await_t info)
    {
        BarrierArriveState& state = get_barrier_arrive_state(info);
        nodepool::id<AssignmentRecordVisNode<K>> list_node_id{};
        auto& list_node = alloc_default_node(&list_node_id);
        list_node.vis_record_id = vis_record_id;
        incref(vis_record_id);
        list_node.camspork_next_id = state.vis_records_head_id;
        state.vis_records_head_id = list_node_id;
    }

    template <VisRecordKind K, typename Callback, typename Logger>
    struct RetireBarrierArriveCommand
    {
        static constexpr MemoizeAction memoize_action = MemoizeAction::EditOne;
        nodepool::id<VisRecordListNode<K>> node_id;
        pending_await_t await_info;
        Callback&& callback;
        Logger&& logger;

        bool operator() (SyncvTable& env, nodepool::id<VisRecordListNode<K>> id)
        {
            env.retire_barrier_arrive_impl(id, await_info, callback, logger);
            return true;
        }
    };

    template <VisRecordKind K, typename Callback, typename Logger>
    void retire_barrier_arrive_impl(
            nodepool::id<VisRecordListNode<K>> vis_record_id,
            pending_await_t await_info,
            Callback&& callback,
            Logger&& logger)
    {
        callback(*this, vis_record_id);

        // Remove PendingAwaitNode.
        nodepool::id<PendingAwaitNode>* p_await_node = &get(vis_record_id).base_data.pending_awaits;
        bool found = false;
        while (*p_await_node) {
            PendingAwaitNode& node = get(*p_await_node);
            if (node.await_id == await_info) {
                remove_and_free_next_node(p_await_node);
                if (false) {
                    fprintf(stderr, "%u, remove ", vis_record_id.id_bits);
                    debug_print(stderr, await_info);
                }
                found = true;
                break;
            }
            else {
                p_await_node = &node.camspork_next_id;
            }
        }
        CAMSPORK_REQUIRE(found, "Remove PendingAwaitNode failed");
    }

    // Find all VisRecords referenced by the BarrierArriveState and remove corresponding pending awaits,
    // then clear the BarrierArriveState.
    // We run the supplied callback (likely AugmentVisRecordCallback) to modify each base-state VisRecord.
    template <typename Callback, typename Logger>
    void retire_barrier_arrive(
            BarrierArriveState* p_state, pending_await_t await_info, Callback&& callback, Logger&& logger)
    {
        auto retire_list = [&] (auto record_id)
        {
            constexpr VisRecordKind K = decltype(record_id)::value_type::vis_record_kind;
            while (record_id) {
                AssignmentRecordVisNode<K>& record_node = get(record_id);
                record_id = record_node.camspork_next_id;
                const nodepool::id<VisRecordListNode<K>> vis_record_id = record_node.vis_record_id;
                VisRecordListNode<K>& vis_node = get(vis_record_id);
                nodepool::id<PendingAwaitNode>* p_await_node = &vis_node.base_data.pending_awaits;
                if (vis_node.is_forwarded()) {
                    CAMSPORK_REQUIRE(!*p_await_node, "pending_awaits should be empty for forwarded VisRecord");
                }
                else {
                    const uint64_t hash = read_hash_helper(vis_record_id);
                    RetireBarrierArriveCommand<K, Callback&, Logger&> command{
                            vis_record_id,
                            await_info,
                            callback,
                            logger};
                    const auto modified_id = for_vis_record_hash_bounds({hash, hash}, command);
                    CAMSPORK_REQUIRE_CMP(modified_id, ==, vis_record_id, "Internal error, EditOne failed");
                    memoize_modified();
                }
                decref(vis_record_id);
                record_node.vis_record_id = {};  // to make things clearer in the debugger.
            }
        };

        nodepool::id<DefaultAssignmentRecordVisNode>& head_id = p_state->vis_records_head_id;
        retire_list(head_id);
        extend_free_list(head_id);
        head_id = {};
    }

    // Big payoff for all this code: function that performs the effects of a synchronization statement with the given
    // sync type and given first/second visibility sets. This affects all visibility records whose visibility set
    // intersects with the first visibility set of the synchronization statement.
    //
    // The real entrypoints are the ones specialized for fence, arrive, join threads.
    // Await works differently, using another code path.
    template <typename Command>
    void update_vis_records_for_sync_impl(Command&& command)
    {
        auto per_qual_tl = [&] (uint8_t qual_tl)
        {
            const auto tid_lo = command.p_cuboid->minimal_superset_interval().tid_lo;
            std::pair<uint64_t, uint64_t> hash_bounds = hash_bounds_for_arrive(qual_tl, tid_lo);
            for_vis_record_hash_bounds(hash_bounds, command);
        };
        if (command.transitive) {
            for (uint8_t qual_tl = 0; qual_tl <= num_qual_tl; ++qual_tl) {
                // qual_tl = num_qual_tl is a special case.
                per_qual_tl(qual_tl);
            }
        }
        else {
            uint32_t qual_bits = command.L1_qual_bits;
            while (qual_bits) {
                uint8_t qual_tl = pop_low_bit_index(&qual_bits);
                per_qual_tl(qual_tl);
            }
        }
    }

    // Augment all visibility records that synchronize with the first visibility set of the fence.
    template <typename Logger>
    void update_vis_records_for_fence(const ThreadCuboid& cuboid, const SyncvFence& fence, Logger& logger)
    {
        FenceUpdateCommand<Logger> command{
                &cuboid, fence.transitive, fence.L1_qual_bits,
                fence.L2_full_qual_bits, fence.L2_temporal_qual_bits, logger};
        update_vis_records_for_sync_impl(command);
    }

    // Save await_id into all visibility records that synchronize with the first visibility set of the arrive.
    template <typename Logger>
    void update_vis_records_for_arrive(
            const ThreadCuboid& cuboid,
            bool transitive,
            qual_bits_t L1_qual_bits,
            std::vector<pending_await_t> pending_awaits,
            Logger& logger)
    {
        if (L1_qual_bits == 0) {
            // Do nothing.
        }
        else {
            ArriveUpdateCommand<Logger> command{
                    &cuboid, transitive, L1_qual_bits, std::move(pending_awaits), logger};
            update_vis_records_for_sync_impl(command);
        }
    }

    template <typename Logger>
    void update_vis_records_for_await(
        uint32_t barrier_index,
        int32_t max_arrive_count,
        const ThreadCuboid& cuboid,
        qual_bits_t L2_full_qual_bits,
        qual_bits_t L2_temporal_qual_bits,
        Logger& logger)
    {
        AugmentVisRecordCallback augment{};
        augment.p_cuboid = &cuboid;
        augment.L2_full_qual_bits = L2_full_qual_bits;
        augment.L2_temporal_qual_bits = L2_temporal_qual_bits;

        // Retire all BarrierArriveState with arrive_count <= max_arrive_count.
        CAMSPORK_C_BOUNDSCHECK(barrier_index, max_live_barriers);
        BinaryTree<int32_t, BarrierArriveState>& arrive_map = barrier_states[barrier_index].arrive_states;
        for (auto iter = arrive_map.begin(); iter != arrive_map.end(); ) {
            auto& pair = *iter;
            const auto arrive_count = pair.first;
            if (int32_t(arrive_count) > max_arrive_count) {
                break;
            }
            retire_barrier_arrive(&pair.second, pack_pending_await(barrier_index, arrive_count), augment, logger);
            arrive_map.erase(iter++);
        }
    }



    // *** Synchronization State Update ***



    template <typename Logger>
    void on_fence(const ThreadCuboid& cuboid, const SyncvFence& fence, Logger&& logger)
    {
        augment_counter++;
        logger.history_set_sync_stmt_info(fence);
        update_vis_records_for_fence(cuboid, fence, logger);
        memoize_modified();
    }

    template <typename Logger>
    void on_arrive(const ThreadCuboid& cuboid, const SyncvArrive& arrive, Logger&& logger)
    {
        // NB augment_counter not changed, as Arrive does not augment any VisRecords.
        const auto home_barrier_index = get_barrier_index(arrive.home_barrier);
        BarrierState& state = barrier_states[home_barrier_index];

        const auto new_arrive_count = state.arrive_count + 1;
        logger.history_set_sync_stmt_info(arrive, state, new_arrive_count);

        const auto count = arrive.barrier_count;
        std::vector<pending_await_t> pending_awaits(count);
        for (uint32_t i = 0; i < count; ++i) {
            pending_awaits[i] = pack_pending_await(get_barrier_index(arrive.all_barriers[i]), state.arrive_count);
        }

        state.arrive_count = new_arrive_count;
        update_vis_records_for_arrive(
                cuboid, arrive.transitive, arrive.L1_qual_bits, std::move(pending_awaits), logger);
        memoize_modified();
    }

    template <typename Logger>
    void on_await(const ThreadCuboid& cuboid, const SyncvAwait& await, Logger&& logger)
    {
        // fprintf(stderr, "\x1b[31mon_await\n\x1b[0m");
        augment_counter++;

        const auto barrier_index = get_barrier_index(await.bar);
        BarrierState& state = barrier_states[barrier_index];

        auto new_await_count = state.await_count;
        const auto N = await.N;
        int32_t max_arrive_count;
        if (N >= 0) {
            // Arrive-indexed barrier.
            max_arrive_count = state.arrive_count - N - 1;
            new_await_count = std::max(max_arrive_count + 1, state.await_count);
        }
        else {
            // Await-indexed barrier.
            int32_t lag = ~N;
            max_arrive_count = state.await_count - lag;
            new_await_count = state.await_count + 1;
        }

        logger.history_set_sync_stmt_info(await, state, new_await_count, max_arrive_count);
        state.await_count = new_await_count;

        update_vis_records_for_await(
                barrier_index,
                max_arrive_count,
                cuboid,
                await.L2_full_qual_bits,
                await.L2_temporal_qual_bits,
                logger);
        // memoize_modified inside retire_barrier_arrive.
    }

    template <typename Logger>
    void on_join_threads(const ThreadCuboid& cuboid, Logger&& logger)
    {
        augment_counter++;
        // TODO
        memoize_modified();
    }



    // *** Access Safety Checking (read/write safety) ***



    static uint32_t assignment_record_window_size(AssignmentRecordWindow window)
    {
        uint32_t prod = 1;
        for (const uint32_t* p = window.begin_inner_extent; p != window.end_inner_extent; ++p) {
            prod *= *p;
        }
        return prod;
    }

    static uint32_t assignment_record_window_size(assignment_record_id*)
    {
        return 1;
    }

    // Copy one of the linked lists of VisRecord references in an AssignmentRecord.
    template <VisRecordKind K>
    nodepool::id<AssignmentRecordVisNode<K>> copy(nodepool::id<AssignmentRecordVisNode<K>> input_id)
    {
        nodepool::id<AssignmentRecordVisNode<K>> output_id{};
        if (input_id) {
            nodepool::id<AssignmentRecordVisNode<K>>* p_tail = &output_id;
            while (input_id) {
                const AssignmentRecordVisNode<K>& input_node = get(input_id);
                AssignmentRecordVisNode<K>& output_node = alloc_default_node(p_tail);
                p_tail = &output_node.camspork_next_id;
                input_id = input_node.camspork_next_id;

                nodepool::id<VisRecordListNode<K>> vis_record_id = input_node.vis_record_id;
                output_node.vis_record_id = vis_record_id;
                incref(vis_record_id);
            }
        }
        return output_id;
    }

    AssignmentRecord& copy(
            const AssignmentRecord& old, uint32_t refcnt, nodepool::id<AssignmentRecord>* out_id)
    {
        CAMSPORK_REQUIRE_CMP(refcnt, !=, 0, "initial refcnt must not be 0");
        AssignmentRecord& assignment_record = alloc_default_node(out_id);
        assignment_record.refcnt = refcnt;
        assignment_record.lazy_last_augment_counter_bits = old.lazy_last_augment_counter_bits;
        assignment_record.mutate_vis_records_head_id = copy(old.mutate_vis_records_head_id);
        assignment_record.read_vis_records_head_id = copy(old.read_vis_records_head_id);
        return assignment_record;
    }

    template <bool IsMutate, bool SharedVisRecord, bool UpdateRecords, typename Input, typename Logger>
    void checked_on_access_impl(
            bool is_convergent,
            Input input,
            const ThreadCuboid& cuboid,
            SyncvAccessInfo access,
            Logger& logger)
    {
        using node_id = nodepool::id<AssignmentRecord>;
        static constexpr VisRecordKind K = VisRecordKind::Default;

        // If the input is a window, take a census of all assignment record IDs in the input window.
        static constexpr bool IsWindow = std::is_same_v<decltype(input), AssignmentRecordWindow>;
        using TrivialCensus = std::array<std::pair<node_id, AssignmentRecordCensusEntry>, 1>;
        std::conditional_t<IsWindow, CensusMap, TrivialCensus> census;

        if constexpr (IsWindow) {
            cuboid_to_intervals<size_t>(
                input.begin_outer_extent, input.end_outer_extent,
                input.begin_offset, input.end_offset,
                input.begin_inner_extent, input.end_inner_extent,
                [&] (size_t lo, size_t hi) {
                    for (size_t i = lo; i < hi; ++i) {
                        node_id id{input.base[i].node_id};
                        if (0 == census[id].count++) {
                            census[id].linear_index_in_input = i;
                        }
                    }
                }
            );
        }
        else {
            census[0].first = node_id{input->node_id};  // Where decltype(input) is assignment_record_id*
            census[0].second.count = 1;
        }

        // We will memoize the new visibility record(s) once.
        // 0 new records if !UpdateRecords
        // 1 new record if SharedVisRecord
        // any # new records if !SharedVisRecord
        using VisRecordID = nodepool::id<VisRecordListNode<K>>;
        using VisRecordList = std::conditional_t<
            !UpdateRecords, std::array<VisRecordID, 0>,
            std::conditional_t<SharedVisRecord, std::array<VisRecordID, 1>, std::vector<VisRecordID>>>;
        VisRecordList new_vis_record_list{};
        const uint32_t vis_record_refcnt = uint32_t(census.size());

        if constexpr (!UpdateRecords) {
        }
        else if (census.empty()) {
        }
        else if constexpr (SharedVisRecord) {
            const VisRecordID new_vis_record_id = memoize_new_vis_record<K>(cuboid, access, vis_record_refcnt);
            logger.history_new_vis_record(*this, new_vis_record_id);
            new_vis_record_list[0] = new_vis_record_id;
        }
        else {
            cuboid.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi) {
                // We model the CPU as of 2025-10-01 as "[almost] all possible threads" [0, UINT32_MAX)
                // and if we pass that here, we will create 4 billion VisRecords.
                CAMSPORK_REQUIRE_CMP(tid_hi, <, UINT32_MAX, "Likely you meant to pass convergent_access_flag");
                for (uint32_t tid = tid_lo; tid < tid_hi; ++tid) {
                    const VisRecordID new_vis_record_id = memoize_new_vis_record<K>(
                        SingleThreadInit{tid},
                        access,
                        vis_record_refcnt
                    );
                    logger.history_new_vis_record(*this, new_vis_record_id);
                    new_vis_record_list.push_back(new_vis_record_id);
                }
            });
        }

        logger.excut_log_assignment_records(*this, input, new_vis_record_list, IsMutate);

        auto check = [&] (node_id id, size_t linear_index)
        {
            if (!id) {
                return;
            }

            // If the mutate is a reduction (read+write), we require vis_flag_full.
            const AssignmentRecord& assignment_record = get(id);
            const auto mut_vis_flag_needed = (
                access.atomic_qual_bits != 0 ? vis_flag_atomic_only :
                access.is_write_only ? vis_flag_temporal : vis_flag_full
            );

            // Check against previous mutate visibility records
            TlSig fail_tl_sig{};
            nodepool::id<DefaultAssignmentRecordVisNode> mutate_id = assignment_record.mutate_vis_records_head_id;
            while (mutate_id) {
                DefaultAssignmentRecordVisNode& node = get(mutate_id);
                const VisRecord& mutate_record = remove_forwarding(&node.vis_record_id);
                logger.history_vis_record_checked(node.vis_record_id, IsMutate);  // Logs memoized (base state) ID.
                const bool visible = visible_to(
                        mutate_record, cuboid, access.extended_qual_bits, mut_vis_flag_needed,
                        is_convergent, &fail_tl_sig);
                if (!visible) {
                    logger.history_vis_record_error(node.vis_record_id, fail_tl_sig);
                    throw SyncvCheckFail{IsMutate ? "WAW HAZARD" : "RAW HAZARD", linear_index};
                }
                mutate_id = node.camspork_next_id;
            }

            // If the access is a mutate, also check against the list of previous read visibility records.
            if constexpr (IsMutate) {
                nodepool::id<DefaultAssignmentRecordVisNode> read_id = assignment_record.read_vis_records_head_id;
                while (read_id) {
                    DefaultAssignmentRecordVisNode& node = get(read_id);
                    const VisRecord& read_record = remove_forwarding(&node.vis_record_id);
                    logger.history_vis_record_checked(node.vis_record_id, IsMutate);  // Logs memoized (base state) ID.
                    const bool visible = visible_to(
                            read_record, cuboid, access.extended_qual_bits, vis_flag_temporal,
                            is_convergent, &fail_tl_sig);
                    if (!visible) {
                        logger.history_vis_record_error(node.vis_record_id, fail_tl_sig);
                        throw SyncvCheckFail{"WAR HAZARD", linear_index};
                    }
                    read_id = node.camspork_next_id;
                }
            }
        };

        auto extend_vis_records = [&] (nodepool::id<AssignmentRecordVisNode<K>>* p_list_head)
        {
            for (const VisRecordID vis_record_id : new_vis_record_list) {
                nodepool::id<DefaultAssignmentRecordVisNode> new_node_id;
                DefaultAssignmentRecordVisNode& node = alloc_default_node(&new_node_id);
                node.vis_record_id = vis_record_id;
                node.camspork_next_id = *p_list_head;
                *p_list_head = new_node_id;
            }
        };

        auto copy_on_write_update = [&] (
                node_id old_id,
                AssignmentRecordCensusEntry& entry)
        {
            // If the old assignment record has ID 0 (doesn't exist ... was presumed empty, no reads, no mutates)
            // or its refcnt exceeds the use count, then we cannot modify the assignment record in-place.
            // Save the ID of the replacement assignment record (new_id), which is the same as the old_id if in-place.
            bool can_modify = false;
            AssignmentRecord* p_assignment_record;
            if (old_id) {
                AssignmentRecord& old = get(old_id);
                const auto old_refcnt = old.refcnt;
                CAMSPORK_REQUIRE_CMP(old_refcnt, >=, entry.count, "corrupt refcnt");
                can_modify = old_refcnt == entry.count;
                if (can_modify) {
                    p_assignment_record = &old;
                    entry.new_node_id = old_id;
                }
                else {
                    // Make a copy of the old AssignmentRecord, and transfer entry.count owning references
                    // from the old to the new one.
                    p_assignment_record = &copy(old, entry.count, &entry.new_node_id);
                    decref(old_id, entry.count);
                }
            }
            else {
                p_assignment_record = &alloc_default_node(&entry.new_node_id);
                p_assignment_record->refcnt = entry.count;
            }
            AssignmentRecord& assignment_record = *p_assignment_record;

            // Add new visibility record (either as new mutate visibility record, or appended read visibility record).
            if constexpr (IsMutate) {
                // Clear out read visibility records upon write.
                // If not atomic, clear mutate visibility records too.
                // Add the new mutate visibility records.
                assignment_record_remove_vis_records(&assignment_record.read_vis_records_head_id);
                if (access.atomic_qual_bits != 0) {
                    extend_vis_records(&assignment_record.mutate_vis_records_head_id);
                    lazy_remove_duplicates(&assignment_record);  // << IMPORTANT for performance
                }
                else {
                    assignment_record_remove_vis_records(&assignment_record.mutate_vis_records_head_id);
                    extend_vis_records(&assignment_record.mutate_vis_records_head_id);
                    assignment_record.lazy_last_augment_counter_bits = get_augment_counter_bits();
                }
            }
            else {
                // Add the new visibility records to the list of read visibility records.
                extend_vis_records(&assignment_record.read_vis_records_head_id);
                lazy_remove_duplicates(&assignment_record);  // << IMPORTANT for performance
            }
        };

        // Check & update all distinct assignment records once.
        for (auto& pair : census) {
            check(pair.first, pair.second.linear_index_in_input);
            if constexpr (UpdateRecords) {
                copy_on_write_update(pair.first, pair.second);
            }
        }

        // Write out new assignment record IDs. Reference counting is already taken care of.
        if constexpr (!UpdateRecords) {
            CAMSPORK_REQUIRE_CMP(new_vis_record_list.size(), ==, 0,
                    "Fix !UpdateRecords code path to not leak vis_record_id");
        }
        else if constexpr (IsWindow) {
            cuboid_to_intervals<size_t>(
                input.begin_outer_extent, input.end_outer_extent,
                input.begin_offset, input.end_offset,
                input.begin_inner_extent, input.end_inner_extent,
                [&] (size_t lo, size_t hi) {
                    for (size_t i = lo; i < hi; ++i) {
                        const node_id id{input.base[i].node_id};
                        const auto iter = census.find(id);
                        CAMSPORK_REQUIRE(iter != census.end(), "fix census code");
                        input.base[i].node_id = iter->second.new_node_id.id_bits;
                    }
                }
            );
            logger.excut_update_assignment_record_ids(census);
        }
        else {
            const nodepool::id<AssignmentRecord> new_id = census[0].second.new_node_id;
            input->node_id = new_id.id_bits;  // Where decltype(input) is assignment_record_id*
            logger.excut_update_assignment_record_ids(new_id);
        }
    }

    // Expect Input = assignment_record_id* or AssignmentRecordWindow.
    template <typename Input, typename Logger>
    void on_r(Input input, const ThreadCuboid& cuboid, SyncvAccessInfo access, Logger&& logger)
    {
        if (no_checking_counter != 0) {
        }
        else if (access.is_convergent || access.force_shared_vis_record) {
            checked_on_access_impl<false, true, true>(access.is_convergent, input, cuboid, access, logger);
        }
        else {
            checked_on_access_impl<false, false, true>(access.is_convergent, input, cuboid, access, logger);
        }
    }

    // Expect Input = assignment_record_id* or AssignmentRecordWindow.
    template <typename Input, typename Logger>
    void on_rw(Input input, const ThreadCuboid& cuboid, SyncvAccessInfo access, Logger&& logger)
    {
        if (no_checking_counter != 0) {
        }
        else if (access.is_convergent || access.force_shared_vis_record) {
            checked_on_access_impl<true, true, true>(access.is_convergent, input, cuboid, access, logger);
        }
        else {
            checked_on_access_impl<true, false, true>(access.is_convergent, input, cuboid, access, logger);
        }
    }

    // Expect Input = assignment_record_id* or AssignmentRecordWindow.
    template <typename Input, typename Logger>
    void on_check_free(Input input, const ThreadCuboid& cuboid, SyncvAccessInfo access, Logger&& logger)
    {
        if (no_checking_counter != 0) {
        }
        else {
            // SharedVisRecord doesn't matter when UpdateRecords=false.
            checked_on_access_impl<true, true, false>(access.is_convergent, input, cuboid, access, logger);
        }
    }

    void clear_visibility(size_t N, assignment_record_id* p_assignment_record_ids)
    {
        for (size_t i = 0; i < N; ++i) {
            nodepool::id<AssignmentRecord> id{p_assignment_record_ids[i].node_id};
            if (id) {
                decref(id);
                p_assignment_record_ids[i].node_id = 0;
            }
        }
    }

    // Resolve forwarding and remove duplicate visibility records.
    // Removing forwarding causes two equivalent visibility records to have identical IDs
    // (both referring to the shared entry in the memoization table).
    template <VisRecordKind K>
    void remove_duplicates(nodepool::id<AssignmentRecordVisNode<K>>* p_list_head)
    {
        using node_id = nodepool::id<AssignmentRecordVisNode<K>>;

        // Remove forwarding (unique ID iff unique record), and clear tmp_is_duplicate to 0.
        // Importantly, we are clearing this flag for base-state VisRecord, not forwarded ones.
        for (node_id id = *p_list_head; id; ) {
            AssignmentRecordVisNode<K>& node = get(id);
            remove_forwarding(&node.vis_record_id).tmp_is_duplicate = 0;
            id = node.camspork_next_id;
        }

        // Remove duplicates, using tmp_is_duplicate to recognize duplicates.
        node_id* p_read_id = p_list_head;
        while (node_id next_id = *p_read_id) {
            AssignmentRecordVisNode<K>& next_node = get(next_id);
            VisRecordListNode<K>& vis_record_node = get(next_node.vis_record_id);
            CAMSPORK_REQUIRE(!vis_record_node.is_forwarded(), "should have resolved forwarding above");

            if (vis_record_node.base_data.tmp_is_duplicate) {
                // Duplicate, remove next node from list (decrements refcount for duplicated vis record).
                // This causes (next_id = *p_read_id) to change, so we don't have to update p_read_id.
                // i.e. since we removed the next node, we're ready to process a new next node next iteration.
                node_id victim_id = remove_next_node(p_read_id);
                CAMSPORK_REQUIRE_CMP(victim_id, ==, next_id, "didn't remove expected node");
                CAMSPORK_REQUIRE(!next_node.camspork_next_id, "next_node should have been removed above");
                decref(next_node.vis_record_id);
                extend_free_list(victim_id);
            }
            else {
                // If next node survives, remember the visibility set ID and move on.
                vis_record_node.base_data.tmp_is_duplicate = 1;
                p_read_id = &get(next_id).camspork_next_id;
            }
        }
    }

    void lazy_remove_duplicates(AssignmentRecord* p_assignment_record)
    {
        // If we leave things as-is, read vis records may build up indefinitely for variables that are written
        // once and read many times. We fix this by removing duplicates; however, this is really expensive,
        // so we only do it once after each fence or await event (synchronization is when memoization kicks
        // in to potentially allow us to recognize duplicates due to duplicated IDs).
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wconversion"
        const auto old_bits = p_assignment_record->lazy_last_augment_counter_bits;
        p_assignment_record->lazy_last_augment_counter_bits = get_augment_counter_bits();
        #pragma GCC diagnostic pop

        if (old_bits != p_assignment_record->lazy_last_augment_counter_bits) {
            // This could fail if the bits of augment_counter overflow exactly.
            // However, this is unlikely, and is only a performance issue if so (we fail to remove duplicates).
            remove_duplicates(&p_assignment_record->mutate_vis_records_head_id);
            remove_duplicates(&p_assignment_record->read_vis_records_head_id);
        }

    }



    // *** Debugging / Testing ***



    // Get IDs of read visibility records of assignment record.
    void debug_get_read_vis_record_ids(const AssignmentRecord& record, std::vector<uint32_t>* out) const
    {
        out->clear();
        nodepool::id<DefaultAssignmentRecordVisNode> id = record.read_vis_records_head_id;
        while (id) {
            const DefaultAssignmentRecordVisNode& node = get(id);
            out->push_back(node.vis_record_id.id_bits);
            id = node.camspork_next_id;
        }
    }

    // Get info for a given visibility record.
    template <VisRecordKind K>
    void debug_get_vis_record_data(nodepool::id<VisRecordListNode<K>> node_id, VisRecordDebugData* out) const
    {
        CAMSPORK_REQUIRE(node_id, "cannot read null VisRecord");
        const VisRecord record = const_resolve_forwarding(node_id);

        out->visibility_set.clear();
        for (nodepool::id<TlSigIntervalListNode> node_id = record.visibility_set; node_id;) {
            const TlSigIntervalListNode& node = get(node_id);
            out->visibility_set.push_back(node.data);
            node_id = node.camspork_next_id;
        }

        out->pending_await_list.clear();
        for (nodepool::id<PendingAwaitNode> node_id = record.pending_awaits; node_id; ) {
            const PendingAwaitNode& node = get(node_id);
            out->pending_await_list.push_back(node.await_id);
            node_id = node.camspork_next_id;
        }
    }

    template <typename ListNode>
    struct RefcntDebug
    {
        std::vector<refcnt_t> refcnts;
        Set<nodepool::id<ListNode>> free_node_ids;

        RefcntDebug(const SyncvTable& self)
          : refcnts(self.debug_node_pool_size<ListNode>())
          , free_node_ids(self.debug_free_node_ids<ListNode>())
        {
        }

        void check_refcnts(const SyncvTable& self)
        {
            for (nodepool::id<ListNode> id : self.debug_get_pool<ListNode>()) {
                const refcnt_t stored_refcnt = self.get(id).get_refcnt();
                const refcnt_t true_refcnt = refcnts[id.node_index()];
                const bool is_free = free_node_ids.count(id);
                if (is_free) {
                    CAMSPORK_REQUIRE_CMP(true_refcnt, ==, 0, "Reference exists to free node");
                }
                else {
                    CAMSPORK_REQUIRE_CMP(true_refcnt, !=, 0, "Unreachable node");
                    if (true_refcnt != stored_refcnt) {
                        fprintf(stderr, "%u, %s\n", id.id_bits, typeid(ListNode).name());
                    }
                    CAMSPORK_REQUIRE_CMP(true_refcnt, ==, stored_refcnt, "wrong refcnt");
                }
            }
        }
    };

    // Massive function that verifies that the current state is legal.
    // This only works if all of the user's arrays of assignment_record_id have been passed.
    // NB this is a const member function to help guard against accidental subtle changes in the course of checking
    // which could cause heisenbugs.
    void debug_validate_state(size_t input_count, const SyncvDebugValidateInput* p_inputs) const
    {
        fprintf(stderr, "SyncvTable::debug_validate_state\n");

        CAMSPORK_REQUIRE_CMP(modified_vis_records.size(), ==, 0, "internal error, missing memoize_modified()");

        std::tuple<
            RefcntDebug<AssignmentRecord>,
            RefcntDebug<TlSigIntervalListNode>,
            RefcntDebug<PendingAwaitNode>,
            RefcntDebug<DefaultVisRecordListNode>,
            RefcntDebug<DefaultAssignmentRecordVisNode>>
        debug_refcnts(
            *this, *this, *this, *this, *this
        );

        if (false) {
            fprintf(stderr, "AssignmentRecord: %u\n", debug_get_pool<AssignmentRecord>().size());
            fprintf(stderr, "TlSigIntervalListNode: %u\n", debug_get_pool<TlSigIntervalListNode>().size());
            fprintf(stderr, "PendingAwaitNode: %u\n", debug_get_pool<PendingAwaitNode>().size());
            fprintf(stderr, "DefaultVisRecordListNode: %u\n", debug_get_pool<DefaultVisRecordListNode>().size());
            fprintf(stderr, "DefaultAssignmentRecordVisNode: %u\n", debug_get_pool<DefaultAssignmentRecordVisNode>().size());
        }

        auto check_all_refcnts = [&]
        {
            std::get<0>(debug_refcnts).check_refcnts(*this);
            std::get<1>(debug_refcnts).check_refcnts(*this);
            std::get<2>(debug_refcnts).check_refcnts(*this);
            std::get<3>(debug_refcnts).check_refcnts(*this);
            std::get<4>(debug_refcnts).check_refcnts(*this);
        };

        auto record_owning = [&] (auto id) -> bool  // First time flag
        {
            std::vector<refcnt_t>& refcnts =
                    std::get<RefcntDebug<typename decltype(id)::value_type>>(debug_refcnts).refcnts;
            if (id) {
                CAMSPORK_REQUIRE_CMP(id.node_index(), <, refcnts.size(), "out-of-bounds node ID");
                auto refcnt_before = refcnts.at(id.node_index())++;
                return refcnt_before == 0;
            }
            return false;
        };

        auto process_assignment_record_list = [&] (auto id)
        {
            while (id) {
                auto& node = get(id);
                CAMSPORK_REQUIRE(node.vis_record_id, "unexpected null");
                record_owning(id);
                record_owning(node.vis_record_id);
                id = node.camspork_next_id;
            }
        };

        auto process_assignment_record = [&] (nodepool::id<AssignmentRecord> id, auto recurse)
        {
            const bool first_time = record_owning(id);
            if (!first_time) {
                return;
            }
            const AssignmentRecord& record = get(id);

            const nodepool::id<DefaultAssignmentRecordVisNode> mutate_id = record.mutate_vis_records_head_id;
            process_assignment_record_list(mutate_id);
            const nodepool::id<DefaultAssignmentRecordVisNode> read_id = record.read_vis_records_head_id;
            process_assignment_record_list(read_id);

            recurse(record.camspork_next_id, recurse);
        };

        // Count ownership references of AssignmentRecord.
        // Further count ownership references from AssignmentRecord to VisRecordListNode, AssignmentRecordVisNode
        for (size_t input_i = 0; input_i < input_count; ++input_i) {
            const assignment_record_id* ptr = p_inputs[input_i].p_records;
            size_t sz = p_inputs[input_i].size;

            for (size_t i = 0; i < sz; ++i) {
                nodepool::id<AssignmentRecord> id{ptr[i].node_id};
                process_assignment_record(id, process_assignment_record);
            }
        }

        // Also count references due to BarrierArriveState
        for (uint32_t barrier_index = 0; barrier_index < max_live_barriers; ++barrier_index) {
            const BarrierState& state = barrier_states[barrier_index];
            for (const auto& pair : state.arrive_states) {
                const nodepool::id<DefaultAssignmentRecordVisNode> head_id = pair.second.vis_records_head_id;
                process_assignment_record_list(head_id);
            }
        }

        // This handles references between PendingAwaitNode
        auto on_PendingAwaitNode = [&] (nodepool::id<PendingAwaitNode> id, auto recurse)
        {
            if (!id) {
                return;
            }
            if (!record_owning(id)) {
                // Not the first time, don't re-scan.
                return;
            }
            id = get(id).camspork_next_id;
            recurse(id, recurse);
        };

        // Count ownership references from live VisRecordListNode objects to other objects:
        //   * TlSigIntervalListNode
        //   * PendingAwaitNode
        //   * forwarded-to VisRecordListNodes
        // Furthermore we validate the following:
        //   * encoding for the visibility set is correct.
        //   * VisRecords are properly stored in BarrierArriveState.
        auto process_vis_record_impl = [&] (auto id, const auto& free_vis_ids)
        {
            if (free_vis_ids.count(id)) {
                return;  // Exit lambda: ignore non-allocated VisRecordListNode.
            }
            const auto& node = get(id);

            if (node.is_forwarded()) {
                CAMSPORK_REQUIRE(node.camspork_next_id, "in forwarding state, but forwarded-to node is null");
                record_owning(node.camspork_next_id);
                CAMSPORK_REQUIRE(!node.base_data.visibility_set, "state should have been cleared upon forwarding");
                CAMSPORK_REQUIRE(!node.base_data.pending_awaits, "state should have been cleared upon forwarding");
            }
            else {
                for (nodepool::id<TlSigIntervalListNode> node_id = node.base_data.visibility_set; node_id; ) {
                    record_owning(node_id);
                    TlSigIntervalListNode this_node = get(node_id);
                    this_node.data.assert_valid();

                    auto next_id = this_node.camspork_next_id;
                    if (next_id) {
                        TlSigIntervalListNode next_node = get(next_id);
                        CAMSPORK_REQUIRE(valid_adjacent(this_node.data, next_node.data), "visibility set intervals not sorted properly");
                        node_id = next_id;
                    }
                    else {
                        break;
                    }
                }

                // Record PendingAwaitNode references.
                on_PendingAwaitNode(node.base_data.pending_awaits, on_PendingAwaitNode);

                // Look for VisRecord reference in BarrierArriveState.
                // The other half of this checking is done in check_BarrierArriveState_VisRecords.
                MultiSet<decltype(PendingAwaitNode::await_id)> await_id_multiset;
                nodepool::id<PendingAwaitNode> await_node_id = node.base_data.pending_awaits;
                while (await_node_id) {
                    const PendingAwaitNode& await_node = get(await_node_id);
                    await_node_id = await_node.camspork_next_id;
                    await_id_multiset.insert(await_node.await_id);
                }
                for (auto await_id : await_id_multiset) {
                    const BarrierArriveState* p_state = get_const_barrier_arrive_state(await_id);
                    if (!p_state) {
                        fprintf(stderr, "%u\n", id.id_bits);
                    }
                    CAMSPORK_REQUIRE(p_state, "Missing BarrierArriveState");
                    const BarrierArriveState& state = *p_state;
                    constexpr VisRecordKind K = node.vis_record_kind;
                    nodepool::id<AssignmentRecordVisNode<K>> record_node_id = state.vis_records_head_id;
                    size_t count = 0;
                    while (record_node_id) {
                        const auto& record_node = get(record_node_id);
                        count += (record_node.vis_record_id == id);
                        record_node_id = record_node.camspork_next_id;
                    }
                    CAMSPORK_REQUIRE_CMP(count, ==, await_id_multiset.count(await_id),
                        "Expect 1:1 VisRecord->PendingAwait and PendingAwait->VisRecord references");
                }
            }
        };

        auto process_all_vis_records = [&] (auto id_for_typing)
        {
            using ListNode = typename decltype(id_for_typing)::value_type;
            RefcntDebug<ListNode>& debug_info = std::get<RefcntDebug<ListNode>>(debug_refcnts);
            for (nodepool::id<ListNode> id : debug_get_pool<ListNode>()) {
                process_vis_record_impl(id, debug_info.free_node_ids);
            }
        };

        process_all_vis_records(nodepool::id<DefaultVisRecordListNode>{});

        // Check that reference counts are correct.
        // For node types without refcnt, the refcnt should just be 0 or 1 (unique ownership).
        check_all_refcnts();

        // Memoization Validation
        // A VisRecord should be in the memoization table iff it's alive and in the base state.

        // (VisRecord in memoization table -> alive and in base state)
        // Also check correct hash keys and hash sorting.
        uint64_t last_hash_bits = 0;

        auto validate_chunk = [this, &last_hash_bits] (const auto& chunk)
        {
            static constexpr VisRecordKind K = chunk.vis_record_kind;
            CAMSPORK_REQUIRE(!chunk.nodes.empty(), "Empty chunk left behind");
            CAMSPORK_REQUIRE_CMP(read_hash_helper(chunk.nodes.back()), ==, chunk.max_hash, "Wrong chunk.max_hash");
            for (nodepool::id<VisRecordListNode<K>> id : chunk.nodes) {
                // VisRecordListNode<K>
                const VisRecordListNode<K>& node = get(id);
                const auto expect_hash_bits = hash_vis_record(node.base_data);
                CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "all memoized VisRecord should have nonzero refcnt");
                CAMSPORK_REQUIRE(!node.is_forwarded(), "forwarding state VisRecord should not be memoized");
                CAMSPORK_REQUIRE(!node.camspork_next_id, "unexpected linked list next");
                CAMSPORK_REQUIRE_CMP(node.base_data.memoize_hash_bits, ==, expect_hash_bits, "wrong hash");
                CAMSPORK_REQUIRE_CMP(last_hash_bits, <=, expect_hash_bits, "Not sorted");
                last_hash_bits = expect_hash_bits;
            }
        };

        last_hash_bits = 0;
        for (const VisRecordChunk<VisRecordKind::Default>& chunk : vis_record_table) {
            validate_chunk(chunk);
        }


        // (VisRecord in memoization table <- alive and in base state)
        // Each VisRecord should be able to find itself in the table; if we fail, it could be because we
        // forgot to memoize it, or something is wrong with the hash search or equality function.
        auto memoize_self_check = [&] (auto id_for_typing)
        {
            using ListNode = typename decltype(id_for_typing)::value_type;
            RefcntDebug<ListNode>& debug = std::get<RefcntDebug<ListNode>>(debug_refcnts);
            for (nodepool::id<ListNode> id : debug_get_pool<ListNode>()) {
                const bool live = debug.refcnts[id.node_index()] != 0;
                if (!live) {
                    continue;
                }
                const auto& node = get(id);
                if (node.is_forwarded()) {
                    continue;
                }

                CAMSPORK_REQUIRE_CMP(read_hash_helper(id), ==, hash_vis_record(node.base_data), "dirty hash");
                try {
                    CAMSPORK_REQUIRE_CMP(id, ==, find_memoized(id), "memoization lookup is buggy");
                }
                catch (...) {
                    for (size_t  chunk_index = 0; chunk_index < vis_record_table.size(); ++chunk_index) {
                        fprintf(stderr, "\n === CHUNK %u ===\n", (unsigned)chunk_index);
                        for (auto id : vis_record_table[chunk_index].nodes) {
                            debug_print(stderr, get(id).base_data);
                        }
                    }
                    fprintf(stderr, "\n === LOOKING FOR ===\n");
                    VisRecord record = node.base_data;
                    debug_print(stderr, record);
                    throw;
                }
            }
        };
        memoize_self_check(nodepool::id<DefaultVisRecordListNode>{});

        // Check correct BarrierArriveState.
        // A base state VisRecord is pointed to by BarrierArriveState iff it contains a corresponding pending await.
        // BarrierArriveState may also point to forwarding state VisRecord.
        // The other half of this checking is in process_vis_record_impl.
        auto check_BarrierArriveState_VisRecords = [&] (auto record_node_id, pending_await_t expected_await_id)
        {
            while (record_node_id) {
                constexpr VisRecordKind K = decltype(record_node_id)::value_type::vis_record_kind;
                const AssignmentRecordVisNode<K>& record_node = get(record_node_id);
                record_node_id = record_node.camspork_next_id;
                const nodepool::id<VisRecordListNode<K>> vis_record_id = record_node.vis_record_id;
                const VisRecordListNode<K>& vis_record = get(vis_record_id);
                if (vis_record.is_forwarded()) {
                    continue;
                }
                nodepool::id<PendingAwaitNode> await_node_id = vis_record.base_data.pending_awaits;
                while (true) {
                    CAMSPORK_REQUIRE(await_node_id, "BarrierArriveState references VisRecord without corresponding pending_await_id");
                    const PendingAwaitNode& await_node = get(await_node_id);
                    await_node_id = await_node.camspork_next_id;
                    if (await_node.await_id == expected_await_id) {
                        break;
                    }
                }
            }
        };
        for (uint32_t barrier_index = 0; barrier_index < max_live_barriers; ++barrier_index) {
            const BarrierState& state = barrier_states[barrier_index];
            for (const auto& pair : state.arrive_states) {
                pending_await_t info = pack_pending_await(barrier_index, pair.first);
                const nodepool::id<DefaultAssignmentRecordVisNode> head_id = pair.second.vis_records_head_id;
                check_BarrierArriveState_VisRecords(head_id, info);
            }
        }
    }

    void debug_print(FILE* f, const VisRecord& record) const
    {
        const unsigned long long ull_hash = record.memoize_hash_bits;
        fprintf(f, "HASH=%llu (0x%llX)\n", ull_hash, ull_hash);
        nodepool::id<TlSigIntervalListNode> interval_id = record.visibility_set;
        while (interval_id) {
            const TlSigIntervalListNode& node = get(interval_id);
            interval_id = node.camspork_next_id;
            const TlSigInterval data = node.data;
            fprintf(f, "[%u, %u, %u, %u, %u, %u]\n",
                    data.tid_lo,
                    data.tid_hi,
                    data.qual_bits_by_vis.array[0],
                    data.qual_bits_by_vis.array[1],
                    data.qual_bits_by_vis.array[2],
                    data.qual_bits_by_vis.array[3]
            );
        }
        static_assert(num_vis_flags == 4);
        nodepool::id<PendingAwaitNode> await_id = record.pending_awaits;
        while (await_id) {
            const PendingAwaitNode& node = get(await_id);
            await_id = node.camspork_next_id;
            debug_print(f, node.await_id);
        }
    }

    void debug_print(FILE* f, pending_await_t await_id) const
    {
        fprintf(f, "PendingAwait(barrier_index=%i, arrive_count=%i)\n",
                int(pending_await_barrier_index(await_id)),
                int(pending_await_arrive_count(await_id)));
    }
};

namespace {

struct SyncvTrivialLogger;  // Defined before SyncvTable; matches SyncvRealLogger interface.

struct SyncvRealLogger
{
    std::string var_str_name;
    std::vector<std::unique_ptr<ExcutBaseAction>>* p_excut_actions;
    std::vector<extent_t> idx_for_single;
    std::vector<ExcutSyncEnvAccess*> actions_to_update;
    VisRecordHistoryLog* p_history_log;

    explicit SyncvRealLogger(const SyncvLogRequest& request)
      : var_str_name(request.var_str_name)
      , p_excut_actions(request.p_excut_actions)
      , idx_for_single(request.idx_for_single)
      , actions_to_update{}
      , p_history_log(request.p_history_log)
    {
    }

    template <typename VisRecordList>
    void excut_log_assignment_records(
            const SyncvTable& env, assignment_record_id* p_id, const VisRecordList& new_vis_record_list, bool is_mutate)
    {
        if (p_excut_actions) {
            nodepool::id<AssignmentRecord> asn_id{p_id->node_id};
            _excut_log_assignment_record_impl(env, asn_id, new_vis_record_list, idx_for_single, is_mutate);
        }
    }

    template <typename VisRecordList>
    void excut_log_assignment_records(
            const SyncvTable& env, AssignmentRecordWindow window, const VisRecordList& new_vis_record_list, bool is_mutate)
    {
        if (p_excut_actions) {
            std::vector<extent_t> idx(window.end_outer_extent - window.begin_outer_extent);
            _excut_recurse_log_window(env, window, idx, 0, 0, new_vis_record_list, is_mutate);
        }
    }

    void excut_update_assignment_record_ids(const CensusMap& census)
    {
        for (ExcutSyncEnvAccess* p : actions_to_update) {
            nodepool::id<AssignmentRecord> key{p->id_before};
            auto iter = census.find(key);
            CAMSPORK_REQUIRE(iter != census.end(), "ExcutSyncEnvAccess::id_before not found in CensusMap");
            const AssignmentRecordCensusEntry& entry = iter->second;
            p->id_after = entry.new_node_id.id_bits;
        }
    }

    void excut_update_assignment_record_ids(nodepool::id<AssignmentRecord> new_id)
    {
        for (ExcutSyncEnvAccess* p : actions_to_update) {
            p->id_after = new_id.id_bits;
        }
    }

  private:
    template <typename VisRecordList>
    void _excut_recurse_log_window(
            const SyncvTable& env, const AssignmentRecordWindow& window, std::vector<extent_t>& idx,
            size_t dim_idx, size_t partial_linear_offset, const VisRecordList& new_vis_record_list, bool is_mutate)
    {
        if (dim_idx >= idx.size()) {
            CAMSPORK_REQUIRE_CMP(idx.size(), ==, dim_idx, "overshot");
            nodepool::id<AssignmentRecord> asn_id{window.base[partial_linear_offset].node_id};
            _excut_log_assignment_record_impl(env, asn_id, new_vis_record_list, idx, is_mutate);
        }
        else {
            const extent_t outer_c = window.begin_outer_extent[dim_idx];
            const extent_t offset_c = window.begin_offset[dim_idx];
            const extent_t end_c = offset_c + window.begin_inner_extent[dim_idx];

            for (extent_t i = offset_c; i < end_c; ++i) {
                idx[dim_idx] = i;
                const auto new_linear_offset = partial_linear_offset * outer_c + i;
                _excut_recurse_log_window(
                        env, window, idx, dim_idx+1, new_linear_offset, new_vis_record_list, is_mutate);
            }
        }
    }

    template <typename VisRecordList>
    void _excut_log_assignment_record_impl(
            const SyncvTable& env,
            nodepool::id<AssignmentRecord> asn_id,
            const VisRecordList& new_vis_record_list,
            std::vector<extent_t> idx,
            bool is_mutate)
    {
        constexpr VisRecordKind K = VisRecordList::value_type::value_type::vis_record_kind;

        // Log top-level assignment record ID, name+idxs of access,
        // and remember to update this with the changed ID later.
        {
            auto p_info = std::make_unique<ExcutSyncEnvAccess>();
            p_info->id_before = asn_id.id_bits;
            p_info->id_after = 0;  // See excut_update_assignment_record_ids
            p_info->name = var_str_name;
            p_info->idx = std::move(idx);
            p_info->mutate_tag = is_mutate ? ExcutMutateTag::Mutate : ExcutMutateTag::Read;
            actions_to_update.push_back(p_info.get());
            p_excut_actions->push_back(std::move(p_info));
        }

        // Log existing VisRecords, tagged as WAR, WAW, or RAW, depending on the relation
        // between the prior VisRecords and the current SyncEnvAccess action.
        if (asn_id) {
            const AssignmentRecord& asn_record = env.get(asn_id);
            if (is_mutate) {
                nodepool::id<DefaultAssignmentRecordVisNode> read_id = asn_record.read_vis_records_head_id;
                while (read_id) {
                    const DefaultAssignmentRecordVisNode& asn_node = env.get(read_id);
                    read_id = asn_node.camspork_next_id;
                    _excut_log_vis_record(env, asn_node.vis_record_id, ExcutMutateTag::WAR);
                }
            }
            nodepool::id<DefaultAssignmentRecordVisNode> mutate_id = asn_record.mutate_vis_records_head_id;
            while (mutate_id) {
                const DefaultAssignmentRecordVisNode& asn_node = env.get(mutate_id);
                mutate_id = asn_node.camspork_next_id;
                _excut_log_vis_record(env, asn_node.vis_record_id, is_mutate ? ExcutMutateTag::WAW : ExcutMutateTag::RAW);
            }
        }

        // Log new VisRecord
        for (nodepool::id<VisRecordListNode<K>> new_vis_id : new_vis_record_list) {
            _excut_log_vis_record(env, new_vis_id, is_mutate ? ExcutMutateTag::Mutate : ExcutMutateTag::Read);
        }
    }

    template <VisRecordKind K>
    void _excut_log_vis_record(
            const SyncvTable& env, nodepool::id<VisRecordListNode<K>> id, ExcutMutateTag mutate_tag)
    {
        const VisRecord& vis_record = env.const_resolve_forwarding(id, &id);
        auto p_excut_vis_record = std::make_unique<ExcutVisRecord>();
        p_excut_vis_record->id = id.id_bits;
        p_excut_vis_record->mutate_tag = mutate_tag;
        p_excut_actions->push_back(std::move(p_excut_vis_record));

        _excut_log_vis_record_data(env, vis_record, mutate_tag);
    }

    void _excut_log_vis_record_data(const SyncvTable& env, const VisRecord& vis_record, ExcutMutateTag mutate_tag)
    {
        nodepool::id<TlSigIntervalListNode> tl_id = vis_record.visibility_set;
        while (tl_id) {
            const TlSigIntervalListNode& node = env.get(tl_id);
            tl_id = node.camspork_next_id;
            auto p_excut_interval = std::make_unique<ExcutTlSigInterval>();
            p_excut_interval->tid_lo = node.data.tid_lo;
            p_excut_interval->tid_hi = node.data.tid_hi;
            p_excut_interval->qual_bits_by_vis = node.data.qual_bits_by_vis;
            p_excut_interval->mutate_tag = mutate_tag;
            p_excut_actions->push_back(std::move(p_excut_interval));
        }

        nodepool::id<PendingAwaitNode> await_node_id = vis_record.pending_awaits;
        while (await_node_id) {
            const PendingAwaitNode& node = env.get(await_node_id);
            await_node_id = node.camspork_next_id;
            auto p_excut_await = std::make_unique<ExcutPendingAwait>();
            barrier_id tmp_barrier_id;
            env.set_barrier_index(&tmp_barrier_id, pending_await_barrier_index(node.await_id));
            p_excut_await->barrier_id = tmp_barrier_id.data;
            p_excut_await->arrive_count = pending_await_arrive_count(node.await_id);
            p_excut_await->mutate_tag = mutate_tag;
            p_excut_actions->push_back(std::move(p_excut_await));
        }
    }

  public:
    void history_set_sync_stmt_info(const SyncvFence& fence)
    {
        if (p_history_log) {
            LoggedSyncStmtValues values{};
            values.L1_qual_bits = fence.L1_qual_bits;
            values.L2_full_qual_bits = fence.L2_full_qual_bits;
            values.L2_temporal_qual_bits = fence.L2_temporal_qual_bits;
            p_history_log->set_syncv_sync_stmt_info({}, values);
        }
    }

    void history_set_sync_stmt_info(const SyncvArrive& arrive, const BarrierState& state, uint32_t new_arrive_count)
    {
        if (p_history_log) {
            LoggedSyncStmtValues values{};
            values.L1_qual_bits = arrive.L1_qual_bits;
            values.L2_full_qual_bits = 0;
            values.L2_temporal_qual_bits = 0;
            values.arrive_count_before = state.arrive_count;
            values.arrive_count_after = new_arrive_count;
            values.await_count_before = state.await_count;
            values.await_count_after = state.await_count;
            p_history_log->set_syncv_sync_stmt_info(arrive.home_barrier, values);
        }
    }

    void history_set_sync_stmt_info(
            const SyncvAwait& await, const BarrierState& state, uint32_t new_await_count, uint32_t max_arrive_count)
    {
        if (p_history_log) {
            LoggedSyncStmtValues values{};
            values.L1_qual_bits = 0;
            values.L2_full_qual_bits = await.L2_full_qual_bits;
            values.L2_temporal_qual_bits = await.L2_temporal_qual_bits;
            values.arrive_count_before = state.arrive_count;
            values.arrive_count_after = state.arrive_count;
            values.await_count_before = state.await_count;
            values.await_count_after = new_await_count;
            values.await_max_arrive_count = max_arrive_count;
            p_history_log->set_syncv_sync_stmt_info(await.bar, values);
        }
    }

    template <VisRecordKind K>
    void history_new_vis_record(SyncvTable& env, nodepool::id<VisRecordListNode<K>> node_id)
    {
        if (p_history_log) {
            history_log_vis_record_id history_id(node_id);
            p_history_log->log_syncv_new_vis_record(history_id, _get_history_vis_record_data(env, node_id));
        }
    }

    // XXX TODO need to start calling this again.
    template <VisRecordKind K>
    void history_vis_record_change(
            SyncvTable& env,
            nodepool::id<VisRecordListNode<K>> old_id,
            nodepool::id<VisRecordListNode<K>> new_id,
            bool debug_printf)
    {
        if (p_history_log) {
            p_history_log->log_syncv_vis_record_change(
                    history_log_vis_record_id(old_id),
                    history_log_vis_record_id(new_id),
                    _get_history_vis_record_data(env, new_id),
                    debug_printf);
        }
    }

    template <VisRecordKind K>
    void history_vis_record_checked(nodepool::id<VisRecordListNode<K>> id, bool is_mutate)
    {
        if (p_history_log) {
            p_history_log->log_syncv_vis_record_checked(history_log_vis_record_id(id), is_mutate);
        }
    }

    template <VisRecordKind K>
    void history_vis_record_error(
            nodepool::id<VisRecordListNode<K>> id, TlSig fail_tl_sig)
    {
        if (p_history_log) {
            p_history_log->log_syncv_vis_record_error(history_log_vis_record_id(id), fail_tl_sig);
        }
    }
  private:
    template <VisRecordKind K>
    LoggedVisRecordData _get_history_vis_record_data(SyncvTable& env, nodepool::id<VisRecordListNode<K>> node_id)
    {
        VisRecordDebugData debug;
        env.debug_get_vis_record_data(node_id, &debug);
        LoggedVisRecordData data{};
        data.visibility_set = std::move(debug.visibility_set);
        const auto& pending_awaits = debug.pending_await_list;
        for (size_t i = 0; i < pending_awaits.size(); ++i) {
            const pending_await_t await_id = pending_awaits[i];
            barrier_id id;
            env.set_barrier_index(&id, pending_await_barrier_index(await_id));
            const auto arrive_count = pending_await_arrive_count(await_id);
            data.pending_await_list.push_back(
                LoggedPendingAwait{p_history_log->get_barrier_name(id), arrive_count}
            );
        }
        return data;
    }
};

}  // end namespace



// *** Primary Implemented Interface ***



#define INTERFACE_PROLOGUE(table) \
try { \
    CAMSPORK_REQUIRE(!table->failed, "Cannot continue using env after failure detected");

#define INTERFACE_EPILOGUE(table) \
    CAMSPORK_REQUIRE_CMP(table->modified_vis_records.size(), ==, 0, "internal error, missing memoize_modified()"); \
} \
catch (...) { \
    table->failed = true; \
    throw; \
}

SyncvTable* new_syncv_table(const syncv_init_t& init)
{
    SyncvTable* table = new SyncvTable;
    table->original_memory_budget = init.memory_budget;
    table->current_memory_budget = init.memory_budget;
    return table;
}

SyncvTable* copy_syncv_table(const SyncvTable* table)
{
    return new SyncvTable(*table);
}

void delete_syncv_table(SyncvTable* table)
{
#if 0
    fprintf(stderr, "       VisRecord capacity = %llu\n",
            (long long unsigned)table->debug_get_pool<DefaultVisRecordListNode>().size());
#endif
    delete table;
}

void SyncvTableDeleter::operator() (SyncvTable* victim) const
{
    delete victim;
}

void on_r(
        SyncvTable* table, assignment_record_id* input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, decltype(nullptr))
{
    INTERFACE_PROLOGUE(table)
    table->on_r(input, cuboid, access, SyncvTrivialLogger{});
    INTERFACE_EPILOGUE(table)
}

void on_r(
        SyncvTable* table, assignment_record_id* input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, const SyncvLogRequest& log)
{
    INTERFACE_PROLOGUE(table)
    table->on_r(input, cuboid, access, SyncvRealLogger(log));
    INTERFACE_EPILOGUE(table)
}

void on_r(
        SyncvTable* table, AssignmentRecordWindow input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, decltype(nullptr))
{
    INTERFACE_PROLOGUE(table)
    table->on_r(input, cuboid, access, SyncvTrivialLogger{});
    INTERFACE_EPILOGUE(table)
}

void on_r(
        SyncvTable* table, AssignmentRecordWindow input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, const SyncvLogRequest& log)
{
    INTERFACE_PROLOGUE(table)
    table->on_r(input, cuboid, access, SyncvRealLogger(log));
    INTERFACE_EPILOGUE(table)
}

// on_rw is a bit of a misnomer now that is_write_only is a thing.

void on_rw(
        SyncvTable* table, assignment_record_id* input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, decltype(nullptr))
{
    INTERFACE_PROLOGUE(table)
    table->on_rw(input, cuboid, access, SyncvTrivialLogger{});
    INTERFACE_EPILOGUE(table)
}

void on_rw(
        SyncvTable* table, assignment_record_id* input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, const SyncvLogRequest& log)
{
    INTERFACE_PROLOGUE(table)
    table->on_rw(input, cuboid, access, SyncvRealLogger(log));
    INTERFACE_EPILOGUE(table)
}

void on_rw(
        SyncvTable* table, AssignmentRecordWindow input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, decltype(nullptr))
{
    INTERFACE_PROLOGUE(table)
    table->on_rw(input, cuboid, access, SyncvTrivialLogger{});
    INTERFACE_EPILOGUE(table)
}

void on_rw(
        SyncvTable* table, AssignmentRecordWindow input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, const SyncvLogRequest& log)
{
    INTERFACE_PROLOGUE(table)
    table->on_rw(input, cuboid, access, SyncvRealLogger(log));
    INTERFACE_EPILOGUE(table)
}

void on_check_free(
        SyncvTable* table, AssignmentRecordWindow input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, decltype(nullptr))
{
    INTERFACE_PROLOGUE(table)
    table->on_check_free(input, cuboid, access, SyncvTrivialLogger{});
    INTERFACE_EPILOGUE(table)
}

void on_check_free(
        SyncvTable* table, AssignmentRecordWindow input,
        const ThreadCuboid& cuboid, SyncvAccessInfo access, const SyncvLogRequest& log)
{
    INTERFACE_PROLOGUE(table)
    table->on_check_free(input, cuboid, access, SyncvRealLogger(log));
    INTERFACE_EPILOGUE(table)
}

void clear_visibility(SyncvTable* table, size_t N, assignment_record_id* array)
{
    INTERFACE_PROLOGUE(table)
    table->clear_visibility(N, array);
    INTERFACE_EPILOGUE(table)
}

void alloc_barriers(SyncvTable* table, size_t N, barrier_id* barriers)
{
    INTERFACE_PROLOGUE(table)
    table->alloc_barriers(N, barriers);
    INTERFACE_EPILOGUE(table)
}

void free_barriers(SyncvTable* table, size_t N, barrier_id* barriers, bool check_arrive_await)
{
    INTERFACE_PROLOGUE(table)
    table->free_barriers(N, barriers, check_arrive_await);
    INTERFACE_EPILOGUE(table)
}

void on_fence(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvFence& fence, const SyncvLogRequest& log)
{
    INTERFACE_PROLOGUE(table)
    table->on_fence(cuboid, fence, SyncvRealLogger(log));
    INTERFACE_EPILOGUE(table)
}

void on_fence(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvFence& fence, decltype(nullptr))
{
    INTERFACE_PROLOGUE(table)
    table->on_fence(cuboid, fence, SyncvTrivialLogger{});
    INTERFACE_EPILOGUE(table)
}

void on_arrive(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvArrive& arrive, const SyncvLogRequest& log)
{
    INTERFACE_PROLOGUE(table)
    table->on_arrive(cuboid, arrive, SyncvRealLogger(log));
    INTERFACE_EPILOGUE(table)
}

void on_arrive(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvArrive& arrive, decltype(nullptr))
{
    INTERFACE_PROLOGUE(table)
    table->on_arrive(cuboid, arrive, SyncvTrivialLogger{});
    INTERFACE_EPILOGUE(table)
}

void on_await(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvAwait& await, const SyncvLogRequest& log)
{
    INTERFACE_PROLOGUE(table)
    table->on_await(cuboid, await, SyncvRealLogger(log));
    INTERFACE_EPILOGUE(table)
}

void on_await(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvAwait& await, decltype(nullptr))
{
    INTERFACE_PROLOGUE(table)
    table->on_await(cuboid, await, SyncvTrivialLogger{});
    INTERFACE_EPILOGUE(table)
}

void on_join_threads(SyncvTable* table, const ThreadCuboid& cuboid, const SyncvLogRequest& log)
{
    INTERFACE_PROLOGUE(table)
    table->on_join_threads(cuboid, SyncvRealLogger(log));
    INTERFACE_EPILOGUE(table)
}

void on_join_threads(SyncvTable* table, const ThreadCuboid& cuboid, decltype(nullptr))
{
    INTERFACE_PROLOGUE(table)
    table->on_join_threads(cuboid, SyncvTrivialLogger{});
    INTERFACE_EPILOGUE(table)
}

void begin_no_checking(SyncvTable* table)
{
    table->no_checking_counter++;
}

void end_no_checking(SyncvTable* table)
{
    CAMSPORK_REQUIRE(table->no_checking_counter, "end_no_checking without begin_no_checking");
    table->no_checking_counter--;
}



// *** Debug Inspection Interface ***



void debug_get_vis_record_data(const SyncvTable* table, uint32_t id, VisRecordDebugData* out)
{
    table->debug_get_vis_record_data(nodepool::id<VisRecordListNode<VisRecordKind::Default>>{id}, out);
}

void debug_validate_state(SyncvTable* table, size_t input_count, const SyncvDebugValidateInput* p_inputs)
{
    table->debug_validate_state(input_count, p_inputs);
}


}  // end namespace
