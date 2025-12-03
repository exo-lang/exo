#include "syncv_table.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
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

    uint8_t forwarded_flag;

    // This has nothing to do with the main purpose of the struct; only needed for assignment_record_remove_duplicates.
    // This should be in AssignmentRecordVisNode conceptually, but that would waste 4 bytes.
    uint8_t tmp_is_duplicate;
};

// TODO consider removing IsMutate templatization, and separate memoization of read/mutate VisRecord objects.
// This shouldn't be needed anymore after vis_flag_temporal was added.
template <bool IsMutate>
struct VisRecordListNode
{
    static constexpr bool is_mutate = IsMutate;

    // Count of owning references.
    // AssignmentRecord references (and AssignmentRecordVisNode) are owning.
    // Forwarding references are owning.
    // Memoization table references are non-owning.
    refcnt_t refcnt;

    // If in base state, this is the next node in the memoization bucket.
    // If in the forwarding state, this is an owning reference to the forwarded-to visibility record.
    nodepool::id<VisRecordListNode<IsMutate>> camspork_next_id;

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

using ReadVisRecordListNode = VisRecordListNode<false>;
using MutateVisRecordListNode = VisRecordListNode<true>;

static_assert(sizeof(ReadVisRecordListNode) == 20, "Check that you meant to change this perf-critical struct");

template <bool IsMutate>
struct AssignmentRecordVisNode
{
    // Linked list of read/mutate vis records for an assignment record.
    // Don't use the camspork_next_id in the VisRecord itself ... that is for the memoization table's usage.
    nodepool::id<VisRecordListNode<IsMutate>> vis_record_id;
    nodepool::id<AssignmentRecordVisNode<IsMutate>> camspork_next_id;

    static constexpr bool is_mutate = IsMutate;

    refcnt_t get_refcnt() const
    {
        return 1;  // Replace if refcnt member added
    }
};

using AssignmentRecordReadNode = AssignmentRecordVisNode<false>;
using AssignmentRecordMutateNode = AssignmentRecordVisNode<true>;

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
    nodepool::id<AssignmentRecordMutateNode> mutate_vis_records_head_id{0};

    // Zero or more read visibility records.
    nodepool::id<AssignmentRecordReadNode> read_vis_records_head_id{0};

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
    // A base-state VisRecord is in a list iff the VisRecord has (parent, arrive_count) in its pending_awaits.
    // Forwarding-state VisRecords may be in the lists as well ... ignore them if found.
    //
    // Re-use of "assignment record" structs is just pragmatic (maybe confusing).
    nodepool::id<AssignmentRecordMutateNode> mutate_vis_records_head_id{0};
    nodepool::id<AssignmentRecordReadNode> read_vis_records_head_id{0};
};


struct BarrierState
{
    int32_t arrive_count;
    int32_t await_count;

    // Sorted by arrive_count.
    // Entries removed from the list upon matched await.
    BinaryTree<int32_t, BarrierArriveState> arrive_states;
};

template <uint32_t Level> constexpr uint64_t bucket_level_size = 0;
template<> constexpr uint64_t bucket_level_size<0> = 1;
template<> constexpr uint64_t bucket_level_size<1> = 32;
template<> constexpr uint64_t bucket_level_size<2> = 128;
template<> constexpr uint64_t bucket_level_size<3> = 256;
template<> constexpr uint64_t bucket_level_size<4> = 1024;
template<> constexpr uint64_t bucket_level_size<5> = 4096;
template<> constexpr uint64_t bucket_level_size<6> = 16384;
template<> constexpr uint64_t bucket_level_size<7> = 0x10'0000;
template<> constexpr uint64_t bucket_level_size<8> = 0x400'0000;
template<> constexpr uint64_t bucket_level_size<9> = 0x1'0000'0000;
constexpr uint32_t bucket_level_count = 10;

template <bool IsMutate, uint32_t BucketLevel> struct IntervalBucket;

template <bool IsMutate, uint32_t BucketLevel>
struct IntervalBucketParentPointer
{
    static_assert(BucketLevel < bucket_level_count);
    uint32_t child_index_in_parent = 0;
    IntervalBucket<IsMutate, BucketLevel + 1>* p_parent = nullptr;

    void update_parent_pointer(
        const IntervalBucketParentPointer& other, IntervalBucket<IsMutate, BucketLevel + 1>* new_parent)
    {
        child_index_in_parent = other.child_index_in_parent;
        p_parent = new_parent;
        CAMSPORK_REQUIRE(new_parent, "Expected parent pointer");
    }
};

template <bool IsMutate>
struct IntervalBucketParentPointer<IsMutate, bucket_level_count - 1>
{
};

// Let Sz = bucket_level_size<BucketLevel>; an interval bucket for a given
// BucketLevel encompasses the interval of thread IDs [I * Sz, (I+1) * Sz - 1]
// for some index I. The idea is to store visibility records in the smallest
// possible (i.e. most specific) bucket that is still a superset.
//
// Unless BucketLevel == 0 (single thread buckets),
// the bucket has child interval buckets of one lower level,
// with the original interval evenly subdivided into N-many child buckets
// owned by the parent bucket.
//
// All buckets except the top-level bucket store a back pointer to
// their parent; see IntervalBucketParentPointer (This is a non-owning
// back ptr; child can't outlive parent).
//
// Buckets should be removed from the tree when empty, see
// delete_interval_bucket_if_empty.
template <bool IsMutate, uint32_t BucketLevel>
struct IntervalBucket : IntervalBucketParentPointer<IsMutate, BucketLevel>
{
    static_assert(BucketLevel != 0, "BucketLevel = 0 for illustration only");
    static_assert(BucketLevel != 1, "Should be template specialization");
    static_assert(BucketLevel < bucket_level_count);

    static constexpr uint32_t bucket_level = BucketLevel;
    static constexpr uint32_t child_count = bucket_level_size<BucketLevel> / bucket_level_size<BucketLevel - 1>;
    using child_t = IntervalBucket<IsMutate, BucketLevel - 1>;
    std::unique_ptr<child_t> child_interval_buckets[child_count];

    // Nth bit is set iff child_interval_buckets[N] isn't empty.
    uint64_t nonempty_child_flags = 0;

    // Bucket for this interval.
    // Note: we don't have to deep copy this because it's just indices into the node pool, which is deep copied.
    nodepool::id<VisRecordListNode<IsMutate>> bucket = {0};

    // For making code re-entrant (prevent de-allocation while being visited).
    uint32_t visitor_count = 0;

    IntervalBucket() = default;

    // Deep copy
    IntervalBucket(const IntervalBucket& other, IntervalBucket<IsMutate, bucket_level + 1>* new_parent)
    {
        this->update_parent_pointer(other, new_parent);
        this->copy_impl(other);
    };

    IntervalBucket(const IntervalBucket& other)
    {
        static_assert(BucketLevel == bucket_level_count - 1, "Non-top-level bucket must have parent pointer");
        this->copy_impl(other);
    }

  private:
    void copy_impl(const IntervalBucket& other)
    {
        for (uint32_t i = 0; i < child_count; ++i) {
            const child_t* p_child = other.child_interval_buckets[i].get();
            if (p_child) {
                child_interval_buckets[i].reset(new child_t(*p_child, this));
            }
        }
        nonempty_child_flags = other.nonempty_child_flags;
        bucket = other.bucket;
        visitor_count = other.visitor_count;
        CAMSPORK_REQUIRE_CMP(visitor_count, ==, 0, "Not sure copying is OK while being traversed.");
    }
};

template <bool IsMutate>
struct IntervalBucket<IsMutate, 1> : IntervalBucketParentPointer<IsMutate, 1>
{
    static constexpr uint32_t bucket_level = 1;
    static constexpr uint32_t child_count = bucket_level_size<1>;
    nodepool::id<VisRecordListNode<IsMutate>> child_interval_buckets[child_count] = {};

    // Nth bit is set iff child_interval_buckets[N] isn't empty.
    uint64_t nonempty_child_flags = 0;

    // Bucket for this interval.
    // Note: we don't have to deep copy this because it's just indices into the node pool, which is deep copied.
    nodepool::id<VisRecordListNode<IsMutate>> bucket = {0};

    // For making code re-entrant (prevent de-allocation while being visited).
    uint32_t visitor_count = 0;

    IntervalBucket() = default;

    // Deep copy
    IntervalBucket(const IntervalBucket& other, IntervalBucket<IsMutate, bucket_level + 1>* new_parent)
    {
        this->update_parent_pointer(other, new_parent);
        for (uint32_t i = 0; i < child_count; ++i) {
            child_interval_buckets[i] = other.child_interval_buckets[i];
        }
        nonempty_child_flags = other.nonempty_child_flags;
        bucket = other.bucket;
        visitor_count = other.visitor_count;
        CAMSPORK_REQUIRE_CMP(visitor_count, ==, 0, "Not sure copying is OK while being traversed.");
    }
};


// TODO memoization is horrible right now, move to hash-indexed buckets or something.
// The hierarchical structure was created to optimize for avoiding redundancy for
// Arrive/Fence statements, but overwhelmingly time is used for memoization lookup
// for read/mutate statements. The latter use case entails horrible linear searches now.
//
// If we do this change, we have to be careful not to double-consider items when
// moving between buckets.


template <bool IsMutate, uint32_t BucketLevel>
bool interval_bucket_is_empty(const IntervalBucket<IsMutate, BucketLevel>& bucket) noexcept
{
    return bucket.nonempty_child_flags == 0 && !bucket.bucket && !bucket.visitor_count;
}

// De-allocate the given bucket if it's empty and not the top-level bucket.
// We presume that the bucket is owned by its parent (unique_ptr tree).
//
// We do not make any modifications to the parent except for nulling out the pointer.
// In particular, we don't change nonempty_child_flags, or handle deleting the parent
// if it too is now empty.
template <bool IsMutate, uint32_t BucketLevel>
void delete_interval_bucket_if_empty(IntervalBucket<IsMutate, BucketLevel>* p)
{
    if (interval_bucket_is_empty(*p)) {
        for (const auto& child : p->child_interval_buckets) {
            CAMSPORK_REQUIRE(!child, "nonempty_child_flags was wrong.");
        }

        if constexpr (BucketLevel < bucket_level_count - 1) {
            // Parent pointer should be correct.
            IntervalBucket<IsMutate, BucketLevel + 1>* p_parent = p->p_parent;
            CAMSPORK_REQUIRE(p_parent, "missing parent ptr");
            const uint32_t child_index = p->child_index_in_parent;
            CAMSPORK_REQUIRE_CMP(child_index, <, p_parent->child_count, "child_index out-of-range");
            CAMSPORK_REQUIRE_CMP(p_parent->nonempty_child_flags, >, 0, "should have been deallocated");

            // p is invalidated after this (unique_ptr reset).
            CAMSPORK_REQUIRE_CMP(p_parent->child_interval_buckets[child_index].get(), ==, p, "???");
            p_parent->child_interval_buckets[child_index].reset();
        }
    }
}

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

    template <bool IsMutate>
    history_log_vis_record_id(nodepool::id<VisRecordListNode<IsMutate>> node_id)
    {
        // This is weird ... since the node ID namespaces are separate for read (!IsMutate) and mutate VisRecord,
        // we use the lowest bit to disambiguate. If we stop treating the two as separate types, we can just
        // pass through the ID directly.
        static_assert(sizeof(data) == 4);
        CAMSPORK_REQUIRE_CMP(node_id.id_bits, <=, 0x7FFF'FFFF, "too many node IDs");
        data = node_id.id_bits << 1 | IsMutate;
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
            const SyncvTable&, Input, const VisRecordList&, ExcutMutateTag)
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

    template <bool IsMutate>
    void history_new_vis_record(SyncvTable&, nodepool::id<VisRecordListNode<IsMutate>>)
    {
    }

    template <bool IsMutate>
    void history_vis_record_change(
            SyncvTable&, nodepool::id<VisRecordListNode<IsMutate>>, nodepool::id<VisRecordListNode<IsMutate>>, bool)
    {
    }

    template <bool IsMutate>
    void history_vis_record_checked(nodepool::id<VisRecordListNode<IsMutate>>)
    {
    }

    template <bool IsMutate>
    void history_vis_record_error(nodepool::id<VisRecordListNode<IsMutate>>, TlSig)
    {
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
    uint64_t augment_counter = 0;     // Number of fence+await
    uint64_t bucket_search_call_counter = 0;
    uint64_t bucket_search_iter_counter = 0;

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
        nodepool::Pool<ReadVisRecordListNode>,
        nodepool::Pool<MutateVisRecordListNode>,
        nodepool::Pool<AssignmentRecordReadNode>,
        nodepool::Pool<AssignmentRecordMutateNode>> pool_tuple;

    // Barrier state.
    // The Nth bit is 1 if N is allocated as a barrier ID.
    uint64_t live_barrier_bits[max_live_barriers / 64] = {0};
    BarrierState barrier_states[max_live_barriers];

    // Memoization table state (requires special deep copy support implemented in IntervalBucket).
    IntervalBucket<false, bucket_level_count - 1> read_top_level_bucket;
    IntervalBucket<true, bucket_level_count - 1> mutate_top_level_bucket;



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
    template <bool IsMutate>
    void incref(nodepool::id<VisRecordListNode<IsMutate>> id, uint32_t added_refcnt = 1)
    {
        VisRecordListNode<IsMutate>& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "should not have started with 0 refcnt");
        node.refcnt += added_refcnt;
        CAMSPORK_REQUIRE_CMP(node.refcnt, >, added_refcnt, "reference count overflow");
    }

    // Decrement reference count of visibility record,
    // and handle necessary free-ing in case of 0 refcnt.
    // NB this is not used in memoize_new_vis_record, since we assert here that the deleted VisRecord
    // is memoized, which isn't the case there. This check is lifesaving for sanity in other cases!
    template <bool IsMutate>
    void decref(nodepool::id<VisRecordListNode<IsMutate>> id)
    {
        CAMSPORK_REQUIRE(id, "decref(0)");
        VisRecordListNode<IsMutate>& node = get(id);
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
                auto memoized_id = remove_memoized(&node);
                CAMSPORK_REQUIRE_CMP(id, ==, memoized_id, "should have been found in memoization table");
                CAMSPORK_REQUIRE(!get(memoized_id).camspork_next_id, "Should have been removed from bucket's list.");
                free_single_vis_record(memoized_id);
            }
        }
    }

    void reset_vis_record_data(VisRecord* p_data)
    {
        static_assert(sizeof(*p_data) == 12, "update me");
        extend_free_list(p_data->visibility_set);
        p_data->visibility_set = {};
        extend_free_list(p_data->pending_awaits);
        p_data->pending_awaits = {};
    }

    template <bool IsMutate>
    void free_single_vis_record(nodepool::id<VisRecordListNode<IsMutate>> id)
    {
        CAMSPORK_REQUIRE(id, "unexpected 0 id");
        VisRecordListNode<IsMutate>& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, ==, 0, "unexpected nonzero refcnt");
        reset_vis_record_data(&node.base_data);
        node.camspork_next_id = {};  // Avoid freeing entire list.
        extend_free_list(id);
    }

    template <bool IsMutate>
    void assignment_record_remove_vis_records(nodepool::id<AssignmentRecordVisNode<IsMutate>>* p_head_id)
    {
        // Decref visibility records
        const auto head_id = *p_head_id;
        auto id = head_id;
        while (id) {
            AssignmentRecordVisNode<IsMutate>& node = get(id);
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
    template <bool IsMutate, typename ThreadInit>
    VisRecordListNode<IsMutate>& alloc_vis_record(
            const ThreadInit& thread_init, SyncvAccessInfo access, nodepool::id<VisRecordListNode<IsMutate>>* out)
    {
        nodepool::id<VisRecordListNode<IsMutate>> vis_record_id;
        VisRecordListNode<IsMutate>& vis_record = alloc_default_node(&vis_record_id);
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
        CAMSPORK_REQUIRE_CMP(0, !=, input.qual_bits_by_vis.array[0], "non-empty input check");
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
        static_assert(sizeof(a) == 12, "Update me");

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
        // Check for any intersections between TlSigIntervals generated by ThreadCuboid + qual_bits.
        // and those stored in the VisRecord, with vis_flag_issue, and also vis_flag_full if transitive.
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
    }



    // *** Memoization ***



    // Get the smallest possible tl_sig interval that is a superset of the visibility set (ignoring atomic-only).
    // This is needed to index into the correct bucket (the smallest one possible containing the visibility set).
    // Note, at time of writing the qual_bits_by_vis aren't used for bucketing, but maybe they should be.
    TlSigBucketKey minimal_superset_interval(nodepool::id<TlSigIntervalListNode> id) const
    {
        TlSigBucketKey key;
        key.tid_lo = UINT32_MAX;
        key.tid_hi = 0u;

        CAMSPORK_REQUIRE(id, "unsupported: empty interval");
        while (id) {
            const TlSigIntervalListNode& node = get(id);
            id = node.camspork_next_id;
            const TlSigInterval data = node.data;
            static_assert(vis_flag_atomic_only == 1, "hard-wired here we skip qual_bits_by_vis[0]");
            auto q_bits = data.qual_bits_by_vis.array[1];
            for (int32_t i = 2; i < num_vis_flags; ++i) {
                q_bits |= data.qual_bits_by_vis.array[i];
            }
            if (q_bits != 0) {
                key.tid_lo = std::min(key.tid_lo, data.tid_lo);
                key.tid_hi = std::max(key.tid_hi, data.tid_hi);
            }
        }
        CAMSPORK_REQUIRE_CMP(key.tid_hi, !=, 0, "unsupported: empty interval");  // akeley98/camspork_0_threads
        return key;
    }

    // "Remove forwarding"; replace ID of forwarded visibility record
    // with ID of base visibility record that the original record forwarded to.
    // Assumes the given ID is intended as an owning ID.
    // Return record data.
    template <bool IsMutate>
    VisRecord& remove_forwarding(nodepool::id<VisRecordListNode<IsMutate>>* p_id)
    {
        const nodepool::id<VisRecordListNode<IsMutate>> old_id = *p_id;
        nodepool::id<VisRecordListNode<IsMutate>> id = old_id;
        CAMSPORK_REQUIRE(id, "null input to remove_forwarding");
        VisRecordListNode<IsMutate>* p_node = &get(id);
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
    template <bool IsMutate>
    VisRecord const_resolve_forwarding(
            nodepool::id<VisRecordListNode<IsMutate>> id,
            nodepool::id<VisRecordListNode<IsMutate>>* p_out_id=nullptr) const
    {
        CAMSPORK_REQUIRE(id, "null input to const_resolve_forwarding");
        const VisRecordListNode<IsMutate>* p_node = &get(id);
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


    enum class BucketProcessType
    {
        Find = 0,
        Insert = 1,
        MapAll = 2,
    };

    // Skeleton code for modifying the memoization table, while maintaining
    // internal consistency.
    // This is based on this function being available
    //
    //     this->process_bucket(nodepool::id<VisRecordListNode<IsMutate>>*, Command)
    //
    // which may modify or delete the bucket (linked list) that has been passed.
    //
    // Only buckets that intersect the minimal_superset are processed.
    // Furthermore, if the operation is "exact", only the smallest bucket
    // containing the minimal_superset is processed.
    //
    // The operation details depend on BucketProcessType:
    //
    // Find: exact; callback skipped if bucket empty.
    //   Returns ID given by process_bucket.
    //
    // Insert: exact; create and process new empty child bucket if needed.
    //   Returns ID given by process_bucket.
    //
    // MapAll: not exact; returns 0 ID.
    //   We process smaller buckets after larger buckets, on the assumption
    //   that process_bucket may move items from smaller to larger buckets
    //   (so we need to avoid double-processing). This is a subtle thing to
    //   account for if we modify the bucketing scheme.
    //   TODO: is this reasoning correct?
    template <bool IsMutate, BucketProcessType Type, typename Command>
    nodepool::id<VisRecordListNode<IsMutate>> for_buckets(TlSigBucketKey minimal_superset, Command&& command)
    {
        if constexpr (IsMutate) {
            return this->for_buckets_impl<Type>(&mutate_top_level_bucket,
                                                minimal_superset.tid_lo,
                                                minimal_superset.tid_hi, command);
        }
        else {
            return this->for_buckets_impl<Type>(&read_top_level_bucket,
                                                minimal_superset.tid_lo,
                                                minimal_superset.tid_hi, command);
        }
    }

    template <BucketProcessType Type, uint32_t BucketLevel, typename Command, bool IsMutate>
    nodepool::id<VisRecordListNode<IsMutate>> for_buckets_impl(
            IntervalBucket<IsMutate, BucketLevel>* p_bucket,
            int64_t relative_tid_lo,
            int64_t relative_tid_hi,
            Command&& command)
    {
        if constexpr (Type != BucketProcessType::Insert && BucketLevel < bucket_level_count - 1) {
            CAMSPORK_REQUIRE(!interval_bucket_is_empty(*p_bucket), "Left behind empty bucket that should have been de-allocated.");
        }

        CAMSPORK_REQUIRE_CMP(relative_tid_lo, <, relative_tid_hi, "Input interval needs to be non-empty");
        constexpr bool ExactType = Type != BucketProcessType::MapAll;
        nodepool::id<VisRecordListNode<IsMutate>> result_id{};

        // Calculate inclusive range of child buckets that intersect the input interval.
        constexpr uint32_t child_size{bucket_level_size<BucketLevel - 1>};
        const uint32_t child_min_index = relative_tid_lo < 0 ? 0u : uint32_t(relative_tid_lo) / child_size;
        const uint32_t child_max_index = std::min(uint32_t(relative_tid_hi - 1) / child_size,
                                                  uint32_t(p_bucket->child_count - 1));

        auto visit_child = [this, p_bucket, relative_tid_lo, relative_tid_hi, &command] (uint32_t child_index)
        {
            nodepool::id<VisRecordListNode<IsMutate>> lambda_result_id = {};
            CAMSPORK_REQUIRE_CMP(child_index, <, p_bucket->child_count, "out-of-range child_index");
            auto& child_ref = p_bucket->child_interval_buckets[child_index];

            if (!child_ref && Type != BucketProcessType::Insert) {
                // Skip empty bucket if not inserting.
                return lambda_result_id;
            }

            try {
                if (!child_ref && Type == BucketProcessType::Insert) {
                    // Speculate that the child bucket will be filled.
                    // We will undo this later if wrong.
                    static_assert(p_bucket->child_count <= 64);
                    p_bucket->nonempty_child_flags |= uint64_t(1) << child_index;

                    if constexpr (BucketLevel != 1) {
                        // Create child interval bucket (unique_ptr).
                        child_ref.reset(new IntervalBucket<IsMutate, BucketLevel - 1>);
                        child_ref->p_parent = p_bucket;
                        child_ref->child_index_in_parent = child_index;
                    }
                    else {
                        // Bottom-level interval bucket is just a node list.
                    }
                }

                // Process the child bucket.
                // This may result in child_ref being nulled out.
                if constexpr (BucketLevel == 1) {
                    if constexpr (ExactType) {
                        lambda_result_id = this->process_bucket(&child_ref, command);
                    }
                    else {
                        this->process_bucket(&child_ref, command);
                    }
                }
                else {
                    const uint64_t offset = child_index * child_size;
                    lambda_result_id = this->for_buckets_impl<Type>(child_ref.get(),
                                                                    relative_tid_lo - offset, relative_tid_hi - offset,
                                                                    command);
                }
            }
            catch (...) {
                if (!child_ref) {
                    CAMSPORK_REQUIRE_CMP(p_bucket->nonempty_child_flags, >, 0, "should have been deleted");
                    static_assert(p_bucket->child_count <= 64);
                    p_bucket->nonempty_child_flags &= ~(uint64_t(1) << child_index);
                }
                throw;
            }

            // Child bucket may have been deallocated for being empty.
            // Note, flags used to be count, this failed because it wasn't re-entrant.
            if (!child_ref) {
                CAMSPORK_REQUIRE_CMP(p_bucket->nonempty_child_flags, >, 0, "should have been deleted");
                static_assert(p_bucket->child_count <= 64);
                p_bucket->nonempty_child_flags &= ~(uint64_t(1) << child_index);
            }
            return lambda_result_id;
        };

        try {
            p_bucket->visitor_count++;

            if constexpr (ExactType) {
                if (child_min_index == child_max_index) {
                    // tid interval fits in child bucket; visit it.
                    result_id = visit_child(child_min_index);
                }
                else if (Type == BucketProcessType::Insert || p_bucket->bucket) {
                    // tid interval doesn't fit in child, so the current bucket is the correct (exact) one.
                    result_id = this->process_bucket(&p_bucket->bucket, command);
                }
            }
            else {
                // Non-exact; we process smaller (child) buckets after larger (this level's) buckets.
                if (p_bucket->bucket) {
                    this->process_bucket(&p_bucket->bucket, command);
                }
                for (uint32_t child_index = child_min_index; child_index <= child_max_index; ++child_index) {
                    visit_child(child_index);
                }
            }
        }
        catch (...) {
            // For the most part, I don't care for exception safety in this code
            // but I make a defensive exception here.
            p_bucket->visitor_count--;
            delete_interval_bucket_if_empty(p_bucket);
            throw;
        }

        p_bucket->visitor_count--;
        delete_interval_bucket_if_empty(p_bucket);
        return result_id;
    }

    // Find visibility record in memoization bucket for which lambda(const VisRecord&) returns true.
    // Returns pointer to ID of record found (non-owning), or null if not found.
    template <bool IsMutate, typename Lambda>
    nodepool::id<VisRecordListNode<IsMutate>>* bucket_search(
            nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head,
            Lambda&& lambda)
    {
        bucket_search_call_counter++;
        using node_id = nodepool::id<VisRecordListNode<IsMutate>>;
        node_id* p_id = p_bucket_head;

        for (node_id id; (id = *p_id); ) {
            bucket_search_iter_counter++;
            VisRecordListNode<IsMutate>& node = get(id);
            CAMSPORK_REQUIRE(!node.is_forwarded(), "Should not be in memoization table.");

            if (lambda(node.base_data)) {
                return p_id;
            }

            p_id = &node.camspork_next_id;
        }
        return nullptr;
    }

    // Add a new visibility record, or return existing memoized one, constructed from the given thread cuboid
    // + qual_bits_by_vis (in TlSigInterval format).
    // The returned ID is an owning reference (ownership count given by added_refcnt).
    template <bool IsMutate, typename ThreadInit>
    [[nodiscard]] nodepool::id<VisRecordListNode<IsMutate>> memoize_new_vis_record(
            const ThreadInit& thread_init, SyncvAccessInfo access, uint32_t added_refcnt)
    {
        nodepool::id<VisRecordListNode<IsMutate>> new_vis_id;
        auto& new_vis = alloc_vis_record<IsMutate>(thread_init, access, &new_vis_id);
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

    template <bool IsMutate>
    struct RemoveMemoizedCommand
    {
        const VisRecordListNode<IsMutate>* p_node;
    };

    // This removes the given node from the memoization table, but does not decrement the reference count or free it.
    // Recall that the memoization table does not own (reference count) the VisRecords contained.
    template <bool IsMutate>
    [[nodiscard]] nodepool::id<VisRecordListNode<IsMutate>> remove_memoized(
            const VisRecordListNode<IsMutate>* p_node)
    {
        CAMSPORK_REQUIRE(p_node, "unexpected null");
        CAMSPORK_REQUIRE(!p_node->is_forwarded(), "forwarding state VisRecord would not be memoized");

        RemoveMemoizedCommand<IsMutate> command{p_node};
        auto bucket_key = minimal_superset_interval(p_node->base_data.visibility_set);
        return for_buckets<IsMutate, BucketProcessType::Find>(bucket_key, command);
    }

    // Find and remove node in bucket.
    template <bool IsMutate>
    nodepool::id<VisRecordListNode<IsMutate>> process_bucket(nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head,
                                                             RemoveMemoizedCommand<IsMutate> command)
    {
        auto lambda = [this, command] (const VisRecord& record) {
            return equal(record, command.p_node->base_data);
        };
        nodepool::id<VisRecordListNode<IsMutate>>* p_id = bucket_search(p_bucket_head, lambda);
        if (p_id) {
            return remove_next_node(p_id);
        }
        else {
            return {};
        }
    }

    template <bool IsMutate>
    struct FindMemoizedCommand
    {
        const VisRecordListNode<IsMutate>* p_node;
    };

    template <bool IsMutate>
    nodepool::id<VisRecordListNode<IsMutate>> find_memoized(const VisRecordListNode<IsMutate>* p_node) const
    {
        CAMSPORK_REQUIRE(p_node, "unexpected null");
        CAMSPORK_REQUIRE(!p_node->is_forwarded(), "forwarding state VisRecord would not be memoized");

        FindMemoizedCommand<IsMutate> command{p_node};
        auto bucket_key = minimal_superset_interval(p_node->base_data.visibility_set);
        return const_cast<SyncvTable*>(this)->for_buckets<IsMutate, BucketProcessType::Find>(bucket_key, command);
    }

    template <bool IsMutate>
    nodepool::id<VisRecordListNode<IsMutate>> process_bucket(
            nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head,
            FindMemoizedCommand<IsMutate> command)
    {
        auto lambda = [this, command] (const VisRecord& record) {
            return equal(record, command.p_node->base_data);
        };
        nodepool::id<VisRecordListNode<IsMutate>>* p_id = bucket_search(p_bucket_head, lambda);
        if (p_id) {
            return *p_id;
        }
        else {
            return {};
        }
    }

    template <bool IsMutate>
    struct MemoizeOrForwardCommand
    {
        nodepool::id<VisRecordListNode<IsMutate>> input_id;
    };

    // Given an existing visibility record in the base state that's not in the memoization table, either
    //   * Add it to the memoization table, if it's unique. Return itself.
    //   * Put it in the forwarding state (discard existing state) and forward to equal already-memoized record.
    //     Return ID of memoized record.
    template <bool IsMutate>
    nodepool::id<VisRecordListNode<IsMutate>> memoize_or_forward(nodepool::id<VisRecordListNode<IsMutate>> id)
    {
        CAMSPORK_REQUIRE(id, "unexpected null");
        VisRecordListNode<IsMutate>& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "should have nonzero refcnt");
        CAMSPORK_REQUIRE(!node.is_forwarded(), "should not already be forwarded");
        CAMSPORK_REQUIRE(!node.camspork_next_id, "shouldn't be in any linked list (memoization bucket or forwarded?)");

        MemoizeOrForwardCommand<IsMutate> command{id};
        auto bucket_key = minimal_superset_interval(node.base_data.visibility_set);
        return for_buckets<IsMutate, BucketProcessType::Insert>(bucket_key, command);
    }

    template <bool IsMutate>
    nodepool::id<VisRecordListNode<IsMutate>> process_bucket(
            nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head,
            MemoizeOrForwardCommand<IsMutate> command)
    {
        VisRecordListNode<IsMutate>& input_node = get(command.input_id);
        VisRecord input_vis_record = input_node.base_data;

        auto lambda = [this, input_vis_record] (const VisRecord& record) {
            return equal(record, input_vis_record);
        };

        nodepool::id<VisRecordListNode<IsMutate>>* p_id = bucket_search(p_bucket_head, lambda);
        if (p_id) {
            // If equivalent memoized node found in bucket, forward input node to it.
            const nodepool::id fwd_id = *p_id;
            CAMSPORK_REQUIRE(fwd_id, "unexpected null");
            CAMSPORK_REQUIRE_CMP(fwd_id, !=, command.input_id, "Trying to memoize something already in the memoization table.");

            reset_vis_record_data(&input_node.base_data);
            input_node.camspork_next_id = fwd_id;
            input_node.base_data.forwarded_flag = 1;
            CAMSPORK_REQUIRE(input_node.is_forwarded(), "should now be in forwarding state");
            incref(fwd_id);  // Forwarding reference is owning.
            return fwd_id;
        }
        else {
            // Insert input node to memoization bucket. No refcnt changes needed for memoization.
            // IMPORTANT: this memoization is at the start of the bucket. This means if the caller of this function
            // is processing this bucket, the caller probably won't encounter this node. See process_buckets_for_sync.
            insert_next_node(p_bucket_head, command.input_id);
            return command.input_id;
        }

    }

    struct AugmentVisRecordCallback
    {
        const ThreadCuboid* p_cuboid;
        qual_bits_t L2_full_qual_bits;
        qual_bits_t L2_temporal_qual_bits;

        template <bool IsMutate>
        void operator() (SyncvTable& env, nodepool::id<VisRecordListNode<IsMutate>> vis_record_id)
        {
            auto& node = env.get(vis_record_id);
            CAMSPORK_REQUIRE(!node.is_forwarded(), "Unexpected modification of forwarding state VisRecord");
            p_cuboid->to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
                {
                    QualBitsByVis q_by_vis{};
                    const auto q_temporal = L2_temporal_qual_bits;
                    q_by_vis.array[get_low_bit_index(vis_flag_atomic_only)] = q_temporal;
                    q_by_vis.array[get_low_bit_index(vis_flag_temporal)] = q_temporal;
                    q_by_vis.array[get_low_bit_index(vis_flag_full)] = L2_full_qual_bits;
                    env.union_tl_sig_interval(&node.base_data, TlSigInterval{tid_lo, tid_hi, q_by_vis});
                }
            );
        }
    };

    template <typename Logger>
    struct FenceUpdateCommand
    {
        const ThreadCuboid* p_cuboid;
        bool transitive;
        qual_bits_t L1_qual_bits, L2_full_qual_bits, L2_temporal_qual_bits;
        Logger& logger;

        static constexpr bool enable_debug_printf = false;

        template <bool IsMutate>
        bool update_for_sync(SyncvTable& env, nodepool::id<VisRecordListNode<IsMutate>> vis_record_id) const
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

    const BarrierArriveState& get_const_barrier_arrive_state(pending_await_t info) const
    {
        const auto barrier_index = pending_await_barrier_index(info);
        const auto arrive_count = pending_await_arrive_count(info);
        const auto& map = barrier_states[barrier_index].arrive_states;
        auto it = map.find(arrive_count);
        CAMSPORK_REQUIRE(it != map.end(), "Missing BarrierArriveState");
        return it->second;
    }

    template <typename Logger>
    struct ArriveUpdateCommand
    {
        const ThreadCuboid* p_cuboid;
        bool transitive;
        qual_bits_t L1_qual_bits;
        std::vector<pending_await_t> pending_awaits;
        Logger& logger;

        static constexpr bool enable_debug_printf = false;

        ArriveUpdateCommand(ArriveUpdateCommand&&) = delete;

        template <bool IsMutate>
        bool update_for_sync(SyncvTable& env, nodepool::id<VisRecordListNode<IsMutate>> vis_record_id)
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
    template <bool IsMutate>
    void extend_barrier_arrive_state(
        nodepool::id<VisRecordListNode<IsMutate>> vis_record_id,
        pending_await_t info)
    {
        BarrierArriveState& state = get_barrier_arrive_state(info);
        nodepool::id<AssignmentRecordVisNode<IsMutate>> list_node_id{};
        auto& list_node = alloc_default_node(&list_node_id);
        list_node.vis_record_id = vis_record_id;
        incref(vis_record_id);
        if constexpr (IsMutate) {
            list_node.camspork_next_id = state.mutate_vis_records_head_id;
            state.mutate_vis_records_head_id = list_node_id;
        }
        else {
            list_node.camspork_next_id = state.read_vis_records_head_id;
            state.read_vis_records_head_id = list_node_id;
        }
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
            constexpr bool IsMutate = decltype(record_id)::value_type::is_mutate;
            while (record_id) {
                AssignmentRecordVisNode<IsMutate>& record_node = get(record_id);
                record_id = record_node.camspork_next_id;
                const nodepool::id<VisRecordListNode<IsMutate>> vis_record_id = record_node.vis_record_id;
                VisRecordListNode<IsMutate>& vis_node = get(vis_record_id);
                nodepool::id<PendingAwaitNode>* p_await_node = &vis_node.base_data.pending_awaits;
                if (vis_node.is_forwarded()) {
                    CAMSPORK_REQUIRE(!*p_await_node, "pending_awaits should be empty for forwarded VisRecord");
                }
                else {
                    // We will remove and re-memoize the base-state VisRecord.
                    const auto tmp = remove_memoized(&vis_node);
                    CAMSPORK_REQUIRE_CMP(tmp, ==, vis_record_id, "memoization broken?");
                    callback(*this, vis_record_id);

                    // Remove PendingAwaitNode.
                    bool found = false;
                    while (*p_await_node) {
                        PendingAwaitNode& node = get(*p_await_node);
                        if (node.await_id == await_info) {
                            remove_and_free_next_node(p_await_node);
                            found = true;
                            break;
                        }
                        p_await_node = &node.camspork_next_id;
                    }
                    CAMSPORK_REQUIRE(found, "Remove PendingAwaitNode failed");
                    const auto new_id = memoize_or_forward(vis_record_id);
                    logger.history_vis_record_change(*this, vis_record_id, new_id, false);
                }
                // Note, this could cause the newly-memoized VisRecord to be immediately destroyed.
                decref(vis_record_id);
                record_node.vis_record_id = {};  // to make things clearer in the debugger.
            }
        };

        nodepool::id<AssignmentRecordMutateNode>& mutate_id = p_state->mutate_vis_records_head_id;
        retire_list(mutate_id);
        extend_free_list(mutate_id);
        mutate_id = {};

        nodepool::id<AssignmentRecordReadNode>& read_id = p_state->read_vis_records_head_id;
        retire_list(read_id);
        extend_free_list(read_id);
        read_id = {};
    }

    // Big payoff for all this code: function that performs the effects of a synchronization statement with the given
    // sync type and given first/second visibility sets. This affects all visibility records whose visibility set
    // intersects with the first visibility set of the synchronization statement.
    //
    // The real entrypoints are the ones specialized for fence, arrive, await.
    template <typename Command>
    void update_vis_records_for_sync_impl(Command&& command)
    {
        // Only records with unordered visibility sets that intersect the first visibility set (V1)
        // can be updated by this sync. TODO this vocabulary will change.
        const TlSigBucketKey minimal_superset = command.p_cuboid->minimal_superset_interval();
        for_buckets<false, BucketProcessType::MapAll>(minimal_superset, command);
        for_buckets<true, BucketProcessType::MapAll>(minimal_superset, command);
    }

    template <bool IsMutate, typename Command>
    void process_bucket_for_sync_impl(nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head, Command&& command)
    {
        // The bucket update process for handling the effects of synchronization on visibility records is quite
        // risky actually. When we modify a visibility record, we temporarily remove it from the memoization bucket,
        // modify it, then attempt to re-insert it. This is fundamentally needed since the modification may
        // cause a duplicate to be created, or a node to be in the wrong bucket.
        //
        // However, this re-insertion may cause the memoization table to be modified unexpectedly.
        // We have to be very careful when traversing the bucket's linked list, and this also explains
        // the IntervalBucket::visitor_count value, if it still exists; (see for_buckets_impl).
        //
        // Note, everything will be left in an inconsistent state in case an exception is thrown.

        using node_id = nodepool::id<VisRecordListNode<IsMutate>>;
        node_id* p_id = p_bucket_head;

        // This might be really confusing. p_id is a pointer to a node ID.
        // It could be a pointer to the bucket (itself the ID of the head of the bucket node list) or it
        // could be the pointer to the camspork_next_id member of the PREVIOUS node (relative to current_node).
        while (node_id current_node_id = *p_id) {
            // Now temporarily remove the current node from the bucket linked list.
            // ("next_node" reflects the "pointer to previous node" viewpoint explained above).
            // *p_id will now be the ID of the node that formerly was after current_node, which (if not ID = 0)
            // is the node that we should process on the next iteration.
            const node_id modified_id = remove_next_node(p_id);
            VisRecordListNode<IsMutate>& current_node = get(modified_id);
            CAMSPORK_REQUIRE(!current_node.camspork_next_id, "Should have been removed from list.");
            CAMSPORK_REQUIRE(!current_node.is_forwarded(), "forwarding state memoized?");

            // Update the visibility record stored in the node.
            const bool syncs = command.update_for_sync(*this, modified_id);
            CAMSPORK_REQUIRE_CMP(p_id, !=, &current_node.camspork_next_id, "something happened");

            // This is where the node might get re-inserted to the memoization table.
            // *p_id might change value here again, but it's guaranteed p_id doesn't point inside &current_node.
            const auto new_node_id = memoize_or_forward(current_node_id);
            const bool debug_printf = command.enable_debug_printf;
            if (syncs) {
                command.logger.history_vis_record_change(*this, current_node_id, new_node_id, debug_printf);
            }

            // This part is dicey. We removed the node from the memoization table, then possibly re-inserted it,
            // either into another bucket, or at the head of this bucket. See the weird assert below.
            // Also see process_bucket(, MemoizeOrForwardCommand).
            // It's possible we re-inserted exactly into its old place, so we need to do some special logic
            // to avoid getting stuck in an infinite loop.
            if (*p_id == current_node_id) {
                // I'm fairly sure this is the only reason this branch should happen, due to how we insert nodes
                // only at the head of buckets. If this assert goes off, the code may still be correct;
                // this is just a warning-to-self to check that my mental model is correct.
                CAMSPORK_REQUIRE_CMP(p_id, ==, p_bucket_head, "see source code note above");
                p_id = &get(current_node_id).camspork_next_id;
            }
        }
    }

    // Augment all visibility records that synchronize with the first visibility set of the fence.
    template <typename Logger>
    void update_vis_records_for_fence(const ThreadCuboid& cuboid, const SyncvFence& fence, Logger& logger)
    {
        // Augment V_A, V_U, and V_O.
        FenceUpdateCommand<Logger> command{
                &cuboid, fence.transitive, fence.L1_qual_bits,
                fence.L2_full_qual_bits, fence.L2_temporal_qual_bits, logger};
        update_vis_records_for_sync_impl(command);
    }

    template <bool IsMutate, typename Logger>
    void process_bucket(
            nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head,
            const FenceUpdateCommand<Logger>& command)
    {
        process_bucket_for_sync_impl(p_bucket_head, command);
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

    template <bool IsMutate, typename Logger>
    void process_bucket(
            nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head,
            ArriveUpdateCommand<Logger>& command)
    {
        process_bucket_for_sync_impl(p_bucket_head, command);
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
    }

    template <typename Logger>
    void on_await(const ThreadCuboid& cuboid, const SyncvAwait& await, Logger&& logger)
    {
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
    template <bool IsMutate>
    nodepool::id<AssignmentRecordVisNode<IsMutate>> copy(nodepool::id<AssignmentRecordVisNode<IsMutate>> input_id)
    {
        nodepool::id<AssignmentRecordVisNode<IsMutate>> output_id{};
        if (input_id) {
            nodepool::id<AssignmentRecordVisNode<IsMutate>>* p_tail = &output_id;
            while (input_id) {
                const AssignmentRecordVisNode<IsMutate>& input_node = get(input_id);
                AssignmentRecordVisNode<IsMutate>& output_node = alloc_default_node(p_tail);
                p_tail = &output_node.camspork_next_id;
                input_id = input_node.camspork_next_id;

                nodepool::id<VisRecordListNode<IsMutate>> vis_record_id = input_node.vis_record_id;
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
        using VisRecordID = nodepool::id<VisRecordListNode<IsMutate>>;
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
            const VisRecordID new_vis_record_id = memoize_new_vis_record<IsMutate>(cuboid, access, vis_record_refcnt);
            logger.history_new_vis_record(*this, new_vis_record_id);
            new_vis_record_list[0] = new_vis_record_id;
        }
        else {
            cuboid.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi) {
                // We model the CPU as of 2025-10-01 as "[almost] all possible threads" [0, UINT32_MAX)
                // and if we pass that here, we will create 4 billion VisRecords.
                CAMSPORK_REQUIRE_CMP(tid_hi, <, UINT32_MAX, "Likely you meant to pass convergent_access_flag");
                for (uint32_t tid = tid_lo; tid < tid_hi; ++tid) {
                    const VisRecordID new_vis_record_id = memoize_new_vis_record<IsMutate>(
                        SingleThreadInit{tid},
                        access,
                        vis_record_refcnt
                    );
                    logger.history_new_vis_record(*this, new_vis_record_id);
                    new_vis_record_list.push_back(new_vis_record_id);
                }
            });
        }

        logger.excut_log_assignment_records(*this, input, new_vis_record_list,
                IsMutate ? ExcutMutateTag::Mutate : ExcutMutateTag::Read);

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
            nodepool::id<AssignmentRecordMutateNode> mutate_id = assignment_record.mutate_vis_records_head_id;
            while (mutate_id) {
                AssignmentRecordMutateNode& node = get(mutate_id);
                const VisRecord& mutate_record = remove_forwarding(&node.vis_record_id);
                logger.history_vis_record_checked(node.vis_record_id);  // Logs memoized (base state) ID.
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
                nodepool::id<AssignmentRecordReadNode> read_id = assignment_record.read_vis_records_head_id;
                while (read_id) {
                    AssignmentRecordReadNode& node = get(read_id);
                    const VisRecord& read_record = remove_forwarding(&node.vis_record_id);
                    logger.history_vis_record_checked(node.vis_record_id);  // Logs memoized (base state) ID.
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

        auto extend_vis_records = [&] (nodepool::id<AssignmentRecordVisNode<IsMutate>>* p_list_head)
        {
            for (const VisRecordID vis_record_id : new_vis_record_list) {
                nodepool::id<AssignmentRecordVisNode<IsMutate>> new_node_id;
                AssignmentRecordVisNode<IsMutate>& node = alloc_default_node(&new_node_id);
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
    template <bool IsMutate>
    void remove_duplicates(nodepool::id<AssignmentRecordVisNode<IsMutate>>* p_list_head)
    {
        using node_id = nodepool::id<AssignmentRecordVisNode<IsMutate>>;

        // Remove forwarding (unique ID iff unique record), and clear tmp_is_duplicate to 0.
        // Importantly, we are clearing this flag for base-state VisRecord, not forwarded ones.
        for (node_id id = *p_list_head; id; ) {
            AssignmentRecordVisNode<IsMutate>& node = get(id);
            uint8_t& is_duplicate = remove_forwarding(&node.vis_record_id).tmp_is_duplicate;
            is_duplicate = 0;
            id = node.camspork_next_id;
        }

        // Remove duplicates, using tmp_is_duplicate to recognize duplicates.
        node_id* p_read_id = p_list_head;
        while (node_id next_id = *p_read_id) {
            AssignmentRecordVisNode<IsMutate>& next_node = get(next_id);
            VisRecordListNode<IsMutate>& vis_record_node = get(next_node.vis_record_id);
            CAMSPORK_REQUIRE(!vis_record_node.is_forwarded(), "should have resolved forwarding above");
            uint8_t& is_duplicate = vis_record_node.base_data.tmp_is_duplicate;

            if (is_duplicate) {
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
                is_duplicate = 1;
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
        nodepool::id<AssignmentRecordReadNode> id = record.read_vis_records_head_id;
        while (id) {
            const AssignmentRecordReadNode& node = get(id);
            out->push_back(node.vis_record_id.id_bits);
            id = node.camspork_next_id;
        }
    }

    // Get info for a given visibility record.
    template <bool IsMutate>
    void debug_get_vis_record_data(nodepool::id<VisRecordListNode<IsMutate>> node_id, VisRecordDebugData* out) const
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
        std::tuple<
            RefcntDebug<AssignmentRecord>,
            RefcntDebug<TlSigIntervalListNode>,
            RefcntDebug<PendingAwaitNode>,
            RefcntDebug<ReadVisRecordListNode>,
            RefcntDebug<MutateVisRecordListNode>,
            RefcntDebug<AssignmentRecordReadNode>,
            RefcntDebug<AssignmentRecordMutateNode>>
        debug_refcnts(
            *this, *this, *this, *this, *this, *this, *this
        );

        if (false) {
            fprintf(stderr, "AssignmentRecord: %u\n", debug_get_pool<AssignmentRecord>().size());
            fprintf(stderr, "TlSigIntervalListNode: %u\n", debug_get_pool<TlSigIntervalListNode>().size());
            fprintf(stderr, "PendingAwaitNode: %u\n", debug_get_pool<PendingAwaitNode>().size());
            fprintf(stderr, "ReadVisRecordListNode: %u\n", debug_get_pool<ReadVisRecordListNode>().size());
            fprintf(stderr, "MutateVisRecordListNode: %u\n", debug_get_pool<MutateVisRecordListNode>().size());
            fprintf(stderr, "AssignmentRecordReadNode: %u\n", debug_get_pool<AssignmentRecordReadNode>().size());
            fprintf(stderr, "AssignmentRecordMutateNode: %u\n", debug_get_pool<AssignmentRecordMutateNode>().size());
        }

        auto check_all_refcnts = [&]
        {
            std::get<0>(debug_refcnts).check_refcnts(*this);
            std::get<1>(debug_refcnts).check_refcnts(*this);
            std::get<2>(debug_refcnts).check_refcnts(*this);
            std::get<3>(debug_refcnts).check_refcnts(*this);
            std::get<4>(debug_refcnts).check_refcnts(*this);
            std::get<5>(debug_refcnts).check_refcnts(*this);
            std::get<6>(debug_refcnts).check_refcnts(*this);
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

            const nodepool::id<AssignmentRecordMutateNode> mutate_id = record.mutate_vis_records_head_id;
            process_assignment_record_list(mutate_id);
            const nodepool::id<AssignmentRecordReadNode> read_id = record.read_vis_records_head_id;
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
                const nodepool::id<AssignmentRecordMutateNode> mutate_id = pair.second.mutate_vis_records_head_id;
                process_assignment_record_list(mutate_id);
                const nodepool::id<AssignmentRecordReadNode> read_id = pair.second.read_vis_records_head_id;
                process_assignment_record_list(read_id);
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
                nodepool::id<PendingAwaitNode> await_node_id = node.base_data.pending_awaits;
                while (await_node_id) {
                    const PendingAwaitNode& await_node = get(await_node_id);
                    await_node_id = await_node.camspork_next_id;
                    const BarrierArriveState& state = get_const_barrier_arrive_state(await_node.await_id);
                    constexpr bool IsMutate = node.is_mutate;
                    nodepool::id<AssignmentRecordVisNode<IsMutate>> record_node_id;
                    if constexpr (node.is_mutate) {
                        record_node_id = state.mutate_vis_records_head_id;
                    }
                    else {
                        record_node_id = state.read_vis_records_head_id;
                    }
                    while (1) {
                        CAMSPORK_REQUIRE(record_node_id, "Missing VisRecord reference in BarrierArriveState");
                        const auto& record_node = get(record_node_id);
                        if (record_node.vis_record_id == id) {
                            break;
                        }
                        record_node_id = record_node.camspork_next_id;
                    }
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

        process_all_vis_records(nodepool::id<ReadVisRecordListNode>{});
        process_all_vis_records(nodepool::id<MutateVisRecordListNode>{});

        // Check that reference counts are correct.
        // For node types without refcnt, the refcnt should just be 0 or 1 (unique ownership).
        check_all_refcnts();

        // Memoization Validation
        // A VisRecord should be in the memoization table iff it's alive and in the base state.

        // (VisRecord in memoization table -> alive and in base state)
        // We also check that no empty IntervalBucket(s) left behind (besides the top level bucket)
        // and that the tree state is consistent (correct back pointer to parent, correct non-empty child counts).
        auto validate_bucket_linked_list = [this] (auto id)
        {
            while (id) {
                // VisRecordListNode<IsMutate>
                const auto& node = get(id);
                CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "all memoized VisRecord should have nonzero refcnt");
                CAMSPORK_REQUIRE(!node.is_forwarded(), "forwarding state VisRecord should not be memoized");
                id = node.camspork_next_id;
            }
        };

        auto validate_child_buckets = [this, validate_bucket_linked_list] (const auto& bucket, auto validate)
        {
            CAMSPORK_REQUIRE_CMP(bucket.visitor_count, ==, 0, "Should always be 0 outside for_buckets<...>(...) otherwise the bucket is immortal.");
            uint64_t real_nonempty_child_flags = 0;

            for (uint32_t child_index = 0; child_index < bucket.child_count; ++child_index) {
                const auto& child_bucket_id_or_ptr = bucket.child_interval_buckets[child_index];
                if (child_bucket_id_or_ptr) {
                    CAMSPORK_REQUIRE_CMP(child_index, <, 64, "need to use more than 64 bit flags");
                    real_nonempty_child_flags |= uint64_t(1) << child_index;
                }
                if constexpr (bucket.bucket_level != 1) {
                    if (child_bucket_id_or_ptr) {
                        auto& child_bucket = *child_bucket_id_or_ptr;
                        CAMSPORK_REQUIRE_CMP(child_bucket.p_parent, ==, &bucket, "wrong parent ptr");
                        CAMSPORK_REQUIRE_CMP(child_bucket.child_index_in_parent, ==, child_index, "wrong child_index");
                        CAMSPORK_REQUIRE(!interval_bucket_is_empty(child_bucket), "should have been deallocated");
                        validate(child_bucket, validate);
                    }
                }
                else {
                    // Level 1 bucket holds level 0 buckets directly (rather than with an extra wrapper
                    // IntervalBucket<0> structure).
                    validate_bucket_linked_list(child_bucket_id_or_ptr);
                }
            }

            CAMSPORK_REQUIRE_CMP(bucket.nonempty_child_flags, ==, real_nonempty_child_flags, "wrong child flags");
            validate_bucket_linked_list(bucket.bucket);
        };
        validate_child_buckets(read_top_level_bucket, validate_child_buckets);
        validate_bucket_linked_list(read_top_level_bucket.bucket);
        validate_child_buckets(mutate_top_level_bucket, validate_child_buckets);
        validate_bucket_linked_list(mutate_top_level_bucket.bucket);


        // (VisRecord in memoization table <- alive and in base state)
        // Each VisRecord should be able to find itself in the table; if we fail, it could be because we
        // forgot to memoize it, or something is wrong with the bucket search or equality function.
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

                try {
                    CAMSPORK_REQUIRE_CMP(id, ==, find_memoized(&node), "memoization lookup is buggy");
                }
                catch (...) {
                    VisRecord record = node.base_data;
                    nodepool::id<TlSigIntervalListNode> interval_id = record.visibility_set;
                    while (interval_id) {
                        const TlSigIntervalListNode& node = get(interval_id);
                        interval_id = node.camspork_next_id;
                        const TlSigInterval data = node.data;
                        fprintf(stderr, "[%u, %u, %u, %u, %u, %u]\n",
                                data.tid_lo,
                                data.tid_hi,
                                data.qual_bits_by_vis.array[0],
                                data.qual_bits_by_vis.array[1],
                                data.qual_bits_by_vis.array[2],
                                data.qual_bits_by_vis.array[3]
                        );
                    }
                    static_assert(num_vis_flags == 4);
                    throw;
                }
            }
        };
        memoize_self_check(nodepool::id<ReadVisRecordListNode>{});
        memoize_self_check(nodepool::id<MutateVisRecordListNode>{});

        // Check correct BarrierArriveState.
        // A base state VisRecord is pointed to by BarrierArriveState iff it contains a corresponding pending await.
        // BarrierArriveState may also point to forwarding state VisRecord.
        // The other half of this checking is in process_vis_record_impl.
        auto check_BarrierArriveState_VisRecords = [&] (auto record_node_id, pending_await_t expected_await_id)
        {
            while (record_node_id) {
                constexpr bool IsMutate = decltype(record_node_id)::value_type::is_mutate;
                const AssignmentRecordVisNode<IsMutate>& record_node = get(record_node_id);
                record_node_id = record_node.camspork_next_id;
                const nodepool::id<VisRecordListNode<IsMutate>> vis_record_id = record_node.vis_record_id;
                const VisRecordListNode<IsMutate>& vis_record = get(vis_record_id);
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
                const nodepool::id<AssignmentRecordMutateNode> mutate_id = pair.second.mutate_vis_records_head_id;
                check_BarrierArriveState_VisRecords(mutate_id, info);
                const nodepool::id<AssignmentRecordReadNode> read_id = pair.second.read_vis_records_head_id;
                check_BarrierArriveState_VisRecords(read_id, info);
            }
        }
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
            const SyncvTable& env, assignment_record_id* p_id, const VisRecordList& new_vis_record_list,
            ExcutMutateTag mutate_tag)
    {
        if (p_excut_actions) {
            nodepool::id<AssignmentRecord> asn_id{p_id->node_id};
            _excut_log_assignment_record_impl(env, asn_id, new_vis_record_list, idx_for_single, mutate_tag);
        }
    }

    template <typename VisRecordList>
    void excut_log_assignment_records(
            const SyncvTable& env, AssignmentRecordWindow window, const VisRecordList& new_vis_record_list,
            ExcutMutateTag mutate_tag)
    {
        if (p_excut_actions) {
            std::vector<extent_t> idx(window.end_outer_extent - window.begin_outer_extent);
            _excut_recurse_log_window(env, window, idx, 0, 0, new_vis_record_list, mutate_tag);
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
            size_t dim_idx, size_t partial_linear_offset, const VisRecordList& new_vis_record_list,
            ExcutMutateTag mutate_tag)
    {
        if (dim_idx >= idx.size()) {
            CAMSPORK_REQUIRE_CMP(idx.size(), ==, dim_idx, "overshot");
            nodepool::id<AssignmentRecord> asn_id{window.base[partial_linear_offset].node_id};
            _excut_log_assignment_record_impl(env, asn_id, new_vis_record_list, idx, mutate_tag);
        }
        else {
            const extent_t outer_c = window.begin_outer_extent[dim_idx];
            const extent_t offset_c = window.begin_offset[dim_idx];
            const extent_t end_c = offset_c + window.begin_inner_extent[dim_idx];

            for (extent_t i = offset_c; i < end_c; ++i) {
                idx[dim_idx] = i;
                const auto new_linear_offset = partial_linear_offset * outer_c + i;
                _excut_recurse_log_window(
                        env, window, idx, dim_idx+1, new_linear_offset, new_vis_record_list, mutate_tag);
            }
        }
    }

    template <typename VisRecordList>
    void _excut_log_assignment_record_impl(
            const SyncvTable& env,
            nodepool::id<AssignmentRecord> asn_id,
            const VisRecordList& new_vis_record_list,
            std::vector<extent_t> idx,
            ExcutMutateTag mutate_tag)
    {
        constexpr bool IsMutate = VisRecordList::value_type::value_type::is_mutate;

        // Log top-level assignment record ID, name+idxs of access,
        // and remember to update this with the changed ID later.
        {
            auto p_info = std::make_unique<ExcutSyncEnvAccess>();
            p_info->id_before = asn_id.id_bits;
            p_info->id_after = 0;  // See excut_update_assignment_record_ids
            p_info->name = var_str_name;
            p_info->idx = std::move(idx);
            p_info->mutate_tag = mutate_tag;
            actions_to_update.push_back(p_info.get());
            p_excut_actions->push_back(std::move(p_info));
        }

        // Log existing VisRecords, tagged as WAR, WAW, or RAW, depending on the relation
        // between the prior VisRecords and the current SyncEnvAccess action.
        if (asn_id) {
            const AssignmentRecord& asn_record = env.get(asn_id);
            if constexpr (IsMutate) {
                nodepool::id<AssignmentRecordReadNode> read_id = asn_record.read_vis_records_head_id;
                while (read_id) {
                    const AssignmentRecordReadNode& asn_node = env.get(read_id);
                    read_id = asn_node.camspork_next_id;
                    _excut_log_vis_record(env, asn_node.vis_record_id, ExcutMutateTag::WAR);
                }
            }
            nodepool::id<AssignmentRecordMutateNode> mutate_id = asn_record.mutate_vis_records_head_id;
            while (mutate_id) {
                const AssignmentRecordMutateNode& asn_node = env.get(mutate_id);
                mutate_id = asn_node.camspork_next_id;
                _excut_log_vis_record(env, asn_node.vis_record_id, IsMutate ? ExcutMutateTag::WAW : ExcutMutateTag::RAW);
            }
        }

        // Log new VisRecord
        for (nodepool::id<VisRecordListNode<IsMutate>> new_vis_id : new_vis_record_list) {
            _excut_log_vis_record(env, new_vis_id, mutate_tag);
        }
    }

    template <bool IsMutate>
    void _excut_log_vis_record(
            const SyncvTable& env, nodepool::id<VisRecordListNode<IsMutate>> id, ExcutMutateTag mutate_tag)
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

    template <bool IsMutate>
    void history_new_vis_record(SyncvTable& env, nodepool::id<VisRecordListNode<IsMutate>> node_id)
    {
        if (p_history_log) {
            history_log_vis_record_id history_id(node_id);
            p_history_log->log_syncv_new_vis_record(history_id, _get_history_vis_record_data(env, node_id));
        }
    }

    template <bool IsMutate>
    void history_vis_record_change(
            SyncvTable& env,
            nodepool::id<VisRecordListNode<IsMutate>> old_id,
            nodepool::id<VisRecordListNode<IsMutate>> new_id,
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

    template <bool IsMutate>
    void history_vis_record_checked(nodepool::id<VisRecordListNode<IsMutate>> id)
    {
        if (p_history_log) {
            p_history_log->log_syncv_vis_record_checked(history_log_vis_record_id(id), IsMutate);
        }
    }

    template <bool IsMutate>
    void history_vis_record_error(
            nodepool::id<VisRecordListNode<IsMutate>> id, TlSig fail_tl_sig)
    {
        if (p_history_log) {
            p_history_log->log_syncv_vis_record_error(history_log_vis_record_id(id), fail_tl_sig);
        }
    }
  private:
    template <bool IsMutate>
    LoggedVisRecordData _get_history_vis_record_data(SyncvTable& env, nodepool::id<VisRecordListNode<IsMutate>> node_id)
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
#if 1
    fprintf(stderr, "bucket_search_call_counter = %llu\n", (long long unsigned)table->bucket_search_call_counter);
    fprintf(stderr, "bucket_search_iter_counter = %llu\n", (long long unsigned)table->bucket_search_iter_counter);
    fprintf(stderr, "ratio = %.1f\n",
            (double)table->bucket_search_iter_counter / (double)table->bucket_search_call_counter);
    fprintf(stderr, "Read   VisRecord capacity = %llu\n",
            (long long unsigned)table->debug_get_pool<VisRecordListNode<false>>().size());
    fprintf(stderr, "Mutate VisRecord capacity = %llu\n",
            (long long unsigned)table->debug_get_pool<VisRecordListNode<true>>().size());
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



void debug_get_read_vis_record_data(const SyncvTable* table, uint32_t id, VisRecordDebugData* out)
{
    table->debug_get_vis_record_data(nodepool::id<VisRecordListNode<false>>{id}, out);
}

void debug_get_mutate_vis_record_data(const SyncvTable* table, uint32_t id, VisRecordDebugData* out)
{
    table->debug_get_vis_record_data(nodepool::id<VisRecordListNode<true>>{id}, out);
}

void debug_validate_state(SyncvTable* table, size_t input_count, const SyncvDebugValidateInput* p_inputs)
{
    table->debug_validate_state(input_count, p_inputs);
}

void debug_pre_delete_check(SyncvTable* table)
{
    // This could go off due to the user not free-ing their own stuff, which I consider valid
    // (if suboptimal) usage, since deleting SyncvTable cleans up all physical memory allocations anyway.
    const bool all_empty = interval_bucket_is_empty(table->read_top_level_bucket)
            && interval_bucket_is_empty(table->mutate_top_level_bucket);
    assert(table->failed || all_empty);
}


}  // end namespace
