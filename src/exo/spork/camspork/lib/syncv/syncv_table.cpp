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
#include <utility>

#include "tl_sig.hpp"
#include "../util/bit_util.hpp"
#include "../util/cuboid_util.hpp"
#include "../util/node_pool.hpp"
#include "../util/require.hpp"

// Maybe replace later
#include <unordered_map>
#include <unordered_set>
template <typename K, typename V> using Map = std::unordered_map<K, V>;
template <typename V> using Set = std::unordered_set<V>;

namespace camspork
{

namespace
{

using refcnt_t = uint32_t;

// We attach a "linked list" of pending awaits to a non-forwarded VisRecord.
// However, two different linked lists may share the same tail, hence the "tree" name and the refcnt.
struct PendingAwaitTreeNode
{
    // Owning reference to the next node in the list (may be shared tail).
    nodepool::id<PendingAwaitTreeNode> camspork_next_id;
    uint32_t refcnt;
    pending_await_t await_id;

    refcnt_t get_refcnt() const
    {
        return refcnt;
    }
};

// We encode a visibility set as a list of sorted, minimal
// tl-sig intervals. The intervals are sorted in that
// a.tid_hi <= b.tid_lo for a before b in the list, and the list
// is minimal in that no more intervals are used than needed
// (mostly by merging adjacent intervals with the same bitfield).
//
// Given V_A \superset V_U \superset V_O [atomic-only, unordered, ordered],
// we have that
//     V_O = union(val: TlSigInterval where val.vis_level() >= vis_level_ordered)
//     V_U = union(val: TlSigInterval where val.vis_level() >= vis_level_unordered)
//     V_A = union(val: TlSigInterval where val.vis_level() >= vis_level_atomic_only)
struct TlSigIntervalListNode
{
    TlSigInterval data;
    nodepool::id<TlSigIntervalListNode> camspork_next_id;

    refcnt_t get_refcnt() const
    {
        return 1;  // Replace if refcnt member added
    }
};

static_assert(sizeof(TlSigIntervalListNode) == 16, "Check that you meant to change this perf-critical struct");

struct VisRecord
{
    // Owning reference to singly-linked list.
    nodepool::id<TlSigIntervalListNode> visibility_set;

    // Owning reference to tree node.
    nodepool::id<PendingAwaitTreeNode> pending_awaits;

    uint8_t original_qual_tl;

    // This has nothing to do with the main purpose of the struct; only needed for assignment_record_remove_duplicates.
    // This should be in AssignmentRecordVisNode conceptually, but that would waste 4 bytes.
    uint8_t tmp_is_duplicate;
};

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

    // Empty visibility set is never valid.
    // We will use that to indicate forwarding.
    // TODO this has to change. Will be valid soon.
    bool is_forwarded() const
    {
        return !base_data.visibility_set;
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

    // TODO update this
    nodepool::id<AssignmentRecordMutateNode> mutate_vis_records_head_id{0};

    // Zero or more read visibility records.
    nodepool::id<AssignmentRecordReadNode> read_vis_records_head_id{0};

    // See assignment_record_remove_duplicates.
    uint32_t last_augment_counter_bits : 16;

    refcnt_t get_refcnt() const
    {
        return refcnt;
    }
};

struct BarrierState
{
    uint32_t arrive_count;
    uint32_t await_count;
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

    // Count of non-null pointers in child_interval_buckets.
    uint32_t nonempty_child_count = 0;

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
        nonempty_child_count = other.nonempty_child_count;
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

    // Count of non-null single_thread_buckets.
    uint32_t nonempty_child_count = 0;

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
        nonempty_child_count = other.nonempty_child_count;
        bucket = other.bucket;
        visitor_count = other.visitor_count;
        CAMSPORK_REQUIRE_CMP(visitor_count, ==, 0, "Not sure copying is OK while being traversed.");
    }
};


// TODO consider further sub-bucketing, e.g. by qual_bits.
// If we do this, we have to be careful not to double-consider items when
// moving between buckets. e.g. bucket by lowest to highest qual_bit count.


template <bool IsMutate, uint32_t BucketLevel>
bool interval_bucket_is_empty(const IntervalBucket<IsMutate, BucketLevel>& bucket) noexcept
{
    return bucket.nonempty_child_count == 0 && !bucket.bucket && !bucket.visitor_count;
}

// De-allocate the given bucket if it's empty and not the top-level bucket.
// We presume that the bucket is owned by its parent (unique_ptr tree).
//
// We do not make any modifications to the parent except for nulling out the pointer.
// In particular, we don't change nonempty_child_count, or handle deleting the parent
// if it too is now empty.
template <bool IsMutate, uint32_t BucketLevel>
void delete_interval_bucket_if_empty(IntervalBucket<IsMutate, BucketLevel>* p) noexcept
{
    if (interval_bucket_is_empty(*p)) {
        for (const auto& child : p->child_interval_buckets) {
            CAMSPORK_REQUIRE(!child, "nonempty_child_count was wrong.");
        }

        if constexpr (BucketLevel < bucket_level_count - 1) {
            // Parent pointer should be correct.
            IntervalBucket<IsMutate, BucketLevel + 1>* p_parent = p->p_parent;
            CAMSPORK_REQUIRE(p_parent, "missing parent ptr");
            const uint32_t child_index = p->child_index_in_parent;
            CAMSPORK_REQUIRE_CMP(child_index, <, p_parent->child_count, "child_index out-of-range");
            CAMSPORK_REQUIRE_CMP(p_parent->nonempty_child_count, >, 0, "should have been deallocated");

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
    uint64_t augment_counter = 0;     // Number of fence+arrive

    // Memory pool state.
    uintptr_t original_memory_budget = 0;
    uintptr_t current_memory_budget = 0;
    std::tuple<
        nodepool::Pool<AssignmentRecord>,
        nodepool::Pool<TlSigIntervalListNode>,
        nodepool::Pool<PendingAwaitTreeNode>,
        nodepool::Pool<ReadVisRecordListNode>,
        nodepool::Pool<MutateVisRecordListNode>,
        nodepool::Pool<AssignmentRecordReadNode>,
        nodepool::Pool<AssignmentRecordMutateNode>> pool_tuple;

    // Barrier state.
    // The Nth bit is 1 if N is allocated as a barrier ID.
    uint64_t live_barrier_bits[max_live_barriers / 64] = {0};
    BarrierState barrier_states[max_live_barriers];

    // Memoization table state (requires special deep copy support).
    IntervalBucket<false, bucket_level_count - 1> read_top_level_bucket;
    IntervalBucket<true, bucket_level_count - 1> mutate_top_level_bucket;



    // *** Memory Pool Allocators; Linked List Manipulation ***
    // See nodepool for more info.



    template <typename ListNode>
    ListNode& alloc_default_node(nodepool::id<ListNode>* out_id) noexcept
    {
        using TypedPool = nodepool::Pool<ListNode>;
        TypedPool& pool = std::get<TypedPool>(pool_tuple);
        return pool.alloc_default_node(&current_memory_budget, out_id);
    }

    template <typename ListNode>
    void extend_free_list(nodepool::id<ListNode> head_id) noexcept
    {
        using TypedPool = nodepool::Pool<ListNode>;
        TypedPool& pool = std::get<TypedPool>(pool_tuple);
        pool.extend_free_list(head_id);
    }

    template <typename ListNode>
    void insert_next_node(nodepool::id<ListNode>* p_insert_after, nodepool::id<ListNode> insert_me) noexcept
    {
        using TypedPool = nodepool::Pool<ListNode>;
        TypedPool& pool = std::get<TypedPool>(pool_tuple);
        return pool.insert_next_node(p_insert_after, insert_me);
    }

    // Given a pointer to the camspork_next_id member of a node in a list,
    // but don't add it to the free chain: the node is returned to the caller.
    template <typename ListNode>
    [[nodiscard]] nodepool::id<ListNode> remove_next_node(nodepool::id<ListNode>* p_id) noexcept
    {
        using TypedPool = nodepool::Pool<ListNode>;
        TypedPool& pool = std::get<TypedPool>(pool_tuple);
        return pool.remove_next_node(p_id);
    }

    // Given a pointer to the camspork_next_id member of a node in a list,
    // remove the next node of the list and add its memory to the free chain.
    // This shouldn't be used if the ListNode itself owns stuff.
    template <typename ListNode>
    void remove_and_free_next_node(nodepool::id<ListNode>* p_id) noexcept
    {
        nodepool::id<ListNode> victim_id = remove_next_node(p_id);
        CAMSPORK_REQUIRE(!get(victim_id).camspork_next_id, "Should have been removed from list");
        extend_free_list(victim_id);               // so this free only adds 1 node to free chain.
    }

    template <typename ListNode>
    ListNode& get(nodepool::id<ListNode> id) noexcept
    {
        using TypedPool = nodepool::Pool<ListNode>;
        return std::get<TypedPool>(pool_tuple).get(id);
    }

    template <typename ListNode>
    const ListNode& get(nodepool::id<ListNode> id) const noexcept
    {
        using TypedPool = nodepool::Pool<ListNode>;
        return std::get<TypedPool>(pool_tuple).get(id);
    }

    template <typename ListNode>
    uint32_t debug_node_pool_size() const noexcept
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
    void incref(nodepool::id<AssignmentRecord> id) noexcept
    {
        AssignmentRecord& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "should not have started with 0 refcnt");
        node.refcnt++;
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "reference count overflow");
    }

    // Decrement reference count of assignment record.
    void decref(nodepool::id<AssignmentRecord> id, uint32_t nref = 1) noexcept
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
    void incref(nodepool::id<VisRecordListNode<IsMutate>> id) noexcept
    {
        VisRecordListNode<IsMutate>& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "should not have started with 0 refcnt");
        node.refcnt++;
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "reference count overflow");
    }

    // Decrement reference count of visibility record,
    // and handle necessary free-ing in case of 0 refcnt.
    template <bool IsMutate>
    void decref(nodepool::id<VisRecordListNode<IsMutate>> id) noexcept
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

    void incref(nodepool::id<PendingAwaitTreeNode> id) noexcept
    {
        PendingAwaitTreeNode& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "should not have started with 0 refcnt");
        node.refcnt++;
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "reference count overflow");
    }

    void decref(nodepool::id<PendingAwaitTreeNode> id) noexcept
    {
        PendingAwaitTreeNode& node = get(id);
        if (0 != --node.refcnt) {
            return;
        }
        auto victim_id = remove_next_node(&id);
        extend_free_list(victim_id);
        if (id) {
            decref(id);
        }
    }

    void reset_vis_record_data(VisRecord* p_data) noexcept
    {
        static_assert(sizeof(*p_data) == 12, "update me");
        p_data->original_qual_tl = ~0;
        extend_free_list(p_data->visibility_set);
        p_data->visibility_set = {};
        if (auto& id = p_data->pending_awaits) {
            decref(id);
            id = nodepool::id<PendingAwaitTreeNode>{};
        }
    }

    template <bool IsMutate>
    void free_single_vis_record(nodepool::id<VisRecordListNode<IsMutate>> id) noexcept
    {
        CAMSPORK_REQUIRE(id, "unexpected 0 id");
        VisRecordListNode<IsMutate>& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, ==, 0, "unexpected 0 refcnt");
        reset_vis_record_data(&node.base_data);
        node.camspork_next_id = {};  // Avoid freeing entire list.
        extend_free_list(id);
    }

    template <bool IsMutate>
    void assignment_record_remove_vis_records(nodepool::id<AssignmentRecordVisNode<IsMutate>> head_id)
    {
        // Decref visibility records
        auto id = head_id;
        while (id) {
            AssignmentRecordVisNode<IsMutate>& node = get(id);
            decref(node.vis_record_id);
            id = node.camspork_next_id;
        }

        // Free physical storage of linked list
        extend_free_list(head_id);
    }

    void reset_assignment_record(AssignmentRecord* p_record) noexcept
    {
        assignment_record_remove_vis_records(p_record->mutate_vis_records_head_id);
        p_record->mutate_vis_records_head_id = {};

        assignment_record_remove_vis_records(p_record->read_vis_records_head_id);
        p_record->read_vis_records_head_id = {};

        p_record->last_augment_counter_bits = 0;
    }



    // *** Operations on Visibility Records ***
    // If the visibility record is modified, you need to be careful to update the memoization table.



    // Allocate a new visibility record.
    // This will later need to be added to the memoization table.
    // This must be kept in-sync with equal(const VisRecord& a, const ThreadCuboid& cuboid, uint32_t bitfield).
    template <bool IsMutate>
    nodepool::id<VisRecordListNode<IsMutate>> alloc_visibility_record(
            const ThreadCuboid& cuboid, uint32_t bitfield)
    {
        nodepool::id<VisRecordListNode<IsMutate>> vis_record_id;
        VisRecordListNode<IsMutate>& vis_record = alloc_default_node(&vis_record_id);
        vis_record.refcnt = 1;
        vis_record.base_data.original_qual_tl = TlSigInterval::get_unique_qual_tl(bitfield);

        // Initialize visibility set = linked list of intervals generated from the initial thread cuboid.
        nodepool::id<TlSigIntervalListNode>* p_node_id = &vis_record.base_data.visibility_set;
        cuboid.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
        {
            TlSigIntervalListNode& tl_sigs_node = alloc_default_node(p_node_id);
            tl_sigs_node.data = TlSigInterval{tid_lo, tid_hi, bitfield};
            p_node_id = &tl_sigs_node.camspork_next_id;
        });

        return vis_record_id;
    }

    // Union tl_sig interval into the visibility set(s).
    // Caller will have to make changes to the memoization table afterwards.
    // Recall V_U (unordered visibility set) and V_O (ordered visibility set) has V_O \subseteq V_U
    // and vis_level() == vis_level_ordered on an interval means to include it in both V_U and V_O.
    void union_tl_sig_interval(VisRecord* p, TlSigInterval input)
    {
        CAMSPORK_REQUIRE_CMP(input.vis_level(), ==, vis_level_ordered, "Only support vis_level_ordered for now");

        // Non-empty input check (cartesian product of non-empty thread interval and non-empty qual-tl set).
        CAMSPORK_REQUIRE_CMP(input.tid_hi, >, input.tid_lo, "non-empty input check");
        CAMSPORK_REQUIRE_CMP(0, !=, input.qual_bits(), "non-empty input check");
        using node_id = nodepool::id<TlSigIntervalListNode>;

        // Note, assignment of 0, 1, 3, allows for bitwise-or to "promote" to vis_level_ordered.
        static_assert(vis_level_atomic_only == 0);
        static_assert(vis_level_unordered == 1);
        static_assert(vis_level_ordered == 3);

        // Visibility set must not be created empty (or VisRecord is in forwarding state). See alloc_visibility_record.
        // TODO fix this.
        assert(p->visibility_set);

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
            new_interval.bitfield = input.bitfield;
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
            // since only the overlapped portion should have its bitfield modified.
            //
            // 1st interval: keeps original bits, left of intersection.
            // 2nd interval: bitfield augmented, footprint of intersection.
            // 3rd interval: keeps original bits, right of intersection.
            TlSigIntervalListNode& next_node = get(original_next_node_id);
            const TlSigInterval original_data = next_node.data;
            const uint32_t intersect_tid_lo = std::max(original_data.tid_lo, input.tid_lo);
            const uint32_t intersect_tid_hi = std::min(original_data.tid_hi, input.tid_hi);
            const uint32_t added_bits = input.bitfield & ~original_data.bitfield;
            const bool change_needed = added_bits != 0 && (intersect_tid_lo < intersect_tid_hi);

            // Possibly add 1st interval.
            if (change_needed && original_data.tid_lo < intersect_tid_lo) {
                node_id new_node_id{};
                TlSigIntervalListNode& new_node = alloc_default_node(&new_node_id);
                new_node.data.tid_lo = original_data.tid_lo;
                new_node.data.tid_hi = intersect_tid_lo;
                new_node.data.bitfield = original_data.bitfield;
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
                next_node.data.bitfield |= added_bits;

                // Possibly add 3rd interval, insert after 2nd interval.
                if (intersect_tid_hi < original_data.tid_hi) {
                    node_id new_node_id{};
                    TlSigIntervalListNode& new_node = alloc_default_node(&new_node_id);
                    new_node.data.tid_lo = intersect_tid_hi;
                    new_node.data.tid_hi = original_data.tid_hi;
                    new_node.data.bitfield = original_data.bitfield;
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

            if (current.tid_hi == next.tid_lo && current.bitfield == next.bitfield) {
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
          && (first.tid_hi < second.tid_lo || first.bitfield != second.bitfield);
    }

    // Check if visibility records are equal.
    bool equal(const VisRecord& a, const VisRecord& b) const
    {
        static_assert(sizeof(a) == 12, "Update me");

        // Must not have empty visibility set (forwarding state passed?)
        // TODO fix me
        assert(a.visibility_set);
        assert(b.visibility_set);

        // Check equal original qual-tl.
        if (a.original_qual_tl != b.original_qual_tl) {
            return false;
        }

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
        // NB false negative shouldn't happen but won't mess things up too badly.
        // i.e. it'll be OK if the lists are equal, but didn't have the same ID.
        return a.pending_awaits == b.pending_awaits;
    }

    // Check if a visibility record matches what would have been constructed
    // from the given tl_sig interval set by alloc_visibility_record.
    bool equal(const VisRecord& a, const ThreadCuboid& cuboid, uint32_t bitfield)
    {
        static_assert(sizeof(a) == 12, "Update me");

        if (a.pending_awaits) {
            return false;
        }

        if (TlSigInterval::qual_bits(bitfield) != (1u << a.original_qual_tl)) {
            return false;
        }

        // Check if existing intervals equal those that would be generated from ThreadCuboid.
        nodepool::id<TlSigIntervalListNode> node_id = a.visibility_set;
        bool equal = true;
        cuboid.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
        {
            if (!node_id) {
                equal = false;
            }
            else {
                const TlSigIntervalListNode& node = get(node_id);
                equal &= node.data == TlSigInterval{tid_lo, tid_hi, bitfield};
                node_id = node.camspork_next_id;
            }
        });
        return equal && !node_id;
    }

    template <bool OrderedOnly, bool Transitive>
    bool visible_to_impl(const VisRecord& vis_record, TlSigInterval access_set)
    {
        // Must not have empty visibility set (forwarding state passed?)
        // TODO fix me
        assert(vis_record.visibility_set);

        const uint32_t qual_bits_mask = Transitive ? ~uint32_t(0) : uint32_t(1) << vis_record.original_qual_tl;

        nodepool::id<TlSigIntervalListNode> id = vis_record.visibility_set;
        while (id) {
            const TlSigIntervalListNode& current_node = get(id);
            id = current_node.camspork_next_id;
            CAMSPORK_REQUIRE(!id || valid_adjacent(current_node.data, get(id).data), "invalid adjacent intervals");

            if (current_node.data.intersects(access_set, qual_bits_mask)) {
                if (!OrderedOnly || current_node.data.vis_level() == vis_level_ordered) {
                    return true;
                }
            }
        }
        return false;
    }

    // Check if the visibility record is visible-to an access with the given tl_sig access set.
    bool visible_to(const VisRecord& vis_record, TlSigInterval accessor_set)
    {
        // TODO remove
        return visible_to_impl<true, true>(vis_record, accessor_set);
    }

    // Check if the visibility record synchronizes-with a synchronization statement with the given first visibility set.
    template <bool Transitive>
    bool synchronizes_with(const VisRecord& vis_record, TlSigInterval V1)
    {
        // TODO remove
        return visible_to_impl<false, Transitive>(vis_record, V1);
    }

    template <bool Transitive>
    bool synchronizes_with(const VisRecord& vis_record, const ThreadCuboid& cuboid, uint32_t bitfield)
    {
        // Check for any intersections between TlSigIntervals generated by ThreadCuboid + bitfield
        // and those stored in the VisRecord. We check the unordered visibility set.
        bool intersects = false;
        nodepool::id<TlSigIntervalListNode> node_id = vis_record.visibility_set;
        cuboid.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
        {
            while (true) {
                if (!node_id) {
                    return;  // Exit lambda
                }
                const TlSigIntervalListNode& node = get(node_id);
                TlSigInterval vis_set_interval = node.data;
                TlSigInterval gen_interval{tid_lo, tid_hi, bitfield};
                if (vis_set_interval.vis_level() >= vis_level_unordered) {
                    intersects |= vis_set_interval.intersects(gen_interval);
                }
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

    bool visible_to(const VisRecord& vis_record, const ThreadCuboid& cuboid, uint32_t bitfield)
    {
        // Check that all threads generated by the ThreadCuboid have at least one qual-tl in common with
        // an overlapping interval in the ordered visibility set of the vis_record.
        const uint32_t want_qual_bits = TlSigInterval::qual_bits(bitfield);
        bool all_visible = true;
        cuboid.to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
        {
            nodepool::id<TlSigIntervalListNode> node_id = vis_record.visibility_set;
            // Interval of threads [missing_threads_lo, tid_hi) not yet found to overlap with
            // a valid TlSigInterval in the visibility set.
            uint32_t missing_threads_lo = tid_lo;

            while (node_id) {
                // Search TlSigIntervals in the VisRecord, skipping any that don't have
                // the required vis_level or with no qual-tl in common.
                const TlSigIntervalListNode& node = get(node_id);
                node_id = node.camspork_next_id;
                const TlSigInterval data = node.data;
                if (data.vis_level() < vis_level_ordered) {
                    continue;
                }
                if (0 == (data.qual_bits() & want_qual_bits)) {
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
            all_visible &= local_visible;
        });
        return all_visible;
    }



    // *** Barrier ID Allocation ***
    // For now barrier_id::data only stores the barrier ID number + 1, but this could change.



    uint32_t get_barrier_id(const barrier_id* bar)
    {
        CAMSPORK_REQUIRE_CMP(bar->data, !=, 0, "null barrier");
        const auto id = (bar->data - 1);
        CAMSPORK_REQUIRE_CMP(id, <, max_live_barriers, "max_live_barriers limit exceeded");
        return uint32_t(id);
    }

    void set_barrier_id(barrier_id* bar, uint32_t id)
    {
        CAMSPORK_REQUIRE_CMP(id, <, max_live_barriers, "max_live_barriers limit exceeded");
        bar->data = id + 1;
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
                set_barrier_id(&barriers[num_allocated], barrier_index);
                live_barrier_bits[word_index] = ~negated_bits;
                barrier_states[barrier_index] = {};
                if (++num_allocated >= N) {
                    return;
                }
            }
        }

        CAMSPORK_REQUIRE(false, "Exceeded implementation limit (max number of barriers per program)");
    }

    void free_barriers(size_t N, barrier_id* barriers)
    {
        for (size_t i = 0; i < N; ++i) {
            if (!barriers[i]) {
                continue;
            }
            const auto barrier_id = get_barrier_id(&barriers[i]);
            const BarrierState& state = barrier_states[barrier_id];
            if (state.arrive_count != state.await_count) {
                std::string message =
                    "Arrive count (" + std::to_string(state.arrive_count) + ") != Await count ("
                    + std::to_string(state.await_count) + ")";
                throw SyncvCheckFail{std::move(message)};
            }

            uint64_t& word = live_barrier_bits[barrier_id / 64u];
            const uint64_t bit = uint64_t(1) << (barrier_id & 63u);
            CAMSPORK_REQUIRE_CMP((word & bit), !=, 0, "Barrier ID was not allocated");
            word &= ~bit;
            barriers[i].data = 0;
        }
    }



    // *** Memoization ***



    // Get the smallest possible tl_sig interval that is a superset of
    // the given visibility set (ignore vis_level); assumes non-empty input set.
    // This is needed to index into the correct bucket (the smallest one possible containing the visibility set).
    // Note, at time of writing the qual_bits aren't used for bucketing, but maybe they should be.
    TlSigInterval minimal_superset_interval(nodepool::id<TlSigIntervalListNode> id) const
    {
        CAMSPORK_REQUIRE(id, "null");
        const TlSigIntervalListNode* p_node = &get(id);
        p_node->data.assert_valid();
        TlSigInterval ret = p_node->data;

        while (1) {
            id = p_node->camspork_next_id;
            if (!id) {
                ret.bitfield = ret.qual_bits();
                ret.assert_valid();
                return ret;
            }

            p_node = &get(id);
            CAMSPORK_REQUIRE_CMP(p_node->data.tid_lo, >=, ret.tid_hi, "Not sorted?");
            p_node->data.assert_valid();
            ret.tid_hi = p_node->data.tid_hi;
            ret.bitfield |= p_node->data.bitfield;
        }
    }

    // "Remove forwarding"; replace ID of forwarded visibility record
    // with ID of base visibility record that the original record forwarded to.
    // Assumes the given ID is intended as an owning ID.
    // Return record data.
    template <bool IsMutate>
    VisRecord& remove_forwarding(nodepool::id<VisRecordListNode<IsMutate>>* p_id) noexcept
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
    // NB could easily modify this to return the ID as well, but not needed for now.
    template <bool IsMutate>
    VisRecord const_resolve_forwarding(nodepool::id<VisRecordListNode<IsMutate>> id) const noexcept
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
    nodepool::id<VisRecordListNode<IsMutate>> for_buckets(TlSigInterval minimal_superset, Command&& command)
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
                    p_bucket->nonempty_child_count++;

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
                    CAMSPORK_REQUIRE_CMP(p_bucket->nonempty_child_count, >, 0, "should have been deleted");
                    p_bucket->nonempty_child_count--;
                }
                throw;
            }

            // Child bucket may have been deallocated for being empty.
            if (!child_ref) {
                CAMSPORK_REQUIRE_CMP(p_bucket->nonempty_child_count, >, 0, "should have been deleted");
                p_bucket->nonempty_child_count--;
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
        using node_id = nodepool::id<VisRecordListNode<IsMutate>>;
        node_id* p_id = p_bucket_head;

        for (node_id id; (id = *p_id); ) {
            VisRecordListNode<IsMutate>& node = get(id);
            CAMSPORK_REQUIRE(!node.is_forwarded(), "Should not be in memoization table.");

            if (lambda(node.base_data)) {
                return p_id;
            }

            p_id = &node.camspork_next_id;
        }
        return nullptr;
    }

    struct NewVisRecordCommand
    {
        const ThreadCuboid* p_cuboid;
        uint32_t bitfield;
        uint32_t added_refcnt;
    };

    // Add a new visibility record, or return existing memoized one, constructed from the given thread cuboid
    // + bitfield (in TlSigInterval format).
    // The returned ID is an owning reference (ownership count given by added_refcnt).
    template <bool IsMutate>
    [[nodiscard]] nodepool::id<VisRecordListNode<IsMutate>> memoize_new_vis_record(const ThreadCuboid& cuboid,
                                                                                   uint32_t bitfield,
                                                                                   uint32_t added_refcnt)
    {
        CAMSPORK_REQUIRE_CMP(added_refcnt, !=, 0, "cannot memoize w/ zero refcnt");

        NewVisRecordCommand command{&cuboid, bitfield, added_refcnt};
        const TlSigInterval key = cuboid.minimal_superset_interval(bitfield);
        nodepool::id<VisRecordListNode<IsMutate>> id = for_buckets<IsMutate, BucketProcessType::Insert>(key, command);
        CAMSPORK_REQUIRE(id, "BucketProcessType::Insert search should not have given null");
        CAMSPORK_REQUIRE(equal(const_resolve_forwarding(id), cuboid, bitfield), "memoization didn't work");
        return id;
    }

    template <bool IsMutate>
    nodepool::id<VisRecordListNode<IsMutate>> process_bucket(nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head,
                                                             NewVisRecordCommand command)
    {
        auto lambda = [this, command] (const VisRecord& record) {
            return equal(record, *command.p_cuboid, command.bitfield);
        };
        nodepool::id<VisRecordListNode<IsMutate>>* p_found_id = bucket_search(p_bucket_head, lambda);

        nodepool::id<VisRecordListNode<IsMutate>> new_id;

        if (p_found_id) {
            // Existing memoized entry found.
            new_id = *p_found_id;
            CAMSPORK_REQUIRE(new_id, "unexpected null from memoization table");
            get(new_id).refcnt += command.added_refcnt;
        }
        else {
            // Add memoized base visibility set entry to bucket of memoization table.
            new_id = alloc_visibility_record<IsMutate>(*command.p_cuboid, command.bitfield);
            CAMSPORK_REQUIRE(!get(new_id).camspork_next_id, "should have been initialized to null");
            get(new_id).refcnt = command.added_refcnt;
            CAMSPORK_REQUIRE(!get(new_id).is_forwarded(), "should not be initialized in forwarding state");
            insert_next_node(p_bucket_head, new_id);
        }
        CAMSPORK_REQUIRE(new_id, "unexpected null");
        return new_id;
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
            const VisRecordListNode<IsMutate>* p_node) noexcept
    {
        CAMSPORK_REQUIRE(p_node, "unexpected null");
        CAMSPORK_REQUIRE_CMP(p_node->refcnt, ==, 0, "should still be memoized if kept alive elsewhere");
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
    nodepool::id<VisRecordListNode<IsMutate>> find_memoized(const VisRecordListNode<IsMutate>* p_node) const noexcept
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
            FindMemoizedCommand<IsMutate> command) noexcept
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
    //   * Add it to the memoization table, if it's unique.
    //   * Put it in the forwarding state (discard existing state) and forward to equal already-memoized record.
    template <bool IsMutate>
    void memoize_or_forward(nodepool::id<VisRecordListNode<IsMutate>> id)
    {
        CAMSPORK_REQUIRE(id, "unexpected null");
        VisRecordListNode<IsMutate>& node = get(id);
        CAMSPORK_REQUIRE_CMP(node.refcnt, !=, 0, "should have nonzero refcnt");
        CAMSPORK_REQUIRE(!node.is_forwarded(), "should not already be forwarded");
        CAMSPORK_REQUIRE(!node.camspork_next_id, "shouldn't be in any linked list (memoization bucket or forwarded?)");

        MemoizeOrForwardCommand<IsMutate> command{id};
        auto bucket_key = minimal_superset_interval(node.base_data.visibility_set);
        for_buckets<IsMutate, BucketProcessType::Insert>(bucket_key, command);
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

            reset_vis_record_data(&input_node.base_data);  // Clear data to put visibility record into forwarding state.
            input_node.camspork_next_id = fwd_id;
            CAMSPORK_REQUIRE(input_node.is_forwarded(), "should now be in forwarding state");
            incref(fwd_id);  // Forwarding reference is owning.
            // fprintf(stderr, "FWD %u -> %u [IsMutate=%i]\n", command.input_id.id_bits, fwd_id.id_bits, IsMutate);
        }
        else {
            // Insert input node to memoization bucket. No refcnt changes needed for memoization.
            // IMPORTANT: this memoization is at the start of the bucket. This means if the caller of this function
            // is processing this bucket, the caller probably won't encounter this node. See process_buckets_for_sync.
            insert_next_node(p_bucket_head, command.input_id);
        }

        return command.input_id;
    }

    template <bool Transitive>
    struct FenceUpdateCommand
    {
        const ThreadCuboid* p_cuboid;
        uint32_t L1_bitfield, L2_full_bitfield, L2_temporal_bitfield;

        template <bool IsMutate>
        void update_for_sync(SyncvTable& env, VisRecord* p_record) const
        {
            if (env.synchronizes_with<Transitive>(*p_record, *p_cuboid, L1_bitfield)) {
                p_cuboid->to_intervals([&] (uint32_t tid_lo, uint32_t tid_hi)
                    {
                        const auto bitfield = IsMutate ? L2_full_bitfield : L2_temporal_bitfield;
                        env.union_tl_sig_interval(p_record, TlSigInterval{tid_lo, tid_hi, bitfield});
                    }
                );
            }
        };

        TlSigInterval minimal_superset_interval() const
        {
            return p_cuboid->minimal_superset_interval(L1_bitfield);
        }
    };

    template <bool Transitive>
    struct ArriveUpdateCommand
    {
        const ThreadCuboid* p_cuboid;
        uint32_t L1_bitfield;
        std::vector<pending_await_t> pending_awaits;

        // Used internally, to avoid creating redundant PendingAwaitTreeNode.
        // This also helps memoization ... VisRecord with equivalent sets of pending awaits
        // will hopefull also use equivalent node ID for the pending_awaits list.
        Map<nodepool::id<PendingAwaitTreeNode>, nodepool::id<PendingAwaitTreeNode>> node_id_map;

        template <bool IsMutate>
        void update_for_sync(SyncvTable& env, VisRecord* p_record)
        {
            if (env.synchronizes_with<Transitive>(*p_record, *p_cuboid, L1_bitfield)) {
                nodepool::id<PendingAwaitTreeNode> old_await_node = p_record->pending_awaits;
                nodepool::id<PendingAwaitTreeNode>* p_new_await_node = &node_id_map[old_await_node];
                if (const auto new_await_node = *p_new_await_node) {
                    // Recycle the list created in the else stmt.
                    env.incref(new_await_node);
                    env.decref(old_await_node);  // critically after incref, in case the two nodes are the same.
                    p_record->pending_awaits = new_await_node;
                }
                else {
                    *p_new_await_node = old_await_node;
                    // Add nodes to the head of the list.
                    // Don't have to manipulate refcnt here, actually.
                    for (auto iter = pending_awaits.rbegin(); iter != pending_awaits.rend(); ++iter) {
                        const auto tmp_id = *p_new_await_node;
                        PendingAwaitTreeNode& node = env.alloc_default_node(p_new_await_node);
                        node.camspork_next_id = tmp_id;
                        node.refcnt = 1;
                        node.await_id = *iter;
                    }
                    p_record->pending_awaits = *p_new_await_node;
                }
            }
        }

        TlSigInterval minimal_superset_interval() const
        {
            return p_cuboid->minimal_superset_interval(L1_bitfield);
        }
    };

    // Big payoff for all this code: function that performs the effects of a synchronization statement with the given
    // sync type and given first/second visibility sets. This affects all visibility records whose visibility set
    // intersects with the first visibility set of the synchronization statement.
    //
    // The real entrypoints are the ones specialized for fence, arrive, await.
    template <typename Command>
    void update_vis_records_for_sync_impl(Command&& command)
    {
        // Only visibility sets that intersect the first visibility set (V1) can be updated by this sync.
        const TlSigInterval minimal_superset = command.minimal_superset_interval();
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
            VisRecordListNode<IsMutate>& current_node = get(remove_next_node(p_id));
            CAMSPORK_REQUIRE(!current_node.camspork_next_id, "Should have been removed from list.");
            CAMSPORK_REQUIRE(!current_node.is_forwarded(), "forwarding state memoized?");

            // Update the visibility record stored in the node.
            command.template update_for_sync<IsMutate>(*this, &current_node.base_data);
            CAMSPORK_REQUIRE_CMP(p_id, !=, &current_node.camspork_next_id, "something happened");

            // This is where the node might get re-inserted to the memoization table.
            // *p_id might change value here again, but it's guaranteed p_id doesn't point inside &current_node.
            memoize_or_forward(current_node_id);

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
    void update_vis_records_for_fence(bool transitive, const ThreadCuboid& cuboid,
            uint32_t L1_bitfield, uint32_t L2_full_bitfield, uint32_t L2_temporal_bitfield)
    {
        L2_full_bitfield |= TlSigInterval::ordered_bits;  // Augment V_A, V_U, and V_O.
        L2_temporal_bitfield |= TlSigInterval::ordered_bits;
        if (transitive) {
            FenceUpdateCommand<true> command{&cuboid, L1_bitfield, L2_full_bitfield, L2_temporal_bitfield};
            update_vis_records_for_sync_impl(command);
        }
        else {
            FenceUpdateCommand<false> command{&cuboid, L1_bitfield, L2_full_bitfield, L2_temporal_bitfield};
            update_vis_records_for_sync_impl(command);
        }
    }

    template <bool IsMutate, bool Transitive>
    void process_bucket(
            nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head,
            const FenceUpdateCommand<Transitive>& command)
    {
        process_bucket_for_sync_impl(p_bucket_head, command);
    }

    // Save await_id into all visibility records that synchronize with the first visibility set of the fence.
    void update_vis_records_for_arrive(
            bool transitive,
            const ThreadCuboid& cuboid,
            uint32_t L1_bitfield,
            std::vector<pending_await_t> pending_awaits)
    {
        if (transitive) {
            ArriveUpdateCommand<true> command{&cuboid, L1_bitfield, std::move(pending_awaits), {}};
            update_vis_records_for_sync_impl(command);
        }
        else {
            ArriveUpdateCommand<false> command{&cuboid, L1_bitfield, std::move(pending_awaits), {}};
            update_vis_records_for_sync_impl(command);
        }
    }

    template <bool IsMutate, bool Transitive>
    void process_bucket(
            nodepool::id<VisRecordListNode<IsMutate>>* p_bucket_head,
            ArriveUpdateCommand<Transitive>& command)
    {
        process_bucket_for_sync_impl(p_bucket_head, command);
    }



    // *** Synchronization State Update ***



    void on_fence(bool transitive, const ThreadCuboid& cuboid,
            uint32_t L1_bitfield, uint32_t L2_full_bitfield, uint32_t L2_temporal_bitfield)
    {
        augment_counter++;
        update_vis_records_for_fence(transitive, cuboid, L1_bitfield, L2_full_bitfield, L2_temporal_bitfield);
    }

    void on_arrive(barrier_id* home_barrier, uint32_t barrier_count, barrier_id* all_barriers,
            bool transitive, const ThreadCuboid& cuboid, uint32_t L1_bitfield)
    {
        const auto home_barrier_id = get_barrier_id(home_barrier);
        BarrierState& state = barrier_states[home_barrier_id];
        const auto await_id = pack_pending_await(home_barrier_id, state.arrive_count);

        std::vector<pending_await_t> pending_awaits(barrier_count);
        for (uint32_t i = 0; i < barrier_count; ++i) {
            pending_awaits[i] = pack_pending_await(get_barrier_id(&all_barriers[i]), state.arrive_count);
        }
        state.arrive_count++;

        update_vis_records_for_arrive(transitive, cuboid, L1_bitfield, std::move(pending_awaits));
    }

    void on_await(barrier_id* bar)
    {
        const auto barrier_id = get_barrier_id(bar);
        BarrierState& state = barrier_states[barrier_id];

        state.await_count++;

        assert(state.arrive_count >= state.await_count);  // TODO should not be assertion

        augment_counter++;
        CAMSPORK_REQUIRE(0, "Implement me");
        // update_vis_records_for_await(V1, V2_full, V2_temporal, await_id);
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
        assignment_record.last_augment_counter_bits = old.last_augment_counter_bits;
        assignment_record.mutate_vis_records_head_id = copy(old.mutate_vis_records_head_id);
        assignment_record.read_vis_records_head_id = copy(old.read_vis_records_head_id);
        return assignment_record;
    }


    template <bool IsMutate, bool UpdateRecords, typename Input>
    void checked_on_access_impl(
            Input input,
            const ThreadCuboid& cuboid,
            uint32_t bitfield)
    {
        using node_id = nodepool::id<AssignmentRecord>;

        // If the input is a window, take a census of all assignment record IDs in the input window.
        static constexpr bool IsWindow = std::is_same_v<decltype(input), AssignmentRecordWindow>;
        using CensusMap = Map<node_id, AssignmentRecordCensusEntry>;
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
                        census[id].count++;
                    }
                }
            );
        }
        else {
            census[0].first = node_id{input->node_id};  // Where decltype(input) is assignment_record_id*
            census[0].second.count = 1;
        }

        auto check = [&] (node_id id)
        {
            if (!id) {
                return;
            }
            const AssignmentRecord& assignment_record = get(id);

            // Check against previous mutate visibility records
            nodepool::id<AssignmentRecordMutateNode> mutate_id = assignment_record.mutate_vis_records_head_id;
            while (mutate_id) {
                AssignmentRecordMutateNode& node = get(mutate_id);
                const VisRecord& mutate_record = remove_forwarding(&node.vis_record_id);
                if (!visible_to(mutate_record, cuboid, bitfield)) {
                    throw SyncvCheckFail{IsMutate ? "WAW Hazard" : "RAW Hazard"};
                }
                mutate_id = node.camspork_next_id;
            }

            // If the access is a mutate, also check against the list of previous read visibility records.
            if constexpr (IsMutate) {
                nodepool::id<AssignmentRecordReadNode> read_id = assignment_record.read_vis_records_head_id;
                while (read_id) {
                    AssignmentRecordReadNode& node = get(read_id);
                    const VisRecord& read_record = remove_forwarding(&node.vis_record_id);
                    if (!visible_to(read_record, cuboid, bitfield)) {
                        throw SyncvCheckFail{"WAR Hazard"};
                    }
                    read_id = node.camspork_next_id;
                }
            }
        };

        auto copy_on_write_update = [&] (
                node_id old_id,
                AssignmentRecordCensusEntry& entry,
                nodepool::id<VisRecordListNode<IsMutate>> new_vis_record_id)
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
                // Clear out assignment record on write and add the single mutate visibility record.
                // TODO this will change for atomic operations.
                reset_assignment_record(&assignment_record);
                AssignmentRecordMutateNode& node = alloc_default_node(&assignment_record.mutate_vis_records_head_id);
                node.vis_record_id = new_vis_record_id;
                assignment_record.last_augment_counter_bits = augment_counter;
            }
            else {
                // Add the new visibility record to the list of read visibility records.
                nodepool::id<AssignmentRecordReadNode> read_id;
                AssignmentRecordReadNode& read_node = alloc_default_node(&read_id);
                read_node.vis_record_id = new_vis_record_id;
                insert_next_node(&assignment_record.read_vis_records_head_id, read_id);

                // If we leave things as-is, read vis records may build up indefinitely for variables that are written
                // once and read many times. We fix this by removing duplicates; however, this is really expensive,
                // so we only do it once after each fence or await event (synchronization is when memoization kicks
                // in to potentially allow us to recognize duplicates due to duplicated IDs).
                const auto old_bits = assignment_record.last_augment_counter_bits;
                assignment_record.last_augment_counter_bits = augment_counter;
                if (old_bits != assignment_record.last_augment_counter_bits) {
                    // This could fail if the bits of augment_counter overflow exactly.
                    // However, this is unlikely, and is only a performance issue if so (we fail to remove duplicates).
                    assignment_record_remove_duplicates(&assignment_record);
                }
            }
        };

        // We will memoize the new visibility record once.
        nodepool::id<VisRecordListNode<IsMutate>> vis_record_id{};
        if (UpdateRecords && !census.empty()) {
            const uint32_t initial_refcnt = uint32_t(census.size());
            // fprintf(stderr, "INITIAL_REFCNT %u\n", initial_refcnt);
            vis_record_id = memoize_new_vis_record<IsMutate>(cuboid, bitfield, initial_refcnt);
        }

        // Check & update all distinct assignment records once.
        for (auto& pair : census) {
            check(pair.first);
            if constexpr (UpdateRecords) {
                copy_on_write_update(pair.first, pair.second, vis_record_id);
            }
        }

        // Write out new assignment record IDs. Reference counting is already taken care of.
        if constexpr (!UpdateRecords) {
            CAMSPORK_REQUIRE(!vis_record_id, "Fix !UpdateRecords code path to not leak vis_record_id");
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
        }
        else {
            input->node_id = census[0].second.new_node_id.id_bits;  // Where decltype(input) is assignment_record_id*
        }
    }

    void on_r(assignment_record_id* p_record, const ThreadCuboid& cuboid, uint32_t bitfield)
    {
        if (no_checking_counter == 0) {
            checked_on_access_impl<false, true>(p_record, cuboid, bitfield);
        }
    }

    void on_rw(assignment_record_id* p_record, const ThreadCuboid& cuboid, uint32_t bitfield)
    {
        if (no_checking_counter == 0) {
            checked_on_access_impl<true, true>(p_record, cuboid, bitfield);
        }
    }

    void on_r(AssignmentRecordWindow window, const ThreadCuboid& cuboid, uint32_t bitfield)
    {
        if (no_checking_counter == 0) {
            checked_on_access_impl<false, true>(window, cuboid, bitfield);
        }
    }

    void on_rw(AssignmentRecordWindow window, const ThreadCuboid& cuboid, uint32_t bitfield)
    {
        if (no_checking_counter == 0) {
            checked_on_access_impl<true, true>(window, cuboid, bitfield);
        }
    }

    void on_check_free(AssignmentRecordWindow window, const ThreadCuboid& cuboid, uint32_t bitfield)
    {
        if (no_checking_counter == 0) {
            checked_on_access_impl<true, false>(window, cuboid, bitfield);
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

    // Resolve forwarding and remove duplicate read visibility records.
    // Removing forwarding causes two equivalent read visibility records to have identical IDs
    // (both referring to the shared entry in the memoization table).
    void assignment_record_remove_duplicates(AssignmentRecord* p_assignment_record)
    {
        using node_id = nodepool::id<AssignmentRecordReadNode>;

        // Remove forwarding (unique ID iff unique record), and clear tmp_is_duplicate to 0.
        for (node_id id = p_assignment_record->read_vis_records_head_id; id; ) {
            AssignmentRecordReadNode& node = get(id);
            uint8_t& is_duplicate = remove_forwarding(&node.vis_record_id).tmp_is_duplicate;
            is_duplicate = 0;
            id = node.camspork_next_id;
        }

        // Remove duplicates, using tmp_is_duplicate to recognize duplicates.
        node_id* p_read_id = &p_assignment_record->read_vis_records_head_id;
        while (node_id next_id = *p_read_id) {
            AssignmentRecordReadNode& next_node = get(next_id);
            ReadVisRecordListNode& vis_record_node = get(next_node.vis_record_id);
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
    void debug_get_vis_record_data(uint32_t id, VisRecordDebugData* out) const
    {
        CAMSPORK_REQUIRE(id, "unexpected null");
        const VisRecord record = const_resolve_forwarding(nodepool::id<VisRecordListNode<IsMutate>>{id});

        out->original_qual_tl = record.original_qual_tl;

        out->visibility_set.clear();
        for (nodepool::id<TlSigIntervalListNode> node_id = record.visibility_set; node_id;) {
            const TlSigIntervalListNode& node = get(node_id);
            out->visibility_set.push_back(node.data);
            node_id = node.camspork_next_id;
        }

        out->pending_await_list.clear();
        for (nodepool::id<PendingAwaitTreeNode> node_id = record.pending_awaits; node_id; ) {
            const PendingAwaitTreeNode& node = get(node_id);
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
                const refcnt_t tested_refcnt = self.get(id).get_refcnt();
                const refcnt_t expected_refcnt = refcnts[id.node_index()];
                const bool is_free = free_node_ids.count(id);
                if (is_free) {
                    CAMSPORK_REQUIRE_CMP(expected_refcnt, ==, 0, "node on free list is referenced");
                }
                else {
                    CAMSPORK_REQUIRE_CMP(expected_refcnt, ==, tested_refcnt, "wrong refcnt");
                }
            }
        }
    };

    // Massive function that verifies that the current state is legal.
    // This only works if all of the user's arrays of assignment_record_id have been passed.
    void debug_validate_state(size_t input_count, const SyncvDebugValidateInput* p_inputs) const
    {
        std::tuple<
            RefcntDebug<AssignmentRecord>,
            RefcntDebug<TlSigIntervalListNode>,
            RefcntDebug<PendingAwaitTreeNode>,
            RefcntDebug<ReadVisRecordListNode>,
            RefcntDebug<MutateVisRecordListNode>,
            RefcntDebug<AssignmentRecordReadNode>,
            RefcntDebug<AssignmentRecordMutateNode>>
        debug_refcnts(
            *this, *this, *this, *this, *this, *this, *this
        );

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

        auto process_assignment_record = [&] (nodepool::id<AssignmentRecord> id, auto recurse)
        {
            const bool first_time = record_owning(id);
            if (!first_time) {
                return;
            }
            const AssignmentRecord& record = get(id);

            nodepool::id<AssignmentRecordMutateNode> mutate_id = record.mutate_vis_records_head_id;
            while (mutate_id) {
                const AssignmentRecordMutateNode& mutate_node = get(mutate_id);
                CAMSPORK_REQUIRE(mutate_node.vis_record_id, "unexpected null mutate_node");
                record_owning(mutate_id);
                record_owning(mutate_node.vis_record_id);
                mutate_id = mutate_node.camspork_next_id;
            }

            nodepool::id<AssignmentRecordReadNode> read_id = record.read_vis_records_head_id;
            while (read_id) {
                const AssignmentRecordReadNode& read_node = get(read_id);
                CAMSPORK_REQUIRE(read_node.vis_record_id, "unexpected null read_node");
                record_owning(read_id);
                record_owning(read_node.vis_record_id);
                read_id = read_node.camspork_next_id;
            }
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

        // This handles references between PendingAwaitTreeNode
        auto on_PendingAwaitTreeNode = [&] (nodepool::id<PendingAwaitTreeNode> id, auto recurse)
        {
            if (!id) {
                return;
            }
            if (!record_owning(id)) {
                // Not the first time, don't re-scan.
            }
            id = get(id).camspork_next_id;
            recurse(id, recurse);
        };

        // Count ownership references from live VisRecordListNode objects to other objects:
        //   * TlSigIntervalListNode
        //   * PendingAwaitTreeNode
        //   * forwarded-to VisRecordListNodes
        // Furthermore we validate that the encoding for the visibility set is correct.
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
                    CAMSPORK_REQUIRE_CMP(this_node.data.tid_hi, >, this_node.data.tid_lo, "invalid interval");
                    CAMSPORK_REQUIRE_CMP(this_node.data.qual_bits(), !=, 0, "invalid empty qual-tl set");

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

                on_PendingAwaitTreeNode(node.base_data.pending_awaits, on_PendingAwaitTreeNode);
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
            uint32_t real_nonempty_child_count = 0;

            for (uint32_t child_index = 0; child_index < bucket.child_count; ++child_index) {
                const auto& child_bucket_id_or_ptr = bucket.child_interval_buckets[child_index];
                real_nonempty_child_count += child_bucket_id_or_ptr ? 1u : 0u;
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

            CAMSPORK_REQUIRE_CMP(bucket.nonempty_child_count, ==, real_nonempty_child_count, "wrong child count");
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
                        fprintf(stderr, "[%u, %u, %u, %u]\n",
                                data.tid_lo, data.tid_hi, data.qual_bits(), data.vis_level());
                    }
                    throw;
                }
            }
        };
        memoize_self_check(nodepool::id<ReadVisRecordListNode>{});
        memoize_self_check(nodepool::id<MutateVisRecordListNode>{});
    }
};



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
    delete table;
}

void SyncvTableDeleter::operator() (SyncvTable* victim) const
{
    delete victim;
}

void on_r(SyncvTable* table, assignment_record_id* p_record, const ThreadCuboid& cuboid, uint32_t bitfield)
{
    INTERFACE_PROLOGUE(table)
    table->on_r(p_record, cuboid, bitfield);
    INTERFACE_EPILOGUE(table)
}

void on_rw(SyncvTable* table, assignment_record_id* p_record, const ThreadCuboid& cuboid, uint32_t bitfield)
{
    INTERFACE_PROLOGUE(table)
    table->on_rw(p_record, cuboid, bitfield);
    INTERFACE_EPILOGUE(table)
}

void on_r(SyncvTable* table, AssignmentRecordWindow window, const ThreadCuboid& cuboid, uint32_t bitfield)
{
    INTERFACE_PROLOGUE(table)
    table->on_r(window, cuboid, bitfield);
    INTERFACE_EPILOGUE(table)
}

void on_rw(SyncvTable* table, AssignmentRecordWindow window, const ThreadCuboid& cuboid, uint32_t bitfield)
{
    INTERFACE_PROLOGUE(table)
    table->on_rw(window, cuboid, bitfield);
    INTERFACE_EPILOGUE(table)
}

void on_check_free(SyncvTable* table, AssignmentRecordWindow window, const ThreadCuboid& cuboid, uint32_t bitfield)
{
    INTERFACE_PROLOGUE(table)
    table->on_check_free(window, cuboid, bitfield);
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

void free_barriers(SyncvTable* table, size_t N, barrier_id* barriers)
{
    INTERFACE_PROLOGUE(table)
    table->free_barriers(N, barriers);
    INTERFACE_EPILOGUE(table)
}

void on_fence(SyncvTable* table, bool transitive, const ThreadCuboid& cuboid,
        uint32_t L1_bitfield, uint32_t L2_full_bitfield, uint32_t L2_temporal_bitfield)
{
    INTERFACE_PROLOGUE(table)
    table->on_fence(transitive, cuboid, L1_bitfield, L2_full_bitfield, L2_temporal_bitfield);
    INTERFACE_EPILOGUE(table)
}

void on_arrive(SyncvTable* table, barrier_id* home_barrier, uint32_t barrier_count, barrier_id* all_barriers,
        bool transitive, const ThreadCuboid& cuboid, uint32_t L1_bitfield)
{
    INTERFACE_PROLOGUE(table)
    table->on_arrive(home_barrier, barrier_count, all_barriers, transitive, cuboid, L1_bitfield);
    INTERFACE_EPILOGUE(table)
}

// void on_await(SyncvTable* table, barrier_id* bar, TlSigInterval V2_full, TlSigInterval V2_temporal)
// {
//     INTERFACE_PROLOGUE(table)
//     table->on_await(bar, V2_full, V2_temporal);
//     INTERFACE_EPILOGUE(table)
// }

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
    table->debug_get_vis_record_data<false>(id, out);
}

void debug_get_mutate_vis_record_data(const SyncvTable* table, uint32_t id, VisRecordDebugData* out)
{
    table->debug_get_vis_record_data<true>(id, out);
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
