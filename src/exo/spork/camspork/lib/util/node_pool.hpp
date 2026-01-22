#pragma once
#include <cassert>
#include <memory>
#include <new>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "require.hpp"

#ifndef CAMSPORK_NODE_POOL_DEBUG_BITS
#define CAMSPORK_NODE_POOL_DEBUG_BITS 4
#endif

namespace camspork
{
namespace nodepool
{

// Linked list nodes are referenced based on their 1-based index in the pool.
// 0 is reserved as "null"; this is to ensure default initialization to 0
// does the expected thing (pretty important for C APIs).
// The bottom CAMSPORK_NODE_POOL_DEBUG_BITS-many bits are reserved to detect use-after-free.
// Each time a node is freed, we update the expected debug bits for the node at that index
// so we can detect usage of the old stale ID.
template <typename ListNode>
struct id
{
    using value_type = ListNode;

    uint32_t id_bits;

    explicit operator bool() const
    {
        return id_bits != 0;
    }

    uint32_t node_index() const
    {
        return (id_bits >> CAMSPORK_NODE_POOL_DEBUG_BITS) - 1u;
    }

    uint32_t debug_bits() const
    {
        return id_bits & ((1u << CAMSPORK_NODE_POOL_DEBUG_BITS) - 1u);
    }

    bool operator<(id other) const
    {
        // Note that as the debug bits are the low bits,
        // enabling/disabling this debug feature
        // should not impact the sorting order.
        return id_bits < other.id_bits;
    }

    bool operator==(id other) const
    {
        return id_bits == other.id_bits;
    }

    bool operator!=(id other) const
    {
        return id_bits != other.id_bits;
    }
};

template <typename Stream, typename ListNode>
Stream& operator<<(Stream& stream, id<ListNode> node_id)
{
    return stream << node_id.id_bits;
}

constexpr uint32_t max_node_index()
{
    return uint32_t(uint64_t(1) << (32u - CAMSPORK_NODE_POOL_DEBUG_BITS)) - 2u;
}

constexpr uint32_t pack_id_bits(uint32_t node_index, uint8_t debug_bits)
{
    const uint32_t index_bits_32 = (node_index + 1) << CAMSPORK_NODE_POOL_DEBUG_BITS;
    if constexpr (CAMSPORK_NODE_POOL_DEBUG_BITS > 0) {
        const uint32_t debug_bits_32 = debug_bits & ((1u << CAMSPORK_NODE_POOL_DEBUG_BITS) - 1u);
        return index_bits_32 | debug_bits_32;
    }
    else {
        return index_bits_32;
    }
}

constexpr uint32_t pack_id_bits(uint32_t node_index, uint32_t debug_bits) = delete;

template <typename ListNode>
struct AllNodesIterator;

// Non-threadsafe memory pool for allocating singly-linked list nodes.
// Each node is referenced by integer index, rather than pointer;
// the index only refers to nodes as long as the pool hasn't been deleted.
//
// ListNode is expected to have an id<ListNode> member named camspork_next_id.
template <typename ListNode>
class Pool
{
    static constexpr uint32_t chunk_size = 4096;
    static_assert(sizeof(ListNode) <= 64, "Re-evaluate chunk_size");

    struct Chunk
    {
        ListNode storage[chunk_size];
#if CAMSPORK_NODE_POOL_DEBUG_BITS > 0
        static_assert(CAMSPORK_NODE_POOL_DEBUG_BITS <= 8);
        uint8_t expected_debug_bits[chunk_size];
#endif
    };

    // [node_index() / chunk_size][node_index() % chunk_size]
    std::vector<std::unique_ptr<Chunk>> chunks;

    // ID of first item in the free list.
    id<ListNode> free_list_head{0};

  public:
    Pool() = default;
    Pool(Pool&&) = default;
    Pool& operator=(Pool&&) = default;
    ~Pool() = default;

    Pool(const Pool& other)
    {
        const size_t n_chunks = other.chunks.size();
        chunks.reserve(n_chunks);
        for (size_t i = 0; i < n_chunks; ++i) {
            std::unique_ptr<Chunk> p_new(new Chunk(*other.chunks[i]));
            chunks.emplace_back(std::move(p_new));
        }
        free_list_head = other.free_list_head;
    };

    Pool& operator=(const Pool& other)
    {
        return (*this = Pool(other));
    };

    const ListNode& get(id<ListNode> _id) const
    {
        return const_cast<Pool<ListNode>*>(this)->get(_id);
    }

    ListNode& get(id<ListNode> _id, bool no_debug=false)
    {
        CAMSPORK_REQUIRE(_id, "null node dereferenced");
        auto _0_index = _id.node_index();
        CAMSPORK_C_BOUNDSCHECK(_0_index, chunks.size() * chunk_size);
        const auto chunk_index = _0_index / chunk_size;
        const auto r = _0_index % chunk_size;
        if constexpr (CAMSPORK_NODE_POOL_DEBUG_BITS) {
            if (!no_debug) {
                const auto expected_id_bits = id_from_node_index(_0_index).id_bits;
                CAMSPORK_REQUIRE_CMP(_id.id_bits, ==, expected_id_bits, "likely use-after-free of node");
            }
        }
        return chunks[chunk_index]->storage[r];
    }

    // Allocate a new node.
    // The node is default initialized, including setting camspork_next_id to 0.
    ListNode& alloc_default_node(uintptr_t* p_memory_budget, id<ListNode>* out_id)
    {
        if (!free_list_head) {
            // Allocate new chunk if allowed by memory budget.
            if (*p_memory_budget < sizeof(Chunk)) {
                throw std::bad_alloc{};
            }
            const uint32_t old_chunk_count = uint32_t(chunks.size());
            chunks.push_back(std::make_unique<Chunk>());

            // ID overflow check
            CAMSPORK_REQUIRE_CMP(chunks.size() * chunk_size, <, max_node_index(), "node ID int overflow");
            if (chunks.size() * chunk_size > max_node_index() / 2) {
                if (old_chunk_count * chunk_size <= max_node_index() / 2) {
                    fprintf(stderr, "Warning: Halfway to exhausting node IDs (%s:%i)\n", __FILE__, __LINE__);
                }
            }

            // Record allocation against memory budget.
            Chunk& new_chunk = *chunks.back();
            *p_memory_budget -= sizeof new_chunk;

            // Organize new chunk into the new free list.
            uint32_t id_offset = old_chunk_count * chunk_size + 1;
            for (uint32_t i = 0; i < chunk_size - 1; ++i) {
                new_chunk.storage[i].camspork_next_id.id_bits = pack_id_bits(id_offset + i, uint8_t(0));
                if constexpr (CAMSPORK_NODE_POOL_DEBUG_BITS > 0) {
                    new_chunk.expected_debug_bits[i] = 0xFF;
                }
            }
            if constexpr (CAMSPORK_NODE_POOL_DEBUG_BITS > 0) {
                new_chunk.expected_debug_bits[chunk_size - 1] = 0xFF;
            }
            new_chunk.storage[chunk_size - 1].camspork_next_id.id_bits = 0;
            free_list_head.id_bits = pack_id_bits(old_chunk_count * chunk_size, uint8_t(0));

            CAMSPORK_REQUIRE_CMP(&new_chunk.storage[0], ==, &get(free_list_head, true), "node free list broken");
        }
        CAMSPORK_REQUIRE(free_list_head, "unexpected empty free list");
        id<ListNode> ret = id_from_node_index(free_list_head.node_index());
        ListNode& node = get(ret, true);
        free_list_head = node.camspork_next_id;

        node = ListNode{};
        node.camspork_next_id = id<ListNode>{0};
        *out_id = ret;
        return node;
    }

    // Move the entire list given to the free list.
    // i.e. append the current free list to the tail of the given list,
    // then let the head of the given list be the new head of the free list.
    void extend_free_list(id<ListNode> head_id) noexcept
    {
        if (!head_id) {
            return;
        }
        id<ListNode> tmp_id = head_id;
        while (1) {
            ListNode& node = get(tmp_id);
            if constexpr (CAMSPORK_NODE_POOL_DEBUG_BITS) {
                const auto node_index = tmp_id.node_index();
                chunks[node_index / chunk_size]->expected_debug_bits[node_index % chunk_size]--;
            }
            if (node.camspork_next_id) {
                tmp_id = node.camspork_next_id;
            }
            else {
                node.camspork_next_id = free_list_head;
                free_list_head = head_id;
                return;
            }
        }
    }

    // Given a pointer to the camspork_next_id member of a node,
    // insert the given insert_me node after said node.
    void insert_next_node(id<ListNode>* p_insert_after, id<ListNode> insert_me) noexcept
    {
        assert(p_insert_after);
        ListNode& inserted_node = get(insert_me);
        assert(!inserted_node.camspork_next_id);
        inserted_node.camspork_next_id = *p_insert_after;
        *p_insert_after = insert_me;
    }

    // Given a pointer to the camspork_next_id member of a node,
    // remove the node AFTER said node, and return the removed node.
    //
    // The returned node automatically has its camspork_next_id nulled,
    // but is not automatically free'd; pass to extend_free_list later.
    [[nodiscard]] id<ListNode> remove_next_node(id<ListNode>* p_id) noexcept
    {
        assert(*p_id);
        const id<ListNode> ret = *p_id;
        ListNode& removed_node = get(ret);
        *p_id = removed_node.camspork_next_id;
        removed_node.camspork_next_id.id_bits = 0;
        return ret;
    };

    // Get number of nodes in pool (both allocated and free)
    // This is mainly for testing and debugging.
    uint32_t size() const noexcept
    {
        const size_t sz = chunks.size() * chunk_size;
        CAMSPORK_REQUIRE_CMP(sz, <=, UINT32_MAX, "nodepool::Pool 32-bit overflow");
        return uint32_t(sz);
    }

    // ID from node index, circumventing use-after-free checking.
    id<ListNode> id_from_node_index(uint32_t node_index) const
    {
        CAMSPORK_C_BOUNDSCHECK(node_index, size());
        uint8_t debug_bits = 0;
        if constexpr (CAMSPORK_NODE_POOL_DEBUG_BITS) {
            debug_bits = chunks[node_index / chunk_size]->expected_debug_bits[node_index % chunk_size];
        }
        return id<ListNode>{pack_id_bits(node_index, debug_bits)};
    }

    // Get IDs of nodes on free chain. Mostly for debugging.
    template <typename Set>
    void get_free_ids(Set* p_set) const
    {
        auto id = free_list_head;
        while (id) {
            id = id_from_node_index(id.node_index());
            p_set->insert(id);
            id = get(id).camspork_next_id;
        }
    }

    friend AllNodesIterator<ListNode>;
};

template <typename ListNode>
struct AllNodesIterator
{
    const Pool<ListNode>* p_pool;
    uint32_t node_index;

    bool operator==(AllNodesIterator<ListNode> other)
    {
        return p_pool == other.p_pool && node_index == other.node_index;
    }

    bool operator!=(AllNodesIterator<ListNode> other)
    {
        return !(*this == other);
    }

    id<ListNode> operator*() const
    {
        return p_pool->id_from_node_index(node_index);
    }

    AllNodesIterator<ListNode>& operator++()
    {
        ++node_index;
        return *this;
    }
};

// Iterate over each valid node ID for the pool.
// Debug feature: you will get IDs to nodes on the free list that appear valid to the use-after-free checker.
// The caller must have a way to distinguish such nodes.
template <typename ListNode>
AllNodesIterator<ListNode> begin(const Pool<ListNode>& pool)
{
    return AllNodesIterator<ListNode>{&pool, 0};
}
template <typename ListNode>
AllNodesIterator<ListNode> end(const Pool<ListNode>& pool)
{
    return AllNodesIterator<ListNode>{&pool, uint32_t(pool.size())};
}

}
}

namespace std
{
template <typename ListNode>
struct hash<camspork::nodepool::id<ListNode>>
{
    size_t operator()(const camspork::nodepool::id<ListNode>& id) const
    {
        return id.id_bits;
    }
};
}
