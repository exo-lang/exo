#pragma once
#include <cassert>
#include <memory>
#include <new>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace camspork
{
namespace nodepool
{

// Linked list nodes are referenced based on their 1-based index in the pool.
// 0 is reserved as "null"; this is to ensure default initialization to 0
// does the expected thing (pretty important for C APIs).
template <typename ListNode>
struct id
{
    using value_type = ListNode;

    uint32_t _1_index;

    explicit operator bool() const
    {
        return _1_index != 0;
    }

    bool operator<(id other) const
    {
        return _1_index < other._1_index;
    }

    bool operator==(id other) const
    {
        return _1_index == other._1_index;
    }

    bool operator!=(id other) const
    {
        return _1_index != other._1_index;
    }
};

template <typename Stream, typename ListNode>
Stream& operator<<(Stream& stream, id<ListNode> node_id)
{
    return stream << node_id._1_index;
}

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
    };

    // [(_1_index - 1) / chunk_size][(_1_index - 1) % chunk_size]
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

    const ListNode& get(id<ListNode> _id) const noexcept
    {
        assert(_id._1_index != 0);
        auto _0_index = _id._1_index - 1u;
        assert(_0_index < chunks.size() * chunk_size);
        return chunks[_0_index / chunk_size]->storage[_0_index % chunk_size];
    }

    ListNode& get(id<ListNode> _id) noexcept
    {
        const auto& const_self = *this;
        return const_cast<ListNode&>(const_self.get(_id));
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
            assert(chunks.size() * chunk_size < UINT32_MAX);
            if (chunks.size() * chunk_size > INT32_MAX) {
                if (old_chunk_count * chunk_size <= INT32_MAX) {
                    fprintf(stderr, "Warning: Halfway to exhausting 32-bit IDs (%s:%i)\n", __FILE__, __LINE__);
                }
            }

            // Record allocation against memory budget.
            Chunk& new_chunk = *chunks.back();
            *p_memory_budget -= sizeof new_chunk;

            // Organize new chunk into the new free list.
            uint32_t id_offset = old_chunk_count * chunk_size + 2;
            for (uint32_t i = 0; i < chunk_size - 1; ++i) {
                new_chunk.storage[i].camspork_next_id._1_index = id_offset + i;
            }
            new_chunk.storage[chunk_size - 1].camspork_next_id._1_index = 0;
            free_list_head._1_index = old_chunk_count * chunk_size + 1;

            assert(&new_chunk.storage[0] == &get(free_list_head));
        }
        assert(free_list_head);
        id<ListNode> ret = free_list_head;
        ListNode& node = get(ret);
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
        removed_node.camspork_next_id._1_index = 0;
        return ret;
    };

    // Get number of nodes in pool (both allocated and free)
    // This is mainly for testing and debugging.
    uint32_t size() const noexcept
    {
        const size_t sz = chunks.size() * chunk_size;
        assert(sz <= UINT32_MAX);
        return uint32_t(sz);
    }

    // Get IDs of nodes on free chain. Mostly for debugging.
    template <typename Set>
    void get_free_ids(Set* p_set) const
    {
        auto id = free_list_head;
        while (id) {
            p_set->insert(id);
            id = get(id).camspork_next_id;
        }
    }
};

}
}

namespace std
{
template <typename ListNode>
struct hash<camspork::nodepool::id<ListNode>>
{
    size_t operator()(const camspork::nodepool::id<ListNode>& id) const
    {
        return id._1_index;
    }
};
}
