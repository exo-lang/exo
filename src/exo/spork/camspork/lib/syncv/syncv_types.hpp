#pragma once

#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>

#include "tl_sig.hpp"
#include "../util/cuboid_util.hpp"
#include "../util/require.hpp"

namespace camspork
{

inline constexpr uint32_t one_shot_arrive_flag = 1;
inline constexpr uint32_t one_shot_await_flag = 2;
inline constexpr uint32_t vis_record_before_alloc_flag = 1;
inline constexpr uint32_t vis_record_freed_flag = 2;  // For abstract machine use-after-free error, not free in C++.

struct assignment_record_id
{
    uint32_t node_id = 0;

    explicit operator bool() const
    {
        return node_id != 0;
    }
};

struct barrier_id
{
    uint32_t data = 0;

    explicit operator bool() const
    {
        return data != 0;
    }

    bool operator< (barrier_id other) const
    {
        return data < other.data;
    }

    bool operator== (barrier_id other) const
    {
        return data == other.data;
    }

    bool operator!= (barrier_id other) const
    {
        return data != other.data;
    }
};

struct syncv_init_t
{
    // TODO remove unused stuff (or use it).
    const char* filename;
    uint64_t memory_budget;
};

struct SyncvCheckFail : std::runtime_error
{
    size_t _linear_index_in_input;

    SyncvCheckFail(std::string msg, size_t linear_index_in_input)
      : std::runtime_error(std::move(msg))
      , _linear_index_in_input(linear_index_in_input)
    {
    }

    size_t linear_index_in_input() const
    {
        return _linear_index_in_input;
    }
};

struct SyncvBarrierFail : std::runtime_error
{
    barrier_id _hamster_barrier_id;

    SyncvBarrierFail(std::string msg, barrier_id _id)
      : std::runtime_error(std::move(msg))
      , _hamster_barrier_id(_id)
    {
    }

    barrier_id hamster_barrier_id() const
    {
        return _hamster_barrier_id;
    }
};

struct SyncvTable;

struct SyncvTableDeleter
{
    void operator() (SyncvTable* victim) const;
};

using SyncvTable_unique_ptr = std::unique_ptr<SyncvTable, SyncvTableDeleter>;

struct ThreadCuboid
{
    uint32_t task_index = 0;
    // Note: might replace these with std::vector or something.
    // Hence it's best to access this data with the accessor functions,
    // and don't assume uint32_t* for the future iterator type.
    uint32_t dim_data = 0;
    static constexpr uint32_t max_dim = 8;
    uint32_t domain_data[max_dim];
    uint32_t offset_data[max_dim];
    uint32_t box_data[max_dim];

    // State access.
    uint32_t dim() const { return dim_data; }
    uint32_t* domain() { return domain_data; }
    const uint32_t* domain() const { return domain_data; }
    uint32_t* offset() { return offset_data; }
    const uint32_t* offset() const { return offset_data; }
    uint32_t* box() { return box_data; }
    const uint32_t* box() const { return box_data; }

    uint32_t domain_num_threads() const
    {
        uint32_t prod = 1;
        for (uint32_t i = 0; i < dim(); ++i) {
            prod *= domain()[i];
        }
        return prod;
    };

    // Initialize (end-begin)-dimensional domain with all threads active
    // i.e. offset = 0, box = domain.
    template <typename Iterator>
    static ThreadCuboid full(Iterator begin, Iterator end)
    {
        ThreadCuboid cuboid;
        cuboid.task_index = 0;
        const ptrdiff_t dim = end - begin;
        CAMSPORK_REQUIRE_CMP(dim, >=, 0, "iterators in wrong order?");
        CAMSPORK_REQUIRE_CMP(dim, <=, max_dim, "implementation limit: ThreadCuboid::max_dim exceeded");
        cuboid.dim_data = uint32_t(dim);
        for (Iterator it = begin; it != end; ++it) {
            const auto i = it - begin;
            const uint32_t c = uint32_t(*it);
            cuboid.domain_data[i] = c;
            cuboid.offset_data[i] = 0;
            cuboid.box_data[i] = c;
        }
        return cuboid;
    }

    // Wrapper around cuboid_to_intervals
    template <typename Callback>
    void to_intervals(Callback&& callback) const
    {
        const uint32_t task_offset = domain_num_threads() * task_index;
        cuboid_to_intervals<uint32_t>(
            domain(), domain() + dim(), offset(), offset() + dim(), box(), box() + dim(),
            [&callback, task_offset] (uint32_t local_lo, uint32_t local_hi)
            {
                callback(task_offset + local_lo, task_offset + local_hi);
            }
        );
    }

    void reshape(uint32_t new_dim, const uint32_t* new_domain)
    {
        CAMSPORK_REQUIRE_CMP(new_dim, <=, max_dim, "Implementation limit exceeded: maximum DomainReshape dimensions");
        CAMSPORK_REQUIRE_CMP(new_dim, >=, dim(), "Invalid reshape that reduces dimensionality");

        uint32_t new_num_threads = 1;
        for (uint32_t i = 0; i < new_dim; ++i) {
            CAMSPORK_REQUIRE_CMP(new_domain[i], >=, 2, "Domain coordinate must be at least 2");
            new_num_threads *= new_domain[i];
        }
        CAMSPORK_REQUIRE_CMP(new_num_threads, ==, domain_num_threads(), "wrong thread count for DomainReshape");

        // Update offset & box for new domain.
        // Each original coordinate splits into 1 or more new coordinates.
        uint32_t new_i = new_dim;
        for (uint32_t old_i = dim(); old_i > 0; ) {
            --old_i;
            const uint32_t old_box_c = box_data[old_i];
            const uint32_t old_offset_c = offset_data[old_i];
            const uint32_t old_domain_c = domain_data[old_i];

            // This old dimension splits into 1 or more new dimensions.
            uint32_t domain_prod = 1;
            uint32_t box_quot = old_box_c;
            uint32_t offset_quot = old_offset_c;
            while (domain_prod < old_domain_c) {
                CAMSPORK_REQUIRE_CMP(new_i, >, 0, "Shouldn't happen given earlier thread count check");
                --new_i;
                const uint32_t new_domain_c = new_domain[new_i];
                domain_prod *= new_domain_c;

                CAMSPORK_REQUIRE_CMP(new_i, >=, old_i, "In-place overwrite is unsafe");
                if (box_quot <= new_domain_c) {
                    box_data[new_i] = box_quot;
                    box_quot = 1;
                }
                else {
                    box_data[new_i] = new_domain_c;
                    CAMSPORK_REQUIRE_CMP(box_quot % new_domain_c, ==, 0, "thread box misaligned for DomainReshape");
                    box_quot = box_quot / new_domain_c;
                }
                offset_data[new_i] = offset_quot % new_domain_c;
                offset_quot = offset_quot / new_domain_c;
                domain_data[new_i] = new_domain_c;
            }
            CAMSPORK_REQUIRE_CMP(domain_prod, ==, old_domain_c, "DomainReshape can only split dimensions");
        }

        dim_data = new_dim;
    }

    uint32_t get_tid_lo() const
    {
        uint32_t tid_lo = task_index;
        for (uint32_t dim_i = 0; dim_i < dim(); ++dim_i) {
            tid_lo *= domain()[dim_i];
            tid_lo += offset()[dim_i];
        }
        return tid_lo;
    }
};

// ThreadCuboid-like interface for holding a continuous thread range.
// Mainly for use by SyncvTable internally.
struct SimpleThreadInit
{
    uint32_t tid_lo, tid_hi;

    template <typename Callback>
    void to_intervals(Callback&& callback) const
    {
        callback(tid_lo, tid_hi);
    }
};

struct EmptyThreadInit
{
    template <typename Callback>
    void to_intervals(Callback&&) const
    {
    }
};

template <typename Stream>
Stream&& operator<<(Stream&& s, const ThreadCuboid& cuboid)
{
    const uint32_t dim = cuboid.dim();
    CAMSPORK_REQUIRE_CMP(dim, <=, cuboid.max_dim, "tried to print ThreadCuboid with too many dimensions");
    auto print_list = [dim, &s] (auto p)
    {
        s << "[";
        if (dim > 0) {
            s << p[0];
            for (uint32_t i = 1; i < dim; ++i) {
                s << ", " << p[i];
            }
        }
        s << "]";
    };
    s << "{\"task_index\": " << cuboid.task_index;
    s << ", \"domain\": ";
    print_list(cuboid.domain());
    s << ", \"offset\": ";
    print_list(cuboid.offset());
    s << ", \"box\": ";
    print_list(cuboid.box());
    s << "}";
    return static_cast<Stream&&>(s);
}

}
