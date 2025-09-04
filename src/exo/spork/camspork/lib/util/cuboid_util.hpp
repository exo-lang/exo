#pragma once

#include "require.hpp"

namespace camspork {

// Imagine we have a tensor T whose size is given by the outer_begin and
// outer_end iterators (i.e. the tensor is (outer_end-outer_begin)-dimensional).
// Take the set of all positions S = {T[i_0, ... i_N] where
// offset_begin[n] <= i_n < (offset_begin[n] + inner_begin[n])}
//
// If the tensor T is stored in C-order in T_storage, then we need to
// find intervals such that
//
// for (auto interval: intervals) {
//     for (IntT i = interval.lo, i < interval.hi; ++i) {
//         T_storage[i];
//     }
// }
//
// loops over each element in S exactly once.
//
// The intervals will:
//   * not be adjacent (no "redundant" intervals)
//   * be delivered by calling callback(interval.lo, interval.hi)
//     in sorted order.
//
// An explicit IntT type parameter must be given.
template <typename IntT, typename Callback, typename OuterIterator, typename OffsetIterator, typename InnerIterator>
void cuboid_to_intervals(
    OuterIterator outer_begin, OuterIterator outer_end,
    OffsetIterator offset_begin, OffsetIterator offset_end,
    InnerIterator inner_begin, InnerIterator inner_end,
    Callback&& callback)
{
    const auto dim = outer_end - outer_begin;
    CAMSPORK_REQUIRE_CMP(dim, ==, offset_end - offset_begin, "mismatched dimensions");
    CAMSPORK_REQUIRE_CMP(dim, ==, inner_end - inner_begin, "mismatched dimensions");

    constexpr IntT discontinuity_sentinel = ~IntT(0);

    auto recurse = [&callback, outer_end, offset_end, inner_end] (
            IntT partial_offset,
            OuterIterator outer_iter, OffsetIterator offset_iter, InnerIterator inner_iter,
            auto recurse) -> IntT
    {
        if (outer_iter == outer_end) {
            return IntT{1};
        }
        const IntT outer_coord = IntT(*outer_iter);
        const IntT offset_coord = IntT(*offset_iter);
        const IntT inner_coord = IntT(*inner_iter);
        partial_offset = partial_offset * outer_coord + offset_coord;

        if (inner_coord == IntT(0)) {
            return IntT(0);
        }

        const IntT leaf_size = recurse(partial_offset, outer_iter + 1, offset_iter + 1, inner_iter + 1, recurse);

        if (leaf_size == IntT(0)) {
            return IntT(0);
        }

        if (leaf_size == discontinuity_sentinel) {
            // Generate remaining intervals (skips i = 0 case already generated).
            for (IntT i = IntT(1); i < inner_coord; ++i) {
                recurse(partial_offset + i, outer_iter + 1, offset_iter + 1, inner_iter + 1, recurse);
            }
            // Caller must also execute this case, to loop over the generation of all intervals.
            return discontinuity_sentinel;
        }
        else {
            if (offset_coord == 0) {
                if (inner_coord == outer_coord) {
                    // This dimension is full, and all dimensions to the right are full.
                    // Inform caller of the size of the product of all dimensions.
                    // Some caller will generate the actual interval, which is a superset of this.
                    return leaf_size * inner_coord;
                }
            }
            else {
                // else case and > (instead of >=) prevents moronic "unsigned comparison with 0" warnings...
                CAMSPORK_REQUIRE_CMP(offset_coord, >, 0, "Negative offset not allowed");
            }
            CAMSPORK_REQUIRE_CMP(offset_coord + inner_coord, <=, outer_coord, "out-of-bounds cuboid extent");
            // This dimension introduces a discontinuity, but all dimensions to the right don't.
            // We will invoke the callback at this level.
            const IntT scalar_offset = partial_offset * leaf_size;
            callback(scalar_offset, scalar_offset + leaf_size * inner_coord);
            return discontinuity_sentinel;
        }
    };

    const IntT leaf_size = recurse(IntT(0), outer_begin, offset_begin, inner_begin, recurse);
    if (leaf_size != discontinuity_sentinel && leaf_size != 0) {
        // If all offsets were 0 and all inner extents equal outer extents, then no recursive function
        // calls generated any intervals and we have to handle that here.
        callback(IntT(0), IntT(leaf_size));
    }
}

}
