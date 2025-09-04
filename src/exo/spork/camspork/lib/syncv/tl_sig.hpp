#pragma once

#include <cassert>
#include <stdint.h>

#include "../util/bit_util.hpp"
#include "../util/require.hpp"
#include "syncv_types.hpp"

namespace camspork
{

// Note, assignment of 0, 1, 3, allows for bitwise-or to "promote" to the next vis_level.
static constexpr uint32_t vis_level_atomic_only = 0;
static constexpr uint32_t vis_level_unordered = 1;
static constexpr uint32_t vis_level_ordered = 3;

// A single timeline signature consists (conceptually) of a pair of
// (thread ID, qual-tl) [qualitative timeline]. We never store this directly.
// Instead, we work with sets of timeline signatures (tl-sig).
//
// A tl-sig interval is the cartesian product
//     [tid_lo, tid_hi) \times L
// where L is a set of qual-tl defined by the bits set in
//     bitfield & qual_bits()
//
// The vis_level() is used elsewhere, to store the different visibility sets
// compactly. Given V_A \superset V_U \superset V_O [atomic-only, unordered, ordered],
// we have that
//     V_O = union(val: TlSigInterval where val.vis_level() >= vis_level_ordered)
//     V_U = union(val: TlSigInterval where val.vis_level() >= vis_level_unordered)
//     V_A = union(val: TlSigInterval where val.vis_level() >= vis_level_atomic_only)
//
// LEGACY TERMS:
//   sigthread = tl-sig (timeline signature)
//   actor signature = qual-tl (qualitative timeline)
//   async visibility = unordered visibility
//   sigbits = qual_bits
struct TlSigInterval
{
    // Thread index range [tid_lo, tid_hi)
    uint32_t tid_lo, tid_hi;

    // qual_bits() | vis_level() << 30
    uint32_t bitfield;

    static constexpr uint32_t unordered_bits = vis_level_unordered << 30;
    static constexpr uint32_t ordered_bits = vis_level_ordered << 30;

    void assert_valid() const
    {
        CAMSPORK_REQUIRE_CMP(qual_bits(), !=, 0, "Invalid TlSigInterval");
        CAMSPORK_REQUIRE_CMP(tid_lo, <=, tid_hi, "Invalid TlSigInterval");
    }

    uint32_t vis_level() const
    {
        return bitfield >> 30;
    }

    uint32_t qual_bits() const
    {
        return bitfield & ((1u << 30) - 1);
    }

    static uint32_t vis_level(uint32_t bitfield)
    {
        return bitfield >> 30;
    }

    static uint32_t qual_bits(uint32_t bitfield)
    {
        return bitfield & ((1u << 30) - 1);
    }

    // Requires that exactly one qual-tl bit is set.
    // Return the bit index of that qual-tl (e.g. 8 -> 3)
    uint8_t get_unique_qual_tl() const
    {
        return get_unique_qual_tl(bitfield);
    }

    static uint8_t get_unique_qual_tl(uint32_t bitfield)
    {
        const auto bits = qual_bits(bitfield);
        CAMSPORK_REQUIRE_CMP(bits, !=, 0, "Require exactly one qual-tl bit set");
        uint8_t bit_index = get_low_bit_index(bits);
        CAMSPORK_REQUIRE_CMP(bits, ==, 1u << bit_index, "Require exactly one qual-tl bit set");
        return bit_index;
    }

    bool operator==(TlSigInterval other) const
    {
        uint32_t diff = tid_lo ^ other.tid_lo;
        diff |= tid_hi ^ other.tid_hi;
        diff |= bitfield ^ other.bitfield;
        return diff == 0;
    }

    bool operator!=(TlSigInterval other) const
    {
        return !(*this == other);
    }

    bool intersects(const TlSigInterval& other, uint32_t qual_bits_mask = ~uint32_t(0)) const
    {
        // <= due to tid_hi being an exclusive bound.
        const bool tid_disjoint = tid_hi <= other.tid_lo || other.tid_hi <= tid_lo;
        uint32_t this_qual_bits = qual_bits();
        const uint32_t other_qual_bits = other.qual_bits();
        CAMSPORK_REQUIRE_CMP(this_qual_bits, !=, 0, "Invalid empty qual-tl set");
        CAMSPORK_REQUIRE_CMP(other_qual_bits, !=, 0, "Invalid empty qual-tl set");
        return !tid_disjoint && 0 != (this_qual_bits & other_qual_bits & qual_bits_mask);
    }
};


}  // end namespace
