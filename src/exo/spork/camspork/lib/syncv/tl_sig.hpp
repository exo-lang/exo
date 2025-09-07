#pragma once

#include <cassert>
#include <stdint.h>

#include "../util/bit_util.hpp"
#include "../util/require.hpp"

namespace camspork
{

static constexpr int32_t vis_level_none = -1;
static constexpr int32_t vis_level_atomic_only = 0;
static constexpr int32_t vis_level_unordered = 1;
static constexpr int32_t vis_level_ordered = 2;

inline const char* vis_level_name(int32_t vis_level)
{
    CAMSPORK_REQUIRE_CMP(vis_level, >=, -1, "Invalid vis_level enum");
    CAMSPORK_REQUIRE_CMP(vis_level, <=, 2, "Invalid vis_level enum");
    static const char* strs[4] = {
        "vis_level_none",
        "vis_level_atomic_only",
        "vis_level_unordered",
        "vis_level_ordered",
    };
    return strs[vis_level + 1];
}

struct QualBitsByVis
{
    uint32_t array[3];

    bool operator== (const QualBitsByVis& other) const
    {
        return diff_bits(other) == 0;
    }

    bool operator!= (const QualBitsByVis& other) const
    {
        return !(*this == other);
    }

    QualBitsByVis& operator|= (const QualBitsByVis& other)
    {
        for (uint32_t i = 0; i < 3; ++i) {
            array[i] |= other.array[i];
        }
        return *this;
    }

    QualBitsByVis operator| (const QualBitsByVis& other) const
    {
        QualBitsByVis result = *this;
        result |= other;
        return result;
    }

    uint32_t diff_bits(const QualBitsByVis& other) const
    {
        uint32_t diff = array[0] ^ other.array[0];
        diff |= array[1] ^ other.array[1];
        diff |= array[2] ^ other.array[2];
        return diff;
    }
};

// A single timeline signature consists of a pair of
// (thread ID, qual-tl) [qualitative timeline]. We don't store this directly.
// Instead, we work with sets of timeline signatures (tl-sig).
//
// A tl-sig interval is the cartesian product
//     [tid_lo, tid_hi) \times L
// where L is a set of qual-tl, delivered as a bitfield.
//
// For compactness, we store the three visibility sets together.
// Given V_A \superset V_U \superset V_O [atomic-only, unordered, ordered],
// we have that
//     V_A = union(val: TlSigInterval where L = qual_bits_by_vis.array[vis_level_atomic_only])
//     V_U = union(val: TlSigInterval where L = qual_bits_by_vis.array[vis_level_unordered])
//     V_O = union(val: TlSigInterval where L = qual_bits_by_vis.array[vis_level_ordered])
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

    QualBitsByVis qual_bits_by_vis;

    void assert_valid() const
    {
        const uint32_t (&qual_tl_bits) [3] = qual_bits_by_vis.array;
        CAMSPORK_REQUIRE_CMP(qual_tl_bits[1] & qual_tl_bits[2], ==, qual_tl_bits[2], "TlSigInterval, invalid subset");
        CAMSPORK_REQUIRE_CMP(qual_tl_bits[0] & qual_tl_bits[1], ==, qual_tl_bits[1], "TlSigInterval, invalid subset");
        CAMSPORK_REQUIRE_CMP(qual_tl_bits[0], !=, 0, "Invalid TlSigInterval empty qual-tl bits");
        CAMSPORK_REQUIRE_CMP(tid_lo, <=, tid_hi, "Invalid TlSigInterval");
    }

    // Requires that exactly one qual-tl bit is set.
    // Return the bit index of that qual-tl (e.g. 8 -> 3)
    uint8_t get_unique_qual_tl() const
    {
        // TODO explain why we choose vis_level_unordered.
        return get_unique_qual_tl(qual_bits_by_vis.array[vis_level_unordered]);
    }

    static uint8_t get_unique_qual_tl(uint32_t qual_bits)
    {
        CAMSPORK_REQUIRE_CMP(qual_bits, !=, 0, "Require exactly one qual-tl bit set");
        uint8_t bit_index = get_low_bit_index(qual_bits);
        CAMSPORK_REQUIRE_CMP(qual_bits, ==, 1u << bit_index, "Require exactly one qual-tl bit set");
        return bit_index;
    }

    bool operator==(TlSigInterval other) const
    {
        uint32_t diff = tid_lo ^ other.tid_lo;
        diff |= tid_hi ^ other.tid_hi;
        diff |= qual_bits_by_vis.diff_bits(other.qual_bits_by_vis);
        return diff == 0;
    }

    bool operator!=(TlSigInterval other) const
    {
        return !(*this == other);
    }

    bool unordered_intersects(const TlSigInterval& other, uint32_t qual_bits_mask) const
    {
        // <= due to tid_hi being an exclusive bound.
        const bool tid_disjoint = tid_hi <= other.tid_lo || other.tid_hi <= tid_lo;
        uint32_t this_qual_bits = qual_bits_by_vis.array[vis_level_unordered];
        uint32_t other_qual_bits = other.qual_bits_by_vis.array[vis_level_unordered];
        return !tid_disjoint && 0 != (this_qual_bits & other_qual_bits & qual_bits_mask);
    }
};


}  // end namespace
