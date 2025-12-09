#pragma once

#include <cassert>
#include <stdint.h>

#include "../util/bit_util.hpp"
#include "../util/require.hpp"

namespace camspork
{

// TODO we are not really using this consistently, maybe -Wconversion can help if we need to widen this.
// However, a better way may be to change Exo to allow assigning different bits to stand for different QualTL
// on a per-program basis, as it's unlikely a single program needs more than 32 QualTL.
using qual_bits_t = uint32_t;
static constexpr uint32_t num_qual_tl = 32;

static constexpr int32_t vis_flag_index_atomic_only = 0;
static constexpr int32_t vis_flag_index_temporal = 1;
static constexpr int32_t vis_flag_index_full = 2;
static constexpr int32_t vis_flag_index_issue = 3;
static constexpr int32_t num_vis_flags = 4;

static constexpr int32_t vis_flag_atomic_only = 1 << vis_flag_index_atomic_only;
static constexpr int32_t vis_flag_temporal = 1 << vis_flag_index_temporal;
static constexpr int32_t vis_flag_full = 1 << vis_flag_index_full;
static constexpr int32_t vis_flag_issue = 1 << vis_flag_index_issue;
static constexpr int32_t vis_flags_all = (1 << num_vis_flags) - 1;

inline const char* vis_flag_name(int32_t vis_flag)
{
    static const char* table[num_vis_flags] = {"atomic-only", "temporal", "full", "issue"};
    const int vis_flag_index = get_low_bit_index(vis_flag);
    CAMSPORK_REQUIRE_CMP(vis_flag, ==, (1 << vis_flag_index), "not a vis flag");
    CAMSPORK_REQUIRE_CMP(vis_flag_index, <, num_vis_flags, "not a vis flag");
    return table[vis_flag_index];
}

struct QualBitsByVis
{
    qual_bits_t array[num_vis_flags];

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
        for (uint32_t i = 0; i < num_vis_flags; ++i) {
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
        auto diff = array[0] ^ other.array[0];
        for (int32_t i = 1; i < num_vis_flags; ++i) {
            diff |= array[i] ^ other.array[i];
        }
        return diff;
    }

    bool intersects(const QualBitsByVis& other) const
    {
        auto q_mask = array[0] & other.array[0];
        for (int32_t i = 1; i < num_vis_flags; ++i) {
            q_mask |= array[i] & other.array[i];
        }
        return q_mask != 0;
    }
};

inline QualBitsByVis qual_vis_product(qual_bits_t qual_bits, int32_t vis_flags)
{
    QualBitsByVis qv;
    for (int32_t i = 0; i < num_vis_flags; ++i) {
        qual_bits_t mask = qual_bits_t(0) - qual_bits_t(1 & (vis_flags >> i));
        qv.array[i] = qual_bits & mask;
    }
    return qv;
}

// See TlSigInterval for most uses.
// This is mainly used for formatting debug and error messages.
struct TlSig
{
    uint32_t tid;
    uint8_t qual_tl;
    int32_t vis_flag;
};

// A single timeline signature is the 3-tuple (thread ID, qual-tl, visibility flag)
// We usually don't store this directly. Instead, we work with sets of tl-sig.
// A TlSigInterval encodes the subset of timeline signatures (t, q, v) where
//
// tid_lo <= t < tid_hi
// 0 != ((1 << q) & qual_bits_by_vis.array[get_low_bit_index(v)]
//
// LEGACY TERMS:
//   sigthread = tl-sig (before visibility flag was invented)
//   actor signature = qual-tl (qualitative timeline)
//   async visibility = unordered visibility
//   sigbits = qual_bits
//   visibility level = visibility flag
struct TlSigInterval
{
    // Thread index range [tid_lo, tid_hi)
    uint32_t tid_lo, tid_hi;

    QualBitsByVis qual_bits_by_vis;

    void assert_valid() const
    {
        CAMSPORK_REQUIRE(qual_bits_by_vis.intersects(qual_bits_by_vis), "Empty qual_bits_by_vis");
        CAMSPORK_REQUIRE_CMP(tid_lo, <=, tid_hi, "Invalid TlSigInterval");
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

    bool intersects(const TlSigInterval& other) const
    {
        // <= due to tid_hi being an exclusive bound.
        const bool tid_disjoint = tid_hi <= other.tid_lo || other.tid_hi <= tid_lo;
        return !tid_disjoint && qual_bits_by_vis.intersects(other.qual_bits_by_vis);
    }

    bool intersects_threads(uint32_t arg_tid_lo, uint32_t arg_tid_hi)
    {
        // <= due to tid_hi being an exclusive bound.
        const bool tid_disjoint = tid_hi <= arg_tid_lo || arg_tid_hi <= tid_lo;
        return !tid_disjoint;
    }

    bool is_atomic_only() const
    {
        // TlSigInterval contains only (t, q, v) where v = vis_flag_atomic_only?
        static_assert(vis_flag_index_atomic_only == 0, "Hard-wired code for skipping atomic-only");
        auto q_bits = qual_bits_by_vis.array[1];
        for (int i = 2; i < num_vis_flags; ++i) {
            q_bits |= popcount(qual_bits_by_vis.array[i]);
        }
        return q_bits == 0;
    }

    uint64_t num_non_atomic_timeline_signatures() const
    {
        // Count of (t, q, v) in TlSigInterval where v != vis_flag_atomic_only.
        uint64_t pop = 0;
        static_assert(vis_flag_index_atomic_only == 0, "Hard-wired code for skipping atomic-only");
        for (int i = 1; i < num_vis_flags; ++i) {
            pop += popcount(qual_bits_by_vis.array[i]);
        }
        return pop * (tid_hi - tid_lo);
    }
};


}  // end namespace
