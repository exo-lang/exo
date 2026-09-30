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

// A single timeline signature (tl-sig) is the 2-tuple (thread ID, qual-tl)
// We usually don't store this directly. Instead, we work with sets of tl-sig.
// A TlSigInterval encodes the subset of timeline signatures (t, q) where
//
// tid_lo <= t < tid_hi
// 0 != ((1 << q) & qual_bits)
//
// Some qual-tl have special properties, given by the qual-tl class bitmasks passed to SyncvTable on construction
// (syncv_init_t): out-of-order qual-tl, and atomic qual-tl.
//
// LEGACY TERMS:
//   sigthread = tl-sig (before visibility flag was invented)
//   actor signature = qual-tl (qualitative timeline)
//   async visibility = unordered visibility
//   sigbits = qual_bits
//   visibility level = visibility flag
//   visibility flag: removed; tl-sig used to be the 3-tuple (thread ID, qual-tl, visibility flag)
struct TlSigInterval
{
    // Thread index range [tid_lo, tid_hi)
    uint32_t tid_lo, tid_hi;

    qual_bits_t qual_bits;

    void assert_valid() const
    {
        CAMSPORK_REQUIRE(qual_bits != 0, "Empty qual_bits");
        CAMSPORK_REQUIRE_CMP(tid_lo, <, tid_hi, "Invalid TlSigInterval");
    }

    bool operator==(TlSigInterval other) const
    {
        uint32_t diff = tid_lo ^ other.tid_lo;
        diff |= tid_hi ^ other.tid_hi;
        diff |= qual_bits ^ other.qual_bits;
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
        return !tid_disjoint && (qual_bits & other.qual_bits) != 0;
    }

    bool intersects_threads(uint32_t arg_tid_lo, uint32_t arg_tid_hi)
    {
        // <= due to tid_hi being an exclusive bound.
        const bool tid_disjoint = tid_hi <= arg_tid_lo || arg_tid_hi <= tid_lo;
        return !tid_disjoint;
    }

    bool is_atomic_only(qual_bits_t atomic_qual_mask) const
    {
        // TlSigInterval contains only (t, q) where q is an atomic qual-tl?
        return (qual_bits & ~atomic_qual_mask) == 0;
    }
};


}  // end namespace
