#include "camspork_excut.hpp"

#include "../util/require.hpp"

namespace camspork {

#define RETURN_MUTATE_TAGGED(name) \
    const uint32_t idx = static_cast<uint32_t>(mutate_tag); \
    CAMSPORK_REQUIRE_CMP(idx, <, 5, "invalid ExcutMutateTag"); \
    static const char* table[] = {name "::Read", name "::Mutate", name "::RAW", name "::WAR", name "::WAW"}; \
    return table[idx];

const char* ExcutSyncEnvAccess::action_name() const
{
    RETURN_MUTATE_TAGGED("SyncEnvAccess");
}

void ExcutSyncEnvAccess::write_args(FILE* file) const
{
    fprintf(file, "[\"int:%u\", \"int:%u\", \"str:%s\"", id_before, id_after, name.c_str());
    for (extent_t n : idx) {
        fprintf(file, ", \"int:%u\"", n);
    }
    fprintf(file, "]");
}

const char* ExcutVisRecord::action_name() const
{
    RETURN_MUTATE_TAGGED("VisRecord");
}

void ExcutVisRecord::write_args(FILE* file) const
{
    fprintf(file, "[\"int:%u\"]", id);
}

const char* ExcutTlSigInterval::action_name() const
{
    RETURN_MUTATE_TAGGED("TlSigInterval");
}

void ExcutTlSigInterval::write_args(FILE* file) const
{
    static_assert(num_vis_flags == 4);
    static_assert(sizeof(qual_bits_by_vis.array[0]) == 4);
    fprintf(file, "[\"int:%u\", \"int:%u\", \"int:%u\", \"int:%u\", \"int:%u\", \"int:%u\"]",
            tid_lo, tid_hi,
            qual_bits_by_vis.array[0],
            qual_bits_by_vis.array[1],
            qual_bits_by_vis.array[2],
            qual_bits_by_vis.array[3]);
}

const char* ExcutPendingAwait::action_name() const
{
    RETURN_MUTATE_TAGGED("PendingAwait");
}

void ExcutPendingAwait::write_args(FILE* file) const
{
    fprintf(file, "[\"int:%u\", \"int:%u\"]", barrier_id, arrive_count);
}

const char* ExcutBarrierAlloc::action_name() const
{
    return "barrier_id";
}

void ExcutBarrierAlloc::write_args(FILE* file) const
{
    fprintf(file, "[\"int:%u\", \"str:%s\"", id, name.c_str());
    for (extent_t n : idx) {
        fprintf(file, ", \"int:%u\"", n);
    }
    fprintf(file, "]");
}

}
