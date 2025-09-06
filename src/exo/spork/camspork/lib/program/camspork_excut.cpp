#include "camspork_excut.hpp"

#include "../util/require.hpp"
#include "../syncv/tl_sig.hpp"

namespace camspork {

void ExcutSyncEnvAccess::write_args(FILE* file) const
{
    fprintf(file, "[\"int:%u\", \"int:%u\", \"str:%s\"", id_before, id_after, name.c_str());
    for (extent_t n : idx) {
        fprintf(file, ", \"int:%u\"", n);
    }
    fprintf(file, "]");
}

void ExcutVisRecord::write_args(FILE* file) const
{
    fprintf(file, "[\"int:%u\", \"int:%u\"]", id, original_qual_bit);
}

void ExcutTlSigInterval::write_args(FILE* file) const
{
    fprintf(file, "[\"int:%u\", \"int:%u\", \"int:%u\", \"str:%s\"]",
            tid_lo, tid_hi, qual_bits, vis_level_name(vis_level));
}

void ExcutPendingAwait::write_args(FILE* file) const
{
    fprintf(file, "[\"int:%u\", \"int:%u\"]", id, arrive_count);
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
