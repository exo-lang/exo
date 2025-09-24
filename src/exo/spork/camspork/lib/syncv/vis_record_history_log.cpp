#include "vis_record_history_log.hpp"

#include "../util/require.hpp"

// NOTE: for the most part, I try to keep this "core" syncv (synchronization validation) code
// separate from the less-than-great program interpreter. To decouple this code from the interpreter,
// remove the "add_remarks" functions.
#include "../program/exec.hpp"

namespace camspork {

namespace {

}  // end anonymous namespace

void VisRecordHistoryLog::set_qual_tl_name(uint32_t index, std::string name)
{
    static_assert(sizeof(qual_bits_t) == 4, "Update the below");
    CAMSPORK_REQUIRE_CMP(index, <, 32, "Only support up to 32 Qual_tl");
    qual_tl_names[index] = std::move(name);
}

void VisRecordHistoryLog::set_barrier_name(barrier_id bar, std::string name)
{
    barrier_name_map[bar] = std::move(name);
}

void VisRecordHistoryLog::add_error_remarks(ProgramEnv* p_env)
{
    // TODO add additional info about the error site.
    add_history_remarks(p_env, error_vis_record_version);
}

void VisRecordHistoryLog::add_last_checked_read_remarks(ProgramEnv* p_env)
{
    add_history_remarks(p_env, last_read_vis_record_version);
}

void VisRecordHistoryLog::add_last_checked_mutate_remarks(ProgramEnv* p_env)
{
    add_history_remarks(p_env, last_mutate_vis_record_version);
}

void VisRecordHistoryLog::add_history_remarks(ProgramEnv* p_env, vis_record_version_t version)
{
    for (const auto& pair : last_sync_stmt_info_map) {
    
    }
}

}
