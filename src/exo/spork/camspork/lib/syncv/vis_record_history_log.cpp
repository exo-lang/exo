#include "vis_record_history_log.hpp"

#include <sstream>

#include "../util/require.hpp"

// NOTE: for the most part, I try to keep this "core" syncv (synchronization validation) code
// separate from the less-than-great program interpreter. To decouple this code from the interpreter,
// remove the "add_remarks" functions.
#include "../program/exec.hpp"

namespace camspork {

namespace {

template <typename Stream>
void stream_sync_stmt_event(Stream& s, VisRecordHistoryLog& log, const LoggedSyncStmtEvent& event)
{
    const LoggedSyncStmtValues v = event.values;
    const bool is_await = v.L1_qual_bits == 0;
    const bool is_arrive = v.L2_temporal_qual_bits == 0;
    s << "  thread cuboid: " << event.thread_cuboid << '\n';
    if (is_arrive || is_await) {
        s << "  barrier: " << event.barrier_name << '\n';
        s << "  arrive_count: " << v.arrive_count_before << " -> " << v.arrive_count_after << '\n';
        s << "  await_count:  " << v.await_count_before << " -> " << v.await_count_after << '\n';
    }
    if (!is_await) {
        s << "  First qual-tl set:\n";
        for (uint32_t bit_index = 0; bit_index < log.num_qual_tl; ++bit_index) {
            if (1 & (v.L1_qual_bits >> bit_index)) {
                s << "               " << log.lazy_get_qual_tl_name(bit_index) << '\n';
            }
        }
    }
    if (!is_arrive) {
        s << "  Second qual-tl set:\n";
        for (uint32_t bit_index = 0; bit_index < log.num_qual_tl; ++bit_index) {
            if (1 & (v.L2_full_qual_bits >> bit_index)) {
                s << "               " << log.lazy_get_qual_tl_name(bit_index) << '\n';
            }
            else if (1 & (v.L2_temporal_qual_bits >> bit_index)) {
                s << "    (temporal) " << log.lazy_get_qual_tl_name(bit_index) << '\n';
            }
        }
    }
}

}  // end anonymous namespace

void VisRecordHistoryLog::set_qual_tl_name(uint32_t index, std::string name)
{
    static_assert(sizeof(qual_bits_t) == 4, "Update the below");
    CAMSPORK_REQUIRE_CMP(index, <, num_qual_tl, "Out-of-range Qual_tl index");
    qual_tl_names[index] = std::move(name);
}

void VisRecordHistoryLog::set_barrier_name(barrier_id bar, std::string name)
{
    barrier_name_map[bar] = std::move(name);
}

const std::string& VisRecordHistoryLog::lazy_get_qual_tl_name(uint32_t i)
{
    CAMSPORK_REQUIRE_CMP(i, <, num_qual_tl, "out of range qual-tl index");
    std::string& name = qual_tl_names[i];
    if (name.empty()) {
        name = "qual_bits_t(" + std::to_string(1u << i) + ")";
    }
    return name;
}

void VisRecordHistoryLog::set_syncv_sync_stmt_info(barrier_id bar, LoggedSyncStmtValues values)
{
    LoggedSyncStmtEvent& event = last_sync_stmt_map[current_stmt_id_bits];
    event.values = values;
    event.barrier_name = barrier_name_map[bar];
    event.thread_cuboid = current_thread_cuboid;
    current_sync_stmt_event = event;
}

void VisRecordHistoryLog::log_syncv_new_vis_record(vis_record_id_t id, LoggedVisRecordData data)
{
    const vis_record_version_t version = new_version_id(id);
    version_origin.emplace_back();
    LoggedVisRecordOrigin& origin = version_origin.back();
    version_data.push_back(std::move(data));

    origin.previous_version = vis_record_version_t{0};
    origin.thread_cuboid = current_thread_cuboid;
    origin.stmt_id_bits = current_stmt_id_bits;

    CAMSPORK_REQUIRE_CMP(version_data.size(), ==, version._1_index,
            "new vis_record_version_t should refer to the last vector element");
    CAMSPORK_REQUIRE_CMP(version_origin.size(), ==, version._1_index,
            "new vis_record_version_t should refer to the last vector element");
}

void VisRecordHistoryLog::log_syncv_vis_record_change(
        vis_record_id_t old_id, vis_record_id_t new_id, LoggedVisRecordData new_data)
{
    const vis_record_version_t old_version = current_version_id(old_id);

    const vis_record_version_t version = new_version_id(new_id);
    version_origin.emplace_back();
    LoggedVisRecordOrigin& origin = version_origin.back();
    version_data.push_back(std::move(new_data));

    origin.previous_version = old_version;
    origin.thread_cuboid = current_thread_cuboid;
    origin.stmt_id_bits = current_stmt_id_bits;

    origin.sync_stmt_event = current_sync_stmt_event;

    CAMSPORK_REQUIRE_CMP(version_data.size(), ==, version._1_index,
            "new vis_record_version_t should refer to the last vector element");
    CAMSPORK_REQUIRE_CMP(version_origin.size(), ==, version._1_index,
            "new vis_record_version_t should refer to the last vector element");
}

void VisRecordHistoryLog::log_syncv_vis_record_checked(vis_record_id_t id, bool is_mutate)
{
    const vis_record_version_t version = current_version_id(id);
    if (is_mutate) {
        last_mutate_vis_record_version = version;
    }
    else {
        last_read_vis_record_version = version;
    }
}

void VisRecordHistoryLog::log_syncv_vis_record_error(vis_record_id_t id, TlSig fail_tl_sig)
{
    error_stmt_id_bits = current_stmt_id_bits;
    error_vis_record_version = current_version_id(id);
    error_tl_sig = fail_tl_sig;
    error_thread_cuboid = current_thread_cuboid;
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
    for (const auto& pair : last_sync_stmt_map) {
        const uint32_t stmt_id_bits = pair.first;
        const LoggedSyncStmtEvent& event = pair.second;
        const StmtRef stmt{stmt_id_bits};
        static_assert(sizeof(stmt_id_bits) == sizeof(stmt));
        std::stringstream s;
        s << "Last barrier recorded:\n";
        stream_sync_stmt_event(s, *this, event);
        p_env->add_remark(stmt, s.str());
    }
}

}
