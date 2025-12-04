#pragma once

#include <map>
#include <stdint.h>
#include <string>
#include <vector>

#include "tl_sig.hpp"
#include "syncv_types.hpp"
#include "../util/require.hpp"

namespace camspork {

struct vis_record_version_t
{
    uint64_t _1_index;

    explicit operator bool() const
    {
        return _1_index != 0;
    }
};

struct LoggedPendingAwait
{
    std::string barrier_name;
    int32_t arrive_count;
};

struct LoggedVisRecordData
{
    uint8_t original_qual_tl;
    std::vector<TlSigInterval> visibility_set;
    std::vector<LoggedPendingAwait> pending_await_list;
};

struct LoggedSyncStmtValues
{
    uint32_t L1_qual_bits;
    uint32_t L2_full_qual_bits;
    uint32_t L2_temporal_qual_bits;
    uint32_t arrive_count_before;
    uint32_t arrive_count_after;
    uint32_t await_count_before;
    uint32_t await_count_after;
    int32_t await_max_arrive_count;
};

struct LoggedSyncStmtEvent
{
    LoggedSyncStmtValues values;
    std::string barrier_name;  // empty for fence
    ThreadCuboid thread_cuboid;
};

struct LoggedVisRecordOrigin
{
    using stmt_id_bits_t = uint32_t;
    vis_record_version_t previous_version;  // 0 if no previous version
    LoggedSyncStmtEvent sync_stmt_event;  // Unused if previous_version = 0, except thread_cuboid.
    stmt_id_bits_t stmt_id_bits;  // Stmt that led to this VisRecord being created (access) or changing (sync).
};

class ProgramEnv;

class VisRecordHistoryLog
{
  public:
    using stmt_id_bits_t = LoggedVisRecordOrigin::stmt_id_bits_t;
    using vis_record_id_t = uint32_t;

  private:
    // State passed through to logged values.
    stmt_id_bits_t current_stmt_id_bits = 0;
    ThreadCuboid current_thread_cuboid{};
    LoggedSyncStmtEvent current_sync_stmt_event{};

    // Internal tables.
    // We store the history of each VisRecord ever recorded, so we have to translate vis_record_id_t
    // (which may be re-used) to vis_record_version_t (which refers to a specific snapshot of a VisRecord
    // independent of any future changes made to it).
    std::vector<vis_record_version_t> id_to_version;
    vis_record_version_t version_counter{0};
    std::map<barrier_id, std::string> barrier_name_map;
    std::map<stmt_id_bits_t, LoggedSyncStmtEvent> last_sync_stmt_map;
    std::vector<LoggedVisRecordData> version_data;  // Indexed by [vis_record_version_t::_1_index - 1]
    std::vector<LoggedVisRecordOrigin> version_origin;  // Indexed by [vis_record_version_t::_1_index - 1]
    std::string qual_tl_names[num_qual_tl];

    // Error info, if detected.
    stmt_id_bits_t error_stmt_id_bits = 0;
    vis_record_version_t error_vis_record_version{};
    TlSig error_tl_sig{};
    ThreadCuboid error_thread_cuboid{};

    // Last read VisRecord and mutate VisRecord that was checked, i.e.,
    // added by a read/mutate and then later encountered by a different mutate/read.
    vis_record_version_t last_read_vis_record_version{};
    vis_record_version_t last_mutate_vis_record_version{};

    int debug_printf_counter = 0;

  public:
    // ******************************************************************************************
    // Callbacks intended for the program interpreter.
    // ******************************************************************************************
    void set_stmt_id_bits(stmt_id_bits_t bits)
    {
        current_stmt_id_bits = bits;
        debug_printf_counter++;
    }
    void set_thread_cuboid(const ThreadCuboid& arg)
    {
        current_thread_cuboid = arg;
    }
    void set_qual_tl_name(uint32_t index, std::string name);
    void set_barrier_name(barrier_id bar, std::string name);
    const std::string& lazy_get_qual_tl_name(uint32_t i);
    const std::string& get_barrier_name(barrier_id bar) const;

    // ******************************************************************************************
    // Callbacks intended for the syncv (synchronization validation) implementation.
    // ******************************************************************************************
    void set_syncv_sync_stmt_info(barrier_id, LoggedSyncStmtValues);  // Affects later log_syncv_vis_record_change.
    void log_syncv_new_vis_record(vis_record_id_t id, LoggedVisRecordData data);
    void log_syncv_vis_record_change(
            vis_record_id_t old_id, vis_record_id_t new_id, LoggedVisRecordData new_data, bool debug_printf);
    void log_syncv_vis_record_checked(vis_record_id_t id, bool is_mutate);
    void log_syncv_vis_record_error(vis_record_id_t id, TlSig fail_tl_sig);

    // ******************************************************************************************
    // Insert remarks tracking the history of a certain VisRecord.
    // ******************************************************************************************
    void add_error_remarks(ProgramEnv* p_env);
    void add_last_checked_read_remarks(ProgramEnv* p_env);
    void add_last_checked_mutate_remarks(ProgramEnv* p_env);
    void add_history_remarks(ProgramEnv*, vis_record_version_t last_version);  // For debugging syncv impl

  private:
    vis_record_version_t current_version_id(vis_record_id_t id)
    {
        CAMSPORK_REQUIRE_CMP(id, <, id_to_version.size(), "id never seen before");
        const vis_record_version_t version = id_to_version[id];
        CAMSPORK_REQUIRE(version, "id never seen before");
        return version;
    }

    // version_offset is the first version id allocated.
    vis_record_version_t new_version_id(vis_record_id_t id)
    {
        if (id_to_version.size() <= id) {
            id_to_version.resize(id + 1);
        }
        ++version_counter._1_index;
        id_to_version[id] = version_counter;
        return version_counter;
    }
};

}
