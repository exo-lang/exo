#include "vis_record_history_log.hpp"

#include <sstream>
#include <utility>
#include <vector>

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
    if (is_arrive ^ is_await) {
        s << "  barrier: " << event.barrier_name << '\n';
        s << "  arrive_count: " << v.arrive_count_before << " -> " << v.arrive_count_after << '\n';
        s << "  await_count:  " << v.await_count_before << " -> " << v.await_count_after << '\n';
        if (is_await) {
            s << "  > Updates pending awaits with arrive_count <= " << v.await_max_arrive_count << '\n';
        }
    }
    if (!is_await) {
        s << "  First qual-tl set:\n";
        for (uint32_t bit_index = 0; bit_index < num_qual_tl; ++bit_index) {
            if (1 & (v.L1_qual_bits >> bit_index)) {
                s << "               " << log.lazy_get_qual_tl_name(bit_index) << '\n';
            }
        }
    }
    if (!is_arrive) {
        s << "  Second qual-tl set:\n";
        for (uint32_t bit_index = 0; bit_index < num_qual_tl; ++bit_index) {
            if (1 & (v.L2_full_qual_bits >> bit_index)) {
                s << "               " << log.lazy_get_qual_tl_name(bit_index) << '\n';
            }
            else if (1 & (v.L2_temporal_qual_bits >> bit_index)) {
                s << "    (temporal) " << log.lazy_get_qual_tl_name(bit_index) << '\n';
            }
        }
    }
}

template <typename Stream>
void stream_tid(Stream& s, uint32_t tid, const ThreadCuboid& cuboid_for_domain)
{
    if (tid + 2 <= 1) {
        s << "...";
        return;
    }
    std::vector<uint32_t> coords(cuboid_for_domain.dim());
    const uint32_t domain_num_threads = cuboid_for_domain.domain_num_threads();
    const uint32_t task_index = tid / domain_num_threads;
    s << "task_index = " << task_index;
    uint32_t tmp_tid = tid % domain_num_threads;
    for (uint32_t dim_i = cuboid_for_domain.dim(); dim_i > 0; ) {
        --dim_i;
        const uint32_t domain_c = cuboid_for_domain.domain()[dim_i];
        coords[dim_i] = tmp_tid % domain_c;
        tmp_tid = tmp_tid / domain_c;
    }
    CAMSPORK_REQUIRE_CMP(coords.size(), !=, 0, "invalid empty domain");
    s << " [" << coords[0];
    for (size_t i = 1; i < coords.size(); ++i) {
        s << ", " << coords[i];
    }
    s << ']';
}

template <typename Stream>
void stream_domain(Stream& s, const ThreadCuboid& cuboid_for_domain)
{
    const uint32_t domain_dim = cuboid_for_domain.dim();
    CAMSPORK_REQUIRE_CMP(domain_dim, >=, 1, "invalid empty ThreadCuboid::domain");
    s << '[';
    s << cuboid_for_domain.domain()[0];
    for (uint32_t dim_i = 1; dim_i < domain_dim; ++dim_i) {
        s << ", " << cuboid_for_domain.domain()[dim_i];
    }
    s << ']';
}

template <typename Stream>
void stream_vis_record(
        Stream& s,
        VisRecordHistoryLog& log,
        const ThreadCuboid& cuboid_for_domain,
        const LoggedVisRecordData& data,
        [[maybe_unused]] bool extra_data=false)
{
    for (const TlSigInterval& t: data.visibility_set) {
        s << "  threads: [";
        stream_tid(s, t.tid_lo, cuboid_for_domain);
        s << ",\n            ";
        stream_tid(s, t.tid_hi - 1, cuboid_for_domain);
        s << "], inclusive, formatted w/ domain ";
        stream_domain(s, cuboid_for_domain);
        s << '\n';
        for (uint32_t q_bit_index = 0; q_bit_index < num_qual_tl; ++q_bit_index) {
            int32_t vis_flags = 0;
            for (int32_t i = 0; i < num_vis_flags; ++i) {
                if ((t.qual_bits_by_vis.array[i] >> q_bit_index) & 1) {
                    vis_flags |= 1 << i;
                }
            }
            if (vis_flags != 0) {
                // Print the vis flags for the qual-tl (don't print if empty)
                const std::string& name = log.lazy_get_qual_tl_name(q_bit_index);
                for (size_t i = name.size(); i < 30; ++i) {
                    s << ' ';
                }
                s << name;
                s << " ->";
                for (int i = 0; i < num_vis_flags; ++i) {
                    int32_t vis_flag = 1 << i;
                    const char* v_name = vis_flag_name(vis_flag);
                    bool v_true = 0 != (vis_flag & vis_flags);
                    s << ' ';
                    while (char c = *v_name++) {
                        s << (v_true ? c : '.');
                    }
                }
                s << '\n';
            }
        }
    }
    for (const LoggedPendingAwait& pending_await : data.pending_await_list) {
        s << "  pending await: " << pending_await.barrier_name;
        s << " arrive_count=" << pending_await.arrive_count;
        s << '\n';
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

const std::string& VisRecordHistoryLog::get_barrier_name(barrier_id bar) const
{
    auto iter = barrier_name_map.find(bar);
    if (iter == barrier_name_map.end()) {
        static const std::string missing = "<missing barrier name>";
        return missing;
    }
    else {
        return iter->second;
    }
}

void VisRecordHistoryLog::set_syncv_sync_stmt_info(barrier_id bar, LoggedSyncStmtValues values)
{
    LoggedSyncStmtEvent& event = last_sync_stmt_map[current_stmt_id_bits];
    event.values = std::move(values);
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
    origin.sync_stmt_event.thread_cuboid = current_thread_cuboid;
    origin.stmt_id_bits = current_stmt_id_bits;

    CAMSPORK_REQUIRE_CMP(version_data.size(), ==, version._1_index,
            "new vis_record_version_t should refer to the last vector element");
    CAMSPORK_REQUIRE_CMP(version_origin.size(), ==, version._1_index,
            "new vis_record_version_t should refer to the last vector element");
}

void VisRecordHistoryLog::log_syncv_vis_record_change(
        vis_record_id_t old_id, vis_record_id_t new_id, LoggedVisRecordData new_data, bool debug_printf)
{
    const vis_record_version_t old_version = current_version_id(old_id);

    const vis_record_version_t version = new_version_id(new_id);
    version_origin.emplace_back();
    LoggedVisRecordOrigin& origin = version_origin.back();
    version_data.push_back(std::move(new_data));

    origin.previous_version = old_version;
    origin.stmt_id_bits = current_stmt_id_bits;

    origin.sync_stmt_event = current_sync_stmt_event;

    CAMSPORK_REQUIRE_CMP(version_data.size(), ==, version._1_index,
            "new vis_record_version_t should refer to the last vector element");
    CAMSPORK_REQUIRE_CMP(version_origin.size(), ==, version._1_index,
            "new vis_record_version_t should refer to the last vector element");

    if (debug_printf) {
        std::stringstream before;
        stream_vis_record(before, *this, current_thread_cuboid, version_data.at(old_version._1_index - 1), true);
        std::stringstream after;
        stream_vis_record(after, *this, current_thread_cuboid, version_data.back(), true);
        if (false) {
            printf("[%i] ID %s\n", debug_printf_counter, old_id == new_id ? "SAME" : "CHANGED");
        }
        else {
            printf("[%i] %u -> %u\n", debug_printf_counter, unsigned(old_id), unsigned(new_id));
            printf("vis_record_version_t(%llu)\n", static_cast<long long unsigned>(version._1_index));
        }
        printf("BEFORE:%s\nAFTER:%s\n", std::move(before).str().c_str(), std::move(after).str().c_str());
    }
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

void VisRecordHistoryLog::log_syncv_vis_record_error(vis_record_id_t id, LoggedMissingTlSig fail_tl_sig)
{
    error_stmt_id_bits = current_stmt_id_bits;
    error_vis_record_version = current_version_id(id);
    error_tl_sig = fail_tl_sig;
    error_thread_cuboid = current_thread_cuboid;
}

void VisRecordHistoryLog::add_error_remarks(ProgramEnv* p_env)
{
    add_history_remarks(p_env, error_vis_record_version);

    // Add additional info about the error site.
    if (error_vis_record_version) {
        std::stringstream s;
        s << "VisRecord did not have vis flag \"" << vis_flag_index_name(error_tl_sig.vis_flag_index) << "\" for\n";
        s << "  thread:  ";
        stream_tid(s, error_tl_sig.tid, error_thread_cuboid);
        s << "; domain=";
        stream_domain(s, error_thread_cuboid);
        auto tmp_qual_bits = error_tl_sig.qual_bits;
        s << "\n  qual-tl: ";
        const char* p_sep = "";
        while (tmp_qual_bits) {
            s << p_sep << lazy_get_qual_tl_name(pop_low_bit_index(&tmp_qual_bits));
            p_sep = " OR ";
        }
        s << "\nVisRecord info:\n";

        const auto v_index = error_vis_record_version._1_index - 1;
        CAMSPORK_C_BOUNDSCHECK(v_index, version_data.size());
        stream_vis_record(s, *this, error_thread_cuboid, version_data[v_index]);

        const StmtRef stmt{error_stmt_id_bits};
        static_assert(sizeof(error_stmt_id_bits) == sizeof(stmt));
        p_env->add_remark(stmt, std::move(s).str());
    }
}

void VisRecordHistoryLog::add_last_checked_read_remarks(ProgramEnv* p_env)
{
    add_history_remarks(p_env, last_read_vis_record_version);
}

void VisRecordHistoryLog::add_last_checked_mutate_remarks(ProgramEnv* p_env)
{
    add_history_remarks(p_env, last_mutate_vis_record_version);
}

void VisRecordHistoryLog::add_history_remarks(ProgramEnv* p_env, vis_record_version_t last_version)
{
    // Earlier remarks should appear higher than later remarks for the same stmt.
    // We will first remark about the last executed sync for each sync stmt.
    for (const auto& pair : last_sync_stmt_map) {
        const uint32_t stmt_id_bits = pair.first;
        const LoggedSyncStmtEvent& event = pair.second;
        const StmtRef stmt{stmt_id_bits};
        static_assert(sizeof(stmt_id_bits) == sizeof(stmt));
        std::stringstream s;
        s << "Last sync recorded:\n";
        stream_sync_stmt_event(s, *this, event);
        p_env->add_remark(stmt, std::move(s).str());
    }

    // We will now remark on the history of the given VisRecord.
    // We search back in time until we found when the VisRecord was originally created
    // and remark on it + all transitions to the present state.
    // We want old state to appear first, so we have to buffer remarks and reverse them.
    std::vector<std::pair<StmtRef, std::string>> remarks_buffer;
    vis_record_version_t tmp_v = last_version;
    while (tmp_v) {
        std::stringstream s;
        const auto v_index = tmp_v._1_index - 1;
        CAMSPORK_C_BOUNDSCHECK(v_index, version_data.size());
        CAMSPORK_C_BOUNDSCHECK(v_index, version_origin.size());
        const LoggedVisRecordData& data = version_data[v_index];
        const LoggedVisRecordOrigin& origin = version_origin[v_index];

        if (origin.previous_version) {
            s << origin.sync_stmt_event.values.sync_stmt_name << " recorded, which caused VisRecord change:\n";
            stream_sync_stmt_event(s, *this, origin.sync_stmt_event);
            s << "BEFORE:\n";
            const auto previous_v_index = origin.previous_version._1_index - 1;
            CAMSPORK_C_BOUNDSCHECK(previous_v_index, version_data.size());
            const LoggedVisRecordData& previous_data = version_data[previous_v_index];
            stream_vis_record(s, *this, origin.sync_stmt_event.thread_cuboid, previous_data);
            s << "AFTER:\n";
        }
        else {
            s << "New VisRecord:\n";
        }
        stream_vis_record(s, *this, origin.sync_stmt_event.thread_cuboid, data);

        const StmtRef stmt{origin.stmt_id_bits};
        static_assert(sizeof(origin.stmt_id_bits) == sizeof(stmt));
        remarks_buffer.emplace_back(stmt, std::move(s).str());

        tmp_v = origin.previous_version;
    }

    while (!remarks_buffer.empty()) {
        const auto& pair = remarks_buffer.back();
        p_env->add_remark(pair.first, std::move(pair.second));
        remarks_buffer.pop_back();
    }
}

}
