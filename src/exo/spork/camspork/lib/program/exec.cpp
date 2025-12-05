#include "exec.hpp"

#include <array>
#include <errno.h>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "builder.hpp"
#include "camspork_excut.hpp"
#include "grammar.hpp"
#include "log_request.hpp"
#include "print.hpp"
#include "../syncv/vis_record_history_log.hpp"
#include "../util/cuboid_util.hpp"

#define CAMSPORK_EXEC_ALWAYS_INLINE __attribute__((always_inline))
// #define CAMSPORK_EXEC_ALWAYS_INLINE

namespace camspork
{

class SwapThreadCuboid
{
    ThreadCuboid saved;
    ThreadCuboid* p_restore;
  public:
    // Sets *p_cuboid = new_cuboid, and restores old value upon destruction.
    [[nodiscard]] SwapThreadCuboid(ThreadCuboid* p_cuboid, ThreadCuboid new_cuboid)
    {
        saved = *p_cuboid;
        p_restore = p_cuboid;
        *p_cuboid = new_cuboid;
    };

    [[nodiscard]] SwapThreadCuboid(ThreadCuboid* p_cuboid)
    {
        saved = *p_cuboid;
        p_restore = p_cuboid;
    };

    SwapThreadCuboid(SwapThreadCuboid&&) = delete;

    ~SwapThreadCuboid()
    {
        *p_restore = saved;
    }
};

template <bool AllowLog>
class ProgramExecLogBase
{
  protected:
    FILE* excut_file = nullptr;

    // if AllowLog && excut_file, then the per-stmt/per-expr callbacks in ProgramExec may push excut actions to log.
    // These should be flushed for each statement using flush_excut_log, if applicable.
    std::vector<std::unique_ptr<ExcutBaseAction>> excut_actions;

    bool excut_first_time = true;

    ProgramExecLogBase() = default;
    ProgramExecLogBase(ProgramExecLogBase&&) = delete;

    ~ProgramExecLogBase()
    {
        if (excut_file) {
            fprintf(excut_file, "]\n");
            fclose(excut_file);
        }
    }
};

template <>
struct ProgramExecLogBase<false>
{
};

// Borrowed reference wrapper around ProgramEnv, to implement actual per-node-type execution.
template <bool AllowLog>
class ProgramExec : public ProgramExecLogBase<AllowLog>
{
    size_t buffer_size;
    const char* p_buffer;
    ProgramEnv& env;
    SinglePositionFilter single_position_filter;
    std::vector<extent_t> single_position_all_ones;  // [1 for x in single_position_filter.idx]

    std::vector<extent_t> tmp_extent;
    std::vector<extent_t> tmp_offset;
    std::vector<barrier_id> tmp_all_barriers;
    StmtRef current_stmt{};
    bool added_error_remark = false;

  public:
    ProgramExec(ProgramEnv* p_self, SinglePositionFilter _single_position_filter)
      : buffer_size(p_self->program_buffer_size)
      , p_buffer(p_self->p_program_buffer.get())
      , env(*p_self)
      , single_position_filter(std::move(_single_position_filter))
    {
        if (single_position_filter) {
            single_position_all_ones = std::vector<extent_t>(single_position_filter.idx.size(), 1);
        }
    }

    ProgramExec(ProgramEnv* p_self, const char* p_excut_filename, SinglePositionFilter _single_position_filter)
      : ProgramExec(p_self, std::move(_single_position_filter))
    {
        static_assert(AllowLog, "Can't open excut log file if C++ functionality not enabled");
        if (p_excut_filename) {
            FILE*& file = this->excut_file;
            file = fopen(p_excut_filename, "w");
            if (!file) {
                throw std::runtime_error(std::string(p_excut_filename) + ": " + strerror(errno));
            }
            fprintf(file, "[\n");
        }
    }

    // ******************************************************************************************
    // Many nodes define array indices as a VLA of ExprRef.
    // We provide a stripped-down iterator over these exprs, evaluated as values.
    // ******************************************************************************************
    struct ExprIterator
    {
        const ExprRef* p_node_ref;
        const ProgramExec<AllowLog>* p_exec;

        intptr_t operator-(ExprIterator other) const
        {
            return p_node_ref - other.p_node_ref;
        }

        ExprIterator operator+(intptr_t i) const
        {
            return ExprIterator{p_node_ref + i, p_exec};
        }

        ExprIterator& operator++ ()
        {
            p_node_ref++;
            return *this;
        }

        value_t operator* () const
        {
            return p_exec->eval(*p_node_ref);
        }

        bool operator==(ExprIterator other) const
        {
            return p_node_ref == other.p_node_ref;
        }

        bool operator!=(ExprIterator other) const
        {
            return p_node_ref != other.p_node_ref;
        }
    };

    template <typename Node>
    ExprIterator expr_vla_begin(const Node* node) const
    {
        return ExprIterator{&node_vla_get_unsafe(node, 0), this};
    };

    template <typename Node>
    ExprIterator expr_vla_end(const Node* node) const
    {
        return expr_vla_begin(node) + node->camspork_vla_size;
    };

    // Evaluate expressions as tuple of extent values and store into tmp_extent.
    template <typename Node>
    void eval_tmp_extent(const Node* node)
    {
        const uint32_t dim = node->camspork_vla_size;
        tmp_extent.resize(dim);
        for (uint32_t i = 0; i < dim; ++i) {
            if constexpr (std::is_same_v<typename Node::camspork_vla_type, OffsetExtentExpr>) {
                tmp_extent[i] = eval_extent_t(node_vla_get(node, i).extent_e);
            }
            else {
                tmp_extent[i] = eval_extent_t(node_vla_get(node, i));
            }
        }
    }

    // Evaluate expressions as tuple of offset values and store into tmp_offset.
    template <typename Node>
    void eval_tmp_offset(const Node* node)
    {
        const uint32_t dim = node->camspork_vla_size;
        tmp_offset.resize(dim);
        for (uint32_t i = 0; i < dim; ++i) {
            if constexpr (std::is_same_v<typename Node::camspork_vla_type, OffsetExtentExpr>) {
                tmp_offset[i] = eval_extent_t(node_vla_get(node, i).offset_e);
            }
            else if constexpr (std::is_same_v<typename Node::camspork_vla_type, ArriveIdx>) {
                tmp_offset[i] = eval_extent_t(node_vla_get(node, i).idx);
            }
            else {
                tmp_offset[i] = eval_extent_t(node_vla_get(node, i));
            }
        }
    }

    template <typename Stream>
    static void print_idx_helper(Stream& stream, const std::vector<extent_t>& idx)
    {
        if (!idx.empty()) {
            stream << '[' << idx[0];
            for (auto it = idx.begin() + 1; it != idx.end(); ++it) {
                stream << ", " << *it;
            }
            stream << ']';
        }
    }

    template <typename Input>
    [[nodiscard]] bool filter_single_position_input(
            Varname name, const VarSlotEntry<assignment_record_id>& slot, Input* p_input)
    {
        if (!single_position_filter) {
            return true;
        }
        if (name.slot() != single_position_filter.name.slot()) {
            return false;
        }
        const std::vector<extent_t>& extent = slot.extent();
        const size_t dim = extent.size();
        CAMSPORK_REQUIRE_CMP(dim, ==, single_position_filter.idx.size(),
                             "Wrong dimensionality for indices of single_position_filter");
        if constexpr (std::is_same_v<Input, AssignmentRecordWindow>) {
            AssignmentRecordWindow& window = *p_input;
            CAMSPORK_REQUIRE_CMP(size_t(window.end_offset - window.begin_offset), == , dim, "internal error");
            CAMSPORK_REQUIRE_CMP(size_t(window.end_inner_extent - window.begin_inner_extent), == , dim, "internal error");
            for (size_t dim_i = 0; dim_i < dim; ++dim_i) {
                static_assert(std::is_unsigned_v<extent_t>, "Relies on wraparound to work");
                const extent_t tmp = single_position_filter.idx[dim_i] - window.begin_offset[dim_i];
                const bool in_bounds = tmp < window.begin_inner_extent[dim_i];
                if (!in_bounds) {
                    return false;
                }
            }
            // Adjust the window so it only covers the single filtered position.
            CAMSPORK_REQUIRE_CMP(single_position_all_ones.size(), ==, dim, "internal error");
            window.begin_offset = &*single_position_filter.idx.begin();
            window.end_offset = &*single_position_filter.idx.end();
            window.begin_inner_extent = &*single_position_all_ones.begin();
            window.end_inner_extent = &*single_position_all_ones.end();
            return true;
        }
        else {
            // Note, we do not assume in this code that the single_position_filter.idx is in-bounds
            // for the current size of the allocation.
            static_assert(std::is_same_v<Input, assignment_record_id*>);
            assignment_record_id* p_id = *p_input;
            size_t linear_index = size_t(p_id - slot.data());
            size_t filter_linear_index = 0;
            for (size_t dim_i = 0; dim_i < dim; ++dim_i) {
                filter_linear_index *= extent[dim_i];
                filter_linear_index += single_position_filter.idx[dim_i];
            }
            return linear_index == filter_linear_index;
        }
    }

    template <typename Node>
    auto prepare_logger(const Node* node, const ThreadCuboid& thread_cuboid, StmtRef stmt)
    {
        using Logger = std::conditional_t<AllowLog, SyncvLogRequest, decltype(nullptr)>;
        Logger logger{};
        if constexpr (AllowLog) {
            if constexpr (!std::is_same_v<Node, Fence> && !std::is_same_v<Node, JoinThreads>) {
                logger.var_str_name = env.str_name(node->name);
            }
            logger.p_excut_actions = this->excut_file ? &this->excut_actions : nullptr;
            if (env.history_enable) {
                static_assert(sizeof(VisRecordHistoryLog::stmt_id_bits_t) == sizeof(stmt.raw_data));
                logger.p_history_log = &env.history_log;
                logger.p_history_log->set_thread_cuboid(thread_cuboid);
                logger.p_history_log->set_stmt_id_bits(stmt.raw_data);
            }
        }
        return logger;
    }

    template <typename Node>
    auto prepare_logger(const Node* node, const ThreadCuboid& thread_cuboid)
    {
        return prepare_logger(node, thread_cuboid, env.stmt_ref_from_ptr(node));
    }


    // ******************************************************************************************
    // EXECUTE STATEMENT
    // ******************************************************************************************
    CAMSPORK_EXEC_ALWAYS_INLINE
    void exec(StmtRef s)
    {
        if (s) {
            s.dispatch(*this, buffer_size, p_buffer);
        }
    }

    template <typename Stmt>
    void operator() (const Stmt* node)
    {
        // Specialized per-Stmt-type execution (exec_impl) wrapped with common code.
        added_error_remark = false;
        const StmtRef stmt_before = current_stmt;
        auto Finally = [stmt_before, this] {
            current_stmt = stmt_before;
            this->flush_excut_log();
        };
        try {
            current_stmt = env.stmt_ref_from_ptr(node);
            exec_impl(node);
        }
        catch (const std::runtime_error& err) {
            if (!added_error_remark) {
                env.add_remark(env.stmt_ref_from_ptr(node), err.what());
                added_error_remark = true;
            }
            Finally();
            throw;
        }
        catch (...) {
            Finally();
            throw;
        }
        Finally();
    }

    void exec_impl(const SyncEnvReadSingle* node)
    {
        exec_sync_env_impl(node, env.stmt_ref_from_ptr(node));
    }

    void exec_impl(const SyncEnvReadWindow* node)
    {
        exec_sync_env_impl(node, env.stmt_ref_from_ptr(node));
    }

    void exec_impl(const SyncEnvReadMulticast* node)
    {
        exec_sync_env_impl(node, env.stmt_ref_from_ptr(node));
    }

    void exec_impl(const SyncEnvMutateSingle* node)
    {
        exec_sync_env_impl(node, env.stmt_ref_from_ptr(node));
    }

    void exec_impl(const SyncEnvMutateWindow* node)
    {
        exec_sync_env_impl(node, env.stmt_ref_from_ptr(node));
    }

    void exec_impl(const SyncEnvMutateMulticast* node)
    {
        exec_sync_env_impl(node, env.stmt_ref_from_ptr(node));
    }

    void operator() (const TrailingBarrierExpr* node)
    {
        fill_tmp_offset_barriers(node);
    }

    template <bool IsMutate, bool IsWindow, bool IsMulticast>
    void exec_sync_env_impl(const SyncEnvAccessNode<IsMutate, IsWindow, IsMulticast>* node, StmtRef stmt_ref)
    {
        const ThreadCuboid& thread_cuboid = env.prepare_thread_cuboid();

        if (!single_position_filter.accepts_name(node->name)) {
            return;
        }

        CAMSPORK_REQUIRE_CMP(IsMutate, ==, bool(node->access_flags & access_flag_mutate),
                "implementation didn't set static IsMutate type to match requested mutate flag");

        SyncvAccessInfo access{};
        access.is_ooo = bool(node->access_flags & access_flag_ooo);
        access.is_convergent = bool(node->access_flags & access_flag_convergent);
        access.is_write_only = bool(node->access_flags & access_flag_write_only);
        access.force_shared_vis_record = bool(node->access_flags & access_flag_force_shared_vis_record);
        access.initial_qual_bit = node->initial_qual_bit;
        access.extended_qual_bits = node->extended_qual_bits;
        access.atomic_qual_bits = node->get_atomic_qual_bits();
        access.barrier_count = 0;
        access.trailing_barriers = nullptr;

        // Prepare tmp_all_barriers (this uses tmp_offset as well).
        if (node->trailing_barrier_expr) {
            node->trailing_barrier_expr.dispatch(*this, buffer_size, p_buffer);
            access.barrier_count = uint32_t(tmp_all_barriers.size());
            access.trailing_barriers = tmp_all_barriers.data();
        }

        // Prepare input: window or single assignment record.
        // If multicasting, the input is a list of individual assignment records to update.
        using Input = std::conditional_t<node->is_window, AssignmentRecordWindow, assignment_record_id*>;
        using InputList = std::conditional_t<node->is_multicast, std::vector<Input>, std::array<Input, 1>>;
        InputList input_list;
        VarSlotEntry<assignment_record_id>& slot = env.sync_slot(node->name);
        bool have_input = true;

        if constexpr (node->is_multicast) {
            static_assert(!node->is_window, "we can only multicast a single position");
            auto callback = [&] (auto& callback_slot, auto linear_idx)
            {
                input_list.push_back(&callback_slot.data()[linear_idx]);
            };
            eval_tmp_offset_multicast(node, slot, callback);
        }
        else if constexpr (node->is_window) {
            const std::vector<extent_t>& alloc_extent = slot.extent();
            const size_t dim = alloc_extent.size();
            eval_tmp_offset(node);
            eval_tmp_extent(node);
            CAMSPORK_REQUIRE_CMP(tmp_extent.size(), ==, dim, "Wrong dimensionality for indexing");
            CAMSPORK_REQUIRE_CMP(tmp_offset.size(), ==, dim, "Wrong dimensionality for indexing");

            // Clip window to allocation.
            for (size_t i = 0; i < dim; ++i) {
                if (tmp_offset[i] >= alloc_extent[i]) {
                    have_input = false;
                }
                else {
                    tmp_extent[i] = std::min(tmp_extent[i], alloc_extent[i] - tmp_offset[i]);
                }
            }

            AssignmentRecordWindow& input = input_list[0];
            input.base = slot.data();
            input.begin_outer_extent = &*slot.extent().begin();
            input.end_outer_extent = &*slot.extent().end();
            input.begin_offset = &*tmp_offset.begin();
            input.end_offset = &*tmp_offset.end();
            input.begin_inner_extent = &*tmp_extent.begin();
            input.end_inner_extent = &*tmp_extent.end();
        }
        else {
            eval_tmp_offset(node);
            const std::vector<extent_t>& alloc_extent = slot.extent();
            const size_t dim = alloc_extent.size();
            CAMSPORK_REQUIRE_CMP(tmp_offset.size(), ==, dim, "Wrong dimensionality for indexing");

            // Bounds check, and disable checking if out-of-bounds.
            for (size_t i = 0; i < dim; ++i) {
                have_input &= tmp_offset[i] < alloc_extent[i];
            }
            if (have_input) {
                input_list[0] = &slot.idx(tmp_offset.begin(), tmp_offset.end());
            }
        }

        if (!have_input) {
            return;
        }

        for (Input& input : input_list) {
            // Prepare excut debug logger if applicable.
            auto logger = prepare_logger(node, thread_cuboid, stmt_ref);
            if constexpr (AllowLog) {
                if constexpr (!node->is_window) {
                    logger.idx_for_single = slot.idx_from_linear(input - slot.data());
                }
            }

            // Call into syncv table (a lot of duplicated code with SyncEnvFreeShard, not great)
            try {
                if (!filter_single_position_input(node->name, slot, &input)) {
                    // Skip if instructed to by SinglePositionFilter.
                }
                else if constexpr (node->is_mutate) {
                    on_rw(env.p_syncv_table.get(), input, thread_cuboid, access, logger);
                }
                else {
                    on_r(env.p_syncv_table.get(), input, thread_cuboid, access, logger);
                }
            }
            catch (const SyncvCheckFail& exc) {
                // If !is_window, we can't trust linear_index_in_input
                // as we passed an already-offset pointer to SyncvTable.
                size_t linear_index;
                if constexpr (node->is_window) {
                    linear_index = exc.linear_index_in_input();
                }
                else {
                    linear_index = size_t(input - slot.data());
                }
                env._syncv_fail_var = node->name;
                env._syncv_fail_idx = slot.idx_from_linear(linear_index);
                std::stringstream s;
                s << exc.what() << " @ " << env.str_name(node->name);
                print_idx_helper(s, env._syncv_fail_idx);
                env.add_remark(stmt_ref, s.str());
                added_error_remark = true;
                throw;
            }
        }
        env.maybe_syncv_debug_validate();
    }

    void exec_impl(const SyncEnvFreeShard* node)
    {
        const ThreadCuboid& thread_cuboid = env.prepare_thread_cuboid();

        if (!single_position_filter.accepts_name(node->name)) {
            return;
        }

        constexpr bool IsMutate = true;
        SyncvAccessInfo access{};
        access.is_ooo = false;
        access.is_convergent = false;  // All threads must be prepared to free the memory.
        access.force_shared_vis_record = false;
        access.initial_qual_bit = 0;
        access.extended_qual_bits = node->extended_qual_bits;
        access.atomic_qual_bits = 0;
        access.barrier_count = 0;
        access.trailing_barriers = nullptr;

        // Translate given indices into a window.
        // The given indices are "points" and the remainder are "intervals"
        // comprising the full allocated extent on that dimension.
        VarSlotEntry<assignment_record_id>& slot = env.sync_slot(node->name);
        eval_tmp_offset(node);
        const auto& alloc_extent = slot.extent();
        const size_t alloc_dim = alloc_extent.size();
        CAMSPORK_REQUIRE_CMP(tmp_offset.size(), <=, alloc_dim, "Too many dimensions for SyncEnvFreeShard");
        std::unique_ptr<extent_t[]> p_data(new extent_t[2 * alloc_dim]);
        extent_t* p_offset = &p_data[0];
        extent_t* p_inner_extent = &p_data[alloc_dim];
        for (size_t i = 0; i < alloc_dim; ++i) {
            if (i < tmp_offset.size()) {
                p_offset[i] = tmp_offset[i];
                p_inner_extent[i] = 1;
            }
            else {
                p_offset[i] = 0;
                p_inner_extent[i] = alloc_extent[i];
            }
        }

        // Prepare input window.
        AssignmentRecordWindow input;
        input.base = slot.data();
        input.begin_outer_extent = &alloc_extent[0];
        input.end_outer_extent = &alloc_extent[alloc_dim];
        input.begin_offset = &p_offset[0];
        input.end_offset = &p_offset[alloc_dim];
        input.begin_inner_extent = &p_inner_extent[0];
        input.end_inner_extent = &p_inner_extent[alloc_dim];

        // Prepare debug logger if applicable.
        auto logger = prepare_logger(node, thread_cuboid);

        // Call into syncv table (a lot of duplicated code with SyncEnvAccessNode, not great).
        try {
            if (!filter_single_position_input(node->name, slot, &input)) {
                // Skip if instructed to by SinglePositionFilter.
            }
            else {
                on_check_free(env.p_syncv_table.get(), input, thread_cuboid, access, logger);
            }
        }
        catch (const SyncvCheckFail& exc) {
            // Unlike SyncEnvAccessNode, we wrap the error message with free(...)
            env._syncv_fail_var = node->name;
            env._syncv_fail_idx = slot.idx_from_linear(exc.linear_index_in_input());
            std::stringstream s;
            s << exc.what() << " @ free(" << env.str_name(node->name);
            print_idx_helper(s, env._syncv_fail_idx);
            s << ")";
            env.add_remark(env.stmt_ref_from_ptr(node), s.str());
            added_error_remark = true;
            throw;
        }
        env.maybe_syncv_debug_validate();
    }

    void exec_impl(const MutateValue* node)
    {
        value_t& lhs = env.value_slot(node->name).idx(expr_vla_begin(node), expr_vla_end(node));
        const value_t rhs = eval(node->rhs);
        lhs = eval_binop(node->op, lhs, rhs);
    }

    void exec_impl(const Fence* node)
    {
        SyncvFence param{};
        param.transitive = bool(node->V1_transitive);
        param.L1_qual_bits = node->L1_qual_bits;
        param.L2_full_qual_bits = node->L2_full_qual_bits;
        param.L2_temporal_qual_bits = node->L2_temporal_qual_bits;
        const ThreadCuboid& thread_cuboid = env.prepare_thread_cuboid();
        on_fence(env.p_syncv_table.get(), thread_cuboid, param, prepare_logger(node, thread_cuboid));
        env.maybe_syncv_debug_validate();
    }

    void exec_impl(const Arrive* node)
    {
        const barrier_id home_barrier = fill_tmp_offset_barriers(node);
        SyncvArrive param{};
        param.home_barrier = home_barrier;
        param.barrier_count = uint32_t(tmp_all_barriers.size());
        param.all_barriers = tmp_all_barriers.data();
        param.transitive = node->V1_transitive;
        param.L1_qual_bits = node->L1_qual_bits;

        // Pass to SyncvTable.
        const ThreadCuboid& thread_cuboid = env.prepare_thread_cuboid();
        on_arrive(env.p_syncv_table.get(), thread_cuboid, param, prepare_logger(node, thread_cuboid));
        env.maybe_syncv_debug_validate();
    }

    template <typename Node, typename Slot, typename Callback>
    void eval_tmp_offset_multicast(const Node* node, Slot& slot, Callback&& callback)
    {
        // Requires Node has ArriveIdx as VLA type.
        // Fill tmp_offset based on the index, and resolve multicasting.
        // callback(slot, linear_idx) is called for each (linearized, C order) position
        // covered by multicasting.
        const std::vector<extent_t>& extent = slot.extent();
        const uint32_t dim = node->camspork_vla_size;
        CAMSPORK_REQUIRE_CMP(dim, ==, extent.size(), "dimension mismatch");

        // Evaluate concrete indices (for barriers this is the "home barrier" index).
        eval_tmp_offset(node);

        auto recurse = [&] (
                uint32_t dim_idx, uint32_t partial_idx, uint32_t equality_mask, auto recurse)
        {
            if (dim_idx >= dim) {
                if (equality_mask != 0) {
                    callback(slot, partial_idx);
                }
                return;
            }
            const extent_t extent_coord = extent[dim_idx];
            const extent_t var_value = tmp_offset[dim_idx];
            const ArriveIdx arrive_idx = node_vla_get(node, dim_idx);
            for (extent_t i = 0; i < extent_coord; ++i) {
                const uint32_t tmp_mask = (i == var_value) ? ~uint32_t(0) : arrive_idx.multicast_per_expr;
                recurse(dim_idx + 1, partial_idx * extent_coord + i, equality_mask & tmp_mask, recurse);
            }
        };
        recurse(0, 0, ~uint32_t(0), recurse);
    }

    template <typename Node>
    barrier_id fill_tmp_offset_barriers(const Node* node)
    {
        // Fill tmp_offset and tmp_all_barriers. Return home barrier.
        VarSlotEntry<barrier_id>& _slot = env.barrier_slot(node->name);

        // Find all barriers matching at least one BarrierExpr.
        tmp_all_barriers.clear();
        auto callback = [&] (auto& slot, auto linear_idx)
        {
            tmp_all_barriers.push_back(slot.data()[linear_idx]);
        };
        eval_tmp_offset_multicast(node, _slot, callback);

        const barrier_id home_barrier = _slot.idx(tmp_offset.begin(), tmp_offset.end());
        return home_barrier;
    }

    void exec_impl(const Await* node)
    {
        // Evaluate concrete indices of barrier.
        SyncvAwait param{};
        VarSlotEntry<barrier_id>& slot = env.barrier_slot(node->name);
        eval_tmp_offset(node);
        param.bar = slot.idx(tmp_offset.begin(), tmp_offset.end());
        param.N = node->N;
        param.L2_full_qual_bits = node->L2_full_qual_bits;
        param.L2_temporal_qual_bits = node->L2_temporal_qual_bits;

        // Pass to SyncvTable.
        const ThreadCuboid& thread_cuboid = env.prepare_thread_cuboid();
        on_await(env.p_syncv_table.get(), thread_cuboid, param, prepare_logger(node, thread_cuboid));
        env.maybe_syncv_debug_validate();
    }

    void exec_impl(const JoinThreads* node)
    {
        const ThreadCuboid& thread_cuboid = env.prepare_thread_cuboid();
        on_join_threads(env.p_syncv_table.get(), thread_cuboid, prepare_logger(node, thread_cuboid));
    }

    void exec_impl(const ValueEnvAlloc* node)
    {
        VarSlotEntry<value_t>& slot = env.value_slot(node->name);
        // Resize if needed.
        eval_tmp_extent(node);
        slot.resize(tmp_extent);
    }

    void exec_impl(const SyncEnvAlloc* node)
    {
        VarSlotEntry<assignment_record_id>& slot = env.sync_slot(node->name);
        slot.clear_sync_env(env.p_syncv_table.get());

        // Resize if needed.
        eval_tmp_extent(node);
        slot.resize(tmp_extent);
        env.maybe_syncv_debug_validate();

        // We rely on the 0-init constructor here.
        // This only runs if the resize(...) caused a reallocation, so we also rely on
        // clear_sync_env calling clear_visibility to 0 things.
        // This is OK even if size < capacity, since the ones in [size, capacity) would have been 0'd in the past
        // at some point, since it was either never used (0-init by alloc) or free'd after last use (clear_sync_env).
    }

    void exec_impl(const ExpectSyncEnvAlloc* node)
    {
        if (single_position_filter.accepts_name(node->name)) {
            VarSlotEntry<assignment_record_id>& slot = env.sync_slot(node->name);

            eval_tmp_extent(node);
            if (slot.extent() != tmp_extent) {
                CAMSPORK_REQUIRE(0, "ExpectSyncEnvAlloc saw wrong size for sync env allocation");
            }
        }
    }

    void exec_impl(const BarrierEnvAlloc* node)
    {
        VarSlotEntry<barrier_id>& slot = env.barrier_slot(node->name);

        // This is needed to return memory to the syncv table.
        // We don't enforce arrive/await equality on this path.
        slot.clear_barrier_env(env.p_syncv_table.get(), false);

        // Resize if needed.
        eval_tmp_extent(node);
        slot.resize(tmp_extent);

        // Allocate new barrier IDs.
        alloc_barriers(env.p_syncv_table.get(), slot.size(), slot.data());
        log_barrier_helper(env.str_name(node->name), slot, {});
        env.maybe_syncv_debug_validate();
    }

    void log_barrier_helper(
            const std::string& var_str_name, const VarSlotEntry<barrier_id>& slot, std::vector<extent_t> idx)
    {
        // Recursively log newly allocated barriers' IDs.
        const std::vector<extent_t>& extent = slot.extent();
        if constexpr (AllowLog) {
            if (idx.size() >= extent.size()) {
                const barrier_id id = slot.idx(idx.begin(), idx.end());
                if (this->excut_file) {
                    auto p_info = std::make_unique<ExcutBarrierAlloc>();
                    p_info->id = id.data;
                    p_info->name = var_str_name;
                    p_info->idx = std::move(idx);
                    this->excut_actions.emplace_back(std::move(p_info));
                }
                if (env.history_enable) {
                    std::stringstream s;
                    s << var_str_name;
                    if (idx.size()) {
                        s << "[";
                        s << idx[0];
                        for (size_t i = 1; i < idx.size(); ++i) {
                            s << ", " << idx[i];
                        }
                        s << "]";
                    }
                    env.history_log.set_barrier_name(id, s.str());
                }
            }
            else {
                const uint32_t c = extent[idx.size()];
                for (uint32_t i = 0; i < c; ++i) {
                    std::vector<extent_t> new_idx = idx;
                    new_idx.push_back(i);
                    log_barrier_helper(var_str_name, slot, new_idx);
                }
            }
        }
    }

    void exec_impl(const DataFree* node)
    {
        env.value_slot(node->name).clear_value_env();
        env.sync_slot(node->name).clear_sync_env(env.p_syncv_table.get());
    }

    void exec_impl(const BarrierFree* node)
    {
        VarSlotEntry<barrier_id>& slot = env.barrier_slot(node->name);
        slot.clear_barrier_env(env.p_syncv_table.get(), true);
        env.maybe_syncv_debug_validate();

        env.sync_slot(node->name).clear_sync_env(env.p_syncv_table.get());
    }

    struct BodyExecImpl
    {
        ProgramExec<AllowLog>& program_exec;
        uint32_t stmts_left;
        const StmtRef* p_stmts;

        template <typename Stmt>
        void operator() (const Stmt* node)
        {
            program_exec(node);
            stmts_left--;
            p_stmts++;
            if (stmts_left <= 0) {
                return;
            }
            // Tail call to execute next stmt.
            // Since this is per-Stmt type, the branch predictor may learn correlations
            // of what the next statement type is with respect to this statement type.
            p_stmts->dispatch(*this, program_exec.buffer_size, program_exec.p_buffer);
        }
    };

    void exec_impl(const StmtBody* node)
    {
        uint32_t num_stmts = node->camspork_vla_size;
        if (num_stmts > 0) {
            const StmtRef* p_stmts = &node_vla_get(node, 0);
            BodyExecImpl impl{*this, num_stmts, p_stmts};
            p_stmts->dispatch(impl, buffer_size, p_buffer);
        }
    }

    void exec_impl(const If* node)
    {
        const bool cond = eval(node->cond);
        try {
            StmtRef s = cond ? node->body : node->orelse;
            exec(s);  // Inlined only once
        }
        catch (...) {
            env.add_remark(node, cond ? "True" : "False");
            throw;
        }
    }

    template <typename TypedFor>
    void exec_for_body(const TypedFor* node, value_t iter_value)
    {
        try {
            env.alloc_scalar_value(node->iter, iter_value);
            exec(node->body);
        }
        catch (...) {
            std::stringstream s;
            s << env.str_name(node->iter) << " = " << iter_value;
            env.add_remark(node, s.str());
            throw;
        }
    }

    void exec_impl(const SeqFor* node)
    {
        const auto lo = eval(node->lo);
        const auto hi = eval(node->hi);
        for (value_t i = lo; i < hi; ++i) {
            exec_for_body(node, i);
        }
    }

    void exec_impl(const TasksFor* node)
    {
        const auto lo = eval(node->lo);
        const auto hi = eval(node->hi);
        for (value_t i = lo; i < hi; ++i) {
            exec_for_body(node, i);
            // Lazy task index. This prompts task_index to change if actually used.
            // This avoids wasting task_index values on every level of the TasksFor loop nest.
            env.dirty_task_index = true;
        }
    }

    void exec_impl(const ThreadsFor* node)
    {
        const uint32_t dim_idx = node->dim_idx;

        const uint32_t offset_c = node->offset;
        const uint32_t box_c = node->box;
        const auto lo = eval(node->lo);
        const auto hi = eval(node->hi);

        // This shouldn't hard to change, but just test it quickly if you change this.
        CAMSPORK_REQUIRE_CMP(lo, ==, 0, "Expected ThreadsFor loop to start from 0 for now");

        // Restores thread cuboid before returning (we have to update task index before saving, otherwise we
        // might restore an incorrect task index).
        env.prepare_thread_cuboid();
        SwapThreadCuboid swap(&env.raw_thread_cuboid);

        CAMSPORK_REQUIRE_CMP(dim_idx, <, env.raw_thread_cuboid.dim(), "ThreadsFor::dim_idx out of range");
        CAMSPORK_REQUIRE_CMP(offset_c + (hi - lo) * box_c, <=, env.raw_thread_cuboid.box()[dim_idx],
                             "ThreadsFor consumes more threads than exists in the current thread box");

        env.raw_thread_cuboid.offset()[dim_idx] += offset_c;
        env.raw_thread_cuboid.box()[dim_idx] = box_c;

        for (value_t i = lo; i < hi; ++i) {
            exec_for_body(node, i);
            // Slide thread box over for the next iteration.
            env.raw_thread_cuboid.offset()[dim_idx] += box_c;
        }
    }

    void exec_impl(const ParallelBlock* node)
    {
        const uint32_t dim = node->camspork_vla_size;
        const uint32_t* begin_dims = &node_vla_get_unsafe(node, 0);
        const uint32_t* end_dims = &node_vla_get_unsafe(node, dim);
        const ThreadCuboid new_cuboid = ThreadCuboid::full(begin_dims, end_dims);

        // Execute body with new thread cuboid, and restore before returning (~SwapThreadCuboid).
        env.prepare_thread_cuboid();
        SwapThreadCuboid swap(&env.raw_thread_cuboid, new_cuboid);
        env.dirty_task_index = false;
        exec(node->body);
        env.dirty_task_index = false;
    }

    void exec_impl(const DomainReshape* node)
    {
        ThreadCuboid new_cuboid = env.prepare_thread_cuboid();  // Must update task_index here!
        new_cuboid.reshape(node->camspork_vla_size, &node_vla_get_unsafe(node, 0));
        // Execute body with new thread cuboid, and restore before returning (~SwapThreadCuboid).
        SwapThreadCuboid swap(&env.raw_thread_cuboid, new_cuboid);
        exec(node->body);
    }

    void flush_excut_log()
    {
        if constexpr (AllowLog) {
            if (this->excut_file) {
                std::vector<std::unique_ptr<ExcutBaseAction>>& actions = this->excut_actions;
                FILE* file = this->excut_file;
                for (const auto& p_action : actions) {
                    bool& first_time = this->excut_first_time;
                    CAMSPORK_REQUIRE(p_action, "null excut action");
                    const char* action_name = p_action->action_name();
                    CAMSPORK_REQUIRE(action_name, "null excut action name");
                    fprintf(file, "  %c[\"%s\", ", (first_time ? ' ' : ','), action_name);
                    p_action->write_args(file);

                    // RISKY: state change in debug logging.
                    const ThreadCuboid& cuboid = env.prepare_thread_cuboid();

                    const bool is_cpu = cuboid.dim() == 1 && 0 == (1 + cuboid.box()[0]);
                    uint32_t local_tid = 0;
                    for (uint32_t i = 0; i < cuboid.dim(); ++i) {
                        local_tid = local_tid * cuboid.domain()[i] + cuboid.offset()[i];
                    }
                    const char* source = "xyzzy.py";  // TODO
                    const int line = 42;  // TODO

                    fprintf(file, ", \"%s\", %u, %u, \"%s\", %i]\n",
                            is_cpu ? "cpu" : "am_threads", cuboid.task_index, local_tid, source, line);
                    first_time = false;
                }
                actions.clear();
            }
        }
    }

    // ******************************************************************************************
    // EVALUATE EXPR
    // ******************************************************************************************
    CAMSPORK_EXEC_ALWAYS_INLINE
    value_t eval(ExprRef e) const
    {
        return e.dispatch(*this, buffer_size, p_buffer);
    }

    extent_t eval_extent_t(ExprRef e) const
    {
        const value_t v = eval(e);
        CAMSPORK_REQUIRE_CMP(v, >=, 0, "Negative value used as array index or extent");
        return extent_t(v);
    }

    CAMSPORK_EXEC_ALWAYS_INLINE
    value_t operator() (const ReadValue* node) const
    {
        return env.value_slot(node->name).idx(expr_vla_begin(node), expr_vla_end(node));
    }

    CAMSPORK_EXEC_ALWAYS_INLINE
    value_t operator() (const Const* node) const
    {
        return node->value;
    }

    value_t operator() (const USub* node) const
    {
        return -eval(node->arg);
    }

    value_t operator() (const BinOp* node) const
    {
        CAMSPORK_REQUIRE_CMP(int(node->op), !=, int(binop::Assign), "binop::Assign is only allowed in MutateValue");
        return eval_binop(node->op, eval(node->lhs), eval(node->rhs));
    }

    static value_t eval_binop(binop op, value_t lhs, value_t rhs)
    {
        switch (op) {
          case binop::Assign:
            return rhs;
          case binop::Add:
            return lhs + rhs;
          case binop::Sub:
            return lhs - rhs;
          case binop::Mul:
            return lhs * rhs;
          case binop::Div:
            // Python-style division
            CAMSPORK_REQUIRE_CMP(rhs, >, 0, "Can only divide by positive numbers");
            {
                const auto q = lhs / rhs;
                return (q < 0) ? q + rhs : q;
            }
          case binop::Mod:
            CAMSPORK_REQUIRE_CMP(rhs, >, 0, "Can only modulo by positive numbers");
            {
                const auto m = lhs % rhs;
                return (m < 0) ? m + rhs : m;
            }
          case binop::Less:
            return lhs < rhs;
          case binop::Leq:
            return lhs <= rhs;
          case binop::Greater:
            return lhs > rhs;
          case binop::Geq:
            return lhs >= rhs;
          case binop::Eq:
            return lhs == rhs;
          case binop::Neq:
            return lhs != rhs;
        }
        return 0;  // XXX should do something
    }

    // ******************************************************************************************
    // First time startup, initialize variable table
    // ******************************************************************************************
    void init_vars(VarConfigTableRef table)
    {
        table.dispatch(*this, buffer_size, p_buffer);
    }

    void operator() (const VarConfigTable* table)
    {
        // Initialize variable tables, then iterate over the variable length array
        // to initialize all the variable slots.
        const auto num_slots = table->camspork_vla_size;
        for (uint32_t i = 0; i < num_slots; ++i) {
            VarConfigRef config = node_vla_get(table, i);
            config.dispatch(*this, buffer_size, p_buffer);
        }
    }

    void operator() (const VarConfig* config)
    {
        const auto num_bytes = config->camspork_vla_size;
        const char* p_str = &node_vla_get(config, 0);
        env.var_slots.push_back({std::string(p_str, p_str + num_bytes), {}, {}, {}});
    }
};
// End ProgramExec

static const syncv_init_t default_table_init
{
    "PLACEHOLDER_FILENAME.txt",
    UINT32_MAX,
};

static const uint32_t static_uint32_max = UINT32_MAX;

ProgramEnv::ProgramEnv(size_t buffer_size, const char* buffer)
  : ProgramEnv(buffer_size, make_shared_program_buffer(buffer_size, buffer))
{
}

ProgramEnv::ProgramEnv(size_t buffer_size, std::shared_ptr<const char[]> buffer)
  : program_buffer_size(buffer_size)
  , p_program_buffer(buffer)
  , header(ProgramHeader::validate(buffer_size, buffer.get()))
  , p_syncv_table(new_syncv_table(default_table_init))
  , raw_thread_cuboid(ThreadCuboid::full(&static_uint32_max, 1 + &static_uint32_max))
{
    ProgramExec<false>(this, no_single_position_filter).init_vars(header.var_config_table);
};

ProgramEnv::ProgramEnv(const ProgramBuilder& builder)
  : ProgramEnv(builder.size(), builder.shared_data())
{
}

void ProgramEnv::exec(StmtRef stmt, const char* p_excut_filename, SinglePositionFilter single_position_filter)
{
    if (p_excut_filename || history_enable) {
        ProgramExec<true>(this, p_excut_filename, std::move(single_position_filter)).exec(stmt);
    }
    else {
        ProgramExec<false>(this, std::move(single_position_filter)).exec(stmt);
    }
}

void ProgramEnv::set_debug_validation_enable(bool flag)
{
    const bool will_check = flag && !debug_validation_enable;
    debug_validation_enable = flag;
    if (will_check) {
        syncv_debug_validate();
    }
}

void ProgramEnv::set_history_enable(bool flag)
{
    history_enable = flag;
}

void ProgramEnv::set_qual_tl_name(uint32_t qual_tl, std::string name)
{
    history_log.set_qual_tl_name(qual_tl, std::move(name));
}

void ProgramEnv::add_error_history_remarks()
{
    history_log.add_error_remarks(this);
}

void ProgramEnv::add_last_checked_read_history_remarks()
{
    history_log.add_last_checked_read_remarks(this);
}

void ProgramEnv::add_last_checked_mutate_history_remarks()
{
    history_log.add_last_checked_mutate_remarks(this);
}

void ProgramEnv::add_debug_version_history_remarks(uint64_t version_id)
{
    static_assert(sizeof(version_id) == sizeof(vis_record_version_t));
    history_log.add_history_remarks(this, vis_record_version_t{version_id});
}

void ProgramEnv::syncv_debug_validate()
{
    std::vector<SyncvDebugValidateInput> inputs;
    for (const VarSlotEnvs& slot : var_slots) {
        const VarSlotEntry<assignment_record_id>* p_sync_env = &slot.sync;
        if (const auto sz = p_sync_env->size()) {
            inputs.push_back({sz, p_sync_env->data()});
        }
    }
    debug_validate_state(p_syncv_table.get(), inputs.size(), inputs.data());
}

void ProgramEnv::stream_program_remarks(std::ostream& stream)
{
    std::unordered_map<camspork::StmtRef, std::vector<const char*>> remarks_map;
    for (const camspork::ProgramExecRemark& remark : _remarks) {
        remarks_map[remark.stmt].push_back(remark.text.c_str());
    }

    const std::vector<const char*> empty;
    auto get_remarks = [&remarks_map, &empty] (camspork::StmtRef stmt) -> const std::vector<const char*>&
    {
        auto iter = remarks_map.find(stmt);
        if (iter == remarks_map.end()) {
            return empty;
        }
        else {
            return iter->second;
        }
    };

    print_program(stream, get_remarks, program_buffer_size, p_program_buffer.get());
}

}  // end namespace camspork

camspork::ProgramEnv* camspork_new_ProgramEnv(const camspork::ProgramBuilder* p_builder)
{
    CAMSPORK_API_PROLOGUE
    return new camspork::ProgramEnv(*p_builder);
    CAMSPORK_API_EPILOGUE(nullptr)
}

camspork::ProgramEnv* camspork_copy_ProgramEnv(const camspork::ProgramEnv* p_original)
{
    CAMSPORK_API_PROLOGUE
    return new camspork::ProgramEnv(*p_original);
    CAMSPORK_API_EPILOGUE(nullptr)
}

void camspork_delete_ProgramEnv(camspork::ProgramEnv* p_victim)
{
    delete p_victim;
}

int camspork_exec_top(
        camspork::ProgramEnv* p_env, const char* p_excut_filename,
        camspork::Varname single_position_name, uint32_t dims, const camspork::extent_t* idx)
{
    CAMSPORK_API_PROLOGUE
    p_env->exec(p_excut_filename, camspork::SinglePositionFilter{single_position_name, {idx, idx + dims}});
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_exec_stmt(
        camspork::ProgramEnv* p_env, camspork::StmtRef stmt, const char* p_excut_filename,
        camspork::Varname single_position_name, uint32_t dims, const camspork::extent_t* idx)
{
    CAMSPORK_API_PROLOGUE
    p_env->exec(stmt, p_excut_filename, camspork::SinglePositionFilter{single_position_name, {idx, idx + dims}});
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_alloc_values(
        camspork::ProgramEnv* p_env, camspork::Varname name, uint32_t dims, const camspork::extent_t* p_extent)
{
    CAMSPORK_API_PROLOGUE
    p_env->alloc_values(name, std::vector<camspork::extent_t>(p_extent, p_extent + dims));
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_alloc_scalar_value(
        camspork::ProgramEnv* p_env, camspork::Varname name, camspork::value_t value)
{
    CAMSPORK_API_PROLOGUE
    p_env->alloc_scalar_value(name, value);
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_alloc_sync(
        camspork::ProgramEnv* p_env, camspork::Varname name, uint32_t dims, const camspork::extent_t* p_extent)
{
    CAMSPORK_API_PROLOGUE
    p_env->alloc_sync(name, std::vector<camspork::extent_t>(p_extent, p_extent + dims));
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_read_value(
        const camspork::ProgramEnv* p_env, camspork::Varname name, uint32_t dims, const camspork::value_t* idxs,
        camspork::value_t* out)
{
    CAMSPORK_API_PROLOGUE
    *out = p_env->value_slot(name).idx(idxs, idxs + dims);
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_set_value(
        camspork::ProgramEnv* p_env, camspork::Varname name, uint32_t dims, const camspork::value_t* idxs,
        camspork::value_t arg)
{
    CAMSPORK_API_PROLOGUE
    p_env->value_slot(name).idx(idxs, idxs + dims) = arg;
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_set_debug_validation_enable(camspork::ProgramEnv* p_env, uint32_t flag)
{
    CAMSPORK_API_PROLOGUE
    p_env->set_debug_validation_enable(flag);
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_set_history_enable(camspork::ProgramEnv* p_env, uint32_t flag)
{
    CAMSPORK_API_PROLOGUE
    p_env->set_history_enable(flag);
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_set_qual_tl_name(camspork::ProgramEnv* p_env, uint32_t qual_tl, const char* name)
{
    CAMSPORK_API_PROLOGUE
    p_env->set_qual_tl_name(qual_tl, name);
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

CAMSPORK_EXPORT int camspork_add_error_history_remarks(camspork::ProgramEnv* p_env)
{
    CAMSPORK_API_PROLOGUE
    p_env->add_error_history_remarks();
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

CAMSPORK_EXPORT int camspork_add_last_checked_read_history_remarks(camspork::ProgramEnv* p_env)
{
    CAMSPORK_API_PROLOGUE
    p_env->add_last_checked_read_history_remarks();
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

CAMSPORK_EXPORT int camspork_add_last_checked_mutate_history_remarks(camspork::ProgramEnv* p_env)
{
    CAMSPORK_API_PROLOGUE
    p_env->add_last_checked_mutate_history_remarks();
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

CAMSPORK_EXPORT int camspork_add_debug_version_history_remarks(camspork::ProgramEnv* p_env, uint64_t version_id)
{
    CAMSPORK_API_PROLOGUE
    p_env->add_debug_version_history_remarks(version_id);
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

CAMSPORK_EXPORT const char* camspork_get_remark(
        const camspork::ProgramEnv* p_env, uint32_t i, camspork::StmtRef* out_stmt)
{
    CAMSPORK_API_PROLOGUE
    const auto& remarks = p_env->get_remarks();
    CAMSPORK_C_BOUNDSCHECK(i, remarks.size());
    const camspork::ProgramExecRemark& remark = remarks[i];
    *out_stmt = remark.stmt;
    return remark.text.c_str();
    CAMSPORK_API_EPILOGUE(nullptr)
}

CAMSPORK_EXPORT int camspork_get_num_remarks(const camspork::ProgramEnv* p_env)
{
    return int(p_env->get_remarks().size());
}

camspork::Varname camspork_syncv_fail_var(const camspork::ProgramEnv* p_env)
{
    // No exception possible, I think.
    return p_env->syncv_fail_var();
}

int camspork_syncv_fail_idx_dim(const camspork::ProgramEnv* p_env)
{
    return int(p_env->syncv_fail_idx().size());
}

const camspork::extent_t* camspork_syncv_fail_idx_ptr(const camspork::ProgramEnv* p_env)
{
    return p_env->syncv_fail_idx().data();
}
