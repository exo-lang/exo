#include "exec.hpp"

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
#include "print.hpp"
#include "../util/cuboid_util.hpp"

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

template <bool EnableExcutLog>
class ProgramExecExcutBase
{
  protected:
    FILE* excut_file = nullptr;

    // if EnableExcutLog, then the per-stmt/per-expr callbacks in ProgramExec may push excut actions to log.
    // These should be flushed for each statement using flush_excut_log, if applicable.
    std::vector<std::unique_ptr<ExcutBaseAction>> excut_actions;

    bool excut_first_time = true;

    ProgramExecExcutBase() = default;
    ProgramExecExcutBase(ProgramExecExcutBase&&) = delete;

    ~ProgramExecExcutBase()
    {
        if (excut_file) {
            fprintf(excut_file, "]\n");
            fclose(excut_file);
        }
    }
};

template <>
struct ProgramExecExcutBase<false>
{
};

// Borrowed reference wrapper around ProgramEnv, to implement actual per-node-type execution.
template <bool EnableExcutLog>
class ProgramExec : public ProgramExecExcutBase<EnableExcutLog>
{
    size_t buffer_size;
    const char* p_buffer;
    ProgramEnv& env;

    std::vector<extent_t> tmp_extent;
    std::vector<extent_t> tmp_offset;
    std::vector<barrier_id> tmp_all_barriers;
    StmtRef current_stmt{};

  public:
    ProgramExec(ProgramEnv* p_self)
      : buffer_size(p_self->program_buffer_size)
      , p_buffer(p_self->p_program_buffer.get())
      , env(*p_self)
    {
    }

    ProgramExec(ProgramEnv* p_self, const char* p_excut_filename) : ProgramExec(p_self)
    {
        static_assert(EnableExcutLog, "Can't open excut log file if C++ functionality not enabled");
        CAMSPORK_REQUIRE(p_excut_filename, "null ptr");
        FILE*& file = this->excut_file;
        file = fopen(p_excut_filename, "w");
        if (!file) {
            throw std::runtime_error(std::string(p_excut_filename) + ": " + strerror(errno));
        }
        fprintf(file, "[\n");
    }

    // ******************************************************************************************
    // Many nodes define array indices as a VLA of ExprRef.
    // We provide a stripped-down iterator over these exprs, evaluated as values.
    // ******************************************************************************************
    struct ExprIterator
    {
        const ExprRef* p_node_ref;
        const ProgramExec<EnableExcutLog>* p_exec;

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


    // ******************************************************************************************
    // EXECUTE STATEMENT
    // ******************************************************************************************
    __attribute__((always_inline))
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
        const StmtRef stmt_before = current_stmt;
        try {
            current_stmt = env.stmt_ref_from_ptr(node);
            exec_impl(node);
        }
        catch (...) {
            current_stmt = stmt_before;
            flush_excut_log();
            throw;
        }
        current_stmt = stmt_before;
        flush_excut_log();
    }

    void exec_impl(const SyncEnvReadSingle* node)
    {
        exec_sync_env_impl(node, env.stmt_ref_from_ptr(node));
    }

    void exec_impl(const SyncEnvReadWindow* node)
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

    template <bool IsMutate, bool IsWindow>
    void exec_sync_env_impl(const SyncEnvAccessNode<IsMutate, IsWindow>* node, StmtRef stmt_ref)
    {
        SyncvQualTlInput q_input;
        q_input.is_ooo = node->is_ooo;
        q_input.initial_qual_bit = node->initial_qual_bit;
        q_input.extended_qual_bits = node->extended_qual_bits;
        q_input.atomic_qual_bits = node->get_atomic_qual_bits();

        // Prepare input: window or single assignment record
        using Input = std::conditional_t<IsWindow, AssignmentRecordWindow, assignment_record_id*>;
        Input input;
        VarSlotEntry<assignment_record_id>& slot = env.sync_slot(node->name);
        eval_tmp_offset(node);
        if constexpr (node->is_window) {
            eval_tmp_extent(node);
            input.base = slot.data();
            input.begin_outer_extent = &*slot.extent().begin();
            input.end_outer_extent = &*slot.extent().end();
            input.begin_offset = &*tmp_offset.begin();
            input.end_offset = &*tmp_offset.end();
            input.begin_inner_extent = &*tmp_extent.begin();
            input.end_inner_extent = &*tmp_extent.end();
        }
        else {
            input = &slot.idx(tmp_offset.begin(), tmp_offset.end());
        }

        // Prepare excut debug logger if applicable.
        using Logger = std::conditional_t<EnableExcutLog, SyncvExcutRequest, decltype(nullptr)>;
        Logger logger{};
        if constexpr (EnableExcutLog) {
            logger.var_str_name = env.str_name(node->name);
            logger.p_out = &this->excut_actions;
            if constexpr (!IsWindow) {
                logger.idx_for_single = tmp_offset;
            }
        }

        // Call into syncv table
        try {
            if constexpr (node->is_mutate) {
                on_rw(env.p_syncv_table.get(), input, env.prepare_thread_cuboid(), q_input, logger);
            }
            else {
                on_r(env.p_syncv_table.get(), input, env.prepare_thread_cuboid(), q_input, logger);
            }
        }
        catch (const SyncvCheckFail& exc) {
            // If !IsWindow, we can't trust linear_index_in_input as we passed an already-offset pointer to SyncvTable.
            env._syncv_fail_var = node->name;
            env._syncv_fail_idx = IsWindow ? slot.idx_from_linear(exc.linear_index_in_input()) : tmp_offset;
            std::stringstream s;
            s << exc.what() << " @ " << env.str_name(node->name);
            print_idx_helper(s, env._syncv_fail_idx);
            env.add_remark(stmt_ref, s.str());
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
        on_fence(env.p_syncv_table.get(), node->V1_transitive, env.prepare_thread_cuboid(),
                node->L1_qual_bits, node->L2_full_qual_bits, node->L2_temporal_qual_bits);
        env.maybe_syncv_debug_validate();
    }

    void exec_impl(const Arrive* node)
    {
        VarSlotEntry<barrier_id>& slot = env.barrier_slot(node->name);
        const std::vector<extent_t>& extent = slot.extent();
        const uint32_t dim = node->camspork_vla_size;

        CAMSPORK_REQUIRE_CMP(dim, ==, extent.size(), "dimension mismatch");

        // Evaluate concrete indices of home barrier.
        eval_tmp_offset(node);
        const barrier_id home_barrier = slot.idx(tmp_offset.begin(), tmp_offset.end());

        // Find all barriers matching at least one BarrierExpr.
        tmp_all_barriers.clear();
        auto fill_barriers = [&] (
                uint32_t dim_idx, uint32_t partial_idx, uint32_t equality_mask, auto recurse)
        {
            if (dim_idx >= dim) {
                if (equality_mask != 0) {
                    tmp_all_barriers.push_back(slot.data()[partial_idx]);
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
        fill_barriers(0, 0, ~uint32_t(0), fill_barriers);

        // Pass to SyncvTable.
        on_arrive(env.p_syncv_table.get(), home_barrier, uint32_t(tmp_all_barriers.size()), tmp_all_barriers.data(),
                node->V1_transitive, env.prepare_thread_cuboid(), node->L1_qual_bits);
        env.maybe_syncv_debug_validate();
    }

    void exec_impl(const Await* node)
    {
        // Evaluate concrete indices of barrier.
        VarSlotEntry<barrier_id>& slot = env.barrier_slot(node->name);
        eval_tmp_offset(node);
        const barrier_id bar = slot.idx(tmp_offset.begin(), tmp_offset.end());

        // Pass to SyncvTable.
        on_await(env.p_syncv_table.get(), bar, node->N, env.prepare_thread_cuboid(),
                node->L2_full_qual_bits, node->L2_temporal_qual_bits);
        env.maybe_syncv_debug_validate();
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

        // Clear every entry.
        // This is needed to return memory to the syncv table.
        clear_visibility(env.p_syncv_table.get(), slot.size(), slot.data());
        slot.mark_empty();

        // Resize if needed.
        eval_tmp_extent(node);
        slot.resize(tmp_extent);
        clear_visibility(env.p_syncv_table.get(), slot.size(), slot.data());
        env.maybe_syncv_debug_validate();
    }

    void exec_impl(const SyncEnvFreeShard*)
    {
        CAMSPORK_REQUIRE(0, "TODO: implement SyncEnvFreeShard");
        env.maybe_syncv_debug_validate();
    }

    void exec_impl(const BarrierEnvAlloc* node)
    {
        VarSlotEntry<barrier_id>& slot = env.barrier_slot(node->name);

        // This is needed to return memory to the syncv table.
        free_barriers(env.p_syncv_table.get(), slot.size(), slot.data());
        slot.mark_empty();

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
        if constexpr (EnableExcutLog) {
            if (idx.size() >= extent.size()) {
                const barrier_id id = slot.idx(idx.begin(), idx.end());
                auto p_info = std::make_unique<ExcutBarrierAlloc>();
                p_info->id = id.data;
                p_info->name = var_str_name;
                p_info->idx = std::move(idx);
                this->excut_actions.emplace_back(std::move(p_info));
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

    void exec_impl(const BarrierEnvFree* node)
    {
        VarSlotEntry<barrier_id>& slot = env.barrier_slot(node->name);
        free_barriers(env.p_syncv_table.get(), slot.size(), slot.data());
        env.maybe_syncv_debug_validate();
    }

    struct BodyExecImpl
    {
        ProgramExec<EnableExcutLog>& program_exec;
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

    void exec_impl(const DomainSplit* node)
    {
        ThreadCuboid new_cuboid = env.prepare_thread_cuboid();  // Must update task_index here!
        const uint32_t split_idx = node->dim_idx;
        const uint32_t split_factor = node->split_factor;
        CAMSPORK_REQUIRE_CMP(split_idx, <, new_cuboid.dim(), "out-of-range DomainSplit::dim_idx");
        CAMSPORK_REQUIRE_CMP(split_factor, >=, 1, "invalid DomainSplit::split_factor");

        const uint32_t domain_c = new_cuboid.domain()[split_idx];
        if (domain_c == split_factor || split_factor == 1) {
            // Unchanged.
        }
        else {
            const uint32_t offset_c = new_cuboid.offset()[split_idx];
            const uint32_t box_c = new_cuboid.box()[split_idx];
            CAMSPORK_REQUIRE_CMP(domain_c % split_factor, ==, 0, "Invalid DomainSplit::split_factor for current env");
            CAMSPORK_REQUIRE_CMP(offset_c % split_factor, ==, 0, "Invalid DomainSplit::split_factor for current env");

            const uint32_t offset_0 = offset_c / split_factor;
            const uint32_t offset_1 = 0;
            const uint32_t domain_0 = domain_c / split_factor;
            const uint32_t domain_1 = split_factor;
            uint32_t box_0, box_1;
            if (box_c < domain_c) {
                box_0 = 1;
                box_1 = box_c;
            }
            else {
                CAMSPORK_REQUIRE_CMP(box_c % split_factor, ==, 0, "Invalid DomainSplit::split_factor for current env");
                box_0 = box_c / split_factor;
                box_1 = split_factor;
            }

            // Insert new domain/offset/box coordinates in place of the old ones.
            new_cuboid.split_replace(split_idx, domain_0, domain_1, offset_0, offset_1, box_0, box_1);
        }

        // Execute body with new thread cuboid, and restore before returning (~SwapThreadCuboid).
        SwapThreadCuboid swap(&env.raw_thread_cuboid, new_cuboid);
        exec(node->body);
    }

    void flush_excut_log()
    {
        if constexpr (EnableExcutLog) {
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

    // ******************************************************************************************
    // EVALUATE EXPR
    // ******************************************************************************************
    __attribute__((always_inline))
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

    __attribute__((always_inline))
    value_t operator() (const ReadValue* node) const
    {
        return env.value_slot(node->name).idx(expr_vla_begin(node), expr_vla_end(node));
    }

    __attribute__((always_inline))
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
    ProgramExec<false>(this).init_vars(header.var_config_table);
};

ProgramEnv::ProgramEnv(const ProgramBuilder& builder)
  : ProgramEnv(builder.size(), builder.shared_data())
{
}

void ProgramEnv::exec(StmtRef stmt, const char* p_excut_filename)
{
    if (p_excut_filename) {
        ProgramExec<true>(this, p_excut_filename).exec(stmt);
    }
    else {
        ProgramExec<false>(this).exec(stmt);
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

int camspork_exec_top(camspork::ProgramEnv* p_env, const char* p_excut_filename)
{
    CAMSPORK_API_PROLOGUE
    p_env->exec(p_excut_filename);
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_exec_stmt(camspork::ProgramEnv* p_env, camspork::StmtRef stmt, const char* p_excut_filename)
{
    CAMSPORK_API_PROLOGUE
    p_env->exec(stmt, p_excut_filename);
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
