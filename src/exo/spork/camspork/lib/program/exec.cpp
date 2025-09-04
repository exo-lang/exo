#include "exec.hpp"

#include <sstream>
#include <stdio.h>
#include <type_traits>
#include <utility>

#include "builder.hpp"
#include "grammar.hpp"
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

// Borrowed reference wrapper around ProgramEnv, to implement actual per-node-type execution.
class ProgramExec
{
    size_t buffer_size;
    const char* p_buffer;
    ProgramEnv& env;

    std::vector<extent_t> tmp_extent;
    std::vector<extent_t> tmp_offset;

  public:
    ProgramExec(ProgramEnv* p_self)
      : buffer_size(p_self->program_buffer_size)
      , p_buffer(p_self->p_program_buffer.get())
      , env(*p_self)
    {
    }

    // ******************************************************************************************
    // Many nodes define array indices as a VLA of ExprRef.
    // We provide a stripped-down iterator over these exprs, evaluated as values.
    // ******************************************************************************************
    struct ExprIterator
    {
        const ExprRef* p_node_ref;
        const ProgramExec* p_exec;

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
            else {
                tmp_offset[i] = eval_extent_t(node_vla_get(node, i));
            }
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

    template <bool IsMutate, bool IsWindow>
    void operator() (const SyncEnvAccessNode<IsMutate, IsWindow>* node)
    {
        CAMSPORK_REQUIRE_CMP(node->initial_qual_bit, ==, node->extended_qual_bits, "TODO");
        uint32_t bitfield = node->initial_qual_bit |
            (node->is_ooo ? TlSigInterval::unordered_bits : TlSigInterval::ordered_bits);

        VarSlotEntry<assignment_record_id>& slot = env.sync_slot(node->name);
        eval_tmp_offset(node);

        if constexpr (node->is_window) {
            eval_tmp_extent(node);

            AssignmentRecordWindow window{};
            window.base = slot.data();
            window.begin_outer_extent = &*slot.extent().begin();
            window.end_outer_extent = &*slot.extent().end();
            window.begin_offset = &*tmp_offset.begin();
            window.end_offset = &*tmp_offset.end();
            window.begin_inner_extent = &*tmp_extent.begin();
            window.end_inner_extent = &*tmp_extent.end();
            if (node->is_mutate) {
                on_rw(env.p_syncv_table.get(), window, env.prepare_thread_cuboid(), bitfield);
            }
            else {
                on_r(env.p_syncv_table.get(), window, env.prepare_thread_cuboid(), bitfield);
            }
        }
        else {
            if (node->is_mutate) {
                on_rw(env.p_syncv_table.get(), &slot.idx(tmp_offset.begin(), tmp_offset.end()),
                    env.prepare_thread_cuboid(), bitfield);
            }
            else {
                on_r(env.p_syncv_table.get(), &slot.idx(tmp_offset.begin(), tmp_offset.end()),
                    env.prepare_thread_cuboid(), bitfield);
            }
        }
        env.maybe_syncv_debug_validate();
    }

    void operator() (const MutateValue* node)
    {
        value_t& lhs = env.value_slot(node->name).idx(expr_vla_begin(node), expr_vla_end(node));
        const value_t rhs = eval(node->rhs);
        lhs = eval_binop(node->op, lhs, rhs);
    }

    void operator() (const Fence* node)
    {
        on_fence(env.p_syncv_table.get(), node->V1_transitive, env.prepare_thread_cuboid(),
                node->L1_qual_bits, node->L2_full_qual_bits, node->L2_temporal_qual_bits);
        env.maybe_syncv_debug_validate();
    }

    void operator() (const Arrive*)
    {
        CAMSPORK_REQUIRE(0, "TODO: implement Arrive");
        env.maybe_syncv_debug_validate();
    }

    void operator() (const Await*)
    {
        CAMSPORK_REQUIRE(0, "TODO: implement Await");
        env.maybe_syncv_debug_validate();
    }

    void operator() (const ValueEnvAlloc* node)
    {
        VarSlotEntry<value_t>& slot = env.value_slot(node->name);
        // Resize if needed.
        eval_tmp_extent(node);
        if (tmp_extent != slot.extent()) {
            slot.reset();
            slot = VarSlotEntry<value_t>(tmp_extent);
        }
    }

    void operator() (const SyncEnvAlloc* node)
    {
        VarSlotEntry<assignment_record_id>& slot = env.sync_slot(node->name);
        // Clear every entry.
        // This is needed to return memory to the syncv table.
        clear_visibility(env.p_syncv_table.get(), slot.size(), slot.data());
        // Resize if needed.
        eval_tmp_extent(node);
        if (tmp_extent != slot.extent()) {
            slot.reset();
            slot = VarSlotEntry<assignment_record_id>(tmp_extent);
        }
        env.maybe_syncv_debug_validate();
    }

    void operator() (const SyncEnvFreeShard*)
    {
        CAMSPORK_REQUIRE(0, "TODO: implement SyncEnvFreeShard");
        env.maybe_syncv_debug_validate();
    }

    void operator() (const BarrierEnvAlloc* node)
    {
        VarSlotEntry<barrier_id>& slot = env.barrier_slot(node->name);
        // This is needed to return memory to the syncv table.
        free_barriers(env.p_syncv_table.get(), slot.size(), slot.data());
        // Resize if needed.
        eval_tmp_extent(node);
        if (tmp_extent != slot.extent()) {
            slot.reset();
            slot = VarSlotEntry<barrier_id>(tmp_extent);
        }
        env.maybe_syncv_debug_validate();
    }

    void operator() (const BarrierEnvFree* node)
    {
        VarSlotEntry<barrier_id>& slot = env.barrier_slot(node->name);
        free_barriers(env.p_syncv_table.get(), slot.size(), slot.data());
        env.maybe_syncv_debug_validate();
    }

    struct BodyExecImpl
    {
        ProgramExec& exec;
        uint32_t stmts_left;
        const StmtRef* p_stmts;

        template <typename Stmt>
        void operator() (const Stmt* node)
        {
            exec(node);
            stmts_left--;
            p_stmts++;
            if (stmts_left <= 0) {
                return;
            }
            // Tail call to execute next stmt.
            // Since this is per-Stmt type, the branch predictor may learn correlations
            // of what the next statement type is with respect to this statement type.
            p_stmts->dispatch(*this, exec.buffer_size, exec.p_buffer);
        }
    };

    void operator() (const StmtBody* node)
    {
        uint32_t num_stmts = node->camspork_vla_size;
        if (num_stmts > 0) {
            const StmtRef* p_stmts = &node_vla_get(node, 0);
            BodyExecImpl impl{*this, num_stmts, p_stmts};
            p_stmts->dispatch(impl, buffer_size, p_buffer);
        }
    }

    void operator() (const If* node)
    {
        StmtRef s = eval(node->cond) ? node->body : node->orelse;
        exec(s);  // Inlined only once
    }

    void operator() (const SeqFor* node)
    {
        const auto lo = eval(node->lo);
        const auto hi = eval(node->hi);
        env.alloc_scalar_value(node->iter, lo);
        for (value_t i = lo; i < hi; ++i) {
            // Look up Varslot each time in case the loop body did something bad!
            env.value_slot(node->iter).scalar() = i;
            exec(node->body);
        }
    }

    void operator() (const TasksFor* node)
    {
        const auto lo = eval(node->lo);
        const auto hi = eval(node->hi);
        env.alloc_scalar_value(node->iter, lo);
        for (value_t i = lo; i < hi; ++i) {
            // Look up Varslot each time in case the loop body did something bad!
            env.value_slot(node->iter).scalar() = i;
            exec(node->body);
            // Lazy task index. This prompts task_index to change if actually used.
            // This avoids wasting task_index values on every level of the TasksFor loop nest.
            env.dirty_task_index = true;
        }
    }

    void operator() (const ThreadsFor* node)
    {
        const uint32_t dim_idx = node->dim_idx;

        const uint32_t offset_c = node->offset;
        const uint32_t box_c = node->box;
        const auto lo = eval(node->lo);
        const auto hi = eval(node->hi);
        env.alloc_scalar_value(node->iter, lo);

        // This shouldn't hard to change, but just test it quickly if you change this.
        CAMSPORK_REQUIRE_CMP(lo, ==, 0, "Expected ThreadsFor loop to start from 0 for now");

        // Restores thread cuboid before returning.
        SwapThreadCuboid swap(&env.raw_thread_cuboid);

        CAMSPORK_REQUIRE_CMP(dim_idx, <, env.raw_thread_cuboid.dim(), "ThreadsFor::dim_idx out of range");
        CAMSPORK_REQUIRE_CMP(offset_c + (hi - lo) * box_c, <=, env.raw_thread_cuboid.box()[dim_idx],
                             "ThreadsFor consumes more threads than exists in the current thread box");

        env.raw_thread_cuboid.offset()[dim_idx] += offset_c;
        env.raw_thread_cuboid.box()[dim_idx] = box_c;

        for (value_t i = lo; i < hi; ++i) {
            if (false) {
                const char* var_c_name = env.var_slots[node->iter.slot()].name.c_str();
                printf("%s = %i, %s\n", var_c_name, i, (std::stringstream() << env.raw_thread_cuboid).str().c_str());
            }

            // Look up Varslot each time in case the loop body did something bad!
            env.value_slot(node->iter).scalar() = i;
            exec(node->body);

            // Slide thread box over for the next iteration.
            env.raw_thread_cuboid.offset()[dim_idx] += box_c;
        }
    }

    void operator() (const ParallelBlock* node)
    {
        const uint32_t dim = node->camspork_vla_size;
        const uint32_t* begin_dims = &node_vla_get_unsafe(node, 0);
        const uint32_t* end_dims = &node_vla_get_unsafe(node, dim);
        const ThreadCuboid new_cuboid = ThreadCuboid::full(begin_dims, end_dims);

        // Execute body with new thread cuboid, and restore before returning (~SwapThreadCuboid).
        SwapThreadCuboid swap(&env.raw_thread_cuboid, new_cuboid);
        env.dirty_task_index = false;
        exec(node->body);
        env.dirty_task_index = false;
    }

    void operator() (const DomainSplit* node)
    {
        ThreadCuboid new_cuboid = env.raw_thread_cuboid;
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
    ProgramExec(this).init_vars(header.var_config_table);
};

ProgramEnv::ProgramEnv(const ProgramBuilder& builder)
  : ProgramEnv(builder.size(), builder.shared_data())
{
}

void ProgramEnv::exec(StmtRef stmt)
{
    ProgramExec(this).exec(stmt);
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

int camspork_exec_top(camspork::ProgramEnv* p_env)
{
    CAMSPORK_API_PROLOGUE
    p_env->exec();
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_exec_stmt(camspork::ProgramEnv* p_env, camspork::StmtRef stmt)
{
    CAMSPORK_API_PROLOGUE
    p_env->exec(stmt);
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
