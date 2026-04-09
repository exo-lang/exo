#include "builder.hpp"

#include "../syncv/syncv_types.hpp"
#include "../syncv/tl_sig.hpp"
#include "../util/require.hpp"

#include <stdio.h>

namespace camspork
{

void BodyBuilder::begin_orelse(ProgramBuilder* p_program_builder)
{
    CAMSPORK_REQUIRE(!is_orelse, "Already called begin_orelse.");
    CAMSPORK_REQUIRE(body_of.type_id() == If_ID, "Must be defining If-statement body to call begin_orelse");
    p_program_builder->end_body_builder(*this);
    stmts.clear();
    is_orelse = true;
}

StmtRef BodyBuilder::save_body_to_nursery(ProgramBuilder* p_program_builder)
{
    StmtRef body = p_program_builder->body_to_nursery(stmts);
    *(is_orelse ? &saved_orelse : &saved_body) = body;
    return body;
}

ProgramBuilder::ProgramBuilder()
{
    // Start out the generated binary with the ProgramHeader.
    ProgramHeader header_template;
    for (uint32_t& magic : header_template.magic_numbers) {
        magic = ProgramHeader::expected_magic_numbers[&magic - &header_template.magic_numbers[0]];
    }
    nursery.add_blob(sizeof header_template, &header_template);

    // Top-level BodyBuilder is special; we never pop it, and it is for building the top-level stmt body.
    body_stack.push_back({{}, {}});
}

void ProgramBuilder::finish()
{
    if (p_shared_finished_buffer) {
        return;
    }

    // Package up top-level statements ... requires all If/For were finished.
    CAMSPORK_REQUIRE_CMP(body_stack.size(), ==, 1, "missing pop_body() before finish()");
    StmtRef top_level_stmt = body_to_nursery(body_stack[0].stmts);

    // Put together variable table.
    std::vector<VarConfigRef> var_configs;
    for (const std::string& name : variable_slot_names) {
        var_configs.push_back(nursery.add_node<VarConfigRef>(VarConfig{}, name.size(), name.data()));
    }
    auto var_config_table = nursery.add_node<VarConfigTableRef>(
        VarConfigTable{}, var_configs.size(), var_configs.data());

    // Fill header; don't access it until we are sure we never call add_node again!
    auto& header = reinterpret_cast<ProgramHeader&>(nursery.data()[0]);
    header.top_level_stmt = top_level_stmt;
    header.var_config_table = var_config_table;

    char* p_shared_data = new char[nursery.size()];
    p_shared_finished_buffer.reset(p_shared_data);
    memcpy(p_shared_data, nursery.data(), size());
}

ExprRef ProgramBuilder::add_ReadValue(Varname name, size_t num_idx, const ExprRef* idx)
{
    check_not_finished();
    return nursery.add_node<ExprRef>(ReadValue{name}, num_idx, idx);
}

ExprRef ProgramBuilder::add_Const(value_t value)
{
    check_not_finished();
    return nursery.add_node<ExprRef>(Const{value});
}

ExprRef ProgramBuilder::add_USub(ExprRef arg)
{
    check_not_finished();
    return nursery.add_node<ExprRef>(USub{arg});
}

ExprRef ProgramBuilder::add_BinOp(binop op, ExprRef lhs, ExprRef rhs)
{
    check_not_finished();
    return nursery.add_node<ExprRef>(BinOp{op, lhs, rhs});
}

TrailingBarrierExprRef ProgramBuilder::add_TrailingBarrierExpr(Varname name, uint32_t num_idx, const ArriveIdx* idx)
{
    check_not_finished();
    return nursery.add_node<TrailingBarrierExprRef>(TrailingBarrierExpr{name}, num_idx, idx);
}

template <typename ReadNode, typename MutateNode, typename IdxType>
StmtRef ProgramBuilder::add_SyncEnvAccess_impl(
    Varname name, size_t num_idx, const IdxType* idx,
    qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, qual_bits_t atomic_qual_bits,
    uint32_t thread_access_granularity, uint32_t access_flags, TrailingBarrierExprRef trailing_barrier_expr)
{
    static_assert(access_flag_all_bits == 15, "update me");
    CAMSPORK_REQUIRE_CMP(access_flags, <=, access_flag_all_bits, "unknown access_flags bit set");
    const bool is_mutate = 0 != (access_flags & access_flag_mutate);
    auto impl = [&] (auto node)
    {
        node.name = name;
        node.initial_qual_bit = initial_qual_bit;
        node.extended_qual_bits = extended_qual_bits;
        node.access_flags = access_flags;
        node.trailing_barrier_expr = trailing_barrier_expr;
        node.thread_access_granularity = thread_access_granularity;
        if constexpr (node.is_mutate) {
            node.atomic_qual_bits = atomic_qual_bits;
        }
        else {
            CAMSPORK_REQUIRE_CMP(atomic_qual_bits, ==, 0, "atomics must have access_flag_mutate set");
        }
        return append_impl(node, num_idx, idx);
    };
    if (is_mutate) {
        return impl(MutateNode{});
    }
    else {
        return impl(ReadNode{});
    }
}

StmtRef ProgramBuilder::add_SyncEnvAccess(
    Varname name, size_t num_idx, const ExprRef* idx,
    qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, qual_bits_t atomic_qual_bits,
    uint32_t thread_access_granularity, uint32_t access_flags, TrailingBarrierExprRef trailing_barrier_expr)
{
    return add_SyncEnvAccess_impl<SyncEnvReadSingle, SyncEnvMutateSingle>(
            name, num_idx, idx, initial_qual_bit, extended_qual_bits, atomic_qual_bits,
            thread_access_granularity, access_flags, trailing_barrier_expr);
}

StmtRef ProgramBuilder::add_SyncEnvAccess(
    Varname name, size_t num_idx, const OffsetExtentExpr* idx,
    qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, qual_bits_t atomic_qual_bits,
    uint32_t thread_access_granularity, uint32_t access_flags, TrailingBarrierExprRef trailing_barrier_expr)
{
    return add_SyncEnvAccess_impl<SyncEnvReadWindow, SyncEnvMutateWindow>(
            name, num_idx, idx, initial_qual_bit, extended_qual_bits, atomic_qual_bits,
            thread_access_granularity, access_flags, trailing_barrier_expr);
}

StmtRef ProgramBuilder::add_SyncEnvAccess(
    Varname name, size_t num_idx, const ArriveIdx* idx,
    qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, qual_bits_t atomic_qual_bits,
    uint32_t thread_access_granularity, uint32_t access_flags, TrailingBarrierExprRef trailing_barrier_expr)
{
    return add_SyncEnvAccess_impl<SyncEnvReadMulticast, SyncEnvMutateMulticast>(
            name, num_idx, idx, initial_qual_bit, extended_qual_bits, atomic_qual_bits,
            thread_access_granularity, access_flags, trailing_barrier_expr);
}

StmtRef ProgramBuilder::add_SyncEnvFreeShard(
        Varname name, size_t num_idx, const ExprRef* idx, qual_bits_t extended_qual_bits)
{
    return append_impl(SyncEnvFreeShard{name, extended_qual_bits}, num_idx, idx);
}


StmtRef ProgramBuilder::add_MutateValue(Varname name, size_t num_idx, const ExprRef* idx, binop op, ExprRef rhs)
{
    return append_impl(MutateValue{name, op, rhs}, num_idx, idx);
}

StmtRef ProgramBuilder::add_Fence(
    qual_bits_t L1_qual_bits,
    qual_bits_t L2_full_qual_bits, qual_bits_t L2_temporal_qual_bits)
{
    CAMSPORK_REQUIRE_CMP(L2_full_qual_bits, ==, L2_full_qual_bits & L2_temporal_qual_bits,
                         "L2_full_qual_bits must be a subset of L2_temporal_qual_bits");
    return append_impl(Fence{L1_qual_bits, L2_full_qual_bits, L2_temporal_qual_bits});
}

StmtRef ProgramBuilder::add_Arrive(
        qual_bits_t L1_qual_bits,
        Varname name, uint32_t num_idx, const ArriveIdx* idx)
{
    return append_impl(Arrive{name, L1_qual_bits}, num_idx, idx);
}

StmtRef ProgramBuilder::add_Await(
        Varname name, uint32_t num_idx, const ExprRef* idx,
        uint32_t L2_full_qual_bits, uint32_t L2_temporal_qual_bits, int32_t N)
{
    CAMSPORK_REQUIRE_CMP(L2_full_qual_bits, ==, L2_full_qual_bits & L2_temporal_qual_bits,
                         "L2_full_qual_bits must be a subset of L2_temporal_qual_bits");
    return append_impl(Await{name, L2_full_qual_bits, L2_temporal_qual_bits, N}, num_idx, idx);
}

StmtRef ProgramBuilder::add_ValueEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent, uint32_t flags)
{
    CAMSPORK_REQUIRE_CMP(flags, ==, 0, "Currently no ValueEnvAlloc flags defined");
    return append_impl(ValueEnvAlloc{name, flags}, num_dims, extent);
}

StmtRef ProgramBuilder::add_SyncEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent, uint32_t flags)
{
    CAMSPORK_REQUIRE_CMP(flags, ==, 0, "Currently no SyncEnvAlloc flags defined");
    return append_impl(SyncEnvAlloc{name, flags}, num_dims, extent);
}

StmtRef ProgramBuilder::add_ExpectSyncEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent)
{
    return append_impl(ExpectSyncEnvAlloc{name}, num_dims, extent);
}

StmtRef ProgramBuilder::add_BarrierEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent, uint32_t flags)
{
    CAMSPORK_REQUIRE_CMP(flags, <=, one_shot_arrive_flag | one_shot_await_flag, "Unknown BarrierEnvAlloc flag");
    return append_impl(BarrierEnvAlloc{name, flags}, num_dims, extent);
}

StmtRef ProgramBuilder::add_DataFree(Varname name)
{
    return append_impl(DataFree{name});
}

StmtRef ProgramBuilder::add_BarrierFree(Varname name)
{
    return append_impl(BarrierFree{name});
}

StmtRef ProgramBuilder::add_JoinThreads()
{
    return append_impl(JoinThreads{});
}

StmtRef ProgramBuilder::push_If(ExprRef cond)
{
    return push_impl(If{cond, {}, {}});
}

void ProgramBuilder::begin_orelse()
{
    body_stack.back().begin_orelse(this);
}

StmtRef ProgramBuilder::push_SeqFor(Varname iter, ExprRef lo, ExprRef hi)
{
    SeqFor node;
    node.iter = iter;
    node.lo = lo;
    node.hi = hi;
    return push_impl(node);
}

StmtRef ProgramBuilder::push_TasksFor(Varname iter, ExprRef lo, ExprRef hi)
{
    TasksFor node;
    node.iter = iter;
    node.lo = lo;
    node.hi = hi;
    return push_impl(node);
}

StmtRef ProgramBuilder::push_ThreadsFor(
    Varname iter, ExprRef lo, ExprRef hi, uint32_t dim_idx, uint32_t offset, uint32_t box)
{
    ThreadsFor node;
    node.iter = iter;
    node.lo = lo;
    node.hi = hi;
    node.dim_idx = dim_idx;
    node.offset = offset;
    node.box = box;
    return push_impl(node);
}

StmtRef ProgramBuilder::push_ParallelBlock(size_t dim, const uint32_t* domain)
{
    return push_impl(ParallelBlock{}, dim, domain);
}

StmtRef ProgramBuilder::push_DomainReshape(size_t dim, const uint32_t* domain)
{
    return push_impl(DomainReshape{}, dim, domain);
}

void ProgramBuilder::check_not_finished() const
{
    CAMSPORK_REQUIRE(!p_shared_finished_buffer, "Cannot modify program after finish()");
}

template <typename...Args>
StmtRef ProgramBuilder::append_impl(Args... a)
{
    check_not_finished();
    StmtRef ref = nursery.add_node<StmtRef>(a...);
    body_stack.back().stmts.push_back(ref);
    return ref;
}

template <typename...Args>
StmtRef ProgramBuilder::push_impl(Args... a)
{
    check_not_finished();
    StmtRef body_of = nursery.add_node<StmtRef>(a...);
    body_stack.back().stmts.push_back(body_of);
    body_stack.push_back({body_of, {}});
    return body_of;
}

void ProgramBuilder::pop_body(StmtRef* out_body, StmtRef* out_orelse)
{
    CAMSPORK_REQUIRE_CMP(body_stack.size(), >=, 2, "Cannot pop last statement body; use finish()");
    BodyBuilder& body_builder = body_stack.back();
    end_body_builder(body_builder);
    if (out_body) {
        *out_body = body_builder.saved_body;
    }
    if (out_orelse) {
        *out_orelse = body_builder.saved_orelse;
    }
    body_stack.pop_back();
}

void ProgramBuilder::end_body_builder(BodyBuilder& builder)
{
    CAMSPORK_REQUIRE(builder.body_of, "Internal error, null builder.body_of");
    builder.save_body_to_nursery(this);
    builder.body_of.dispatch(builder, nursery.size(), nursery.data());
};

StmtRef ProgramBuilder::body_to_nursery(const std::vector<StmtRef>& stmts)
{
    // XXX this could cause the nursery to realloc, which invalidates all Node pointers.
    check_not_finished();
    if (stmts.size() == 1) {
        return stmts[0];
    }
    else if (stmts.size() == 0) {
        return StmtRef{};
    }
    else {
        // XXX note above is due to this.
        return nursery.add_node<StmtRef>(StmtBody{}, stmts);
    }
}

}  // end namespace

camspork::ProgramBuilder* camspork_new_ProgramBuilder()
{
    CAMSPORK_API_PROLOGUE
    return new camspork::ProgramBuilder();
    CAMSPORK_API_EPILOGUE(nullptr)
}

void camspork_delete_ProgramBuilder(camspork::ProgramBuilder* p_builder)
{
    delete p_builder;
}

int camspork_finish_ProgramBuilder(camspork::ProgramBuilder* p_builder)
{
    CAMSPORK_API_PROLOGUE
    p_builder->finish();
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

int camspork_ProgramBuilder_is_finished(const camspork::ProgramBuilder* p_builder)
{
    return p_builder->is_finished();
}

size_t camspork_ProgramBuilder_size(camspork::ProgramBuilder* p_builder)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->size();
    CAMSPORK_API_EPILOGUE(0)
}

const char* camspork_ProgramBuilder_data(camspork::ProgramBuilder* p_builder)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->data();
    CAMSPORK_API_EPILOGUE(nullptr)
}

camspork_RawVarname camspork_add_variable(camspork::ProgramBuilder* p_builder, const char* p_name)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_variable(p_name);
    CAMSPORK_API_EPILOGUE(camspork::Varname())
}

camspork_RawExprRef camspork_add_ReadValue(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_ReadValue(name, num_idx, idx);
    CAMSPORK_API_EPILOGUE(camspork::ExprRef())
}

camspork_RawExprRef camspork_add_Const(camspork::ProgramBuilder* p_builder,
    camspork::value_t value)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_Const(value);
    CAMSPORK_API_EPILOGUE(camspork::ExprRef())
}

camspork_RawExprRef camspork_add_USub(camspork::ProgramBuilder* p_builder,
    camspork_RawExprRef arg)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_USub(arg);
    CAMSPORK_API_EPILOGUE(camspork::ExprRef())
}

camspork_RawExprRef camspork_add_BinOp(camspork::ProgramBuilder* p_builder,
    camspork::binop op, camspork_RawExprRef lhs, camspork_RawExprRef rhs)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_BinOp(op, lhs, rhs);
    CAMSPORK_API_EPILOGUE(camspork::ExprRef())
}

camspork_RawTrailingBarrierExprRef camspork_add_TrailingBarrierExpr(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ArriveIdx* idx)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_TrailingBarrierExpr(name, num_idx, idx);
    CAMSPORK_API_EPILOGUE(camspork::TrailingBarrierExprRef())
}

static camspork::TrailingBarrierExprRef get_trailing_barrier_expr(
    const camspork::TrailingBarrierExprRef* trailing_barrier_expr)
{
    return trailing_barrier_expr ? *trailing_barrier_expr : camspork::TrailingBarrierExprRef();
}

camspork_RawStmtRef camspork_add_SyncEnvAccessSingle(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx,
    camspork::qual_bits_t initial_qual_bit, camspork::qual_bits_t extended_qual_bits,
    camspork::qual_bits_t atomic_qual_bits, uint32_t thread_access_granularity,
    uint32_t access_flags, const camspork::TrailingBarrierExprRef* trailing_barrier_expr)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_SyncEnvAccess(
            name, num_idx, idx, initial_qual_bit, extended_qual_bits, atomic_qual_bits,
            thread_access_granularity, access_flags, get_trailing_barrier_expr(trailing_barrier_expr));
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_SyncEnvAccessWindow(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::OffsetExtentExpr* idx,
    camspork::qual_bits_t initial_qual_bit, camspork::qual_bits_t extended_qual_bits,
    camspork::qual_bits_t atomic_qual_bits, uint32_t thread_access_granularity,
    uint32_t access_flags, const camspork::TrailingBarrierExprRef* trailing_barrier_expr)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_SyncEnvAccess(
            name, num_idx, idx, initial_qual_bit, extended_qual_bits, atomic_qual_bits,
            thread_access_granularity, access_flags, get_trailing_barrier_expr(trailing_barrier_expr));
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_SyncEnvAccessMulticast(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ArriveIdx* idx,
    camspork::qual_bits_t initial_qual_bit, camspork::qual_bits_t extended_qual_bits,
    camspork::qual_bits_t atomic_qual_bits, uint32_t thread_access_granularity,
    uint32_t access_flags, const camspork::TrailingBarrierExprRef* trailing_barrier_expr)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_SyncEnvAccess(
            name, num_idx, idx, initial_qual_bit, extended_qual_bits, atomic_qual_bits,
            thread_access_granularity, access_flags, get_trailing_barrier_expr(trailing_barrier_expr));
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_SyncEnvFreeShard(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx, camspork::qual_bits_t extended_qual_bits)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_SyncEnvFreeShard(name, num_idx, idx, extended_qual_bits);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef());
}

camspork_RawStmtRef camspork_add_MutateValue(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx, camspork::binop op, camspork::ExprRef rhs)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_MutateValue(name, num_idx, idx, op, rhs);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_Fence(camspork::ProgramBuilder* p_builder,
    camspork::qual_bits_t L1_qual_bits,
    camspork::qual_bits_t L2_full_qual_bits, camspork::qual_bits_t L2_temporal_qual_bits)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_Fence(L1_qual_bits, L2_full_qual_bits, L2_temporal_qual_bits);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_Arrive(camspork::ProgramBuilder* p_builder,
        camspork::qual_bits_t L1_qual_bits,
        camspork_RawVarname name, uint32_t num_idx, const camspork::ArriveIdx* idx)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_Arrive(L1_qual_bits, name, num_idx, idx);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_Await(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx,
    uint32_t L2_full_qual_bits, uint32_t L2_temporal_qual_bits, int32_t N)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_Await(name, num_idx, idx, L2_full_qual_bits, L2_temporal_qual_bits, N);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_ValueEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_dims, const camspork::ExprRef* extent, uint32_t flags)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_ValueEnvAlloc(name, num_dims, extent, flags);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_SyncEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_dims, const camspork::ExprRef* extent, uint32_t flags)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_SyncEnvAlloc(name, num_dims, extent, flags);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_ExpectSyncEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_dims, const camspork::ExprRef* extent)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_ExpectSyncEnvAlloc(name, num_dims, extent);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_BarrierEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_dims, const camspork::ExprRef* extent, uint32_t flags)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_BarrierEnvAlloc(name, num_dims, extent, flags);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_DataFree(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_DataFree(name);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_BarrierFree(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_BarrierFree(name);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_add_JoinThreads(camspork::ProgramBuilder* p_builder)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->add_JoinThreads();
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_push_If(camspork::ProgramBuilder* p_builder,
    camspork::ExprRef cond)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->push_If(cond);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

int camspork_begin_orelse(camspork::ProgramBuilder* p_builder)
{
    CAMSPORK_API_PROLOGUE
    p_builder->begin_orelse();
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}

camspork_RawStmtRef camspork_push_SeqFor(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname iter, camspork::ExprRef lo, camspork::ExprRef hi)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->push_SeqFor(iter, lo, hi);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_push_TasksFor(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname iter, camspork::ExprRef lo, camspork::ExprRef hi)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->push_TasksFor(iter, lo, hi);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_push_ThreadsFor(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname iter, camspork::ExprRef lo, camspork::ExprRef hi, uint32_t dim_idx, uint32_t offset, uint32_t box)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->push_ThreadsFor(iter, lo, hi, dim_idx, offset, box);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_push_ParallelBlock(camspork::ProgramBuilder* p_builder,
    uint32_t dim, const uint32_t* domain)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->push_ParallelBlock(dim, domain);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

camspork_RawStmtRef camspork_push_DomainReshape(camspork::ProgramBuilder* p_builder,
    uint32_t dim, const uint32_t* domain)
{
    CAMSPORK_API_PROLOGUE
    return p_builder->push_DomainReshape(dim, domain);
    CAMSPORK_API_EPILOGUE(camspork::StmtRef())
}

int camspork_pop_body(camspork::ProgramBuilder* p_builder, camspork::StmtRef* out_body, camspork::StmtRef* out_orelse)
{
    CAMSPORK_API_PROLOGUE
    p_builder->pop_body(out_body, out_orelse);
    return 1;
    CAMSPORK_API_EPILOGUE(0)
}
