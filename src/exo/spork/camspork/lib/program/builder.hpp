#pragma once

#include <memory>
#include <string>
#include <vector>

#include "grammar.hpp"
#include "../util/api_util.hpp"
#include "../util/require.hpp"

namespace camspork
{

class ProgramBuilder;

struct BodyBuilder
{
    // save_body_to_nursery sets the vector of stmts to be saved_body unless is_orelse is true, then it's saved_orelse.
    // Then dispatching the BodyBuilder updates the body/orelse of the body_of node.
    //
    // IMPORTANT: the two-step design is due to a weakness in the nursery design with realloc.
    // If you call body_to_nursery(...) while holding a Node*, the Node* may be invalidated.
    StmtRef body_of;
    std::vector<StmtRef> stmts;
    bool is_orelse = false;
    StmtRef saved_body={};
    StmtRef saved_orelse={};

    template <uint32_t StmtType>
    void operator() (stmt<StmtType>*) const
    {
        // Fallback for node types not specifically targetted below.
        CAMSPORK_REQUIRE_CMP(StmtType, ==, 0xFFFFFFFF, "Internal error: invalid node type for BodyBuilder");
    }

    void operator() (If* node) const
    {
        if (is_orelse) {
            node->orelse = saved_orelse;
        }
        else {
            node->body = saved_body;
        }
    }

    template <typename Node>
    void set_body_common(Node* node) const
    {
        CAMSPORK_REQUIRE(!is_orelse, "Only If statements may have an orelse");
        node->body = saved_body;
    }

    void operator() (SeqFor* node) const { set_body_common(node); }
    void operator() (TasksFor* node) const { set_body_common(node); }
    void operator() (ThreadsFor* node) const { set_body_common(node); }
    void operator() (ParallelBlock* node) const { set_body_common(node); }
    void operator() (DomainReshape* node) const { set_body_common(node); }

    void begin_orelse(ProgramBuilder* p_program_builder);
    StmtRef save_body_to_nursery(ProgramBuilder* p_program_builder);
};

class ProgramBuilder
{
    std::vector<std::string> variable_slot_names;
    NodeNursery nursery;
    // 0th entry in body_stack corresponds to the top-level program
    // Further levels are used while building If, For, etc.
    std::vector<BodyBuilder> body_stack;
    std::shared_ptr<const char[]> p_shared_finished_buffer;

  public:
    ProgramBuilder();

    // Finalize the program. All push_* must have been paired with pop_body().
    // This prepares p_shared_finished_buffer.
    void finish();

    bool is_finished() const
    {
        return !!p_shared_finished_buffer;
    }

    size_t size() const
    {
        CAMSPORK_REQUIRE(p_shared_finished_buffer, "call ProgramBuilder::finish()");
        return nursery.size();
    }

    const char* data() const
    {
        CAMSPORK_REQUIRE(p_shared_finished_buffer, "call ProgramBuilder::finish()");
        return p_shared_finished_buffer.get();
    }

    const std::shared_ptr<const char[]>& shared_data() const
    {
        return p_shared_finished_buffer;
    }

    // ******************************************************************************************
    // Add variables to the program.
    // ******************************************************************************************
    Varname add_variable(const char* name)
    {
        variable_slot_names.push_back(name);
        return Varname{uint32_t(variable_slot_names.size())};
    }

    // ******************************************************************************************
    // Add expressions to the program.
    // ******************************************************************************************
    ExprRef add_ReadValue(Varname name, size_t num_idx, const ExprRef* idx);
    ExprRef add_Const(value_t value);
    ExprRef add_USub(ExprRef arg);
    ExprRef add_BinOp(binop op, ExprRef lhs, ExprRef rhs);

    TrailingBarrierExprRef add_TrailingBarrierExpr(Varname name, uint32_t num_idx, const ArriveIdx* idx);

    // ******************************************************************************************
    // Add statements that don't have a body to the program.
    // ******************************************************************************************
    StmtRef add_SyncEnvAccess(  // single
        Varname name, size_t num_idx, const ExprRef* idx,
        qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, qual_bits_t atomic_qual_bits,
        uint32_t thread_access_granularity, uint32_t access_flags, TrailingBarrierExprRef trailing_barrier_expr);
    StmtRef add_SyncEnvAccess(  // window
        Varname name, size_t num_idx, const OffsetExtentExpr* idx,
        qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, qual_bits_t atomic_qual_bits,
        uint32_t thread_access_granularity, uint32_t access_flags, TrailingBarrierExprRef trailing_barrier_expr);
    StmtRef add_SyncEnvAccess(  // multicast
        Varname name, size_t num_idx, const ArriveIdx* idx,
        qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, qual_bits_t atomic_qual_bits,
        uint32_t thread_access_granularity, uint32_t access_flags, TrailingBarrierExprRef trailing_barrier_expr);
    StmtRef add_SyncEnvFreeShard(
        Varname name, size_t num_idx, const ExprRef* idx, qual_bits_t extended_qual_bits);
    StmtRef add_MutateValue(Varname name, size_t num_idx, const ExprRef* idx, binop op, ExprRef rhs);
    StmtRef add_Fence(
        qual_bits_t L1_qual_bits,
        qual_bits_t L2_full_qual_bits, qual_bits_t L2_temporal_qual_bits);
    StmtRef add_Arrive(
        qual_bits_t L1_qual_bits,
        Varname name, uint32_t num_idx, const ArriveIdx* idx);
    StmtRef add_Await(
        Varname name, uint32_t num_idx, const ExprRef* idx,
        uint32_t L2_full_qual_bits, uint32_t L2_temporal_qual_bits, int32_t N);
    StmtRef add_ValueEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent, uint32_t flags);
    StmtRef add_SyncEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent, uint32_t flags);
    StmtRef add_ExpectSyncEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent);
    StmtRef add_BarrierEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent, uint32_t flags);
    StmtRef add_DataFree(Varname name);
    StmtRef add_BarrierFree(Varname name);
    StmtRef add_JoinThreads();

    // ******************************************************************************************
    // Add statements with a body to the program.
    // Further statements go into the body of the new statement, until you call pop_body().
    // Use begin_orelse() to switch to adding statements to the orelse of an If statement.
    // ******************************************************************************************
    StmtRef push_If(ExprRef cond);
    void begin_orelse();
    StmtRef push_SeqFor(Varname iter, ExprRef lo, ExprRef hi);
    StmtRef push_TasksFor(Varname iter, ExprRef lo, ExprRef hi);
    StmtRef push_ThreadsFor(Varname iter, ExprRef lo, ExprRef hi, uint32_t dim_idx, uint32_t offset, uint32_t box);
    StmtRef push_ParallelBlock(size_t dim, const uint32_t* domain);
    StmtRef push_DomainReshape(size_t dim, const uint32_t* domain);

  private:
    void check_not_finished() const;

    template <typename...Args>
    StmtRef append_impl(Args... a);

    template <typename...Args>
    StmtRef push_impl(Args... a);

    template <typename ReadNode, typename MutateNode, typename IdxType>
    StmtRef add_SyncEnvAccess_impl(
        Varname name, size_t num_idx, const IdxType* idx,
        qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, qual_bits_t atomic_qual_bits,
        uint32_t thread_access_granularity, uint32_t access_flags, TrailingBarrierExprRef trailing_barrier_expr);
  public:
    void pop_body(StmtRef* out_body=nullptr, StmtRef* out_orelse=nullptr);

    // For use by BodyBuilder.
    void end_body_builder(BodyBuilder& body_builder);
    StmtRef body_to_nursery(const std::vector<StmtRef>& stmts);
};

}  // end namespace

// 0 or null returns signal an error, except for delete, is_finished.

CAMSPORK_EXPORT camspork::ProgramBuilder* camspork_new_ProgramBuilder();
CAMSPORK_EXPORT void camspork_delete_ProgramBuilder(camspork::ProgramBuilder* p_builder);
CAMSPORK_EXPORT int camspork_finish_ProgramBuilder(camspork::ProgramBuilder* p_builder);
CAMSPORK_EXPORT int camspork_ProgramBuilder_is_finished(const camspork::ProgramBuilder* p_builder);
CAMSPORK_EXPORT size_t camspork_ProgramBuilder_size(camspork::ProgramBuilder* p_builder);
CAMSPORK_EXPORT const char* camspork_ProgramBuilder_data(camspork::ProgramBuilder* p_builder);
CAMSPORK_EXPORT camspork_RawVarname camspork_add_variable(camspork::ProgramBuilder* p_builder, const char* p_name);

CAMSPORK_EXPORT camspork_RawExprRef camspork_add_ReadValue(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx);
CAMSPORK_EXPORT camspork_RawExprRef camspork_add_Const(camspork::ProgramBuilder* p_builder,
    camspork::value_t value);
CAMSPORK_EXPORT camspork_RawExprRef camspork_add_USub(camspork::ProgramBuilder* p_builder,
    camspork_RawExprRef arg);
CAMSPORK_EXPORT camspork_RawExprRef camspork_add_BinOp(camspork::ProgramBuilder* p_builder,
    camspork::binop op, camspork_RawExprRef lhs, camspork_RawExprRef rhs);

CAMSPORK_EXPORT camspork_RawTrailingBarrierExprRef camspork_add_TrailingBarrierExpr(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ArriveIdx* idx);

// MacOS ctypes mystery bug: can't pass camspork::TrailingBarrierExprRef by value.
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_SyncEnvAccessSingle(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx,
    camspork::qual_bits_t initial_qual_bit, camspork::qual_bits_t extended_qual_bits,
    camspork::qual_bits_t atomic_qual_bits, uint32_t thread_access_granularity,
    uint32_t access_flags, const camspork::TrailingBarrierExprRef* trailing_barrier_expr);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_SyncEnvAccessWindow(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::OffsetExtentExpr* idx,
    camspork::qual_bits_t initial_qual_bit, camspork::qual_bits_t extended_qual_bits,
    camspork::qual_bits_t atomic_qual_bits, uint32_t thread_access_granularity,
    uint32_t access_flags, const camspork::TrailingBarrierExprRef* trailing_barrier_expr);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_SyncEnvAccessMulticast(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ArriveIdx* idx,
    camspork::qual_bits_t initial_qual_bit, camspork::qual_bits_t extended_qual_bits,
    camspork::qual_bits_t atomic_qual_bits, uint32_t thread_access_granularity,
    uint32_t access_flags, const camspork::TrailingBarrierExprRef* trailing_barrier_expr);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_SyncEnvFreeShard(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx, camspork::qual_bits_t extended_qual_bits);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_MutateValue(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx, camspork::binop op, camspork::ExprRef rhs);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_Fence(camspork::ProgramBuilder* p_builder,
    camspork::qual_bits_t L1_qual_bits,
    camspork::qual_bits_t L2_full_qual_bits, camspork::qual_bits_t L2_temporal_qual_bits);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_Arrive(camspork::ProgramBuilder* p_builder,
    camspork::qual_bits_t L1_qual_bits,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ArriveIdx* idx);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_Await(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_idx, const camspork::ExprRef* idx,
    uint32_t L2_full_qual_bits, uint32_t L2_temporal_qual_bits, int32_t N);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_ValueEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_dims, const camspork::ExprRef* extent, uint32_t flags);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_SyncEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_dims, const camspork::ExprRef* extent, uint32_t flags);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_ExpectSyncEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_dims, const camspork::ExprRef* extent);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_BarrierEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name, uint32_t num_dims, const camspork::ExprRef* extent, uint32_t flags);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_DataFree(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_BarrierFree(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname name);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_add_JoinThreads(camspork::ProgramBuilder* p_builder);

CAMSPORK_EXPORT camspork_RawStmtRef camspork_push_If(camspork::ProgramBuilder* p_builder,
    camspork::ExprRef cond);
CAMSPORK_EXPORT int camspork_begin_orelse(camspork::ProgramBuilder* p_builder);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_push_SeqFor(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname iter, camspork::ExprRef lo, camspork::ExprRef hi);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_push_TasksFor(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname iter, camspork::ExprRef lo, camspork::ExprRef hi);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_push_ThreadsFor(camspork::ProgramBuilder* p_builder,
    camspork_RawVarname iter, camspork::ExprRef lo, camspork::ExprRef hi, uint32_t dim_idx, uint32_t offset, uint32_t box);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_push_ParallelBlock(camspork::ProgramBuilder* p_builder,
    uint32_t dim, const uint32_t* domain);
CAMSPORK_EXPORT camspork_RawStmtRef camspork_push_DomainReshape(camspork::ProgramBuilder* p_builder,
    uint32_t dim, const uint32_t* domain);
CAMSPORK_EXPORT int camspork_pop_body(camspork::ProgramBuilder* p_builder,
    camspork::StmtRef* out_body, camspork::StmtRef* out_orelse);
