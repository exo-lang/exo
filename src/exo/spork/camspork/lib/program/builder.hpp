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
    // When dispatched, this sets the vector of stmts to be the body of the body_of statement,
    // unless is_orelse is true, then we set the vector to be the orelse of the body_of If statement.
    ProgramBuilder* p_program_builder;
    StmtRef body_of;
    std::vector<StmtRef> stmts;
    bool is_orelse = false;
    StmtRef saved_body={};
    StmtRef saved_orelse={};

    template <uint32_t StmtType>
    void operator() (stmt<StmtType>*)
    {
        // Fallback for node types not specifically targetted below.
        CAMSPORK_REQUIRE_CMP(StmtType, ==, -1, "Internal error: invalid node type for BodyBuilder");
    }

    void operator() (If* node)
    {
        StmtRef s = body_to_nursery();
        if (is_orelse) {
            node->orelse = s;
            saved_orelse = s;
        }
        else {
            node->body = s;
            saved_body = s;
        }
    }

    template <typename Node>
    void set_body_common(Node* node)
    {
        StmtRef s = body_to_nursery();
        CAMSPORK_REQUIRE(!is_orelse, "Only If statements may have an orelse");
        node->body = s;
        saved_body = s;
    }

    void operator() (SeqFor* node) { set_body_common(node); }
    void operator() (TasksFor* node) { set_body_common(node); }
    void operator() (ThreadsFor* node) { set_body_common(node); }
    void operator() (ParallelBlock* node) { set_body_common(node); }
    void operator() (DomainSplit* node) { set_body_common(node); }

    void begin_orelse();
    StmtRef body_to_nursery() const;
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

    // ******************************************************************************************
    // Add statements that don't have a body to the program.
    // ******************************************************************************************
    StmtRef add_SyncEnvAccess(  // single
        Varname name, size_t num_idx, const ExprRef* idx,
        qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, uint32_t is_mutate, uint32_t is_ooo);
    StmtRef add_SyncEnvAccess(  // window
        Varname name, size_t num_idx, const OffsetExtentExpr* idx,
        qual_bits_t initial_qual_bit, qual_bits_t extended_qual_bits, uint32_t is_mutate, uint32_t is_ooo);
    StmtRef add_MutateValue(Varname name, size_t num_idx, const ExprRef* idx, binop op, ExprRef rhs);
    StmtRef add_Fence(
        uint32_t V1_transitive, qual_bits_t L1_qual_bits,
        qual_bits_t L2_full_qual_bits, qual_bits_t L2_temporal_qual_bits);
    StmtRef add_ValueEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent);
    StmtRef add_SyncEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent);
    StmtRef add_BarrierEnvAlloc(Varname name, size_t num_dims, const ExprRef* extent);

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
    StmtRef push_DomainSplit(uint32_t dim_idx, uint32_t split_factor);

  private:
    void check_not_finished() const;

    template <typename...Args>
    StmtRef append_impl(Args... a);

    template <typename...Args>
    StmtRef push_impl(Args... a);

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
CAMSPORK_EXPORT camspork::Varname camspork_add_variable(camspork::ProgramBuilder* p_builder, const char* p_name);

CAMSPORK_EXPORT camspork::ExprRef camspork_add_ReadValue(camspork::ProgramBuilder* p_builder,
    camspork::Varname name, uint32_t num_idx, const camspork::ExprRef* idx);
CAMSPORK_EXPORT camspork::ExprRef camspork_add_Const(camspork::ProgramBuilder* p_builder,
    camspork::value_t value);
CAMSPORK_EXPORT camspork::ExprRef camspork_add_USub(camspork::ProgramBuilder* p_builder,
    camspork::ExprRef arg);
CAMSPORK_EXPORT camspork::ExprRef camspork_add_BinOp(camspork::ProgramBuilder* p_builder,
    camspork::binop op, camspork::ExprRef lhs, camspork::ExprRef rhs);

CAMSPORK_EXPORT camspork::StmtRef camspork_add_SyncEnvAccessSingle(camspork::ProgramBuilder* p_builder,
    camspork::Varname name, uint32_t num_idx, const camspork::ExprRef* idx,
    camspork::qual_bits_t initial_qual_bit, camspork::qual_bits_t extended_qual_bits,
    uint32_t is_mutate, uint32_t is_ooo);
CAMSPORK_EXPORT camspork::StmtRef camspork_add_SyncEnvAccessWindow(camspork::ProgramBuilder* p_builder,
    camspork::Varname name, uint32_t num_idx, const camspork::OffsetExtentExpr* idx,
    camspork::qual_bits_t initial_qual_bit, camspork::qual_bits_t extended_qual_bits,
    uint32_t is_mutate, uint32_t is_ooo);
CAMSPORK_EXPORT camspork::StmtRef camspork_add_MutateValue(camspork::ProgramBuilder* p_builder,
    camspork::Varname name, uint32_t num_idx, const camspork::ExprRef* idx, camspork::binop op, camspork::ExprRef rhs);
CAMSPORK_EXPORT camspork::StmtRef camspork_add_Fence(camspork::ProgramBuilder* p_builder,
    uint32_t V1_transitive, camspork::qual_bits_t L1_qual_bits,
    camspork::qual_bits_t L2_full_qual_bits, camspork::qual_bits_t L2_temporal_qual_bits);
CAMSPORK_EXPORT camspork::StmtRef camspork_add_ValueEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork::Varname name, uint32_t num_dims, const camspork::ExprRef* extent);
CAMSPORK_EXPORT camspork::StmtRef camspork_add_SyncEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork::Varname name, uint32_t num_dims, const camspork::ExprRef* extent);
CAMSPORK_EXPORT camspork::StmtRef camspork_add_BarrierEnvAlloc(camspork::ProgramBuilder* p_builder,
    camspork::Varname name, uint32_t num_dims, const camspork::ExprRef* extent);

// TODO Arrive, Await, SyncEnvFreeShard, BarrierEnvFree

CAMSPORK_EXPORT camspork::StmtRef camspork_push_If(camspork::ProgramBuilder* p_builder,
    camspork::ExprRef cond);
CAMSPORK_EXPORT int camspork_begin_orelse(camspork::ProgramBuilder* p_builder);
CAMSPORK_EXPORT camspork::StmtRef camspork_push_SeqFor(camspork::ProgramBuilder* p_builder,
    camspork::Varname iter, camspork::ExprRef lo, camspork::ExprRef hi);
CAMSPORK_EXPORT camspork::StmtRef camspork_push_TasksFor(camspork::ProgramBuilder* p_builder,
    camspork::Varname iter, camspork::ExprRef lo, camspork::ExprRef hi);
CAMSPORK_EXPORT camspork::StmtRef camspork_push_ThreadsFor(camspork::ProgramBuilder* p_builder,
    camspork::Varname iter, camspork::ExprRef lo, camspork::ExprRef hi, uint32_t dim_idx, uint32_t offset, uint32_t box);
CAMSPORK_EXPORT camspork::StmtRef camspork_push_ParallelBlock(camspork::ProgramBuilder* p_builder,
    uint32_t dim, const uint32_t* domain);
CAMSPORK_EXPORT camspork::StmtRef camspork_push_DomainSplit(camspork::ProgramBuilder* p_builder,
    uint32_t dim_idx, uint32_t split_factor);
CAMSPORK_EXPORT int camspork_pop_body(camspork::ProgramBuilder* p_builder,
    camspork::StmtRef* out_body, camspork::StmtRef* out_orelse);
