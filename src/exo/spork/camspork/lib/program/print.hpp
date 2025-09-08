#pragma once

#include <array>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "grammar.hpp"
#include "../util/api_util.hpp"
#include "../util/require.hpp"

namespace camspork
{

template <typename Stream, typename StmtRefRemarksCallback>
class ProgramPrinter
{
    Stream& stream;
    StmtRefRemarksCallback& remarks_callback;  // StmtRef -> sequence_container<obj> where we can do Stream << obj.
    size_t buffer_size;
    const char* program_buffer;
    int indent_levels = 1;

    std::vector<std::string> var_str_table;
    std::set<std::string> var_str_set;
    bool binop_no_parens_flag = false;

  public:
    // Constructor just prints the program, wrapped with print_program(...) later.
    ProgramPrinter(
            Stream& _stream, StmtRefRemarksCallback& _remarks_callback,
            size_t _buffer_size, const char* _program_buffer)
      : stream(_stream)
      , remarks_callback(_remarks_callback)
      , buffer_size(_buffer_size)
      , program_buffer(_program_buffer)
    {
        const ProgramHeader& header = ProgramHeader::validate(buffer_size, program_buffer);
        *this << "@camspork.program\n";
        *this << "def program(b: camspork.ProgramBuilder):\n";

        // Fill var_str_table, and add b.add_variable(...) to output text.
        var_str_set.insert("b");
        header.var_config_table.dispatch(*this, buffer_size, program_buffer);

        *this << header.top_level_stmt;
    }

    void operator() (const VarConfigTable* p_table)
    {
        for (uint32_t i = 0; i < p_table->camspork_vla_size; ++i) {
            node_vla_get(p_table, i).dispatch(*this, buffer_size, program_buffer);
        }
    }

    void operator() (const VarConfig* p_config)
    {
        // Unique-ify variable name.
        const auto strlen = p_config->camspork_vla_size;
        std::string prefix(&node_vla_get_unsafe(p_config, 0), &node_vla_get_unsafe(p_config, strlen));
        for (int i = 0; ; ++i) {
            std::string name = i == 0 ? prefix : prefix + "_" + std::to_string(i);
            if (!var_str_set.count(name)) {
                // Add new variable, and declare in Python builder syntax.
                *this << "  " << name << " = b.add_variable(\"" << prefix << "\")\n";
                var_str_set.insert(name);
                var_str_table.push_back(std::move(name));
                return;
            }
        }
    }

    void operator() (const ReadValue* node)
    {
        *this << node->name;
        print_idx(node);
    }

    void operator() (const Const* node)
    {
        *this << node->value;
    }

    void operator() (const USub* node)
    {
        binop_no_parens_flag = false;
        *this << "-" << node->arg;
    }

    void operator() (const BinOp* node)
    {
        // Just parenthesize everything for now, except maybe for top-level BinOp (binop_no_parens_flag).
        const bool parens = !binop_no_parens_flag;
        binop_no_parens_flag = false;
        if (parens) {
            stream << "(";
        }
        *this << node->lhs << " " << node->op << " " << node->rhs;
        if (parens) {
            stream << ")";
        }
    }

    template <bool IsMutate, bool IsWindow>
    void operator() (const SyncEnvAccessNode<IsMutate, IsWindow>* node)
    {
        print_tabs();
        *this << "b.SyncEnvAccess(" << node->name;
        print_idx(node, true);  // print offset
        *this << ", " << node->initial_qual_bit << ", " << node->extended_qual_bits;
        *this << ", is_mutate=" << (node->is_mutate ? "True" : "False");
        *this << ", is_ooo=" << (node->is_ooo ? "True" : "False");
        if (const qual_bits_t q = node->get_atomic_qual_bits()) {
            *this << ", atomic_qual_bits=" << q;
        }
        if constexpr (node->is_window) {
            *this << ", extent=";
            print_idx(node, false);  // print extent
        }
        *this << ")\n";
    }

    void operator() (const MutateValue* node)
    {
        print_tabs();
        *this << "b.MutateValue(" << node->name;
        print_idx(node);
        *this << ", \"" << node->op << "\", " << node->rhs << ")\n";
    }

    void operator() (const Fence* node)
    {
        print_tabs();
        *this << "b.Fence(" << (node->V1_transitive ? "True" : "False") << ", " << node->L1_qual_bits;
        *this << ", " << node->L2_full_qual_bits << ", " << node->L2_temporal_qual_bits << ")\n";
    }

    void operator() (const Arrive* node)
    {
        print_tabs();
        *this << "b.Arrive(" << (node->V1_transitive ? "True" : "False") << ", " << node->L1_qual_bits;
        *this << ", " << node->name;
        print_idx(node);
        const auto dim = node->camspork_vla_size;
        if (dim > 0) {
            // multicasts: transpose bits in multicast_per_expr to recover this.
            *this << ", multicasts=(";
            for (uint32_t expr_idx = 0; expr_idx < 32; ++expr_idx) {
            CAMSPORK_REQUIRE_CMP(dim, <=, 32, "sorry, too many dims in Arrive to handle");
                uint32_t multicast_flag_bits = 0;
                for (uint32_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
                    ArriveIdx arrive_idx = node_vla_get(node, dim_idx);
                    multicast_flag_bits |= uint32_t(arrive_idx[expr_idx]) << dim_idx;
                }
                if (multicast_flag_bits) {
                    *this << "(";
                    for (uint32_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
                        const auto f = 1u & (multicast_flag_bits >> dim_idx);
                        *this << (f ? "True, " : "False, ");
                    }
                    *this << "), ";
                }
            }
            *this << ")";
        }
        *this << ")\n";
    }

    void operator() (const Await* node)
    {
        print_tabs();
        *this << "b.Await(" << node->name;
        print_idx(node);
        *this << ", " << node->L2_full_qual_bits << ", " << node->L2_temporal_qual_bits << ", N=" << node->N << ")\n";
    }

    void operator() (const ValueEnvAlloc* node)
    {
        print_tabs();
        *this << "b.ValueEnvAlloc(" << node->name;
        print_idx(node);
        *this << ")\n";
    }

    void operator() (const SyncEnvAlloc* node)
    {
        print_tabs();
        *this << "b.SyncEnvAlloc(" << node->name;
        print_idx(node);
        *this << ")\n";
    }

    void operator() (const SyncEnvFreeShard*)
    {
        // TODO
    }

    void operator() (const BarrierEnvAlloc* node)
    {
        print_tabs();
        *this << "b.BarrierEnvAlloc(" << node->name;
        print_idx(node);
        *this << ")\n";
    }

    void operator() (const BarrierEnvFree* node)
    {
        print_tabs();
        *this << "b.BarrierEnvAlloc(" << node->name << ")\n";
    }

    void operator() (const StmtBody* node)
    {
        for (uint32_t i = 0; i < node->camspork_vla_size; ++i) {
            *this << node_vla_get(node, i);
        }
    }

    void operator() (const If* node)
    {
        print_tabs();
        *this << "with b.If(" << node->cond << "):\n";
        indent_levels++;
        *this << node->body;
        if (node->orelse) {
            print_tabs();
            *this << "b.begin_orelse()\n" << node->orelse;
        }
        indent_levels--;
    }

    void operator() (const SeqFor* node)
    {
        print_tabs();
        *this << "with b.SeqFor(" << node->iter << ", " << node->lo << ", " << node->hi << "):\n";
        indent_levels++;
        *this << node->body;
        indent_levels--;
    }

    void operator() (const TasksFor* node)
    {
        print_tabs();
        *this << "with b.TasksFor(" << node->iter << ", " << node->lo << ", " << node->hi << "):\n";
        indent_levels++;
        *this << node->body;
        indent_levels--;
    }

    void operator() (const ThreadsFor* node)
    {
        print_tabs();
        *this << "with b.ThreadsFor(" << node->iter << ", " << node->lo << ", " << node->hi
               << ", " << node->dim_idx << ", " << node->offset << ", " << node->box << "):\n";
        indent_levels++;
        *this << node->body;
        indent_levels--;
    }

    void operator() (const ParallelBlock* node)
    {
        print_tabs();
        *this << "with b.ParallelBlock(";
        for (uint32_t i = 0; i < node->camspork_vla_size; ++i) {
            *this << node_vla_get(node, i) << ", ";
        }
        *this << "):\n";
        indent_levels++;
        *this << node->body;
        indent_levels--;
    }

    void operator() (const DomainSplit* node)
    {
        print_tabs();
        *this << "with b.DomainSplit(" << node->dim_idx << ", " << node->split_factor << "):\n";
        indent_levels++;
        *this << node->body;
        indent_levels--;
    }

  private:
    template <typename T>
    ProgramPrinter<Stream, StmtRefRemarksCallback>& operator<<(T n)
    {
        stream << n;
        return *this;
    }

    ProgramPrinter<Stream, StmtRefRemarksCallback>& operator<<(Varname varname)
    {
        const auto slot = varname.slot();
        CAMSPORK_C_BOUNDSCHECK(slot, var_str_table.size());
        stream << var_str_table[slot];
        return *this;
    }

    ProgramPrinter<Stream, StmtRefRemarksCallback>& operator<<(ExprRef expr)
    {
        binop_no_parens_flag = true;
        expr.dispatch(*this, buffer_size, program_buffer);
        binop_no_parens_flag = false;
        return *this;
    }

    ProgramPrinter<Stream, StmtRefRemarksCallback>& operator<<(StmtRef stmt)
    {
        if (!stmt) {
            print_tabs();
            stream << "pass\n";
        }
        else {
            for (auto&& remark : remarks_callback(stmt)) {
                std::stringstream stringstream;
                stringstream << remark;
                bool had_newline = true;
                for (char c : stringstream.str()) {
                    if (had_newline) {
                        print_tabs();
                        stream << "# ";
                    }
                    stream << c;
                    had_newline = c == '\n';
                }
                if (!had_newline) {
                    stream << '\n';
                }
            }
            stmt.dispatch(*this, buffer_size, program_buffer);
        }
        return *this;
    }

    ProgramPrinter<Stream, StmtRefRemarksCallback>& operator<<(binop op)
    {
        stream << binop_names.get(op);
        return *this;
    }

    template <typename Node>
    void print_idx(const Node* node, bool print_offset=false)
    {
        auto get_e = [&] (auto i)
        {
            auto e = node_vla_get(node, i);
            if constexpr (std::is_same_v<decltype(e), OffsetExtentExpr>) {
                return print_offset ? e.offset_e : e.extent_e;
            }
            else if constexpr (std::is_same_v<decltype(e), ArriveIdx>) {
                return e.idx;
            }
            else {
                return e;
            }
        };

        const uint32_t dim = node->camspork_vla_size;
        if (dim) {
            *this << "[" << get_e(0);
            for (uint32_t i = 1; i < dim; ++i) {
                *this << ", " << get_e(i);
            }
            *this << "]";
        }
    }

    void print_tabs()
    {
        const int spaces = indent_levels * 2;
        for (int i = 0; i < spaces; ++i) {
            stream << ' ';
        }
    }
};

template <typename Stream>
void print_program(Stream& stream, size_t buffer_size, const char* program_buffer)
{
    auto no_remarks = [] (StmtRef)
    {
        return std::array<char, 0>{};
    };
    ProgramPrinter<Stream, decltype(no_remarks)>(stream, no_remarks, buffer_size, program_buffer);
}

template <typename Stream, typename StmtRefRemarksCallback>
void print_program(
        Stream& stream, StmtRefRemarksCallback& remarks_callback, size_t buffer_size, const char* program_buffer)
{
    ProgramPrinter<Stream, StmtRefRemarksCallback>(stream, remarks_callback, buffer_size, program_buffer);
}

class ProgramEnv;

}  // end namespace camspork

// Output, either formatted program or error, goes into thread_local_message.
// 0 = error, 1 = success.
CAMSPORK_EXPORT int camspork_thread_local_print_program(size_t buffer_size, const void* program_buffer);
CAMSPORK_EXPORT int camspork_thread_local_print_program_with_remarks(camspork::ProgramEnv* p_env);
