#pragma once

#include <new>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <vector>

#include "../util/api_util.hpp"
#include "../util/require.hpp"

#define CAMSPORK_NODE_VLA_MEMBER(T) \
    static constexpr bool camspork_vla_member = true; \
    uint32_t camspork_vla_size = 0; \
    using camspork_vla_type = T; \
    static_assert(alignof(T) <= 4); \
    /* Bytes needed for variable-length-array, rounded up to 4 */ \
    uint32_t camspork_vla_bytes() const { return (sizeof(T) * camspork_vla_size + 3) & ~uint32_t(3); } \
    uint32_t camspork_total_bytes() const { return sizeof(*this) + camspork_vla_bytes(); }

#define CAMSPORK_NODE_NO_VLA() \
    static constexpr bool camspork_vla_member = false; \
    uint32_t camspork_vla_bytes() const { return 0; } \
    uint32_t camspork_total_bytes() const { return sizeof(*this) + camspork_vla_bytes(); }

namespace camspork
{

// Change or make polymorphic if needed!
// Will have to update Python ctypes to match
using extent_t = uint32_t;
using value_t = int32_t;

enum class binop : uint32_t
{
    // 0 reserved as error code
    Assign = 1,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Less,
    Leq,
    Greater,
    Geq,
    Eq,
    Neq,
};

class BinOpNames
{
    static constexpr uint32_t _size = 13;
    const char* names[_size];
  public:
    BinOpNames();
    const char* get(binop op) const
    {
        const uint32_t op_enum = static_cast<uint32_t>(op);
        CAMSPORK_REQUIRE_CMP(op_enum, >, 0, "binop(0) reserved as error code");
        CAMSPORK_REQUIRE_CMP(op_enum, <, _size, "invalid binop");
        return names[op_enum];
    }
};

extern const BinOpNames binop_names;

struct BinOpTableEntry
{
    int second_char = -1;
    binop op = static_cast<binop>(0);
};

class BinOpTable
{
    BinOpTableEntry entries_by_char[256][2];
  public:
    BinOpTable();
    binop get(const char* p_str) const;
};

extern const BinOpTable binop_table;


// ******************************************************************************************
// Polymorphic node reference.
// The program object will be delivered as a flat buffer of char (must be 32-bit aligned)
// which is easy to serialize to disk, or move across Python/C ABI boundaries. See ProgramHeader.
//
// Each node is a struct immediately followed by an optional variable-length array.
// (use either the CAMSPORK_NODE_NO_VLA or CAMSPORK_NODE_VLA_MEMBER macros).
// This VLA is often used for array indices or extents (sizes), of length camspork_vla_size.
//
// The NodeRef identifies both the type of the node, and its position in the buffer/file.
// Therefore, it is not possible to dereference a NodeRef without a pointer to the buffer.
// ******************************************************************************************
// Note: definition duplicated as Python ctypes
template <template<uint32_t> typename NodeType, uint32_t NumTypes>
struct NodeRef
{
    static_assert(NumTypes <= 32);

    uint32_t raw_data = 0;

    // Bottom 5 bits holds the ID of the node type.
    uint32_t type_id() const
    {
        return raw_data & 31;
    }

    // Top 27 bits holds the address of the node in the file, multiplied by 4 bytes.
    uint32_t byte_offset() const
    {
        return (raw_data >> 3) & ~uint32_t(3);
    }

    void set_type_byte_offset(uint32_t type, size_t byte_offset)
    {
        CAMSPORK_REQUIRE_CMP(type, <, NumTypes, "internal error, invalid NodeRef type ID");
        raw_data = type | byte_offset << 3;
        CAMSPORK_REQUIRE_CMP(this->byte_offset(), ==, byte_offset, "NodeRef 32 bit overflow");
    }

    // 0 used as "null" value (this can't be valid as the ProgramHeader is at offset 0).
    explicit operator bool() const
    {
        return raw_data != 0;
    }

    void clear()
    {
        raw_data = 0;
    }

    bool operator==(NodeRef other) const
    {
        return this->raw_data == other.raw_data;
    }

    bool operator!=(NodeRef other) const
    {
        return this->raw_data != other.raw_data;
    }

    // Callable must implement operator()(const NodeType<N>*) for N = 0, 1, ..., NumTypes - 1
    template <typename Callable, typename Char>
    __attribute__((always_inline))
    auto dispatch(Callable&& callable, size_t buffer_size, Char* buffer) const
    {
        // Alignment check
        uintptr_t buffer_address = reinterpret_cast<uintptr_t>(buffer);
        CAMSPORK_C_BOUNDSCHECK(buffer_address % 4, 1);

        char* mutable_buffer = const_cast<char*>(buffer);
        const size_t byte_offset = this->byte_offset();

        // NOTE: we pass nodes by pointer to make node_vla_get more safe.
        // We bounds check in debug and release builds to avoid security flaws from evil input files.

        #define CAMSPORK_DISPATCH_CASE(N) \
          case N: \
            if constexpr (N < NumTypes) { \
                using Node = NodeType<N>; \
                CAMSPORK_C_BOUNDSCHECK(byte_offset + sizeof(Node), buffer_size + 1); \
                auto p_node = reinterpret_cast<Node*>(&mutable_buffer[byte_offset]); \
                CAMSPORK_C_BOUNDSCHECK(byte_offset + p_node->camspork_total_bytes(), buffer_size + 1); \
                if constexpr (std::is_const_v<Char>) { \
                    return callable(const_cast<const Node*>(p_node)); \
                } \
                else { \
                    return callable(p_node); \
                } \
            } \
            else { \
                CAMSPORK_C_BOUNDSCHECK(N, NumTypes); \
            }

        switch (type_id()) {
            default:
            CAMSPORK_DISPATCH_CASE(0)
            CAMSPORK_DISPATCH_CASE(1)
            CAMSPORK_DISPATCH_CASE(2)
            CAMSPORK_DISPATCH_CASE(3)
            CAMSPORK_DISPATCH_CASE(4)
            CAMSPORK_DISPATCH_CASE(5)
            CAMSPORK_DISPATCH_CASE(6)
            CAMSPORK_DISPATCH_CASE(7)
            CAMSPORK_DISPATCH_CASE(8)
            CAMSPORK_DISPATCH_CASE(9)
            CAMSPORK_DISPATCH_CASE(10)
            CAMSPORK_DISPATCH_CASE(11)
            CAMSPORK_DISPATCH_CASE(12)
            CAMSPORK_DISPATCH_CASE(13)
            CAMSPORK_DISPATCH_CASE(14)
            CAMSPORK_DISPATCH_CASE(15)
            CAMSPORK_DISPATCH_CASE(16)
            CAMSPORK_DISPATCH_CASE(17)
            CAMSPORK_DISPATCH_CASE(18)
            CAMSPORK_DISPATCH_CASE(19)
            CAMSPORK_DISPATCH_CASE(20)
            CAMSPORK_DISPATCH_CASE(21)
            CAMSPORK_DISPATCH_CASE(22)
            CAMSPORK_DISPATCH_CASE(23)
            CAMSPORK_DISPATCH_CASE(24)
            CAMSPORK_DISPATCH_CASE(25)
            CAMSPORK_DISPATCH_CASE(26)
            CAMSPORK_DISPATCH_CASE(27)
            CAMSPORK_DISPATCH_CASE(28)
            CAMSPORK_DISPATCH_CASE(29)
            CAMSPORK_DISPATCH_CASE(30)
            CAMSPORK_DISPATCH_CASE(31)
        }
    }
};

template <typename Node>
const typename Node::camspork_vla_type& node_vla_get(const Node* p_node, uint32_t i)
{
    static_assert(alignof(Node) == 4);
    static_assert(Node::camspork_vla_member, "Must have CAMSPORK_NODE_VLA_MEMBER in the struct body");
    const char* p_vla = reinterpret_cast<const char*>(p_node) + sizeof(Node);
    CAMSPORK_C_BOUNDSCHECK(i, p_node->camspork_vla_size);
    return reinterpret_cast<const typename Node::camspork_vla_type*>(p_vla)[i];
}

template <typename Node>
const typename Node::camspork_vla_type& node_vla_get_unsafe(const Node* p_node, uint32_t i)
{
    static_assert(alignof(Node) == 4);
    static_assert(Node::camspork_vla_member, "Must have CAMSPORK_NODE_VLA_MEMBER in the struct body");
    const char* p_vla = reinterpret_cast<const char*>(p_node) + sizeof(Node);
    return reinterpret_cast<const typename Node::camspork_vla_type*>(p_vla)[i];
}


// ******************************************************************************************
// Binary builder object.
// Use add_node<NodeRefT>(...) to write a struct (plus the trailing VLA if it exists)
// to the binary. Unfortunately, the NodeRefT type must be given explicitly.
// ******************************************************************************************
class NodeNursery
{
    // p_nursery_data is an allocation of size nursery_capacity bytes.
    // Of that, p_nursery_data[0:nursery_size] holds real data.
    uint32_t nursery_size = 0;
    uint32_t nursery_capacity = 0;
    char* p_nursery_data = 0;

  public:
    NodeNursery() = default;
    NodeNursery(NodeNursery&&) = delete;
    ~NodeNursery()
    {
        free(p_nursery_data);
    }

    template <typename NodeRefT, template<uint32_t> typename NodeType, uint32_t TypeID>
    NodeRefT
    add_node(
        NodeType<TypeID> node, size_t vla_size, const typename NodeType<TypeID>::camspork_vla_type* p_vla)
    {
        node.camspork_vla_size = uint32_t(vla_size);
        CAMSPORK_REQUIRE_CMP(node.camspork_vla_size, ==, vla_size, "32-bit overflow in VLA");
        const uint32_t offset = add_blob(sizeof(node), &node);
        add_blob(node.camspork_vla_bytes(), p_vla);
        NodeRefT node_ref;
        node_ref.set_type_byte_offset(TypeID, offset);
        return node_ref;
    }

    template <typename NodeRefT, template<uint32_t> typename NodeType, uint32_t TypeID>
    NodeRefT
    add_node(
        NodeType<TypeID> node, const std::vector<typename NodeType<TypeID>::camspork_vla_type>& vla)
    {
        return add_node<NodeRefT>(node, vla.size(), vla.data());
    }

    template <typename NodeRefT, template<uint32_t> typename NodeType, uint32_t TypeID>
    NodeRefT
    add_node(NodeType<TypeID> node)
    {
        static_assert(!node.camspork_vla_member, "Need CAMSPORK_NODE_NO_VLA in struct definition");
        const uint32_t offset = add_blob(sizeof(node), &node);
        NodeRefT node_ref;
        node_ref.set_type_byte_offset(TypeID, offset);
        return node_ref;
    }

    uint32_t add_blob(size_t bytes, const void* p_blob)
    {
        const uint32_t offset = nursery_size;
        CAMSPORK_REQUIRE_CMP(bytes % 4, ==, 0, "internal error, expected 32 bit alignment");
        reserve_bytes(nursery_size + bytes);
        memcpy(p_nursery_data + nursery_size, p_blob, bytes);
        size_t new_size = nursery_size + bytes;
        CAMSPORK_REQUIRE_CMP(new_size, <=, UINT32_MAX, "NodeNursery: 32-bit overflow");
        nursery_size = uint32_t(new_size);
        return offset;
    }

    size_t size() const
    {
        return nursery_size;
    }

    char* data()
    {
        return p_nursery_data;
    }

    const char* data() const
    {
        return p_nursery_data;
    }

  private:
    void reserve_bytes(size_t bytes)
    {
        if (bytes > nursery_capacity) {
            // Note: realloc is potentially massively more efficient than C++ due to page remapping.
            bytes = (bytes + 4095) & ~size_t(4095);
            char* p_new = static_cast<char*>(realloc(p_nursery_data, bytes));
            if (p_new == nullptr) {
                throw std::bad_alloc();
            }
            p_nursery_data = p_new;
            CAMSPORK_REQUIRE_CMP(bytes, <=, UINT32_MAX, "NodeNursery: 32-bit overflow");
            nursery_capacity = uint32_t(bytes);
        }
    }
};


// ******************************************************************************************
// Each program variable is identified by a unique index (slot) in a flat table.
//
// There is currently nothing polymorphic about the VarConfig/VarConfigTable types,
// but for consistency we adapt NodeRef to work with it anyway.
// Currently this information is just a string name for the variable, for debugging.
// ******************************************************************************************

// Note: definition duplicated as Python ctypes
struct Varname
{
    uint32_t slot_1_index = 0;  // slot index + 1

    explicit operator bool() const
    {
        return slot_1_index != 0;
    }

    uint32_t slot() const
    {
        return slot_1_index - 1;
    }
};

template <uint32_t IgnoredTypeID>
struct VarConfigNode
{
    CAMSPORK_NODE_VLA_MEMBER(char);
};

using VarConfig = VarConfigNode<0>;
using VarConfigRef = NodeRef<VarConfigNode, 1>;

template <uint32_t IgnoredTypeID>
struct VarConfigTableNode
{
    CAMSPORK_NODE_VLA_MEMBER(VarConfigRef);
};

using VarConfigTable = VarConfigTableNode<0>;
using VarConfigTableRef = NodeRef<VarConfigTableNode, 1>;


// ******************************************************************************************
// Expression node types, all pointed to by polymorphic ExprRef object.
// ******************************************************************************************

template <uint32_t TypeID>
struct expr
{
};

static constexpr uint32_t NumExprTypes = 4;

using ExprRef = NodeRef<expr, NumExprTypes>;

// ReadValue(Varname name, expr* idx)
using ReadValue = expr<0>;
template <>
struct expr<0>
{
    Varname name;
    CAMSPORK_NODE_VLA_MEMBER(ExprRef)

    uint32_t dim() const
    {
        return camspork_vla_size;
    }
};


// Const(int value)
using Const = expr<1>;
template <>
struct expr<1>
{
    int32_t value;
    CAMSPORK_NODE_NO_VLA()
};


// USub(expr arg)
using USub = expr<2>;
template<>
struct expr<2>
{
    ExprRef arg;
    CAMSPORK_NODE_NO_VLA()
};


// BinOp(binop op, expr lhs, expr rhs)
using BinOp = expr<3>;
template<>
struct expr<3>
{
    binop op;
    ExprRef lhs;
    ExprRef rhs;
    CAMSPORK_NODE_NO_VLA()
};

// Update this if you add more expr node types.
static_assert(NumExprTypes == 4);


// Note: definition duplicated as Python ctypes
struct OffsetExtentExpr
{
    ExprRef offset_e;
    ExprRef extent_e;
};



// ******************************************************************************************
// Statement node types, all pointed to by polymorphic StmtRef object (which can be null)
// ******************************************************************************************

using qual_bits_t = uint32_t;

template <uint32_t TypeID>
struct stmt
{
};

static constexpr uint32_t NumStmtTypes = 20;

using StmtRef = NodeRef<stmt, NumStmtTypes>;

template <bool IsWindow>
struct SyncEnvAccessNodeData
{
    Varname name;
    uint32_t initial_qual_bit;
    uint32_t extended_qual_bits;
    static constexpr bool is_window = IsWindow;
    using IdxT = std::conditional_t<IsWindow, OffsetExtentExpr, ExprRef>;
    CAMSPORK_NODE_VLA_MEMBER(IdxT);
};

template <bool IsMutate, bool IsWindow>
struct SyncEnvAccessNode : SyncEnvAccessNodeData<IsWindow>
{
    static constexpr bool is_mutate = IsMutate;
    uint32_t is_ooo;
};

// SyncEnvReadSingle(Varname name, qual_tl initial_qual_bit, qual_tl* extended_qual_bits, bool is_ooo, expr* offset)
using SyncEnvReadSingle = stmt<0>;
template <>
struct stmt<0> : SyncEnvAccessNode<false, false>
{
};

// SyncEnvReadWindow(Varname name, qual_tl initial_qual_bit, qual_tl* extended_qual_bits, bool is_ooo, expr* offset, expr* extent)
using SyncEnvReadWindow = stmt<1>;
template <>
struct stmt<1> : SyncEnvAccessNode<false, true>
{
};

// SyncEnvMutateSingle(Varname name, qual_tl initial_qual_bit, qual_tl* extended_qual_bits, bool is_ooo, expr* offset)
using SyncEnvMutateSingle = stmt<2>;
template <>
struct stmt<2> : SyncEnvAccessNode<true, false>
{
};

// SyncEnvMutateWindow(Varname name, qual_tl initial_qual_bit, qual_tl* extended_qual_bits, bool is_ooo, expr* offset, expr* extent)
using SyncEnvMutateWindow = stmt<3>;
template <>
struct stmt<3> : SyncEnvAccessNode<true, true>
{
};

// MutateValue(Varname name, expr* idx, binop op, expr rhs)
using MutateValue = stmt<4>;
template<>
struct stmt<4>
{
    Varname name;
    binop op;
    ExprRef rhs;
    CAMSPORK_NODE_VLA_MEMBER(ExprRef)
};

// Fence(qual_tl* L1_qual_bits, qual_tl* L2_full_qual_bits, qual_tl* L2_temporal_qual_bits)
using Fence = stmt<5>;
template<>
struct stmt<5>
{
    uint32_t V1_transitive;
    qual_bits_t L1_qual_bits;
    qual_bits_t L2_full_qual_bits;
    qual_bits_t L2_temporal_qual_bits;
    CAMSPORK_NODE_NO_VLA()
};

struct ArriveIdx
{
    Varname idx;
    uint32_t multicast_per_expr;
};

// Arrive(Varname name, bool V1_transitive, qual_tl* L1_qual_bits, Varname* idx, multicast_flag* multicast_flags)
using Arrive = stmt<6>;
template<>
struct stmt<6>
{
    Varname name;
    uint32_t V1_transitive;
    uint32_t L1_qual_bits;
    uint32_t num_barrier_exprs;
    CAMSPORK_NODE_VLA_MEMBER(ArriveIdx)
};

// Await(Varname name, qual_tl* L2_full_qual_bits, qual_tl* L2_temporal_qual_bits, Varname* idx)
using Await = stmt<7>;
template<>
struct stmt<7>
{
    Varname name;
    uint32_t L2_full_qual_bits;
    uint32_t L2_temporal_qual_bits;
    CAMSPORK_NODE_VLA_MEMBER(Varname)
};

// ValueEnvAlloc(Varname name, expr* extent)
using ValueEnvAlloc = stmt<8>;
template<>
struct stmt<8>
{
    Varname name;
    CAMSPORK_NODE_VLA_MEMBER(ExprRef)
};

// SyncEnvAlloc(Varname name, expr* extent)
using SyncEnvAlloc = stmt<9>;
template<>
struct stmt<9>
{
    Varname name;
    CAMSPORK_NODE_VLA_MEMBER(ExprRef)
};

// SyncEnvFreeShard(Varname name, expr* offset, expr* extent)
using SyncEnvFreeShard = stmt<10>;
template<>
struct stmt<10>
{
    Varname name;
    CAMSPORK_NODE_VLA_MEMBER(OffsetExtentExpr)
};

// BarrierEnvAlloc(Varname name, expr* extent)
using BarrierEnvAlloc = stmt<11>;
template<>
struct stmt<11>
{
    Varname name;
    CAMSPORK_NODE_VLA_MEMBER(ExprRef)
};

// BarrierEnvFree(Varname name)
using BarrierEnvFree = stmt<12>;
template<>
struct stmt<12>
{
    Varname name;
    CAMSPORK_NODE_NO_VLA()
};

// StmtBody(stmt* body)
// This is statement composition body[0] ; body[1] ; ...
static constexpr uint32_t StmtBody_ID = 13;
using StmtBody = stmt<13>;
template<>
struct stmt<13>
{
    CAMSPORK_NODE_VLA_MEMBER(StmtRef);
};

// If(expr cond, stmt body, stmt orelse)
static constexpr uint32_t If_ID = 14;
using If = stmt<14>;
template<>
struct stmt<14>
{
    ExprRef cond;
    StmtRef body;
    StmtRef orelse;  // reminder: can be null
    CAMSPORK_NODE_NO_VLA()
};

struct BaseForStmt
{
    Varname iter;
    ExprRef lo;
    ExprRef hi;
    StmtRef body;
    CAMSPORK_NODE_NO_VLA()
};

// SeqFor(Varname iter, expr lo, expr hi, stmt body)
using SeqFor = stmt<15>;
template<>
struct stmt<15> : BaseForStmt
{
};

// TasksFor(Varname iter, expr lo, expr hi, stmt body)
using TasksFor = stmt<16>;
template<>
struct stmt<16> : BaseForStmt
{
};

// ThreadsFor(Varname iter, expr lo, expr hi, int dim_idx, int offset, int box, stmt body)
using ThreadsFor = stmt<17>;
template<>
struct stmt<17> : BaseForStmt
{
    uint32_t dim_idx;
    uint32_t offset;
    uint32_t box;
};

// ParallelBlock(stmt body, int* domain)
using ParallelBlock = stmt<18>;
template<>
struct stmt<18>
{
    StmtRef body;
    CAMSPORK_NODE_VLA_MEMBER(uint32_t)
};

// DomainSplit(stmt body, int dim_idx, int split_factor)
using DomainSplit = stmt<19>;
template<>
struct stmt<19>
{
    StmtRef body;
    uint32_t dim_idx;
    uint32_t split_factor;
    CAMSPORK_NODE_NO_VLA()
};

// Update this if you add more stmt node types.
static_assert(NumStmtTypes == 20);


// ******************************************************************************************
// This is stored at offset 0 in the program buffer.
// ******************************************************************************************
struct ProgramHeader
{
    static const uint32_t expected_magic_numbers[];

    uint32_t magic_numbers[7 + 32 + 32 + 1];
    StmtRef top_level_stmt;
    VarConfigTableRef var_config_table;

    static const ProgramHeader& validate(size_t buffer_size, const char* buffer);
};

static_assert(alignof(ProgramHeader) == 4);

}

CAMSPORK_EXPORT camspork::binop camspork_binop_from_str(const char* p_str);
CAMSPORK_EXPORT const char* camspork_binop_to_str(camspork::binop op);
