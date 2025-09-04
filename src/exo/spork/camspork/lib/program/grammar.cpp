#include "grammar.hpp"

namespace camspork
{

BinOpNames::BinOpNames()
{
    names[0] = "<error>";
    names[static_cast<uint32_t>(binop::Assign)] = "=";
    names[static_cast<uint32_t>(binop::Add)] = "+";
    names[static_cast<uint32_t>(binop::Sub)] = "-";
    names[static_cast<uint32_t>(binop::Mul)] = "*";
    names[static_cast<uint32_t>(binop::Div)] = "/";
    names[static_cast<uint32_t>(binop::Mod)] = "%";
    names[static_cast<uint32_t>(binop::Less)] = "<";
    names[static_cast<uint32_t>(binop::Leq)] = "<=";
    names[static_cast<uint32_t>(binop::Greater)] = ">";
    names[static_cast<uint32_t>(binop::Geq)] = ">=";
    names[static_cast<uint32_t>(binop::Eq)] = "==";
    names[static_cast<uint32_t>(binop::Neq)] = "!=";
}

const BinOpNames binop_names;

BinOpTable::BinOpTable()
{
    entries_by_char['='][0] = BinOpTableEntry{'\0', binop::Assign};
    entries_by_char['+'][0] = BinOpTableEntry{'\0', binop::Add};
    entries_by_char['-'][0] = BinOpTableEntry{'\0', binop::Sub};
    entries_by_char['*'][0] = BinOpTableEntry{'\0', binop::Mul};
    entries_by_char['/'][0] = BinOpTableEntry{'\0', binop::Div};
    entries_by_char['%'][0] = BinOpTableEntry{'\0', binop::Mod};
    entries_by_char['<'][0] = BinOpTableEntry{'\0', binop::Less};
    entries_by_char['<'][1] = BinOpTableEntry{'=', binop::Leq};
    entries_by_char['>'][0] = BinOpTableEntry{'\0', binop::Greater};
    entries_by_char['>'][1] = BinOpTableEntry{'=', binop::Geq};
    entries_by_char['='][1] = BinOpTableEntry{'=', binop::Eq};
    entries_by_char['!'][1] = BinOpTableEntry{'=', binop::Neq};
}

binop BinOpTable::get(const char* p_str) const
{
    uint8_t first_char = uint8_t(p_str[0]);
    // String length must be 1 or 2 to match anything.
    if (first_char != 0 && (p_str[1] == 0 || p_str[2] == 0)) {
        int second_char = int(uint8_t(p_str[1]));
        for (BinOpTableEntry entry : entries_by_char[first_char]) {
            if (second_char == entry.second_char) {
                return entry.op;
            }
        }
    }
    throw std::runtime_error(std::string("BinOpTable::get, unknown: \"") + p_str + "\"");
}

const BinOpTable binop_table;

const uint32_t ProgramHeader::expected_magic_numbers[] = {
    0x800A0D0A,
    0x736d6163,
    0x6b726f70,
    2,  // Format version number
    sizeof(ProgramHeader),
    sizeof(VarConfig),
    sizeof(VarConfigTable),
    sizeof(stmt<0>),
    sizeof(stmt<1>),
    sizeof(stmt<2>),
    sizeof(stmt<3>),
    sizeof(stmt<4>),
    sizeof(stmt<5>),
    sizeof(stmt<6>),
    sizeof(stmt<7>),
    sizeof(stmt<8>),
    sizeof(stmt<9>),
    sizeof(stmt<10>),
    sizeof(stmt<11>),
    sizeof(stmt<12>),
    sizeof(stmt<13>),
    sizeof(stmt<14>),
    sizeof(stmt<15>),
    sizeof(stmt<16>),
    sizeof(stmt<17>),
    sizeof(stmt<18>),
    sizeof(stmt<19>),
    sizeof(stmt<20>),
    sizeof(stmt<21>),
    sizeof(stmt<22>),
    sizeof(stmt<23>),
    sizeof(stmt<24>),
    sizeof(stmt<25>),
    sizeof(stmt<26>),
    sizeof(stmt<27>),
    sizeof(stmt<28>),
    sizeof(stmt<29>),
    sizeof(stmt<30>),
    sizeof(stmt<31>),
    sizeof(expr<0>),
    sizeof(expr<1>),
    sizeof(expr<2>),
    sizeof(expr<3>),
    sizeof(expr<4>),
    sizeof(expr<5>),
    sizeof(expr<6>),
    sizeof(expr<7>),
    sizeof(expr<8>),
    sizeof(expr<9>),
    sizeof(expr<10>),
    sizeof(expr<11>),
    sizeof(expr<12>),
    sizeof(expr<13>),
    sizeof(expr<14>),
    sizeof(expr<15>),
    sizeof(expr<16>),
    sizeof(expr<17>),
    sizeof(expr<18>),
    sizeof(expr<19>),
    sizeof(expr<20>),
    sizeof(expr<21>),
    sizeof(expr<22>),
    sizeof(expr<23>),
    sizeof(expr<24>),
    sizeof(expr<25>),
    sizeof(expr<26>),
    sizeof(expr<27>),
    sizeof(expr<28>),
    sizeof(expr<29>),
    sizeof(expr<30>),
    sizeof(expr<31>),
    0xFFFFFFFF,
};

static_assert(sizeof(ProgramHeader::magic_numbers) == sizeof(ProgramHeader::expected_magic_numbers));

const ProgramHeader& ProgramHeader::validate(size_t buffer_size, const char* buffer)
{
    CAMSPORK_C_BOUNDSCHECK(sizeof(ProgramHeader), buffer_size + 1);
    const auto& header = reinterpret_cast<const ProgramHeader&>(buffer[0]);
    for (const uint32_t& expected_magic : expected_magic_numbers) {
        const uint32_t magic = header.magic_numbers[&expected_magic - &expected_magic_numbers[0]];
        CAMSPORK_REQUIRE_CMP(magic, ==, expected_magic, "incorrect magic number");
    }
    return header;
}

}

camspork::binop camspork_binop_from_str(const char* p_str)
{
    CAMSPORK_API_PROLOGUE
    return camspork::binop_table.get(p_str);
    CAMSPORK_API_EPILOGUE(static_cast<camspork::binop>(0));
}

const char* camspork_binop_to_str(camspork::binop op)
{
    CAMSPORK_API_PROLOGUE
    return camspork::binop_names.get(op);
    CAMSPORK_API_EPILOGUE(nullptr);
}
