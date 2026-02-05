#pragma once

#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <string>

#include "api_util.hpp"

namespace camspork
{

struct RequireFail : std::runtime_error
{
    RequireFail() : std::runtime_error("camspork::RequireFail")
    {
    }
};

std::string& thread_local_message_ref();

template <typename LHS, typename RHS>
[[noreturn]] void require_cmp_fail(
    const char* file, int line,
    LHS lhs, RHS rhs,
    const char* lhs_name, const char* op, const char* rhs_name, const char* msg)
{
    try {
        std::stringstream s;
        s << "Failed requirement: " << lhs_name << " " << op << " " << rhs_name;
        s << " (with " << lhs_name << " = " << lhs << "; ";
        s << rhs_name << " = " << rhs << "; ";
        s << msg << " @ " << file << ":" << line << ")";
        thread_local_message_ref() = s.str();
    }
    catch (...) {
        fprintf(stderr, "Exception while formatting error: %s:%i\n", __FILE__, __LINE__);
    }
    fprintf(stderr, "%s\n", thread_local_message_ref().c_str());
    throw RequireFail{};
}

[[noreturn]] void require_fail(
    const char* file, int line, const char* expr_str, const char* msg);

}  // end namespace

CAMSPORK_EXPORT const char* camspork_thread_local_message_c_str();

#define CAMSPORK_REQUIRE(expr, msg) do { \
    if (!(expr)) { \
        ::camspork::require_fail(__FILE__, __LINE__, #expr, msg); \
    } \
} while (0)

#define CAMSPORK_REQUIRE_CMP(lhs, op, rhs, msg) do { \
    if (!((lhs) op (rhs))) { \
        ::camspork::require_cmp_fail(__FILE__, __LINE__, lhs, rhs, #lhs, #op, #rhs, msg); \
    } \
} while (0)

#define CAMSPORK_C_BOUNDSCHECK(i, bounds) \
    CAMSPORK_REQUIRE_CMP(size_t(i), <, size_t(bounds), "this may be a bug in the C library, or a corrupt input file")
