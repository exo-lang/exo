#pragma once

#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "grammar.hpp"

namespace camspork
{

enum class ExcutMutateTag
{
    Read,
    Mutate,
    RAW,
    WAR,
    WAW,
};

struct ExcutBaseAction
{
    // Action name.
    // Don't include characters that would require escaping in JSON.
    virtual const char* action_name() const = 0;

    // Write list of actions in JSON list syntax.
    virtual void write_args(FILE* file) const = 0;
};

struct ExcutSyncEnvAccess : ExcutBaseAction
{
    uint32_t id_before;
    uint32_t id_after;
    std::string name;
    std::vector<extent_t> idx;
    ExcutMutateTag mutate_tag;

    virtual const char* action_name() const override;
    virtual void write_args(FILE* file) const override;
};

struct ExcutVisRecord : ExcutBaseAction
{
    uint32_t id;
    uint32_t original_qual_bit;
    ExcutMutateTag mutate_tag;

    virtual const char* action_name() const override;
    virtual void write_args(FILE* file) const override;
};

struct ExcutTlSigInterval : ExcutBaseAction
{
    uint32_t tid_lo;
    uint32_t tid_hi;
    uint32_t atomic_only_qual_bits;
    uint32_t unordered_qual_bits;
    uint32_t temporal_ordered_qual_bits;
    uint32_t full_ordered_qual_bits;
    ExcutMutateTag mutate_tag;

    virtual const char* action_name() const override;
    virtual void write_args(FILE* file) const override;
};

struct ExcutPendingAwait : ExcutBaseAction
{
    uint32_t barrier_id;
    uint32_t arrive_count;
    ExcutMutateTag mutate_tag;

    virtual const char* action_name() const override;
    virtual void write_args(FILE* file) const override;
};

struct ExcutBarrierAlloc : ExcutBaseAction
{
    uint32_t id;
    std::string name;
    std::vector<extent_t> idx;

    virtual const char* action_name() const override;
    virtual void write_args(FILE* file) const override;
};

}
