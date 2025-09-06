#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "grammar.hpp"

namespace camspork
{

struct ExcutBaseAction
{
    // Action name.
    // Don't include characters that would require escaping in JSON.
    const char* p_action_name = nullptr;

    // Write list of actions in JSON list syntax.
    virtual void write_args(FILE* file) const = 0;
};

struct ExcutSyncEnvAccess : ExcutBaseAction
{
    uint32_t id_before;
    uint32_t id_after;
    std::string name;
    std::vector<extent_t> idx;

    virtual void write_args(FILE* file) const override;
};

struct ExcutVisRecord : ExcutBaseAction
{
    uint32_t id;
    uint32_t original_qual_bit;

    virtual void write_args(FILE* file) const override;
};

struct ExcutTlSigInterval : ExcutBaseAction
{
    uint32_t tid_lo;
    uint32_t tid_hi;
    uint32_t qual_bits;
    uint32_t vis_level;

    virtual void write_args(FILE* file) const override;
};

struct ExcutPendingAwait : ExcutBaseAction
{
    uint32_t id;
    uint32_t arrive_count;

    virtual void write_args(FILE* file) const override;
};

struct ExcutBarrierAlloc : ExcutBaseAction
{
    uint32_t id;
    std::string name;
    std::vector<extent_t> idx;

    virtual void write_args(FILE* file) const override;
};

}
