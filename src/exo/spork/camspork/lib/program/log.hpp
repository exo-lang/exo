#pragma once

#include <memory>
#include <string>
#include <vector>

#include "int_types.hpp"
#include "camspork_excut.hpp"

namespace camspork {

struct SyncvLogRequest
{
    std::string var_str_name;
    std::vector<extent_t> idx_for_single;

    // May be null
    std::vector<std::unique_ptr<ExcutBaseAction>>* p_excut_actions;
};

}
