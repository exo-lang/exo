#pragma once

#include <memory>
#include <string>
#include <vector>

#include "camspork_excut.hpp"
#include "int_types.hpp"

namespace camspork {

class VisRecordHistoryLog;

struct SyncvLogRequest {
  std::string var_str_name;
  std::vector<extent_t> idx_for_single;

  // May be null
  std::vector<std::unique_ptr<ExcutBaseAction>> *p_excut_actions;

  // May be null
  VisRecordHistoryLog *p_history_log;
};

}  // namespace camspork
