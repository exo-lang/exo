
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Expose the built-in Scheduling operators here

from ..API_scheduling import (
  is_atomic_scheduling_op,
  # basic operations
  simplify,
  rename,
  make_instr,
  #
  # precision, memory, and window annotation setting
  set_precision,
  set_window,
  set_memory,
  #
  # Configuration modifying operations
  bind_config,
  delete_config,
  write_config,
  #
  # buffer and window oriented operations
  expand_dim,
  reuse_buffer,
  inline_window,
  # loop rewriting
  divide_loop,
  # deprecated scheduling operations
  add_unsafe_guard,
)