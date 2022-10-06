# ---------------------------------------------------------------------------- #
# Header shims

configure_file(shims/include/gemmini.h.in include/gemmini.h @ONLY)
configure_file(shims/include/gemmini_testutils.h.in include/gemmini_testutils.h @ONLY)
configure_file(shims/include/gemmini_params.h.in include/gemmini_params.h @ONLY)
configure_file(shims/include/gemmini_counter.h.in include/gemmini_counter.h @ONLY)
configure_file(shims/rocc-software/src/xcustom.h.in rocc-software/src/xcustom.h @ONLY)


# ---------------------------------------------------------------------------- #
# Exo implementations

add_exo_library(gemmini_lib conv/conv.py matmul/matmul.py)
target_sources(gemmini_lib PRIVATE shims/gemmini.c)
target_include_directories(gemmini_lib SYSTEM PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(gemmini_lib PRIVATE exo-gemmini::helpers)
