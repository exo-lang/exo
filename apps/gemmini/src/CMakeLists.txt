# ---------------------------------------------------------------------------- #
# Common helpers

add_subdirectory(common)


# ---------------------------------------------------------------------------- #
# Exo implementations

add_exo_library(gemmini_lib exo/conv.py exo/matmul.py)
target_include_directories(gemmini_lib SYSTEM PRIVATE "${gemmini-rocc-tests_SOURCE_DIR}")
target_link_libraries(gemmini_lib PRIVATE exo-gemmini::helpers)
