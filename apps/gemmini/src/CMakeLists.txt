# ---------------------------------------------------------------------------- #
# Gemmini platform library

add_subdirectory(platform)


# ---------------------------------------------------------------------------- #
# Exo implementations

add_exo_library(
  NAME gemmini_lib
  SOURCES exo/conv.py exo/matmul.py
)
target_include_directories(gemmini_lib SYSTEM PRIVATE "${gemmini-rocc-tests_SOURCE_DIR}")
target_link_libraries(gemmini_lib PUBLIC exo-gemmini::platform)
