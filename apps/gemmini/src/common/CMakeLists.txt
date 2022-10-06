# ---------------------------------------------------------------------------- #
# Library code

add_library(
    exo-gemmini_helpers
    helpers.c
    helpers.h
    gemm_acc_malloc.c
    gemm_acc_malloc.h
    gemm_malloc.c
    gemm_malloc.h
)
add_library(exo-gemmini::helpers ALIAS exo-gemmini_helpers)
target_include_directories(exo-gemmini_helpers PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
