# ---------------------------------------------------------------------------- #
# Library code

add_library(
    exo-gemmini_platform
    gemm_acc_malloc.c
    gemm_acc_malloc.h
    gemm_malloc.c
    gemm_malloc.h
)
add_library(exo-gemmini::platform ALIAS exo-gemmini_platform)
target_include_directories(exo-gemmini_platform PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
