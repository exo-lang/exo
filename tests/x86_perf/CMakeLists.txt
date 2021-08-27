cmake_minimum_required(VERSION 3.21)
project(x86_perf)

include(FetchContent)

##########################################################
# Locate BLIS
find_package(PkgConfig REQUIRED)
pkg_check_modules(blis REQUIRED IMPORTED_TARGET blis)


##########################################################
# Download Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY git@github.com:google/benchmark.git
    GIT_TAG        v1.5.6
)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
FetchContent_MakeAvailable(benchmark)


##########################################################
# Get test-harness-built SYS_TL sgemm kernel
find_library(
    SGEMM_LIBRARY test_avx2_sgemm_6x16.so REQUIRED
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../tmp/"
)

find_path(
    SGEMM_INCLUDE_DIR test_avx2_sgemm_6x16.h REQUIRED
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../tmp/"
)

add_library(avx2_sgemm_6x16 IMPORTED SHARED)
set_target_properties(
    avx2_sgemm_6x16 PROPERTIES IMPORTED_LOCATION "${SGEMM_LIBRARY}")
target_include_directories(
    avx2_sgemm_6x16 INTERFACE "${SGEMM_INCLUDE_DIR}")

##########################################################
# Get test-harness-built SYS_TL sgemm kernel
find_library(
    SGEMM_FULL_LIBRARY test_avx2_sgemm_full.so REQUIRED
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../tmp/"
)

find_path(
    SGEMM_FULL_INCLUDE_DIR test_avx2_sgemm_full.h REQUIRED
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../tmp/"
)

add_library(avx2_sgemm_full IMPORTED SHARED)
set_target_properties(
    avx2_sgemm_full PROPERTIES IMPORTED_LOCATION "${SGEMM_FULL_LIBRARY}")
target_include_directories(
    avx2_sgemm_full INTERFACE "${SGEMM_FULL_INCLUDE_DIR}")


##########################################################
# Build main benchmark
add_executable(run benchmark.cpp)
target_link_libraries(
    run PRIVATE avx2_sgemm_6x16 avx2_sgemm_full
                PkgConfig::blis benchmark::benchmark)
target_compile_options(
    run PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-march=skylake>)
