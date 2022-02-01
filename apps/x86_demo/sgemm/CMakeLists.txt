# ---------------------------------------------------------------------------- #
# SYS_ATL libraries

add_sys_atl_library(sgemm sgemm.py)


# ---------------------------------------------------------------------------- #
# Test harness

add_library(alex_sgemm OBJECT alex_sgemm.cpp)
target_compile_features(alex_sgemm PRIVATE cxx_std_17)

add_executable(bench_sgemm bench_sgemm.cpp)
target_link_libraries(
  bench_sgemm
  PRIVATE
  x86_demo::sgemm
  MKL::MKL
  benchmark::benchmark_main
  alex_sgemm
)
target_compile_features(bench_sgemm PRIVATE cxx_std_17)
target_compile_definitions(bench_sgemm PRIVATE CBLAS_NAME="MKL")

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenBLAS REQUIRED IMPORTED_TARGET
                  openblas)

add_executable(bench_sgemm_openblas bench_sgemm.cpp)
target_link_libraries(
  bench_sgemm_openblas
  PRIVATE
  x86_demo::sgemm
  benchmark::benchmark_main
  PkgConfig::OpenBLAS
  alex_sgemm
)
target_compile_features(bench_sgemm_openblas PRIVATE cxx_std_17)
target_compile_definitions(bench_sgemm_openblas PRIVATE CBLAS_NAME="OpenBLAS")


# ---------------------------------------------------------------------------- #
# Tiny harness

add_executable(run_systl test.cpp)
target_link_libraries(
  run_systl
  PRIVATE
  x86_demo::sgemm
  alex_sgemm
)
target_compile_features(run_systl PRIVATE cxx_std_17)


# ---------------------------------------------------------------------------- #
# CTest configuration

if (BUILD_TESTING)
  add_test(NAME bench_sgemm_sys
           COMMAND bench_sgemm --benchmark_filter=sys)
  add_test(NAME bench_sgemm_mkl
           COMMAND bench_sgemm --benchmark_filter=MKL)
  add_test(NAME bench_sgemm_openblas
           COMMAND bench_sgemm_openblas --benchmark_filter=OpenBLAS)

  add_test(NAME run_systl
           COMMAND run_systl 1000)

  # set_tests_properties(sgemm_100 PROPERTIES ENVIRONMENT "${MKL_ENV}")
endif ()
