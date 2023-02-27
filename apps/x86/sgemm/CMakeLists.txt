# ---------------------------------------------------------------------------- #
# Exo libraries

add_exo_library(
  NAME sgemm
  SOURCES sgemm.py
)


# ---------------------------------------------------------------------------- #
# Test harness

add_library(alex_sgemm OBJECT alex_sgemm.cpp)
target_compile_features(alex_sgemm PRIVATE cxx_std_17)

add_executable(bench_sgemm bench_sgemm.cpp)
target_link_libraries(
  bench_sgemm
  PRIVATE
  x86::sgemm
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
  x86::sgemm
  benchmark::benchmark_main
  PkgConfig::OpenBLAS
  alex_sgemm
)
target_compile_features(bench_sgemm_openblas PRIVATE cxx_std_17)
target_compile_definitions(bench_sgemm_openblas PRIVATE CBLAS_NAME="OpenBLAS")

# ---------------------------------------------------------------------------- #
# CTest configuration

if (BUILD_TESTING)
  add_test(NAME sgemm_100 COMMAND bench_sgemm 100)
  set_tests_properties(sgemm_100 PROPERTIES ENVIRONMENT "${MKL_ENV}")
endif ()

# ---------------------------------------------------------------------------- #
# Tiny harness

add_executable(run_exo test.cpp)
target_link_libraries(
  run_exo
  PRIVATE
  x86::sgemm
  alex_sgemm
)
target_compile_features(run_exo PRIVATE cxx_std_17)
