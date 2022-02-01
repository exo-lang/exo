# ---------------------------------------------------------------------------- #
# SYSTL libraries

add_sys_atl_library(ssyrk ssyrk.py)


# ---------------------------------------------------------------------------- #
# Test harness

add_executable(bench_ssyrk bench_ssyrk.cpp)
target_link_libraries(
  bench_ssyrk
  PRIVATE
  x86_demo::ssyrk
  MKL::MKL
  benchmark::benchmark_main
  apps::util
)
target_compile_features(bench_ssyrk PRIVATE cxx_std_17)

add_executable(test_ssyrk test_ssyrk.cpp)
target_link_libraries(
  test_ssyrk
  PRIVATE
  x86_demo::ssyrk
  MKL::MKL
  GTest::gtest_main
  apps::util
)
target_compile_features(test_ssyrk PRIVATE cxx_std_17)
