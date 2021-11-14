# ---------------------------------------------------------------------------- #
# SYS_ATL libraries

add_sys_atl_library(sgemm sgemm.py)


# ---------------------------------------------------------------------------- #
# Test harness

add_executable(bench_sgemm bench_sgemm.cpp alex_sgemm.cpp)
target_link_libraries(
  bench_sgemm
  PRIVATE
  x86_demo::sgemm
  MKL::MKL
  benchmark::benchmark_main
)
target_compile_features(bench_sgemm PRIVATE cxx_std_17)

# ---------------------------------------------------------------------------- #
# CTest configuration

if (BUILD_TESTING)
  add_test(NAME sgemm_100 COMMAND bench_sgemm 100)
  set_tests_properties(sgemm_100 PROPERTIES ENVIRONMENT "${MKL_ENV}")
endif ()
