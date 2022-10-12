# ---------------------------------------------------------------------------- #
# Test harness

add_executable(bench_ssyrk bench_ssyrk.cpp)
target_link_libraries(
  bench_ssyrk
  PRIVATE
  MKL::MKL
  benchmark::benchmark_main
)
target_compile_features(bench_ssyrk PRIVATE cxx_std_11)
