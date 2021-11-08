# ---------------------------------------------------------------------------- #
# Test harness

add_executable(bench_conv bench_conv.cpp)
target_link_libraries(
  bench_conv
  PRIVATE
  DNNL::dnnl
  benchmark::benchmark_main
)
target_compile_features(bench_conv PRIVATE cxx_std_17)
