# ---------------------------------------------------------------------------- #
# SYS_ATL libraries

add_sys_atl_library(conv conv.py)


# ---------------------------------------------------------------------------- #
# Test harness

add_executable(bench_conv bench_conv.cpp)
target_link_libraries(
  bench_conv
  PRIVATE
  x86_demo::conv
  DNNL::dnnl
  benchmark::benchmark_main
)
target_compile_features(bench_conv PRIVATE cxx_std_17)
