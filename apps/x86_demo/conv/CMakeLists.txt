# ---------------------------------------------------------------------------- #
# SYS_ATL libraries

add_sys_atl_library(conv conv.py)


# ---------------------------------------------------------------------------- #
# OneDNN Conv common libraries

add_library(onednn_conv OBJECT onednn_conv.cpp)
target_link_libraries(onednn_conv PUBLIC DNNL::dnnl)
target_compile_features(onednn_conv PRIVATE cxx_std_17)

# ---------------------------------------------------------------------------- #
# Test harness

add_executable(bench_conv bench_conv.cpp)
target_link_libraries(
  bench_conv
  PRIVATE
  x86_demo::conv
  benchmark::benchmark_main
  onednn_conv
)
target_compile_features(bench_conv PRIVATE cxx_std_17)

add_executable(test_conv test_conv.cpp)
target_link_libraries(
  test_conv
  PRIVATE
  x86_demo::conv
  DNNL::dnnl
  onednn_conv
)
target_compile_features(test_conv PRIVATE cxx_std_17)
