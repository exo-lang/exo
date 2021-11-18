# ---------------------------------------------------------------------------- #
# SYS_ATL libraries

add_sys_atl_library(conv conv.py)


# ---------------------------------------------------------------------------- #
# OneDNN Conv common libraries

add_library(conv_impls OBJECT sys_atl_conv.cpp onednn_conv.cpp)
target_link_libraries(conv_impls PUBLIC DNNL::dnnl PRIVATE x86_demo::conv)
target_compile_features(conv_impls PUBLIC cxx_std_17)


# ---------------------------------------------------------------------------- #
# Test harness

add_executable(bench_conv bench_conv.cpp)
target_link_libraries(
  bench_conv
  PRIVATE
  benchmark::benchmark_main
  conv_impls
)
target_compile_features(bench_conv PRIVATE cxx_std_17)

add_executable(test_conv test_conv.cpp)
target_link_libraries(
  test_conv
  PRIVATE
  conv_impls
)
target_compile_features(test_conv PRIVATE cxx_std_17)
