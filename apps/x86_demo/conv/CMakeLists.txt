# ---------------------------------------------------------------------------- #
# Conv implementations

find_package(Halide REQUIRED)

## SYS_ATL
add_sys_atl_library(conv conv.py)

## Halide
add_executable(halide_conv.gen halide_conv_gen.cpp)
target_link_libraries(halide_conv.gen PRIVATE Halide::Generator)

add_halide_library(halide_conv_kernel FROM halide_conv.gen
                   STMT_HTML _)

## Common object library
add_library(conv_impls OBJECT sys_atl_conv.cpp onednn_conv.cpp halide_conv.cpp)
target_link_libraries(
  conv_impls
  PUBLIC DNNL::dnnl
  PRIVATE x86_demo::conv halide_conv_kernel
)
target_compile_features(conv_impls PUBLIC cxx_std_17)


# ---------------------------------------------------------------------------- #
# Test harness

# Benchmarking
add_executable(bench_conv bench_conv.cpp)
target_link_libraries(bench_conv PRIVATE benchmark::benchmark_main conv_impls)
target_compile_features(bench_conv PRIVATE cxx_std_17)

# Correctness test
add_executable(test_conv test_conv.cpp)
target_link_libraries(test_conv PRIVATE conv_impls)
target_compile_features(test_conv PRIVATE cxx_std_17)


# ---------------------------------------------------------------------------- #
# CTest configuration

if (BUILD_TESTING)
  add_test(NAME test_conv
           COMMAND test_conv)

  add_test(NAME bench_conv
           COMMAND bench_conv --benchmark_filter=102)
endif ()
