cmake_minimum_required(VERSION 3.21)
project(x86_demo LANGUAGES C CXX)

# ---------------------------------------------------------------------------- #
# Project-wide configuration

if (PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif ()


# ---------------------------------------------------------------------------- #
# Dependencies

## SYS_ATL
find_package(Python3 REQUIRED)
find_package(SYS_ATL REQUIRED HINTS "${Python3_SITELIB}")

## oneAPI MKL
set(MKL_ARCH "intel64"
    CACHE STRING "MKL architecture. Options: intel64, ia32")
set(MKL_LINK "static"
    CACHE STRING "MKL link type. Options: dynamic, static, sdl")
set(MKL_THREADING "sequential"
    CACHE STRING "MKL threading model. Options: sequential, intel_thread, gnu_thread, pgi_thread, tbb_thread")
set(MKL_INTERFACE "lp64"
    CACHE STRING "MKL interface type. Options: lp64, ilp64")
set(MKL_MPI "openmpi"
    CACHE STRING "Which MPI interface to use. Options: intelmpi, mpich, openmpi, msmpi, mshpc")

find_package(MKL REQUIRED)

## oneAPI DNNL
set(DNNL_CONFIGURATION "cpu_gomp"
    CACHE STRING "DNNL backend. Options: cpu_dpcpp_gpu_dpcpp, cpu_(gomp|iomp), cpu_tbb")

find_package(dnnl REQUIRED)

## Google Benchmark
find_package(benchmark REQUIRED)


# ---------------------------------------------------------------------------- #
# SYS_ATL libraries

add_sys_atl_library(sgemm sgemm.py)


# ---------------------------------------------------------------------------- #
# Test harness executables

## SGEMM
add_executable(run_sgemm run_sgemm.cpp)
target_link_libraries(
    run_sgemm
    PRIVATE
    x86_demo::sgemm
    MKL::MKL
    benchmark::benchmark_main
)
target_compile_features(run_sgemm PRIVATE cxx_std_11)

## Convolution

## RESNET

## SYRK

## Transformer


# ---------------------------------------------------------------------------- #
# CTest configuration

if (BUILD_TESTING)
  add_test(NAME sgemm_100 COMMAND run_sgemm 100)
  set_tests_properties(sgemm_100 PROPERTIES ENVIRONMENT "${MKL_ENV}")
endif ()
