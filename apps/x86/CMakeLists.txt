cmake_minimum_required(VERSION 3.21)
project(x86 LANGUAGES C CXX)

# ---------------------------------------------------------------------------- #
# Project-wide configuration

if (PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif ()


# ---------------------------------------------------------------------------- #
# Dependencies

## Exo
find_package(Exo REQUIRED)

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
# Benchmarks

add_subdirectory(conv)
add_subdirectory(resnet)
add_subdirectory(sgemm)
add_subdirectory(ssyrk)
