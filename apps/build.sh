#!/bin/bash

## Constants

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
: "${CC:=clang-13}"
: "${CXX:=clang++-13}"

## Build dependencies

# Ensure SYS_ATL is up to date
(cd "${ROOT_DIR}" && pip uninstall -y SYS_ATL && python -m build &&
  pip install dist/*.whl)

# Build and stage Google Benchmark
cmake -G Ninja -S "${ROOT_DIR}/dependencies/benchmark" -B build/benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DBENCHMARK_ENABLE_TESTING=NO \
  -DCMAKE_INSTALL_PREFIX="${PWD}/build/_install"

cmake --build build/benchmark --target install

## Build apps

rm -rf build/apps
cmake -G Ninja -S "${ROOT_DIR}/apps" -B build/apps \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PWD}/build/_install" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_FLAGS="-march=skylake-avx512 $CFLAGS" \
  -DCMAKE_CXX_FLAGS="-march=skylake-avx512 $CXXFLAGS"

cmake --build build/apps

## Run correctness checks

set -e
./build/apps/x86_demo/sgemm/run_systl 1000
./build/apps/x86_demo/conv/test_conv

## Run benchmarks

export HL_NUM_THREADS=1
taskset -c 0 ./build/apps/x86_demo/conv/bench_conv --benchmark_filter=102
taskset -c 0 ./build/apps/x86_demo/sgemm/bench_sgemm_openblas \
  --benchmark_filter=sys_atl
taskset -c 0 ./build/apps/x86_demo/sgemm/bench_sgemm_openblas \
  --benchmark_filter=OpenBLAS
taskset -c 0 ./build/apps/x86_demo/sgemm/bench_sgemm --benchmark_filter=MKL
