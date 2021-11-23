#!/bin/bash

## Constants

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"

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
  -DCMAKE_C_COMPILER=clang-13 \
  -DCMAKE_CXX_COMPILER=clang++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PWD}/build/_install" \
  -DCMAKE_C_FLAGS="-march=skylake-avx512" \
  -DCMAKE_CXX_FLAGS="-march=skylake-avx512"

cmake --build build/apps

## Run correctness checks

set -e
./build/apps/x86_demo/sgemm/run_systl 1000
./build/apps/x86_demo/conv/test_conv

## Run benchmarks

./build/apps/x86_demo/sgemm/bench_sgemm --benchmark_filter=sys
./build/apps/x86_demo/sgemm/bench_sgemm --benchmark_filter=MKL
taskset -c 0 ./build/apps/x86_demo/sgemm/bench_sgemm_openblas \
  --benchmark_filter=OpenBLAS

export HL_NUM_THREADS=1
taskset -c 0 ./build/apps/x86_demo/conv/bench_conv --benchmark_filter=102
