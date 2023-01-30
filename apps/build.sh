#!/bin/bash
set -e

## Constants

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"

## Compiler configuration

: "${CC:=clang-13}"
: "${CXX:=clang++-13}"
: "${CFLAGS:=-march=native}"
: "${CXXFLAGS:=$CFLAGS}"
: "${CMAKE_BUILD_TYPE:=Release}"
export CC CXX CFLAGS CXXFLAGS CMAKE_BUILD_TYPE

## Build dependencies

# Ensure Exo is up to date
(cd "${ROOT_DIR}" && pip uninstall -y exo-lang && python -m build &&
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
  -DCMAKE_PREFIX_PATH="${PWD}/build/_install"

cmake --build build/apps

## Run correctness checks

./build/apps/x86/sgemm/run_exo 1000
./build/apps/x86/conv/test_conv

## Run benchmarks

export HL_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

./build/apps/x86/sgemm/bench_sgemm \
  --benchmark_filter=exo --benchmark_out=sgemm_exo.json
./build/apps/x86/sgemm/bench_sgemm \
  --benchmark_filter=MKL --benchmark_out=sgemm_mkl.json
./build/apps/x86/sgemm/bench_sgemm_openblas \
  --benchmark_filter=OpenBLAS --benchmark_out=sgemm_openblas.json

./build/apps/x86/conv/bench_conv \
  --benchmark_filter=102 --benchmark_out=conv.json
