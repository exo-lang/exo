#!/bin/bash

set -e

## Constants

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
LOCAL_PREFIX="${PWD}/.local"

export CC="${CC:=clang-13}"
export CXX="${CXX:=clang++-13}"

export HL_NUM_THREADS=1

## Build dependencies

# Ensure SYS_ATL is up to date
(cd "${ROOT_DIR}" && pip uninstall -y SYS_ATL && python -m build &&
  pip install dist/*.whl)

# Build and stage Google Benchmark
cmake -G Ninja -S "${ROOT_DIR}/dependencies/benchmark" -B build/benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DBENCHMARK_ENABLE_TESTING=NO \
  -DCMAKE_INSTALL_PREFIX="${LOCAL_PREFIX}"

cmake --build build/benchmark --target install

# Build and stage Google Test
cmake -G Ninja -S "${ROOT_DIR}/dependencies/googletest" -B build/googletest \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_GMOCK=OFF \
  -DCMAKE_INSTALL_PREFIX="${LOCAL_PREFIX}"

cmake --build build/googletest --target install

## Build apps

rm -rf build/apps
cmake -G Ninja -S "${ROOT_DIR}/apps" -B build/apps \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${LOCAL_PREFIX}" \
  -DCMAKE_C_FLAGS="-march=skylake-avx512" \
  -DCMAKE_CXX_FLAGS="-march=skylake-avx512"

cmake --build build/apps

## Run correctness checks and benchmarks

(cd build/apps && taskset -c 0 ctest)
