#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"

# Meant to be run from the root of the repo:
# ./apps/build.sh

rm -rf build/apps
(cd "${ROOT_DIR}" && pip uninstall -y SYS_ATL && python -m build &&
  pip install dist/*.whl)

cmake -G Ninja -S "${ROOT_DIR}/dependencies/benchmark" -B build/benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DBENCHMARK_ENABLE_TESTING=NO \
  -DCMAKE_INSTALL_PREFIX="${PWD}/build/_install"

cmake --build build/benchmark --target install

cmake -G Ninja -S "${ROOT_DIR}/apps" -B build/apps \
  -DCMAKE_C_COMPILER=clang-13 \
  -DCMAKE_CXX_COMPILER=clang++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PWD}/build/_install" \
  -DCMAKE_C_FLAGS="-march=skylake-avx512 -g3" \
  -DCMAKE_CXX_FLAGS="-march=skylake-avx512 -g3"

cmake --build build/apps
