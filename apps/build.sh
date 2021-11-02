#!/bin/bash

echo "WARNING: this is meant to be run on Alex's computer. YMMV"

# Meant to be run from the root of the repo:
# ./apps/build.sh

cmake -G Ninja -S dependencies/benchmark -B build/benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DBENCHMARK_ENABLE_TESTING=NO \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/_install"

cmake --build build/benchmark --target install

cmake -G Ninja -S apps -B build/apps \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build/_install" \
  -DCMAKE_C_FLAGS="-march=skylake-avx512" \
  -DCMAKE_CXX_FLAGS="-march=skylake-avx512"

cmake --build build/apps
