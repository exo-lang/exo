#!/bin/bash

## Constants

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." >/dev/null 2>&1 && pwd)"

## Build dependencies

# Ensure Exo is up to date
if [ "$1" = "update" ]; then
  (cd "${ROOT_DIR}" && pip uninstall -y exo-lang && python -m build &&
     pip install dist/*.whl)
fi

# set up cmake build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64

# do the build
cmake --build build --verbose

# run a single case
./build/test 768
