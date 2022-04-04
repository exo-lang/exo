#!/bin/bash

## Constants

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"

## Build dependencies

# Ensure Exo is up to date
(cd "${ROOT_DIR}" && pip uninstall -y exo-lang && python -m build &&
  pip install dist/*.whl)

# set up cmake build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# do the build
cmake --build build --verbose

# run a single case
./build/test 768