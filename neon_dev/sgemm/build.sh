#!/bin/bash

## Constants

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"

## Build dependencies

# Ensure SYS_ATL is up to date
(cd "${ROOT_DIR}" && pip uninstall -y SYS_ATL && python -m build &&
  pip install dist/*.whl)

# set up cmake build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# do the build
VERBOSE=1 cmake --build build

# run a single case
./build/test 768