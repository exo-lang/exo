#!/bin/bash

set -e

APPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$APPS_DIR"

./build.sh

export HL_NUM_THREADS=1
taskset -c 0 ./build/apps/x86_demo/conv/bench_conv --benchmark_filter=102
