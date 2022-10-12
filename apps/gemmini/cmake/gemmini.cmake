cmake_minimum_required(VERSION 3.22)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV "$ENV{RISCV}" CACHE PATH "Path to RISCV toolchain")

set(CMAKE_SYSROOT "${RISCV}/sysroot")
set(CMAKE_CROSSCOMPILING_EMULATOR "${RISCV}/bin/spike" --extension=gemmini)

set(CMAKE_C_COMPILER "${RISCV}/bin/riscv64-unknown-elf-gcc")
set(CMAKE_CXX_COMPILER "${RISCV}/bin/riscv64-unknown-elf-g++")

add_compile_definitions(BAREMETAL=1)
add_compile_options(-mcmodel=medany -fno-tree-loop-distribute-patterns -fno-builtin-printf -fno-common)
add_link_options(-nostartfiles -nostdlib)
