cmake_minimum_required(VERSION 3.22)

set(RISCV "$ENV{RISCV}" CACHE PATH "Path to RISCV toolchain")
set(ARCH riscv64-unknown-elf)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_SYSROOT "${RISCV}/sysroot")
set(CMAKE_CROSSCOMPILING_EMULATOR "${RISCV}/bin/spike" --extension=gemmini)

set(CMAKE_C_COMPILER "${RISCV}/bin/${ARCH}-gcc")
set(CMAKE_CXX_COMPILER "${RISCV}/bin/${ARCH}-g++")

set(CMAKE_C_FLAGS_INIT "-mcmodel=medany -nostartfiles -nostdlib -fno-tree-loop-distribute-patterns -fno-builtin-printf -fno-common")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
