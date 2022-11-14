cmake_minimum_required(VERSION 3.22)

# NOTICE: this project only has any hope of compiling with the RISC-V toolchain
# provided by UCB-BAR. Thus, the included toolchain file is set by default, even
# though this is typically a bad idea.
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/cmake/gemmini.cmake"
    CACHE FILEPATH "CMake toolchain file")

project(exo-gemmini LANGUAGES C ASM)

enable_testing()

# Naughty
add_compile_options(
    -Wno-pointer-to-int-cast
    -Wno-int-to-pointer-cast
    -Wno-incompatible-pointer-types
    -Wno-discarded-qualifiers
)

# ---------------------------------------------------------------------------- #
# Project-wide configuration

if (PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif ()

# ---------------------------------------------------------------------------- #
# Dependencies

## Exo
find_package(Exo REQUIRED)

## gemmini-rocc-tests
include(FetchContent)

FetchContent_Declare(
  gemmini-rocc-tests
  GIT_REPOSITORY https://github.com/exo-lang/gemmini-rocc-tests.git
  GIT_TAG        exo-v2
)

FetchContent_MakeAvailable(gemmini-rocc-tests)

# ---------------------------------------------------------------------------- #
# Benchmarks

add_subdirectory(src)
add_subdirectory(test)
