cmake_minimum_required(VERSION 3.21)
project(aarch64_sgemm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -ffp-contract=fast -framework Accelerate -save-temps=obj")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -ffp-contract=fast -framework Accelerate -save-temps=obj")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}:../../../src/exo/cmake")

if (EXISTS "$ENV{PYENV_ROOT}")
  execute_process(
    COMMAND "$ENV{PYENV_ROOT}/shims/python" "-c" "import platform ; print(platform.python_version())"
    OUTPUT_VARIABLE python_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(Python3_ROOT_DIR "$ENV{PYENV_ROOT}/versions/${python_version}"
      CACHE PATH "Path to Python3 root directory")
endif ()

find_package(Exo REQUIRED)

# ---------------------------------------------------------------------------- #
# Exo libraries

add_exo_library(
  NAME sgemm
  SOURCES sgemm.py
)

# ---------------------------------------------------------------------------- #

add_executable(test naive_sgemm.cpp test.cpp)
target_link_libraries(test PRIVATE aarch64_sgemm::sgemm)
target_compile_features(test PRIVATE cxx_std_17)
