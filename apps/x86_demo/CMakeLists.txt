cmake_minimum_required(VERSION 3.21)

set(CMAKE_TOOLCHAIN_FILE
    "${CMAKE_CURRENT_LIST_DIR}/../toolchains/skylakex.cmake"
    CACHE FILEPATH "Path to toolchain file.")

project(x86_demo LANGUAGES C CXX)

##
# TODO: this is a bit ugly, but it accommodates pip -e (editable)

find_package(Python3 REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c [[
import SYS_ATL
from pathlib import Path
print(Path(SYS_ATL.__file__).parent)
]]
  OUTPUT_VARIABLE SYS_ATL_HINT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(SYS_ATL REQUIRED CONFIG HINTS "${SYS_ATL_HINT}")

##

add_sys_atl_library(sgemm sgemm.py)

add_executable(run_sgemm run_sgemm.cpp)
target_link_libraries(run_sgemm PRIVATE x86_demo::sgemm)
