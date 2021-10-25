cmake_minimum_required(VERSION 3.21)
project(x86_demo LANGUAGES C CXX)

if (PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif ()

find_package(Python3 REQUIRED)
find_package(SYS_ATL REQUIRED HINTS "${Python3_SITELIB}")

add_sys_atl_library(sgemm sgemm.py)

add_executable(run_sgemm run_sgemm.cpp)
target_link_libraries(run_sgemm PRIVATE x86_demo::sgemm)

if (BUILD_TESTING)
  add_test(NAME sgemm_100
           COMMAND run_sgemm 100)
endif ()
