cmake_minimum_required(VERSION 3.21)
project(apps)

if (NOT PROJECT_IS_TOP_LEVEL)
  message(FATAL_ERROR "This project is not meant to be consumed"
                      "by add_subdirectory, or FetchContent.")
endif ()

include(CTest)

add_library(util INTERFACE util/util.hpp)
add_library(apps::util ALIAS util)

target_include_directories(
  util INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/util>")

add_subdirectory(x86_demo)
