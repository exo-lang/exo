cmake_minimum_required(VERSION 3.21)
project(apps)

if (PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif ()

add_subdirectory(x86)
