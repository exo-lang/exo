cmake_minimum_required(VERSION 3.21)

if (NOT CMAKE_FIND_PACKAGE_NAME STREQUAL "SYS_ATL")
  message(AUTHOR_WARNING "Found SYS_ATL using non-standard name '${CMAKE_FIND_PACKAGE_NAME}'")
endif ()

include(CMakeFindDependencyMacro)
find_dependency(Python3)

find_program(SYS_ATL_CC_EXECUTABLE sysatlcc
             HINTS "${Python3_ROOT_DIR}/bin")
mark_as_advanced(SYS_ATL_CC_EXECUTABLE)

if (NOT SYS_ATL_CC_EXECUTABLE)
    set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
        "Could not find sysatlcc!")
    set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    return()
endif ()

execute_process(
  COMMAND "${SYS_ATL_CC_EXECUTABLE}" --version
  OUTPUT_VARIABLE SYS_ATL_version_output
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

if (SYS_ATL_version_output MATCHES "version ([0-9]+\\.[0-9]+\\.[0-9]+)")
  set(SYS_ATL_VERSION "${CMAKE_MATCH_1}")
else ()
  set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
        "Could not determine version using sysatlcc!")
  set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  return()
endif ()

if (NOT TARGET SYS_ATL::compiler)
  add_executable(SYS_ATL::compiler IMPORTED)
  set_target_properties(
    SYS_ATL::compiler
    PROPERTIES
      IMPORTED_LOCATION "${SYS_ATL_CC_EXECUTABLE}"
  )
endif ()

include("${CMAKE_CURRENT_LIST_DIR}/AddSYS_ATLLibrary.cmake")

foreach (comp IN LISTS SYS_ATL_FIND_COMPONENTS)
  if (NOT SYS_ATL_${comp}_FOUND AND SYS_ATL_FIND_REQUIRED_${comp})
    set(SYS_ATL_FOUND FALSE)
  endif ()
endforeach ()
