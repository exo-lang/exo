cmake_minimum_required(VERSION 3.21)

if (NOT CMAKE_FIND_PACKAGE_NAME STREQUAL "Exo")
  message(AUTHOR_WARNING "Found Exo using non-standard name '${CMAKE_FIND_PACKAGE_NAME}'")
endif ()

include(CMakeFindDependencyMacro)
find_dependency(Python3)

find_program(Exo_CC_EXECUTABLE exocc
             HINTS "${Python3_ROOT_DIR}/bin")
mark_as_advanced(Exo_CC_EXECUTABLE)

if (NOT Exo_CC_EXECUTABLE)
    set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
        "Could not find exocc!")
    set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    return()
endif ()

execute_process(
  COMMAND "${Exo_CC_EXECUTABLE}" --version
  OUTPUT_VARIABLE Exo_version_output
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

if (Exo_version_output MATCHES "version ([0-9]+\\.[0-9]+\\.[0-9]+)")
  set(Exo_VERSION "${CMAKE_MATCH_1}")
else ()
  set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE
        "Could not determine version using exocc!")
  set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  return()
endif ()

if (NOT TARGET Exo::compiler)
  add_executable(Exo::compiler IMPORTED)
  set_target_properties(
    Exo::compiler
    PROPERTIES
      IMPORTED_LOCATION "${Exo_CC_EXECUTABLE}"
  )
endif ()

include("${CMAKE_CURRENT_LIST_DIR}/AddExoLibrary.cmake")

foreach (comp IN LISTS Exo_FIND_COMPONENTS)
  if (NOT Exo_${comp}_FOUND AND Exo_FIND_REQUIRED_${comp})
    set(Exo_FOUND FALSE)
  endif ()
endforeach ()
