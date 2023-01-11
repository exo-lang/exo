cmake_minimum_required(VERSION 3.21)

if (NOT CMAKE_FIND_PACKAGE_NAME STREQUAL "Exo")
  message(AUTHOR_WARNING "Found Exo using non-standard name '${CMAKE_FIND_PACKAGE_NAME}'")
endif ()

include(CMakeFindDependencyMacro)
find_dependency(Python 3.9)

find_program(
  Exo_EXECUTABLE exocc
  HINTS
  "${Python_ROOT_DIR}/bin"
  "${Python_ROOT}/bin"
)
mark_as_advanced(Exo_EXECUTABLE)

if (NOT Exo_EXECUTABLE)
  set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "Could not find exocc!")
  set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  return()
endif ()

if (NOT TARGET Exo::compiler)
  add_executable(Exo::compiler IMPORTED)
  set_target_properties(Exo::compiler PROPERTIES IMPORTED_LOCATION "${Exo_EXECUTABLE}")
endif ()

include("${CMAKE_CURRENT_LIST_DIR}/AddExoLibrary.cmake")

foreach (comp IN LISTS Exo_FIND_COMPONENTS)
  if (NOT Exo_${comp}_FOUND AND Exo_FIND_REQUIRED_${comp})
    set(Exo_FOUND FALSE)
  endif ()
endforeach ()
