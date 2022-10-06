function(add_exo_library target)
  set(source_files "")

  foreach (src IN LISTS ARGN)
    cmake_path(ABSOLUTE_PATH src
               BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
               NORMALIZE)
    list(APPEND source_files "${src}")
  endforeach ()

  set(intdir "${CMAKE_CURRENT_BINARY_DIR}/${target}.exo")
  set(files "${intdir}/${target}.c" "${intdir}/${target}.h")

  add_custom_command(
    OUTPUT ${files}
    COMMAND Exo::compiler -o "${intdir}" --stem "${target}" ${source_files}
    DEPENDS ${source_files}
    VERBATIM
  )

  add_library(${target} ${files})
  add_library(${PROJECT_NAME}::${target} ALIAS ${target})
  target_include_directories(${target} PUBLIC "$<BUILD_INTERFACE:${intdir}>")
endfunction()
