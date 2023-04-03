function(add_exo_library)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "NAME" "SOURCES;PYTHONPATH")

  set(source_files "")

  foreach (src IN LISTS ARG_SOURCES)
    cmake_path(ABSOLUTE_PATH src
               BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
               NORMALIZE)
    list(APPEND source_files "${src}")
  endforeach ()

  set(intdir "${ARG_NAME}.exo")
  set(files "${intdir}/${ARG_NAME}.c" "${intdir}/${ARG_NAME}.h")

  list(TRANSFORM ARG_PYTHONPATH PREPEND "--modify;PYTHONPATH=path_list_append:")

  add_custom_command(
    OUTPUT ${files}
    COMMAND ${CMAKE_COMMAND} -E env ${ARG_PYTHONPATH} --
            $<TARGET_FILE:Exo::compiler> -o "${intdir}" --stem "${ARG_NAME}"
            ${source_files}
    DEPENDS ${source_files}
    DEPFILE "${intdir}/${ARG_NAME}.d"
    VERBATIM
  )

  list(TRANSFORM files PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")
  add_library(${ARG_NAME} ${files})
  add_library(${PROJECT_NAME}::${ARG_NAME} ALIAS ${ARG_NAME})
  target_include_directories(${ARG_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${intdir}>")
endfunction()
