function(add_exo_library target source_file)
  cmake_path(ABSOLUTE_PATH source_file
             BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
             NORMALIZE)

  set(intdir "${CMAKE_CURRENT_BINARY_DIR}/${target}.exo")
  set(files "${intdir}/${target}.c" "${intdir}/${target}.h")

  add_custom_command(
    OUTPUT ${files}
    COMMAND Exo::compiler -o "${intdir}" --stem "${target}" "${source_file}"
    DEPENDS "${source_file}"
    VERBATIM
  )
  add_library(${target} ${files})
  add_library(${PROJECT_NAME}::${target} ALIAS ${target})
  target_include_directories(${target} PUBLIC "$<BUILD_INTERFACE:${intdir}>")
endfunction()
