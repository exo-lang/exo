# ---------------------------------------------------------------------------- #
# Shim library to force upstream tests to use exo pipelines

## Configure header shims
configure_file(shims/include/gemmini.h.in include/gemmini.h @ONLY)
configure_file(shims/include/gemmini_testutils.h.in include/gemmini_testutils.h @ONLY)
configure_file(shims/include/gemmini_params.h.in include/gemmini_params.h @ONLY)
configure_file(shims/include/gemmini_counter.h.in include/gemmini_counter.h @ONLY)
configure_file(shims/rocc-software/src/xcustom.h.in rocc-software/src/xcustom.h @ONLY)

## Create wrapper library
add_library(exo-gemmini_shim shims/gemmini.c)
add_library(exo-gemmini::shim ALIAS exo-gemmini_shim)
target_include_directories(exo-gemmini_shim PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(exo-gemmini_shim PRIVATE exo-gemmini::gemmini_lib)

# ---------------------------------------------------------------------------- #
# Matmul tests

function(test_tiled_matmul_ws suffix i j k)
    add_executable(
        "tiled_matmul_ws${suffix}"
        "${gemmini-rocc-tests_SOURCE_DIR}/bareMetalC/tiled_matmul_ws.c"
    )
    target_compile_definitions(
        "tiled_matmul_ws${suffix}"
        PRIVATE
        "MAT_DIM_I=${i}"
        "MAT_DIM_J=${j}"
        "MAT_DIM_K=${k}"
    )
    target_link_libraries("tiled_matmul_ws${suffix}" PRIVATE exo-gemmini::shim exo-gemmini::rt)
    add_test(
        NAME "tiled_matmul_ws${suffix}"
        COMMAND "tiled_matmul_ws${suffix}"
    )
endfunction()

test_tiled_matmul_ws(""   512   512  512)
test_tiled_matmul_ws("-2" 12544 64   256)
test_tiled_matmul_ws("-3" 12544 256  64 )
test_tiled_matmul_ws("-4" 3136  512  128)
test_tiled_matmul_ws("-5" 3136  128  512)
test_tiled_matmul_ws("-6" 784   1024 256)

# ---------------------------------------------------------------------------- #
# Conv tests

function(test_conv suffix IN_DIM IN_CHANNELS OUT_CHANNELS)
    add_executable(
        "conv${suffix}"
        "${gemmini-rocc-tests_SOURCE_DIR}/bareMetalC/conv.c"
    )
    target_compile_definitions(
        "conv${suffix}"
        PRIVATE
        "IN_DIM=${IN_DIM}"
        "IN_CHANNELS=${IN_CHANNELS}"
        "OUT_CHANNELS=${OUT_CHANNELS}"
        "BATCH_SIZE=4"
        "PADDING=0"
        "STRIDE=1"
    )
    target_link_libraries("conv${suffix}" PRIVATE exo-gemmini::shim exo-gemmini::rt)
    add_test(
        NAME "conv${suffix}"
        COMMAND "conv${suffix}"
    )
endfunction()

test_conv(""   58 64  64 )
test_conv("-2" 30 128 128)
test_conv("-3" 16 256 256)
