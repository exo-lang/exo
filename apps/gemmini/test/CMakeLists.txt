# ---------------------------------------------------------------------------- #
# Bare-metal C rt from riscv-tests

add_library(
    exo-gemmini_rt
    "${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/benchmarks/common/syscalls.c"
	"${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/benchmarks/common/crt.S"
)
add_library(exo-gemmini::rt ALIAS exo-gemmini_rt)
target_compile_definitions(
    exo-gemmini_rt
    PRIVATE
    PREALLOCATE=1
	MULTITHREAD=1
	ID_STRING=
)
target_link_options(
    exo-gemmini_rt
    INTERFACE
    "SHELL:-T ${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/benchmarks/common/test.ld"
)
target_include_directories(
    exo-gemmini_rt
    SYSTEM PUBLIC
    "${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests"
    "${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/env"
    "${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/benchmarks/common"
)

# ---------------------------------------------------------------------------- #
# Upstream bare-metal tests

add_subdirectory(bare-metal)

# ---------------------------------------------------------------------------- #
# Our own tests

## Helper library
add_library(exo-gemmini_helpers helpers.c helpers.h)
add_library(exo-gemmini::helpers ALIAS exo-gemmini_helpers)

function(add_gemmini_test name)
    add_executable("${name}" "${name}.c")
    target_include_directories("${name}" SYSTEM PRIVATE "${gemmini-rocc-tests_SOURCE_DIR}")
    target_link_libraries(
        "${name}"
        PRIVATE
        exo-gemmini::gemmini_lib
        exo-gemmini::rt
        exo-gemmini::helpers
    )
    add_test(NAME "${name}" COMMAND "${name}")
endfunction()

add_gemmini_test(conv_3_main)
add_gemmini_test(conv_17_main)
add_gemmini_test(conv_30_main)
add_gemmini_test(matmul_4_main)
add_gemmini_test(matmul_6_main)
add_gemmini_test(matmul_14_main)
add_gemmini_test(matmul_16_main)
add_gemmini_test(matmul_27_main)
add_gemmini_test(matmul_512x512x512_main)
