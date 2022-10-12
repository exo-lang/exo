# ---------------------------------------------------------------------------- #
# Bare-metal C runtime from riscv-tests

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

function(add_gemmini_test name source)
    add_executable("${name}" "${source}")
    target_include_directories("${name}" SYSTEM PRIVATE "${gemmini-rocc-tests_SOURCE_DIR}")
    target_compile_definitions("${name}" PRIVATE "KERNEL_FN=${name}" ${ARGN})
    target_link_libraries(
        "${name}"
        PRIVATE
        exo-gemmini::gemmini_lib
        exo-gemmini::rt
        exo-gemmini::helpers
    )
    add_test(NAME "${name}" COMMAND "${name}")
endfunction()

add_gemmini_test(
    conv_3 run_conv.c
    BATCH_SIZE=4
    KERNEL_DIM=3
    IN_CHANNEL=64
    IN_DIM=58
    OUT_CHANNEL=64
    OUT_DIM=56
)

add_gemmini_test(
    conv_17 run_conv.c
    BATCH_SIZE=4
    KERNEL_DIM=3
    IN_CHANNEL=128
    IN_DIM=30
    OUT_CHANNEL=128
    OUT_DIM=28
)

add_gemmini_test(
    conv_30 run_conv.c
    BATCH_SIZE=4
    KERNEL_DIM=3
    IN_CHANNEL=256
    IN_DIM=16
    OUT_CHANNEL=256
    OUT_DIM=14
)

add_gemmini_test(matmul_4  run_matmul.c NN=12544 KK=64   MM=256)
add_gemmini_test(matmul_6  run_matmul.c NN=12544 KK=256  MM=64 )
add_gemmini_test(matmul_14 run_matmul.c NN=3136  KK=128  MM=512)
add_gemmini_test(matmul_16 run_matmul.c NN=3136  KK=512  MM=128)
add_gemmini_test(matmul_27 run_matmul.c NN=784   KK=1024 MM=256)
add_gemmini_test(matmul_512x512x512 run_matmul.c NN=512   KK=512  MM=512)
