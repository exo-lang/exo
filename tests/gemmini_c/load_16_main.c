// riscv64-unknown-elf-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc -I/home/yuka/chipyard/generators/gemmini/software/gemmini-rocc-tests/riscv-tests -I/home/yuka/chipyard/generators/gemmini/software/gemmini-rocc-tests/riscv-tests/env -I/home/yuka/chipyard/generators/gemmini/software/gemmini-rocc-tests -I/home/yuka/chipyard/generators/gemmini/software/gemmini-rocc-tests/riscv-tests/benchmarks/common -DID_STRING=  -nostdlib -nostartfiles -static -T /home/yuka/chipyard/generators/gemmini/software/gemmini-rocc-tests/riscv-tests/benchmarks/common/test.ld -DBAREMETAL=1  /home/yuka/SYS_ATL/tests/main.c /home/yuka/SYS_ATL/tests/tmp/test_load_16.c -o main /home/yuka/chipyard/generators/gemmini/software/gemmini-rocc-tests/riscv-tests/benchmarks/common/crt.S /home/yuka/chipyard/generators/gemmini/software/gemmini-rocc-tests/riscv-tests/benchmarks/common/syscalls.c
//
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
//#include <time.h>

#include "include/gemmini_testutils.h"
#include "../tmp/test_load_16.h"

int main() {
    gemmini_flush(0);

    float x[16*16];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            x[16*i + j] = (float)1.0*i*j;
        }
    }
    float *y = (float*) 0;

    ld_16(x, y);

    printf("\nDone\n");

    exit(0);
}
