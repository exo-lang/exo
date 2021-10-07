// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define N 8

#if (N*DIM) > (BANK_NUM*BANK_ROWS)
#error not enough scratchpad space
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  gemmini_flush(0);

  // Set counter and reset
  counter_configure(0, LOAD_ACTIVE_CYCLE, false);
  if(counter_read(0) != 0) {
    printf("Counter Reset Failed (not equal to 0)\n");
    exit(1);
  }

  // Initial matrix
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  static elem_t In[N][DIM][DIM] row_align(1);
  static elem_t Out[N][DIM][DIM] row_align(1);

  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j)
        In[n][i][j] = i*DIM + j + n;

  // Move in
  for (size_t i = 0; i < N; i++) {
    gemmini_mvin(In[i], i*DIM);
    gemmini_mvout(Out[i], i*DIM);
  }

  // Check value (should be increasing right now as Gemmini executes in the background)
  int counter_val = counter_read(0);
  // Take a snapshot
  counter_snapshot_take();
  int snapshot_val = counter_read(0);

  // Print first counter value (Syscall takes a lot of time, and we might not capture a snapshot when
  // the instructions are still being executed)
  printf("Read DMA cycles: %d\n", counter_val);
  if (counter_val == 0) {
    printf("Counter Value failed to increase\n");
    exit(1);
  } 

  // Wait till the operation finish
  gemmini_fence();

  // Check again
  counter_val = counter_read(0);
  printf("Cycle when taking snapshot: %d, Cycle read after operation finished: %d\n",
    snapshot_val, counter_val);
  if (counter_val != snapshot_val) {
    printf("Snapshot changed after taken; test failed\n");
    exit(1);
  }

  // Reset snapshot, and check if cycles changed
  counter_snapshot_reset();
  counter_val = counter_read(0);
  printf("Cycles after snapshot is reset: %d\n", counter_val);
  if (counter_val < snapshot_val + 10) {
    printf("Counter values changed too little after snapshot reset; check if counter continues properly\n");
    exit(1);
  }

  // Global reset
  counter_reset();
  counter_val = counter_read(0);
  printf("Cycles after counter reset: %d\n", counter_val);
  if (counter_val != 0) {
    printf("Cycles did not reset after global reset inst\n");
    exit(1);
  }

  // Check external counter
  counter_configure(7, ROB_LD_COUNT, true);
  for (size_t i = 0; i < N; i++) {
    gemmini_mvin(In[i], i*DIM);
    gemmini_mvout(Out[i], i*DIM);
  }
  // Fused read and take snapshot command
  uint32_t custom_command = (7 & 0x7) << 4 | 0x4;
  gemmini_counter_access(counter_val, custom_command);
  printf("ROB # of load insts after executing %d mvin and mvout insts: %d\n", N-1, counter_val);
  if (counter_val < 3) {
    printf("The load ROB counter value is too small\n");
    exit(1);
  }
  snapshot_val = counter_read(7);
  if (counter_val != snapshot_val) {
    printf("Snapshot value doesn't match the raw value read before snapshot taken\n");
    exit(1);
  }

  exit(0);
}


