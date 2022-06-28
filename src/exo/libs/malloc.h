#define HEAP_SIZE 100000

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void init_mem(void);
void *malloc_dram(long unsigned int bytes);
void free_dram(void *ptr);
