#ifndef CUSTOM_MALLOC_H
#define CUSTOM_MALLOC_H
void init_mem(void);
void *malloc_dram(long unsigned int bytes);
void free_dram(void *ptr);
#endif
