#include "custom_malloc.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef HEAP_SIZE
#define HEAP_SIZE 100000
#endif

uint8_t HEAP[HEAP_SIZE];

// https://stackoverflow.com/questions/5473189/what-is-a-packed-structure-in-c
typedef struct __attribute__((__packed__)) FreeBlock {
  uint32_t size;
  struct FreeBlock *next;
  uint8_t data[0];
} FreeBlock;
FreeBlock *freelist;

void init_mem() {
  FreeBlock *p = (FreeBlock *)HEAP;
  p->next = p + 1;
  p->size = 0;
  freelist = p;

  p++;
  p->next = 0;
  p->size = HEAP_SIZE - sizeof(FreeBlock);
}

void *search(FreeBlock *cur, uint32_t bytes) {
  FreeBlock *prev = freelist;

  for (;;) {
    uint32_t size = sizeof(uint32_t) + sizeof(FreeBlock *);

    if (cur->next == 0 && cur->size < bytes + size) {
      printf("Out of memory!\n");
      return 0;
    } else if (cur->next == 0 && cur->size >= (bytes + size)) {
      // cut cur into bytes blocks and create new
      uint32_t sz = cur->size;
      FreeBlock *new = (void *)cur + bytes + size;
      new->next = 0;
      new->size = sz - bytes - size;
      cur->size = bytes + size;
      prev->next = new;
      break;
    } else if (cur->size >= (bytes + size)) {
      prev->next = cur->next;
      break;
    } else {
      prev = cur;
      cur = cur->next;
    }
  }

  return cur->data;
}

void *malloc_dram(long unsigned int bytes) {
  bytes = bytes < sizeof(FreeBlock) ? sizeof(FreeBlock) : bytes;
  if (bytes == 0)
    return 0;
  FreeBlock *loc = search(freelist, bytes);
  if (loc == 0)
    return 0;
  else {
    return loc;
  }
}

void free_dram(void *ptr) {
  if (ptr == 0)
    return;
  ptr = ptr - sizeof(uint32_t) - sizeof(FreeBlock *);
  FreeBlock *next = freelist->next;
  freelist->next = ptr;
  ((FreeBlock *)ptr)->next = next;

  return;
}
