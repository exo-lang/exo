uint32_t  HEAP_SIZE = {heap_size};
uint8_t   HEAP[HEAP_SIZE];

struct FreeBlock {
  FreeBlock *next;
  uint32_t size;
};
FreeBlock *freelist;

void init_mem() {
  freelist = HEAP;
  freelist->next = 0;
  freelist->size = HEAP_SIZE;
}

void *malloc(uint32_t bytes) {
  // todo
  // return 0 on failure
}

void free(void *ptr) {

}

malloc dram_all

Alloc dram : DRAM
blah

Alloc spmem[8] : GEMM_SCRATCHPAD
for i:
  load to spmem[i]
  multiply
  write back to dram

Alloc dram2 : DRAM
blah
Alloc dram3 : DRAM
blah

blah
