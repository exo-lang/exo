//#define HEAP_SIZE {heap_size}
//#define DIM {dim}
#define HEAP_SIZE 1000
#define DIM 16

#include<stdio.h>
#include<stdint.h>
#include<assert.h>

uint32_t _ceil(float num) {
    int inum = (int)num;
    if (num == (float)inum) {
        return inum;
    }
    return inum + 1;
}

// Get enough HEAP for FreeBlock
uint8_t HEAP[HEAP_SIZE];

// https://stackoverflow.com/questions/5473189/what-is-a-packed-structure-in-c
typedef struct __attribute__((__packed__)) FreeBlock {
    uint32_t size;
    struct FreeBlock *next;
    uint32_t loc;
} FreeBlock;
FreeBlock *freelist;

void init_mem() {
    FreeBlock *p = (FreeBlock *)HEAP;
    p->next = p+1;
    p->size = 0;
    freelist = p;

    p++;
    p->next = 0;
    p->size = (HEAP_SIZE/DIM);
}

void *search(FreeBlock *cur, uint32_t bytes) {
    FreeBlock *prev = freelist;

    for(;;) {
        if (cur->next == 0 && cur->size < bytes) {
            fprintf(stderr, "Out of memory!\n");
            return 0;
        } else if (cur->next == 0 && cur->size >= bytes) {
            // cut cur into bytes blocks and create new
            uint32_t sz = cur->size;
            FreeBlock *new = (void *)cur + sizeof(FreeBlock);
            new->next = 0;
            new->size = sz - bytes;
            new->loc  = cur->loc + bytes;
            cur->size = bytes;
            prev->next = new;
            break;
        } else if (cur->size >= bytes) {
            prev->next = cur->next;
            break;
        } else {
            prev = cur;
            cur = cur->next;
        }
    }

    return cur->loc;
}

uint32_t malloc_gemm(long unsigned int bytes) {
    assert(bytes != 0);
    uint32_t b = _ceil(((float)bytes)/(float)DIM);
    uint32_t loc = search(freelist, b);
    
    return loc;
}

// TODO: How to free?? We don't have address.
// Keep track of "not-free-list"??
void *search_free(uint32_t ptr) {
    FreeBlock *cur = freelist;

    for(;;) {
        printf("cur->loc: %d\n", cur->loc);
        if (cur->next == 0 && cur->loc != ptr) {
            fprintf(stderr, "No such ptr!\n");
            return 0;
        } else if (cur->loc == ptr) {
            return cur;
        } else {
            cur = cur->next;
        }
    }
    return 0;
}

void free_gemm(uint32_t ptr) {
    FreeBlock* p = search_free(ptr);
    FreeBlock *next = freelist->next;
    freelist->next = p;
    ((FreeBlock*)p)->next = next;

    return;
}

int main(void) {
  fprintf(stderr, "calling init_mem\n");
  init_mem();
  uint32_t zero = malloc_gemm(10);
  fprintf(stderr, "zero: %d\n", zero);
  uint32_t one = malloc_gemm(20);
  fprintf(stderr, "one: %d\n", one);
  uint32_t three = malloc_gemm(40);
  fprintf(stderr, "three: %d\n", three);
  uint32_t six = malloc_gemm(100);
  fprintf(stderr, "six: %d\n", six);
  uint32_t _13 = malloc_gemm(200);
  fprintf(stderr, "_13: %d\n", _13);

  free_gemm(one);
  uint32_t one2 = malloc_gemm(20);
  fprintf(stderr, "one2: %d\n", one2);

  return 0;
}
