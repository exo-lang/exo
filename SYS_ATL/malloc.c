#define HEAP_SIZE {heap_size}

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
    p->next = p+1;
    p->size = 0;
    freelist = p;

    p++;
    p->next = 0;
    p->size = HEAP_SIZE - sizeof(FreeBlock);
}

void *search(FreeBlock *cur, uint32_t bytes) {
    FreeBlock *prev = freelist;
    //fprintf(stderr, "bytes = %d: \n", bytes);

    for(;;) {
        //fprintf(stderr, "cur is %p and cur->next is %p. cur->size is %d\n"
        //              ,cur, cur->next, cur->size);
        uint32_t size = sizeof(uint32_t) + sizeof(FreeBlock*);
        //fprintf(stderr, "size: %d\n", size);

        if (cur->next == 0 && cur->size < bytes + size) {
            fprintf(stderr, "Out of memory!\n");
            return 0;
        } else if (cur->next == 0 && cur->size >= (bytes + size)) {
            // cut cur into bytes blocks and create new
            uint32_t sz = cur->size;
            FreeBlock *new = (void *)cur + bytes + size;
            new->next = 0;
            new->size = sz - bytes - size;
            cur->size = bytes + size;
            //fprintf(stderr, "newsize: %d\n", new->size);
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
    //fprintf(stderr,"\n");

    return cur->data;
}

void *malloc(long unsigned int bytes) {
    bytes = bytes < sizeof(FreeBlock) ? sizeof(FreeBlock) : bytes;
    if (bytes == 0) return 0;
    FreeBlock *loc = search(freelist, bytes);
    if (loc == 0)
        return 0;
    else {
        return loc;
    }
}

void free(void *ptr) {
    if (ptr == 0) return;
    ptr = ptr -  sizeof(uint32_t) - sizeof(FreeBlock*);
    FreeBlock *next = freelist->next;
    freelist->next = ptr;
    ((FreeBlock*)ptr)->next = next;
    //int s = (int)((FreeBlock*)ptr)->size;
    //fprintf(stderr,"free ptr->size is: %d\n", s);

    return;
}

/* test free
#include <stdlib.h>
int main(void) {
  fprintf(stderr, "calling init_mem\n");
  init_mem();

  srand(100);

  uint8_t *p[1000];
  uint8_t sz[1000];
  size_t idx = 0;

  uint8_t buf[0x200];

  for(int i=0; i<1000; i++) {
    if(idx == 0 || (idx < 1000 && rand() % 2 == 0)) {
      // malloc
      sz[idx] = rand() % 0x100;
      p[idx] = malloc(sz[idx]);
      if(sz[idx] > 0 && !p[idx]) {
        fprintf(stderr, "malloc returned null\n");
        assert(0);
      }
      memset(p[idx], sz[idx], sz[idx]);
      idx++;
    } else {
      // free
      size_t victim = rand() % idx;
      memset(buf, sz[victim], sz[victim]);
      if(memcmp(p[victim], buf, sz[victim])) {
        fprintf(stderr, "memcmp failed\n");
        assert(0);
      }
      free(p[victim]);
      idx--;
      p[victim] = p[idx];
      sz[victim] = sz[idx];
    }
  }
}
*/

/* malloc test
int main(void) {
  fprintf(stderr, "calling init_mem\n");
  init_mem();
  char *p[100];
  fprintf(stderr, "HEAP: %p\n", HEAP);
  for(int i=0; i<100; i++) {
    fprintf(stderr, "i = %d\n", i);
    p[i] = malloc(i);
    fprintf(stderr, "addr: %p\n", p[i]);
    if(p[i] == 0 && i != 0) {
      fprintf(stderr, "malloc(%d): NULL returned\n", i);
      assert(0);
    }
      memset(p[i], i, i);
  }

  char buf[101] = {};
  for(int i=0; i<100; i++) {
    memset(buf, i, i);
    if(memcmp(buf, p[i], i)) {
        fprintf(stderr, "cmp(%d) failed\n", i);
        fprintf(stderr, "buf is 0x%02x at %p\n", buf[0], buf);
        fprintf(stderr, "p[i] is 0x%02x at %p\n", p[i][0], p[i]);
        assert(0);
    }
  }

  return 0;
}
*/
