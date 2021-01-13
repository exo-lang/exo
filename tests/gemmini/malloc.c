/* Gemmini runnable malloc template
 *
 * Place this file under
 * chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC
 * and run
 * $ cd chipyard/generators/gemmini/software/gemmini-rocc-tests/build
 * $ make
 * $ spike --extension=gemmini bareMetalC/malloc-baremetal 
 *
 */

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <stdint.h>
#include <string.h>

#include "include/gemmini_testutils.h"

#define HEAP_SIZE 100

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

    for(;;) {
        uint32_t size = sizeof(uint32_t) + sizeof(FreeBlock*);

        if (cur->next == 0 && cur->size < bytes + size) {
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

    return;
}

// alloc_nest( x : R[n,m] @IN, y : R[n,m] @IN, res : R[n,m] @OUT )
void alloc_nest( int n, int m, float* x, float* y, float* res) {
    float *rloc = (float*) malloc (m * sizeof(float));
    for (int i=0; i < n; i++) {
        float *xloc = (float*) malloc (m * sizeof(float));
        float *yloc = (float*) malloc (m * sizeof(float));
        for (int j=0; j < m; j++) {
            xloc[j] = x[(i) * m + (j)];
        }
        for (int j=0; j < m; j++) {
            yloc[j] = y[(i) * m + (j)];
        }
        for (int j=0; j < m; j++) {
            rloc[j] = xloc[j] + yloc[j];
        }
        for (int j=0; j < m; j++) {
            res[(i) * m + (j)] = rloc[j];
        }
        free(xloc);
        free(yloc);
    }
    free(rloc);
}

float x[] = {1.0, 2.0, 3.0, 3.2, 4.0, 5.3};
float y[] = {2.6, 3.7, 8.9, 1.3, 2.3, 6.7};

int main() {
    init_mem();

    int n_size = 2;
    int m_size = 3;
    float res[n_size][m_size];

    unsigned long cpu_start = read_cycles();
    alloc_nest(n_size, m_size, x, y, res);
    unsigned long cpu_end = read_cycles();
    printf("\nCycles taken: %u\n", cpu_end-cpu_start);

    printf("printing result..\n");
    for (int i = 0; i < n_size; i++) {
        for (int j = 0; j < m_size; j++) {
            printf("%d ", (int)res[i][j]);
        }
    }
    printf("\ndone\n");

    exit(0);
}
