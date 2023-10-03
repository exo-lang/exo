#include "blur.h"



#include <stdio.h>
#include <stdlib.h>


// consumer(
//     n : size,
//     f : ui8[n + 1] @DRAM,
//     g : ui8[n] @DRAM
// )
static void consumer( void *ctxt, int_fast32_t n, const uint8_t* f, uint8_t* g );

// producer(
//     n : size,
//     f : ui8[n + 1] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
static void producer( void *ctxt, int_fast32_t n, uint8_t* f, const uint8_t* inp );

// blur_staged(
//     n : size,
//     g : ui8[n] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
void blur_staged( void *ctxt, int_fast32_t n, uint8_t* g, const uint8_t* inp ) {
uint8_t *f = (uint8_t*) malloc((n + 1) * sizeof(*f));
producer(ctxt,n,f,inp);
consumer(ctxt,n,f,g);
free(f);
}

// consumer(
//     n : size,
//     f : ui8[n + 1] @DRAM,
//     g : ui8[n] @DRAM
// )
static void consumer( void *ctxt, int_fast32_t n, const uint8_t* f, uint8_t* g ) {
for (int_fast32_t i = 0; i < n; i++) {
  g[i] = (f[i] + f[i + 1]) / 2.0;
}
}

// producer(
//     n : size,
//     f : ui8[n + 1] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
static void producer( void *ctxt, int_fast32_t n, uint8_t* f, const uint8_t* inp ) {
for (int_fast32_t i = 0; i < n + 1; i++) {
  f[i] = (inp[i] + inp[i + 1] + inp[i + 2] + inp[i + 3] + inp[i + 4] + inp[i + 5]) / 6.0;
}
}

