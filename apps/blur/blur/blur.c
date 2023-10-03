#include "blur.h"

#include <stdio.h>
#include <stdlib.h>

// consumer(
//     n : size,
//     f : ui8[n + 1] @DRAM,
//     g : ui8[n] @DRAM
// )
static void consumer(void *ctxt, int_fast32_t n, const uint8_t *f, uint8_t *g);

// producer(
//     n : size,
//     f : ui8[n + 1] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
static void producer(
    void *ctxt, int_fast32_t n, uint8_t *f, const uint8_t *inp);

// blur_compute_at_store_at(
//     n : size,
//     g : ui8[n] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
void blur_compute_at_store_at(
    void *ctxt, int_fast32_t n, uint8_t *g, const uint8_t *inp) {
  for (int_fast32_t i = 0; i < n; i++) {
    uint8_t *f_tmp = (uint8_t *)malloc(2 * sizeof(*f_tmp));
    f_tmp[0] = (inp[i] + inp[1 + i] + inp[2 + i] + inp[3 + i] + inp[4 + i] +
                   inp[5 + i]) /
               6.0;
    f_tmp[1] = (inp[1 + i] + inp[2 + i] + inp[3 + i] + inp[4 + i] + inp[5 + i] +
                   inp[6 + i]) /
               6.0;
    g[i] = (f_tmp[0] + f_tmp[1]) / 2.0;
    free(f_tmp);
  }
}

// blur_compute_at_store_root(
//     n : size,
//     g : ui8[n] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
void blur_compute_at_store_root(
    void *ctxt, int_fast32_t n, uint8_t *g, const uint8_t *inp) {
  uint8_t *f = (uint8_t *)malloc((1 + n) * sizeof(*f));
  for (int_fast32_t i = 0; i < n; i++) {
    f[i] = (inp[i] + inp[1 + i] + inp[2 + i] + inp[3 + i] + inp[4 + i] +
               inp[5 + i]) /
           6.0;
    f[1 + i] = (inp[1 + i] + inp[2 + i] + inp[3 + i] + inp[4 + i] + inp[5 + i] +
                   inp[6 + i]) /
               6.0;
    g[i] = (f[i] + f[1 + i]) / 2.0;
  }
  free(f);
}

// blur_inline(
//     n : size,
//     g : ui8[n] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
void blur_inline(void *ctxt, int_fast32_t n, uint8_t *g, const uint8_t *inp) {
  for (int_fast32_t i = 0; i < n; i++) {
    g[i] = ((inp[i] + inp[1 + i] + inp[2 + i] + inp[3 + i] + inp[4 + i] +
                inp[5 + i]) /
                   6.0 +
               (inp[1 + i] + inp[2 + i] + inp[3 + i] + inp[4 + i] + inp[5 + i] +
                   inp[6 + i]) /
                   6.0) /
           2.0;
  }
}

// blur_staged(
//     n : size,
//     g : ui8[n] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
void blur_staged(void *ctxt, int_fast32_t n, uint8_t *g, const uint8_t *inp) {
  uint8_t *f = (uint8_t *)malloc((n + 1) * sizeof(*f));
  producer(ctxt, n, f, inp);
  consumer(ctxt, n, f, g);
  free(f);
}

// consumer(
//     n : size,
//     f : ui8[n + 1] @DRAM,
//     g : ui8[n] @DRAM
// )
static void consumer(void *ctxt, int_fast32_t n, const uint8_t *f, uint8_t *g) {
  for (int_fast32_t i = 0; i < n; i++) {
    g[i] = (f[i] + f[i + 1]) / 2.0;
  }
}

// producer(
//     n : size,
//     f : ui8[n + 1] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
static void producer(
    void *ctxt, int_fast32_t n, uint8_t *f, const uint8_t *inp) {
  for (int_fast32_t i = 0; i < n + 1; i++) {
    f[i] = (inp[i] + inp[i + 1] + inp[i + 2] + inp[i + 3] + inp[i + 4] +
               inp[i + 5]) /
           6.0;
  }
}
