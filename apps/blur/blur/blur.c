#include "blur.h"

#include <stdio.h>
#include <stdlib.h>

// consumer(
//     n : size,
//     m : size,
//     f : ui8[n + 4, m + 4] @DRAM,
//     g : ui8[n + 4, m + 4] @DRAM
// )
static void consumer(
    void *ctxt, int_fast32_t n, int_fast32_t m, const uint8_t *f, uint8_t *g);

// producer(
//     n : size,
//     m : size,
//     f : ui8[n + 4, m + 4] @DRAM,
//     inp : ui8[n + 4, m + 4] @DRAM
// )
static void producer(
    void *ctxt, int_fast32_t n, int_fast32_t m, uint8_t *f, const uint8_t *inp);

// blur_inline(
//     n : size,
//     m : size,
//     g : ui8[n + 4, m + 4] @DRAM,
//     inp : ui8[n + 4, m + 4] @DRAM
// )
void blur_inline(void *ctxt, int_fast32_t n, int_fast32_t m, uint8_t *g,
    const uint8_t *inp) {
  EXO_ASSUME(n % 128 == 0);
  EXO_ASSUME(m % 256 == 0);
  for (int_fast32_t i = 0; i < n; i++) {
    for (int_fast32_t j = 0; j < m; j++) {
      g[i * (m + 4) + j] =
          ((inp[i * (m + 4) + j] + inp[i * (m + 4) + 1 + j] +
               inp[i * (m + 4) + 2 + j] + inp[i * (m + 4) + 3 + j] +
               inp[i * (m + 4) + 4 + j]) /
                  5.0 +
              (inp[(i + 1) * (m + 4) + j] + inp[(i + 1) * (m + 4) + 1 + j] +
                  inp[(i + 1) * (m + 4) + 2 + j] +
                  inp[(i + 1) * (m + 4) + 3 + j] +
                  inp[(i + 1) * (m + 4) + 4 + j]) /
                  5.0 +
              (inp[(i + 2) * (m + 4) + j] + inp[(i + 2) * (m + 4) + 1 + j] +
                  inp[(i + 2) * (m + 4) + 2 + j] +
                  inp[(i + 2) * (m + 4) + 3 + j] +
                  inp[(i + 2) * (m + 4) + 4 + j]) /
                  5.0 +
              (inp[(i + 3) * (m + 4) + j] + inp[(i + 3) * (m + 4) + 1 + j] +
                  inp[(i + 3) * (m + 4) + 2 + j] +
                  inp[(i + 3) * (m + 4) + 3 + j] +
                  inp[(i + 3) * (m + 4) + 4 + j]) /
                  5.0 +
              (inp[(i + 4) * (m + 4) + j] + inp[(i + 4) * (m + 4) + 1 + j] +
                  inp[(i + 4) * (m + 4) + 2 + j] +
                  inp[(i + 4) * (m + 4) + 3 + j] +
                  inp[(i + 4) * (m + 4) + 4 + j]) /
                  5.0) /
          5.0;
    }
  }
}

// blur_staged(
//     n : size,
//     m : size,
//     g : ui8[n + 4, m + 4] @DRAM,
//     inp : ui8[n + 4, m + 4] @DRAM
// )
void blur_staged(void *ctxt, int_fast32_t n, int_fast32_t m, uint8_t *g,
    const uint8_t *inp) {
  EXO_ASSUME(n % 128 == 0);
  EXO_ASSUME(m % 256 == 0);
  uint8_t *f = (uint8_t *)malloc((n + 4) * (m + 4) * sizeof(*f));
  producer(ctxt, n, m, f, inp);
  consumer(ctxt, n, m, f, g);
  free(f);
}

// blur_tiled(
//     n : size,
//     m : size,
//     g : ui8[n + 4, m + 4] @DRAM,
//     inp : ui8[n + 4, m + 4] @DRAM
// )
void blur_tiled(void *ctxt, int_fast32_t n, int_fast32_t m, uint8_t *g,
    const uint8_t *inp) {
  EXO_ASSUME(n % 128 == 0);
  EXO_ASSUME(m % 256 == 0);
  uint8_t *f = (uint8_t *)malloc(132 * 256 * sizeof(*f));
  for (int_fast32_t io = 0; io < ((n) / (128)); io++) {
    for (int_fast32_t jo = 0; jo < ((m) / (256)); jo++) {
      for (int_fast32_t ii = 0; ii < 132; ii++) {
        for (int_fast32_t ji = 0; ji < 256; ji++) {
          f[ii * 256 + ji] =
              (inp[(ii + 128 * io) * (m + 4) + ji + 256 * jo] +
                  inp[(ii + 128 * io) * (m + 4) + 1 + ji + 256 * jo] +
                  inp[(ii + 128 * io) * (m + 4) + 2 + ji + 256 * jo] +
                  inp[(ii + 128 * io) * (m + 4) + 3 + ji + 256 * jo] +
                  inp[(ii + 128 * io) * (m + 4) + 4 + ji + 256 * jo]) /
              5.0;
        }
      }
      for (int_fast32_t ii = 0; ii < 128; ii++) {
        for (int_fast32_t ji = 0; ji < 256; ji++) {
          g[(ii + 128 * io) * (m + 4) + ji + 256 * jo] =
              (f[ii * 256 + ji] + f[(1 + ii) * 256 + ji] +
                  f[(2 + ii) * 256 + ji] + f[(3 + ii) * 256 + ji] +
                  f[(4 + ii) * 256 + ji]) /
              5.0;
        }
      }
    }
  }
  free(f);
}

// consumer(
//     n : size,
//     m : size,
//     f : ui8[n + 4, m + 4] @DRAM,
//     g : ui8[n + 4, m + 4] @DRAM
// )
static void consumer(
    void *ctxt, int_fast32_t n, int_fast32_t m, const uint8_t *f, uint8_t *g) {
  for (int_fast32_t i = 0; i < n; i++) {
    for (int_fast32_t j = 0; j < m; j++) {
      g[i * (m + 4) + j] =
          (f[i * (m + 4) + j] + f[(i + 1) * (m + 4) + j] +
              f[(i + 2) * (m + 4) + j] + f[(i + 3) * (m + 4) + j] +
              f[(i + 4) * (m + 4) + j]) /
          5.0;
    }
  }
}

// producer(
//     n : size,
//     m : size,
//     f : ui8[n + 4, m + 4] @DRAM,
//     inp : ui8[n + 4, m + 4] @DRAM
// )
static void producer(void *ctxt, int_fast32_t n, int_fast32_t m, uint8_t *f,
    const uint8_t *inp) {
  for (int_fast32_t i = 0; i < n + 4; i++) {
    for (int_fast32_t j = 0; j < m; j++) {
      f[i * (m + 4) + j] =
          (inp[i * (m + 4) + j] + inp[i * (m + 4) + j + 1] +
              inp[i * (m + 4) + j + 2] + inp[i * (m + 4) + j + 3] +
              inp[i * (m + 4) + j + 4]) /
          5.0;
    }
  }
}
