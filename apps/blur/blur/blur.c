#include "blur.h"



#include <stdio.h>
#include <stdlib.h>



// blur(
//     n : size,
//     g : ui8[n] @DRAM,
//     inp : ui8[n + 6] @DRAM
// )
void blur( void *ctxt, int_fast32_t n, uint8_t* g, const uint8_t* inp ) {
uint8_t *f = (uint8_t*) malloc((n + 1) * sizeof(*f));
for (int_fast32_t i = 0; i < n + 1; i++) {
  f[i] = (inp[i] + inp[i + 1] + inp[i + 2] + inp[i + 3] + inp[i + 4] + inp[i + 5]) / 6.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  g[i] = (f[i] + f[i + 1]) / 2.0;
}
free(f);
}

