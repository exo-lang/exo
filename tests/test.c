#include "test.h"

// add_vec( x : [n]R IN, y : [n]R IN, res : [n]R OUT )
void add_vec(size_t n, float* x, float* y, float* res) {
for (int i_5=0; i_5 < n; i_5++) {
res[i_5] = x[i_5] + y[i_5];
}
}

