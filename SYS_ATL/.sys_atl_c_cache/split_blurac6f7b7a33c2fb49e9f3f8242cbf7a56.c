int _floor_div(int num, int quot) {
  int off = (num<0)? quot-1 : 0;
  return (num-off)/quot;
}


// split_blur( image : R[N,M] @IN, kernel : R[K,K] @IN, res : R[N,M] @OUT )
void split_blur( int N, int M, int K, float* image, float* kernel, float* res) {
for (int xhi=0; xhi < _floor_div(N, 16); xhi++) {
  for (int xlo=0; xlo < 16; xlo++) {
    if (16 * xhi + xlo < N) {
      for (int yhi=0; yhi < _floor_div(M, 16); yhi++) {
        for (int ylo=0; ylo < 16; ylo++) {
          if (16 * yhi + ylo < M) {
            res[(16 * xhi + xlo) * M + (16 * yhi + ylo)] = 0.0;
          }
        }
      }
    }
  }
}
for (int xhi=0; xhi < _floor_div(N, 16); xhi++) {
  for (int xlo=0; xlo < 16; xlo++) {
    if (16 * xhi + xlo < N) {
      for (int yhi=0; yhi < _floor_div(M, 16); yhi++) {
        for (int ylo=0; ylo < 16; ylo++) {
          if (16 * yhi + ylo < M) {
            for (int i=0; i < K; i++) {
              for (int j=0; j < K; j++) {
                if (16 * xhi + xlo + i < N && 16 * yhi + ylo + j < M) {
                  res[(16 * xhi + xlo) * M + (16 * yhi + ylo)] += kernel[(i) * K + (j)] * image[(16 * xhi + xlo + i) * M + (16 * yhi + ylo + j)];
                }
              }
            }
          }
        }
      }
    }
  }
}
}
