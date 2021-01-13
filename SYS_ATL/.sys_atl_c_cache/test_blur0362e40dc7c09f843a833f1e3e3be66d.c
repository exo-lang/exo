int _ceil_div(int num, int quot) {
  int off = (num>0)? quot-1 : 0;
  return (num+off)/quot;
}


// test_blur( image : R[N,M] @IN, kernel : R[K,K] @IN, res : R[N,M] @OUT )
void test_blur( int N, int M, int K, float* image, float* kernel, float* res) {
for (int x=0; x < N; x++) {
  for (int y=0; y < M; y++) {
    res[(x) * M + (y)] = 0.0;
  }
}
for (int xhi=0; xhi < _ceil_div(N - 64 + 1, 64); xhi++) {
  for (int yhi=0; yhi < _ceil_div(M - 64 + 1, 64); yhi++) {
    for (int i=0; i < K; i++) {
      for (int j=0; j < K; j++) {
        for (int xlo=0; xlo < 64; xlo++) {
          for (int ylo=0; ylo < 64; ylo++) {
            if (64 * xhi + xlo + i < N && 64 * yhi + ylo + j < M) {
              res[(64 * xhi + xlo) * M + (64 * yhi + ylo)] += kernel[(i) * K + (j)] * image[(64 * xhi + xlo + i) * M + (64 * yhi + ylo + j)];
            }
          }
        }
      }
    }
  }
  for (int xlo=0; xlo < 64; xlo++) {
    for (int ylo=0; ylo < M - _ceil_div(M - 64 + 1, 64) * 64; ylo++) {
      for (int i=0; i < K; i++) {
        for (int j=0; j < K; j++) {
          if (64 * xhi + xlo + i < N && ylo + _ceil_div(M - 64 + 1, 64) * 64 + j < M) {
            res[(64 * xhi + xlo) * M + (ylo + _ceil_div(M - 64 + 1, 64) * 64)] += kernel[(i) * K + (j)] * image[(64 * xhi + xlo + i) * M + (ylo + _ceil_div(M - 64 + 1, 64) * 64 + j)];
          }
        }
      }
    }
  }
}
for (int xlo=0; xlo < N - _ceil_div(N - 64 + 1, 64) * 64; xlo++) {
  for (int y=0; y < M; y++) {
    for (int i=0; i < K; i++) {
      for (int j=0; j < K; j++) {
        if (xlo + _ceil_div(N - 64 + 1, 64) * 64 + i < N && y + j < M) {
          res[(xlo + _ceil_div(N - 64 + 1, 64) * 64) * M + (y)] += kernel[(i) * K + (j)] * image[(xlo + _ceil_div(N - 64 + 1, 64) * 64 + i) * M + (y + j)];
        }
      }
    }
  }
}
}
