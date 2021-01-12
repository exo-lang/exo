int _ceil_div(int num, int quot) {
  int off = (num>0)? quot-1 : 0;
  return (num+off)/quot;
}


// split_blur( image : R[N,M] @IN, kernel : R[K,K] @IN, res : R[N,M] @OUT )
void split_blur( int N, int M, int K, float* image, float* kernel, float* res) {
for (int x=0; x < N; x++) {
  for (int y=0; y < M; y++) {
    res[(x) * M + (y)] = 0.0;
  }
}
for (int xhi=0; xhi < _ceil_div(N - 8 + 1, 8); xhi++) {
  for (int yhi=0; yhi < _ceil_div(M - 8 + 1, 8); yhi++) {
    for (int xlo=0; xlo < 8; xlo++) {
      for (int ylo=0; ylo < 8; ylo++) {
        for (int i=0; i < K; i++) {
          for (int j=0; j < K; j++) {
            if (8 * xhi + xlo + i < N && 8 * yhi + ylo + j < M) {
              res[(8 * xhi + xlo) * M + (8 * yhi + ylo)] += kernel[(i) * K + (j)] * image[(8 * xhi + xlo + i) * M + (8 * yhi + ylo + j)];
            }
          }
        }
      }
    }
  }
  for (int xlo=0; xlo < 8; xlo++) {
    for (int ylo=0; ylo < M - _ceil_div(M - 8 + 1, 8) * 8; ylo++) {
      for (int i=0; i < K; i++) {
        for (int j=0; j < K; j++) {
          if (8 * xhi + xlo + i < N && ylo + _ceil_div(M - 8 + 1, 8) * 8 + j < M) {
            res[(8 * xhi + xlo) * M + (ylo + _ceil_div(M - 8 + 1, 8) * 8)] += kernel[(i) * K + (j)] * image[(8 * xhi + xlo + i) * M + (ylo + _ceil_div(M - 8 + 1, 8) * 8 + j)];
          }
        }
      }
    }
  }
}
for (int xlo=0; xlo < N - _ceil_div(N - 8 + 1, 8) * 8; xlo++) {
  for (int y=0; y < M; y++) {
    for (int i=0; i < K; i++) {
      for (int j=0; j < K; j++) {
        if (xlo + _ceil_div(N - 8 + 1, 8) * 8 + i < N && y + j < M) {
          res[(xlo + _ceil_div(N - 8 + 1, 8) * 8) * M + (y)] += kernel[(i) * K + (j)] * image[(xlo + _ceil_div(N - 8 + 1, 8) * 8 + i) * M + (y + j)];
        }
      }
    }
  }
}
}
