int _floor_div(int num, int quot) {
  int off = (num<0)? quot-1 : 0;
  return (num-off)/quot;
}


// split_blur( image : R[N,M] @IN, kernel : R[K,K] @IN, res : R[N,M] @OUT )
void split_blur( int N, int M, int K, float* image, float* kernel, float* res) {
for (int x=0; x < N; x++) {
  for (int yhi=0; yhi < _floor_div(M, 4); yhi++) {
    for (int ylo=0; ylo < 4; ylo++) {
      if (4 * yhi + ylo < M) {
        res[(x) * M + (4 * yhi + ylo)] = 0.0;
      }
    }
  }
}
for (int x=0; x < N; x++) {
  for (int yhi=0; yhi < _floor_div(M, 4); yhi++) {
    for (int ylo=0; ylo < 4; ylo++) {
      if (4 * yhi + ylo < M) {
        for (int i=0; i < K; i++) {
          for (int j=0; j < K; j++) {
            if (x + i < N && 4 * yhi + ylo + j < M) {
              res[(x) * M + (4 * yhi + ylo)] += kernel[(i) * K + (j)] * image[(x + i) * M + (4 * yhi + ylo + j)];
            }
          }
        }
      }
    }
  }
}
}
