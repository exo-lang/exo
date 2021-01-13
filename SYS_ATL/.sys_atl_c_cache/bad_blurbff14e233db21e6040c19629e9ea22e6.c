int _floor_div(int num, int quot) {
  int off = (num<0)? quot-1 : 0;
  return (num-off)/quot;
}


// bad_blur( image : R[N,M] @IN, kernel : R[K,K] @IN, res : R[N,M] @OUT )
void bad_blur( int N, int M, int K, float* image, float* kernel, float* res) {
for (int y=0; y < M; y++) {
  for (int x=0; x < N; x++) {
    res[(x) * M + (y)] = 0.0;
  }
}
for (int y=0; y < M; y++) {
  for (int x=0; x < N; x++) {
    for (int i=0; i < K; i++) {
      for (int j=0; j < K; j++) {
        if (x + i < N && y + j < M) {
          res[(x) * M + (y)] += kernel[(i) * K + (j)] * image[(x + i) * M + (y + j)];
        }
      }
    }
  }
}
}
