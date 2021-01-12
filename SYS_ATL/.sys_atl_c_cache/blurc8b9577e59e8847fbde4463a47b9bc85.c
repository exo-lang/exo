int _floor_div(int num, int quot) {
  int off = (num<0)? quot-1 : 0;
  return (num-off)/quot;
}


// blur( image : R[N,M] @IN, kernel : R[K,K] @IN, res : R[N,M] @OUT )
void blur( int N, int M, int K, float* image, float* kernel, float* res) {
for (int x=0; x < N; x++) {
  for (int y=0; y < M; y++) {
    res[(x) * M + (y)] = 0.0;
  }
}
for (int x=0; x < N; x++) {
  for (int y=0; y < M; y++) {
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
