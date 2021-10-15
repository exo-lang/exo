#include <immintrin.h>
#include <stdio.h>

void reference(int N, int M, int K, uint8_t* A, uint8_t* B, uint32_t* C) {
  for (int m=0; m<M; m++) {
    for (int k=0; k<K; k++) {
      for (int n=0; n<N; n++) {
        for (int n_in=0; n_in<4; n_in++) {
          C[m*N + n] += A[m*4*K + 4*k+n_in] * B[k*4*N + 4*n+n_in];
        }
      }
    }
  }
}

void amx_stuff(int N, int M, int K, uint8_t* A, uint8_t* B, uint32_t* C) {
  unsigned char config[] = {
        0x01, // ID
        0x00, // start row
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
        4*K, 0x00, // bytes per row tile 0
        4*N, 0x00, // bytes per row tile 1
        4*N, 0x00, // bytes per row tile 2
        0x00, 0x00, // bytes per row tile 3
        0x00, 0x00, // bytes per row tile 4
        0x00, 0x00, // bytes per row tile 5
        0x00, 0x00, // bytes per row tile 6
        0x00, 0x00, // bytes per row tile 7
        0x00, 0x00, // bytes per row tile 8
        0x00, 0x00, // bytes per row tile 9
        0x00, 0x00, // bytes per row tile 10
        0x00, 0x00, // bytes per row tile 11
        0x00, 0x00, // bytes per row tile 12
        0x00, 0x00, // bytes per row tile 13
        0x00, 0x00, // bytes per row tile 14
        0x00, 0x00, // bytes per row tile 15
        M, // rows tile 0
        K, // rows tile 1
        M, // rows tile 2
        0x00, // rows tile 3
        0x00, // rows tile 4
        0x00, // rows tile 5
        0x00, // rows tile 6
        0x00, // rows tile 7
        0x00, // rows tile 8
        0x00, // rows tile 9
        0x00, // rows tile 10
        0x00, // rows tile 11
        0x00, // rows tile 12
        0x00, // rows tile 13
        0x00, // rows tile 14
        0x00 // rows tile 15
    };

    _tile_loadconfig(config);
   
    _tile_zero(2);
    _tile_loadd(0, A, 4*K);
    _tile_loadd(1, B, 4*N);

    _tile_dpbuud(2, 0, 1);
 
    _tile_stored(2, C, 4*N);
}

int main() {
  int N = 10;
  int K = 5;
  int M = 7;
  uint8_t A[M][4*K];
  uint8_t B[K][4*N];
  uint32_t C_amx[M][N]; 
  uint32_t C_ref[M][N]; 
  
  for (int i=0; i<M; i++) {
    for (int j=0; j<4*K; j++) {
      A[i][j] = (i+j);
    }
  }
  for (int i=0; i<K; i++) {
    for (int j=0; j<4*N; j++) {
      B[i][j] = (i+j);
    }
  }
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      C_amx[i][j] = 0;
      C_ref[i][j] = 0;
    }
  }

  reference(N, K, M, A, B, C_ref);

  amx_stuff(N, K, M, A, B, C_amx);
 
  int match = 1;
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      match &= (C_ref[i][j] == C_amx[i][j]);
    }
  }
  if (!match) {
    printf("ERROR: FAILED\n");  
    return -1;
  }

  printf("Success!\n");
  printf("Reference results:\n");
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%d ", C_ref[i][j]);
    }
    printf("\n");
  }
  printf("------------------------------\n");
  printf("AMX results:\n");
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%d ", C_amx[i][j]);
    }
    printf("\n");
  }

  return 0;
}
