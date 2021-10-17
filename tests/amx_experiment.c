#include <immintrin.h>
#include <stdio.h>

void my_dpbuud(int M, int K, int N, uint8_t* A, uint8_t* B, uint32_t* C) {
  /*
    A = M x 4K
    B = K x 4N
    C = M x N (but is uint32_t)
  */
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

void amx_dpbuud(int M, int K, int N, uint8_t* A, uint8_t* B, uint32_t* C) {
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

void matmul_32(int M, int K, int N, uint32_t* A, uint32_t* B, uint32_t* C) {
  for (int m=0; m<M; m++) {
    for (int k=0; k<K; k++) {
      for (int n=0; n<N; n++) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

void my_matmul(int M, int K, int N, uint8_t* A, uint8_t* B, uint32_t* C) {
  /*
    A = M x 4K
    B = K x 4N
    C = M x N (but is uint32_t)
  */
  for (int m=0; m<M; m++) {
    for (int k=0; k<K; k++) {
      for (int n=0; n<N; n++) {
        for (int n_in=0; n_in<4; n_in++) {
          C[m*N + n] += (1 << (8*n_in*2)) * A[m*4*K + 4*k+n_in] * B[k*4*N + 4*n+n_in];
        }
      }
    }
  }
}

void convert_to_uint32_t(int M, int N, uint8_t* A_in, uint32_t* A_out) {
  /*
    A_in = M x 4N
    A_out = M x N
  */
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      A_out[i * N + j] = 0;
      for (int k=0; k<4; k++) { 
        A_out[i*N + j] += A_in[i*4*N + 4*j + k] * (1 << (8*k));
      }
    }
  }
}

int main() {
  int M = 7;
  int K = 5;
  int N = 10;
  
  uint8_t A[M][4*K];
  uint8_t B[K][4*N];


  uint32_t A_32[M][K];
  uint32_t B_32[K][N];
  uint32_t C_amx[M][N]; 
  uint32_t C_ref[M][N]; 
  
  for (int i=0; i<M; i++) {
    for (int j=0; j<4*K; j++) {
      A[i][j] = (j%4 > 0) ? 0 : (i+j);
    }
  }
  for (int i=0; i<K; i++) {
    for (int j=0; j<4*N; j++) {
      B[i][j] = (j%4 > 0) ? 0 : (i+j);
    }
  }
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      C_amx[i][j] = 0;
      C_ref[i][j] = 0;
    }
  }

  my_dpbuud(M, K, N, A, B, C_ref);
  amx_dpbuud(M, K, N, A, B, C_amx);
 
  int match = 1;
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      match &= (C_ref[i][j] == C_amx[i][j]);
    }
  }
  if (!match) {
    printf("ERROR: My DPBUUD failed\n");  
    return -1;
  }

  printf("My DPBUUD succeeded!\n");
  /*
  printf("Reference results:\n");
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%d\t", C_ref[i][j]);
    }
    printf("\n");
  }
  printf("------------------------------\n");
  printf("AMX results:\n");
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%d\t", C_amx[i][j]);
    }
    printf("\n");
  }
  */
  
  uint32_t C_matmul1[M][N];
  uint32_t C_matmul2[M][N];
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      C_matmul1[i][j] = 0;
      C_matmul2[i][j] = 0;
    }
  }

  convert_to_uint32_t(M, K, A, A_32);
  convert_to_uint32_t(K, N, B, B_32);

  /*
  printf("Matrix A:\n");
  for (int i=0; i<M; i++) {
    for (int j=0; j<4*K; j++) {
      printf("%d\t", A[i][j]);
    }
    printf("\n");
  }
  printf("Matrix A_32:\n");
  for (int i=0; i<M; i++) {
    for (int j=0; j<K; j++) {
      printf("%d\t", A_32[i][j]);
    }
    printf("\n");
  }
  printf("Matrix B:\n");
  for (int i=0; i<K; i++) {
    for (int j=0; j<4*N; j++) {
      printf("%d\t", B[i][j]);
    }
    printf("\n");
  }
  printf("Matrix B_32:\n");
  for (int i=0; i<K; i++) {
    for (int j=0; j<N; j++) {
      printf("%d\t", B_32[i][j]);
    }
    printf("\n");
  }
  */

  matmul_32(M, K, N, A_32, B_32, C_matmul1);
  my_matmul(M, K, N, A, B, C_matmul2);
  
  match = 1;
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      match &= (C_matmul1[i][j] == C_matmul2[i][j]);
    }
  }
  if (!match) {
    printf("ERROR: My matmul failed\n");  
    return -1;
  }

  printf("My matmul succeeded!\n");
  
  /*
  printf("Matrix C reference matmul\n");
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%d\t", C_matmul1[i][j]);
    }
    printf("\n");
  }
  printf("Matrix C matmul\n");
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%d\t", C_matmul2[i][j]);
    }
    printf("\n");
  }*/

  return 0;
}
