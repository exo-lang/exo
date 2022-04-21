#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// General way to output a matrix.
#define print_matrix(M, N, A)                                                  \
  for (int i = 0; i < M; i++) {                                                \
    for (int j = 0; j < 4 * K; j++) {                                          \
      printf("%u\t", A[i][j]);                                                 \
    }                                                                          \
    printf("\n");                                                              \
  }

/*
  Reference implementation of AMX's dpbuud. Same signature as amx_dpbuud below.
*/
void ref_dpbuud(int M, int K, int N, uint8_t *A, uint8_t *B, uint32_t *C) {
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        for (int n_in = 0; n_in < 4; n_in++) {
          C[m * N + n] +=
              A[m * 4 * K + 4 * k + n_in] * B[k * 4 * N + 4 * n + n_in];
        }
      }
    }
  }
}

/*
  AMX's implementation of dpbuud.
   - A = M x 4K
   - B = K x 4N
   - C = M x N (but is uint32_t)
*/
void amx_dpbuud(int M, int K, int N, uint8_t *A, uint8_t *B, uint32_t *C) {
  unsigned char config[] = {
      0x01,                                      // ID
      0x00,                                      // start row
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // reserved
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // reserved
      4 * K, 0x00,                               // bytes per row tile 0
      4 * N, 0x00,                               // bytes per row tile 1
      4 * N, 0x00,                               // bytes per row tile 2
      0x01, 0x00,                                // bytes per row tile 3
      0x00, 0x00,                                // bytes per row tile 4
      0x00, 0x00,                                // bytes per row tile 5
      0x00, 0x00,                                // bytes per row tile 6
      0x00, 0x00,                                // bytes per row tile 7
      0x00, 0x00,                                // bytes per row tile 8
      0x00, 0x00,                                // bytes per row tile 9
      0x00, 0x00,                                // bytes per row tile 10
      0x00, 0x00,                                // bytes per row tile 11
      0x00, 0x00,                                // bytes per row tile 12
      0x00, 0x00,                                // bytes per row tile 13
      0x00, 0x00,                                // bytes per row tile 14
      0x00, 0x00,                                // bytes per row tile 15
      M,                                         // rows tile 0
      K,                                         // rows tile 1
      M,                                         // rows tile 2
      0x01,                                      // rows tile 3
      0x00,                                      // rows tile 4
      0x00,                                      // rows tile 5
      0x00,                                      // rows tile 6
      0x00,                                      // rows tile 7
      0x00,                                      // rows tile 8
      0x00,                                      // rows tile 9
      0x00,                                      // rows tile 10
      0x00,                                      // rows tile 11
      0x00,                                      // rows tile 12
      0x00,                                      // rows tile 13
      0x00,                                      // rows tile 14
      0x00                                       // rows tile 15
  };

  _tile_loadconfig(config);

  // const int tile_num = 2;

  _tile_zero(2);
  _tile_loadd(0, A, 4 * K);
  _tile_loadd(1, B, 4 * N);

  _tile_dpbuud(2, 0, 1);

  _tile_stored(2, C, 4 * N);
}

/*
  Takes a matrix old_B, and converts it from its 4M x N representation
  to a M x 4N representation, which matches AMX's tile format.
   - old_B = 4M x N,
   - new_B = M x 4N,
*/
void transform(int M, int N, uint8_t *new_B, uint8_t *old_B) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int m_in = 0; m_in < 4; m_in++) {
        new_B[m * 4 * N + (4 * n + m_in)] = old_B[(4 * m + m_in) * N + n];
      }
    }
  }
}

/*
  Makes use of AMX's tile instruction to perform matmul on uint8_t matrices.
  Requires a memory transform prior to loading data into the tile.
   - A = M x 4K
   - B = 4K x N
   - C = M x N (but is uint32_t)
*/
void amx_matmul(int M, int K, int N, uint8_t *A, uint8_t *B, uint32_t *C) {
  uint8_t new_B[K][4 * N];
  transform(K, N, new_B, B);
  amx_dpbuud(M, K, N, A, new_B, C);
}

/*
  Reference implementation of matmul on uint8_t matrices.
*/
void ref_matmul_8(int M, int K, int N, uint8_t *A, uint8_t *B, uint32_t *C) {
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        // TODO: do I need the 1ul?
        C[m * N + n] += 1ul * A[m * K + k] * B[k * N + n];
      }
    }
  }
}

/*
  Reference implementation of matmul on uint32_t matrices.
*/
void ref_matmul_32(int M, int K, int N, uint32_t *A, uint32_t *B, uint32_t *C) {
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

/*
  Performs matmul on uint8_t matrices, except it interprets 4 consecutive bytes
  as a uint32.
*/
void my_matmul_32(int M, int K, int N, uint8_t *A, uint8_t *B, uint32_t *C) {
  /*
    A = M x 4K
    B = K x 4N
    C = M x N (but is uint32_t)
  */
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        for (int n_in = 0; n_in < 4; n_in++) {
          for (int k_in = 0; k_in < 4; k_in++) {
            uint32_t C_temp = (1ul << (8 * (n_in + k_in))) *
                              A[m * 4 * K + 4 * k + k_in] *
                              B[k * 4 * N + 4 * n + n_in];
            C[m * N + n] += C_temp;
          }
        }
      }
    }
  }
}

/*
  Converts a matrix of uint8_ts into a matrix of uint32_t by interpreting
  consecutive 4 bytes as a uint32_t.
   - A_in = M x 4N
   - A_out = M x N
*/
void convert_to_uint32_t(int M, int N, uint8_t *A_in, uint32_t *A_out) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      A_out[i * N + j] = 0;
      for (int k = 0; k < 4; k++) {
        A_out[i * N + j] += (1ul << (8 * k)) * A_in[i * 4 * N + 4 * j + k];
      }
    }
  }
}

// Test to ensure reference dpbuud implementation matches amx implementation
void test_dpbuud(int M, int K, int N) {
  uint8_t A[M][4 * K];
  uint8_t B[K][4 * N];
  uint32_t C_amx[M][N];
  uint32_t C_ref[M][N];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < 4 * K; j++) {
      A[i][j] = (i * i - 2 * i + 1 + j);
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < 4 * N; j++) {
      B[i][j] = (i + 2 * j * j);
    }
  }

  memset(C_amx, 0, 4 * M * N);
  memset(C_ref, 0, 4 * M * N);

  ref_dpbuud(M, K, N, A, B, C_ref);
  amx_dpbuud(M, K, N, A, B, C_amx);

  int match = 1;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      match &= (C_ref[i][j] == C_amx[i][j]);
    }
  }
  if (!match) {
    printf("ERROR: My DPBUUD failed\n");

    printf("Ref DPBUUD:\n");
    print_matrix(M, N, C_ref);
    printf("------------------------------\n");

    printf("AMX DPBUUD:\n");
    print_matrix(M, N, C_amx);

    return;
  }

  printf("My DPBUUD succeeded!\n");
}

// test to ensure that reference matmul implementation = transform + amx dpbuud
// approach
void test_matmul_8(int M, int K, int N) {
  uint8_t A[M][4 * K];
  uint8_t B[4 * K][N];
  uint32_t C_ref[M][N];  // output of a reference matmul
  uint32_t C_amx[M][N];  // output of dpbuud after an initial transformation

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < 4 * K; j++) {
      A[i][j] = (i * i - 2 * i + 1 + j);
    }
  }
  for (int i = 0; i < 4 * K; i++) {
    for (int j = 0; j < N; j++) {
      B[i][j] = (i + 2 * j * j);
    }
  }

  memset(C_ref, 0, 4 * M * N);
  memset(C_amx, 0, 4 * M * N);

  ref_matmul_8(M, 4 * K, N, A, B, C_ref);
  amx_matmul(M, K, N, A, B, C_amx);

  int match = 1;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      match &= (C_ref[i][j] == C_amx[i][j]);
    }
  }

  if (!match) {
    printf("ERROR: Matmul_8 failed\n");
    printf("Ref Matmul_8:\n");
    print_matrix(M, N, C_ref);
    printf("AMX Matmul_8:\n");
    print_matrix(M, N, C_amx);

    return;
  }
  printf("Matmul_8 test succeeded!\n");
}

void simple_transform_memory() {
  uint8_t src[4 * 16] __attribute__((aligned(512)));
  uint8_t dest[4 * 16] __attribute__((aligned(512)));
  for (int i = 0; i < 4 * 16; i++) {
    src[i] = i;
  }
  __m512i a = _mm512_load_epi32(src);
  for (int i = 0; i < 4 * 16; i++) {
    printf("%2d ", src[i]);
    if ((i + 1) % 16 == 0) {
      printf("\n");
    }
  }

  // step 1: permute across lanes
  uint32_t permute_indices[16] = {
      0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
  __m512i b = _mm512_load_epi32(permute_indices);
  __m512i c = _mm512_permutevar_epi32(b, a);

  // step 2: shuffle within lanes
  uint8_t shuffle_indices[4 * 16] = {
      0,
      4,
      8,
      12,
      1,
      5,
      9,
      13,
      2,
      6,
      10,
      14,
      3,
      7,
      11,
      15,
      0,
      4,
      8,
      12,
      1,
      5,
      9,
      13,
      2,
      6,
      10,
      14,
      3,
      7,
      11,
      15,
      0,
      4,
      8,
      12,
      1,
      5,
      9,
      13,
      2,
      6,
      10,
      14,
      3,
      7,
      11,
      15,
      0,
      4,
      8,
      12,
      1,
      5,
      9,
      13,
      2,
      6,
      10,
      14,
      3,
      7,
      11,
      15,
  };
  __m512i d = _mm512_load_epi32(shuffle_indices);
  __m512i e = _mm512_shuffle_epi8(c, d);

  _mm512_store_epi32(dest, e);
  for (int i = 0; i < 4 * 16; i++) {
    printf("%2d ", dest[i]);
    if ((i + 1) % 16 == 0) {
      printf("\n");
    }
  }
}

int main() {
  unsigned char config[] = {
      0x01,                                      // ID
      0x00,                                      // start row
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // reserved
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // reserved
      64, 0x00,                                  // bytes per row tile 0
      64, 0x00,                                  // bytes per row tile 1
      64, 0x00,                                  // bytes per row tile 2
      0x01, 0x00,                                // bytes per row tile 3
      0x00, 0x00,                                // bytes per row tile 4
      0x00, 0x00,                                // bytes per row tile 5
      0x00, 0x00,                                // bytes per row tile 6
      0x00, 0x00,                                // bytes per row tile 7
      0x00, 0x00,                                // bytes per row tile 8
      0x00, 0x00,                                // bytes per row tile 9
      0x00, 0x00,                                // bytes per row tile 10
      0x00, 0x00,                                // bytes per row tile 11
      0x00, 0x00,                                // bytes per row tile 12
      0x00, 0x00,                                // bytes per row tile 13
      0x00, 0x00,                                // bytes per row tile 14
      0x00, 0x00,                                // bytes per row tile 15
      16,                                        // rows tile 0
      16,                                        // rows tile 1
      16,                                        // rows tile 2
      0x01,                                      // rows tile 3
      0x00,                                      // rows tile 4
      0x00,                                      // rows tile 5
      0x00,                                      // rows tile 6
      0x00,                                      // rows tile 7
      0x00,                                      // rows tile 8
      0x00,                                      // rows tile 9
      0x00,                                      // rows tile 10
      0x00,                                      // rows tile 11
      0x00,                                      // rows tile 12
      0x00,                                      // rows tile 13
      0x00,                                      // rows tile 14
      0x00                                       // rows tile 15
  };

  _tile_loadconfig(config);

  uint8_t A[16 * 64];
  uint32_t B[16 * 16];
  uint8_t C[16 * 64];
  for (int i = 0; i < 16 * 64; i++) {
    A[i] = 1;
  }
  for (int i = 0; i < 16 * 16; i++) {
    B[i] = 2;
  }

  _tile_loadd(0, A, 64);
  _tile_loadd(1, A, 64);
  _tile_loadd(2, B, 64);
  _tile_dpbssd(2, 1, 0);
  _tile_stored(2, C, 64);

  for (int i = 0; i < 16 * 64; i++) {
    printf("%2d ", C[i]);
    if ((i + 1) % 64 == 0) {
      printf("\n");
    }
  }

  // int M = 10;
  // int K = 5;
  // int N = 7;

  // test_dpbuud(M, K, N);
  // test_matmul_8(M, K, N);

  /*
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

    uint32_t x = A_32[0][0];
    uint32_t y = A_32[1][0];
    uint32_t z = A_32[0][1];

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

    printf("Reference matmul\n");
    for (int i=0; i<M; i++) {
      for (int j=0; j<N; j++) {
        printf("%d\t", C_matmul1[i][j]);
      }
      printf("\n");
    }
    printf("My matmul\n");
    for (int i=0; i<M; i++) {
      for (int j=0; j<N; j++) {
        printf("%d\t", C_matmul2[i][j]);
      }
      printf("\n");
    }

    return -1;
  }

  printf("My matmul succeeded!\n");
  */

  return 0;
}
