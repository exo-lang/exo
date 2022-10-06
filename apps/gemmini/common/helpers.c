#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "helpers.h"

void print_2i8(int N, int M, int8_t *data) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++)
      printf("%d ", (int)data[M * i + j]);
    printf("\\n");
  }
}

void print_4i8(int N, int M, int K, int R, int8_t *data) {
  printf("%d %d %d %d\\n", N, M, K, R);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      printf("{ ");
      for (int k = 0; k < K; k++) {
        printf("{ ");
        for (int r = 0; r < R; r++)
          printf("%d ", (int)data[M * K * R * i + K * R * j + R * k + r]);
        printf("}, ");
      }
      printf("}, ");
    }
    printf("\\n");
  }
}

void print_2i32(int N, int M, int32_t *data) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++)
      printf("%d ", (int)data[M * i + j]);
    printf("\\n");
  }
}

bool check_eq_2i8(int N, int M, int8_t *lhs, int8_t *rhs) {
  bool flag = true;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++)
      if (lhs[M * i + j] != rhs[M * i + j])
        flag = false;
  }
  return flag;
}

bool check_eq_4i8(int N, int M, int K, int R, int8_t *lhs, int8_t *rhs) {
  bool flag = true;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++)
      for (int k = 0; k < K; k++)
        for (int r = 0; r < R; r++)
          if (lhs[M * K * R * i + K * R * j + R * k + r] !=
              rhs[M * K * R * i + K * R * j + R * k + r])
            flag = false;
  }
  return flag;
}

bool check_eq_2i32(int N, int M, int32_t *lhs, int32_t *rhs) {
  bool flag = true;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++)
      if (lhs[M * i + j] != rhs[M * i + j])
        flag = false;
  }
  return flag;
}
