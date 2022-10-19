#pragma once

#ifndef EXO_APPS_GEMMINI_HELPERS_H
#define EXO_APPS_GEMMINI_HELPERS_H

#include <stdint.h>

void print_2i8(int N, int M, int8_t *data);
void print_4i8(int N, int M, int K, int R, int8_t *data);
void print_2i32(int N, int M, int32_t *data);
bool check_eq_2i8(int N, int M, int8_t *lhs, int8_t *rhs);
bool check_eq_4i8(int N, int M, int K, int R, int8_t *lhs, int8_t *rhs);
bool check_eq_2i32(int N, int M, int32_t *lhs, int32_t *rhs);

#endif  // EXO_APPS_GEMMINI_HELPERS_H
