
#pragma once
#ifndef UNSHARP_H
#define UNSHARP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#define EXO_ASSUME(expr) ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#define EXO_ASSUME(expr) ((void)(expr))
#endif

struct exo_win_1f32 {
  float *const data;
  const int_fast32_t strides[1];
};
struct exo_win_1f32c {
  const float *const data;
  const int_fast32_t strides[1];
};
// exo_unsharp(
//     W : size,
//     H : size,
//     output : f32[3, H, W] @DRAM,
//     input : f32[3, H + 6, W + 6] @DRAM
// )
void exo_unsharp(void *ctxt, int_fast32_t W, int_fast32_t H, float *output,
    const float *input);

// exo_unsharp_base(
//     W : size,
//     H : size,
//     output : f32[3, H, W] @DRAM,
//     input : f32[3, H + 6, W + 6] @DRAM
// )
void exo_unsharp_base(void *ctxt, int_fast32_t W, int_fast32_t H, float *output,
    const float *input);

// exo_unsharp_vectorized(
//     W : size,
//     H : size,
//     output : f32[3, H, W] @DRAM,
//     input : f32[3, H + 6, W + 6] @DRAM
// )
void exo_unsharp_vectorized(void *ctxt, int_fast32_t W, int_fast32_t H,
    float *output, const float *input);

#ifdef __cplusplus
}
#endif
#endif  // UNSHARP_H
