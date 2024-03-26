
#pragma once
#ifndef BLUR_H
#define BLUR_H

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

struct exo_win_1ui16 {
  uint16_t *const data;
  const int_fast32_t strides[1];
};
struct exo_win_1ui16c {
  const uint16_t *const data;
  const int_fast32_t strides[1];
};
// exo_base_blur(
//     W : size,
//     H : size,
//     blur_y : ui16[H, W] @DRAM,
//     inp : ui16[H + 2, W + 2] @DRAM
// )
void exo_base_blur(void *ctxt, int_fast32_t W, int_fast32_t H, uint16_t *blur_y,
    const uint16_t *inp);

// exo_blur_halide(
//     W : size,
//     H : size,
//     blur_y : ui16[H, W] @DRAM,
//     inp : ui16[H + 2, W + 2] @DRAM
// )
void exo_blur_halide(void *ctxt, int_fast32_t W, int_fast32_t H,
    uint16_t *blur_y, const uint16_t *inp);

#ifdef __cplusplus
}
#endif
#endif  // BLUR_H
