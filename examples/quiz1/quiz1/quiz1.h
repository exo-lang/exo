
#pragma once
#ifndef QUIZ1_H
#define QUIZ1_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


#ifndef EXO_WIN_1F32
#define EXO_WIN_1F32
struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
#endif
#ifndef EXO_WIN_1F32C
#define EXO_WIN_1F32C
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
#endif
// vec_double(
//     N : size,
//     inp : f32[N] @DRAM,
//     out : f32[N] @DRAM
// )
void vec_double( void *ctxt, int_fast32_t N, const float* inp, float* out );

// vec_double_optimized(
//     N : size,
//     inp : f32[N] @DRAM,
//     out : f32[N] @DRAM
// )
void vec_double_optimized( void *ctxt, int_fast32_t N, const float* inp, float* out );



#ifdef __cplusplus
}
#endif
#endif  // QUIZ1_H
