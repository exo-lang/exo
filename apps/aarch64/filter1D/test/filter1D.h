
#pragma once
#ifndef FILTER1D_H
#define FILTER1D_H

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


struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
// filter1D(
//     ow : size,
//     kw : size,
//     x : f32[ow + kw - 1] @DRAM,
//     y : f32[ow] @DRAM,
//     w : f32[kw] @DRAM
// )
void filter1D( void *ctxt, int_fast32_t ow, int_fast32_t kw, const float* x, float* y, const float* w );



#ifdef __cplusplus
}
#endif
#endif  // FILTER1D_H
