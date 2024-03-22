
#pragma once
#ifndef UNSHARP_H
#define UNSHARP_H

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



// unsharp(
//     W : size,
//     H : size,
//     output : f32[H, W] @DRAM,
//     input : f32[H + 6, W + 6, 3] @DRAM
// )
void unsharp( void *ctxt, int_fast32_t W, int_fast32_t H, float* output, const float* input );



#ifdef __cplusplus
}
#endif
#endif  // UNSHARP_H
