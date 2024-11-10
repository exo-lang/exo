
#pragma once
#ifndef QUIZ2_H
#define QUIZ2_H

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



// scaled_add(
//     N : size,
//     a : f32[N] @DRAM,
//     b : f32[N] @DRAM,
//     c : f32[N] @DRAM
// )
void scaled_add( void *ctxt, int_fast32_t N, const float* a, const float* b, float* c );

// scaled_add_scheduled(
//     N : size,
//     a : f32[N] @DRAM,
//     b : f32[N] @DRAM,
//     c : f32[N] @DRAM
// )
void scaled_add_scheduled( void *ctxt, int_fast32_t N, const float* a, const float* b, float* c );



#ifdef __cplusplus
}
#endif
#endif  // QUIZ2_H
