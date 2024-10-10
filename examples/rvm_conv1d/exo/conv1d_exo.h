
#pragma once
#ifndef CONV1D_EXO_H
#define CONV1D_EXO_H

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


#ifndef EXO_WIN_2I32
#define EXO_WIN_2I32
struct exo_win_2i32{
    int32_t * const data;
    const int_fast32_t strides[2];
};
#endif
#ifndef EXO_WIN_2I32C
#define EXO_WIN_2I32C
struct exo_win_2i32c{
    const int32_t * const data;
    const int_fast32_t strides[2];
};
#endif
// exo_conv1d_tile_lt_kw(
//     data : i32[4, 16] @DRAM,
//     kernels : i32[16, 4, 4] @DRAM,
//     out : i32[16, 16] @DRAM
// )
void exo_conv1d_tile_lt_kw( void *ctxt, const int32_t* data, const int32_t* kernels, int32_t* out );



#ifdef __cplusplus
}
#endif
#endif  // CONV1D_EXO_H
