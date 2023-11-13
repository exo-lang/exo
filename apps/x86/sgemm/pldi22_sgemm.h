
#pragma once
#ifndef PLDI22_SGEMM_H
#define PLDI22_SGEMM_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

typedef struct sgemm_Context { 

} sgemm_Context;


// sgemm_exo(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M,K]  @DRAM,
//     B : f32[K,N]  @DRAM,
//     C : f32[M,N]  @DRAM
// )
void sgemm_pldi22_exo( sgemm_Context *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, float* A, float* B, float* C );



#ifdef __cplusplus
}
#endif
#endif  // PLDI22_SGEMM_H
