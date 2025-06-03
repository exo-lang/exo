#ifndef EXO_ARM_SVE_H
#define EXO_ARM_SVE_H

#include <arm_sve.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline __attribute__((always_inline)) void svmla_n_f32_x_vla(
    int64_t N, float32_t *dst, const float32_t *src1, float32_t src2) {
  int64_t i = 0;
  svbool_t pg = svwhilelt_b32(i, N);
  do {
    svst1_f32(pg, &dst[i],
        svmla_n_f32_x(
            pg, svld1_f32(pg, &dst[i]), svld1_f32(pg, &src1[i]), src2));
    i += svcntw();
    pg = svwhilelt_b32(i, N);
  } while (svptest_first(svptrue_b32(), pg));
}

#ifdef __cplusplus
}
#endif

#endif