#ifndef _LDST_I8_16_LIB_H_
#define _LDST_I8_16_LIB_H_
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stdbool.h>

struct systl_win_2i8{
    int8_t *data;
    int strides[2];
};
typedef struct ldst_i8_16_lib_Context { 

} ldst_i8_16_lib_Context;


// ldst_i8_16(
//     x : i8[16,16]  @DRAM,
//     y : i8[16,16]  @DRAM
// )
void ldst_i8_16( ldst_i8_16_lib_Context *ctxt, int8_t* x, int8_t* y );
#ifdef __cplusplus
}
#endif
#endif //_LDST_I8_16_LIB_H_
