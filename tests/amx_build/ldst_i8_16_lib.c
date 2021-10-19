#include "ldst_i8_16_lib.h"
#include <stdint.h>

#include <stdint.h>


inline int _floor_div(int num, int quot) {
  int off = (num>=0)? 0 : quot-1;
  return (num-off)/quot;
}

inline int8_t _clamp_32to8(int32_t x) {
  return (x < -128)? -128 : ((x > 127)? 127 : x);
}

#include <stdio.h>
#include <stdlib.h>




/* relying on the following instruction...
ld_i8(n,m,src,dst)
_tile_loadd({dst}.data, {src}.data, {src}.strides[0]);
*/


/* relying on the following instruction...
st_i8(n,m,src,dst)
_tile_stored({src}.data, {dst}.data, {src}.strides[0]);
*/


/* relying on the following instruction...
config(tile0_bytes,tile0_rows)

  unsigned char config[] = {{
        0x01,
        0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        {tile0_bytes}, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        {tile0_rows},
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00
    }};
 

    _tile_loadconfig(config);

*/

// ldst_i8_16(
//     x : i8[16,16]  @DRAM,
//     y : i8[16,16]  @DRAM
// )
void ldst_i8_16( ldst_i8_16_lib_Context *ctxt, int8_t* x, int8_t* y ) {

  unsigned char config[] = {
        0x01,
        0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        (16), 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        (16),
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00
    };
 

    _tile_loadconfig(config);

int8_t *tmp = (int8_t*) 0;
_tile_loadd(((struct systl_win_2i8){ tmp + (0) * (16) + (0) * (1), { 16,1 } }).data, ((struct systl_win_2i8){ x + (0) * (16) + (0) * (1), { 16,1 } }).data, ((struct systl_win_2i8){ x + (0) * (16) + (0) * (1), { 16,1 } }).strides[0]);
_tile_stored(((struct systl_win_2i8){ tmp + (0) * (16) + (0) * (1), { 16,1 } }).data, ((struct systl_win_2i8){ y + (0) * (16) + (0) * (1), { 16,1 } }).data, ((struct systl_win_2i8){ tmp + (0) * (16) + (0) * (1), { 16,1 } }).strides[0]);

}
