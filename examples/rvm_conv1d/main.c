
/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#include "conv1Di32.h"
#include "exo/conv1d_exo.h"

////////////////////
// CONFIGURATION //
//////////////////

#define TILE 4

/////////////
// MACROS //
///////////

#define CEIL_DIV(a, b) ((((a) % (b)) != 0) ? (((a) / (b)) + 1) : (a) / (b))

int32_t out[OC * N];
int32_t data_tile[TILE][IC * W];
int32_t result[OC * N];
int32_t small_data_tile_a[TILE*TILE];
int32_t small_data_tile_b[TILE*TILE];

////////////////
// MAIN CODE //
//////////////

void conv1d_tile_lt_kw_reord(int32_t *data, int32_t *kernels, int32_t *out)
{
    // should be ceil_div(ic*kw, tile) * tile
    // and initialized to 0
    int tile_i_len = CEIL_DIV(OC, TILE*4);
    int tile_j_len = CEIL_DIV(N, TILE);
    int data_base;
    int cycles;
    int32_t *kernel_base = kernels;
    register int32_t *small_data_tile = small_data_tile_a;
    register int32_t *temp;
    for (int tile_i = 0; tile_i < tile_i_len; tile_i++)
    {
        data_base = 0;
        for (int tile_j = 0; tile_j < tile_j_len; tile_j++)
        {
            asm volatile("mzero m1");            
            asm volatile("mzero m2");
            asm volatile("mzero m3");            
            asm volatile("mzero m4");
            int data_row = 0;
            for (int tile_k = 0; tile_k < IC; tile_k++)
            {
                //CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT, 0x1);
                //CSR_WRITE(CSR_REG_MCYCLE, 0);
                for (int replica = 0; replica < TILE; replica++)
                {
                    int ofs = data_base + replica;
                    int drow_ofs = data_row + ofs;
                    int dtile_ofs = replica*TILE;
                    for (int i = 0; i < W; i++)
                    {
                        // Check that we are not out of bounds of the input in the current channel
                        // this should not block: addresses are different
                        small_data_tile[dtile_ofs] = 0; 
                        if (ofs < N) {
                            small_data_tile[dtile_ofs] = data[drow_ofs];
                        }
                        
                        ofs++;
                        drow_ofs++;
                        dtile_ofs++;
                    }
                    //CSR_READ(CSR_REG_MCYCLE, &cycles);
                    //printf("cyc: %d\n", cycles);
                }
                data_row += N;

                asm volatile("mld.w m0, (%1), %0" ::"r"(TILE * 4), "r"(small_data_tile));
                asm volatile("mld.w m5, (%1), %0" ::"r"(IC * W * 4), "r"(kernel_base));
                asm volatile("mmasa.w m1, m0, m5");
                asm volatile("mld.w m6, (%1), %0" ::"r"(IC * W * 4), "r"(kernel_base+TILE * IC * W));
                asm volatile("mmasa.w m2, m0, m6");
                asm volatile("mld.w m7, (%1), %0" ::"r"(IC * W * 4), "r"(kernel_base+TILE * IC * W*2));
                asm volatile("mmasa.w m3, m0, m7");
                asm volatile("mld.w m5, (%1), %0" ::"r"(IC * W * 4), "r"(kernel_base+TILE * IC * W*3));
                asm volatile("mmasa.w m4, m0, m5");      
                kernel_base += W;
                // swap
                // asm ("xor %0, %0, %1" : "=r"(small_data_tile_cur) : "r"(small_data_tile_old));
                // asm ("xor %0, %0, %1" : "=r"(small_data_tile_old) : "r"(small_data_tile_cur));
                // asm ("xor %0, %0, %1" : "=r"(small_data_tile_cur) : "r"(small_data_tile_old));
            }
            int32_t *outptr = (out + (tile_i * N * 4 + tile_j) * TILE);
            asm volatile("mst.w m1, (%1), %0" ::"r"(N * 4), "r"(outptr));
            asm volatile("mst.w m2, (%1), %0" ::"r"(N * 4), "r"(outptr + TILE*N));
            asm volatile("mst.w m3, (%1), %0" ::"r"(N * 4), "r"(outptr + TILE*N*2));
            asm volatile("mst.w m4, (%1), %0" ::"r"(N * 4), "r"(outptr + TILE*N*3));

            data_base += TILE;
            kernel_base -= W * IC;
        }
        kernel_base += TILE * IC * W*4;
    }
}

#define BRANCHLESS_TERNARY(c, x, y) ((-(c) & x) | (~(-(c)) & y));
void conv1d_cpu(int32_t *data, int32_t *kernels, int32_t *out)
{
    for (int i = 0; i < OC; i++)
    {
        for (int j = 0; j < N; j++)
        {
            out[N * i + j] = 0;
            for (int w_i = 0; w_i < W; w_i++)
            {
                for (int w_j = 0; w_j < IC; w_j++)
                {
                    int data_idx = j + w_i;
                    int kernel_idx = (IC * i + w_j) * W + w_i;
                    int data_at_idx = BRANCHLESS_TERNARY(data_idx < N, data[w_j * N + j + w_i], 0);
                    out[N * i + j] += data_at_idx * kernels[kernel_idx];
                }
            }
        }
    }
}

int check_result(int32_t *result) {
    int err = 0;
    for (int i = 0; i < OC; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (result[N * i + j] != EXPECTED[N * i + j])
            {
                err++;
                printf("exp %d got %d\n\r", EXPECTED[N * i + j], result[N * i + j]);
            }
        }
    }
    return err;
}

int main()
{
    for (int i = 0; i < TILE; i++)
    {
        for (int j = 0; j < TILE; j++)
        {
            small_data_tile_a[i*TILE+j] = 0;
            small_data_tile_b[i*TILE+j] = 0;
        }
    }

    conv1d_tile_lt_kw_reord(DATA, KERNELS, result);
    printf("handwritten err: %d\n\r", check_result(result));

    for (int i = 0; i < OC; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*N+j] = 0;
        }
    }

    exo_conv1d_tile_lt_kw(NULL, DATA, KERNELS, result);
    printf("exo err: %d\n\r", check_result(result));

    return 0;
}
