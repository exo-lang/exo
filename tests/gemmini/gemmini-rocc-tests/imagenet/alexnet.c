#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "alexnet_params.h"
#include "alexnet_images.h"

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type=WS;

    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check=false;
    
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool conv=true;
    
    if (argc < 4) {
        conv = false;
    } else if (strcmp(argv[3], "conv") == 0) {
        conv = true;
    } else if (strcmp(argv[3], "matmul") == 0) {
        conv = false;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check] [conv]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;

    // conv_1
    if (!conv) {
      start = read_cycles();

        im2col(conv_1_params.batch_size, conv_1_params.in_channels, conv_1_params.in_dim,
            conv_1_params.I, conv_1_params.K,
            images, conv_1_in, &conv_1_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_1_params.I, conv_1_params.J, conv_1_params.K,
            conv_1_in, conv_1_w, conv_1_b, conv_1_out,
            RELU, conv_1_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_1");

        end = read_cycles();
        matmul_cycles += end - start;

      start = read_cycles();

        pool_with_col2im(conv_1_params.I, conv_1_params.J,
            conv_1_params.batch_size, conv_1_params.out_channels, conv_1_params.out_dim_pooled,
            conv_1_out, conv_1_out_pooled, &conv_1_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
            conv_1_params.out_channels, conv_1_params.out_dim,
            conv_1_params.stride, conv_1_params.padding, conv_1_params.kernel_size,

            (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

            RELU, conv_1_params.output_scale, 0,
            conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    printf("conv_1 cycles: %llu \n", end - start);


    // conv_2
    if (!conv) {
      start = read_cycles();

        im2col(conv_2_params.batch_size, conv_2_params.in_channels, conv_2_params.in_dim,
            conv_2_params.I, conv_2_params.K,
            conv_1_out_pooled, conv_2_in, &conv_2_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_2_in, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_2");

        end = read_cycles();
        matmul_cycles += end - start;

      start = read_cycles();

        pool_with_col2im(conv_2_params.I, conv_2_params.J,
            conv_2_params.batch_size, conv_2_params.out_channels, conv_2_params.out_dim_pooled,
            conv_2_out, conv_2_out_pooled, &conv_2_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_2_params.batch_size, conv_2_params.in_dim, conv_2_params.in_channels,
            conv_2_params.out_channels, conv_2_params.out_dim,
            conv_2_params.stride, conv_2_params.padding, conv_2_params.kernel_size,

            (elem_t*)conv_1_out_pooled, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_2_out_pooled,

            RELU, conv_2_params.output_scale, 0,
            conv_2_params.pool_size, conv_2_params.pool_stride, conv_2_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    printf("conv_2 cycles: %llu \n", end - start);


    // conv_3
    if (!conv) {
      start = read_cycles();

        im2col(conv_3_params.batch_size, conv_3_params.in_channels, conv_3_params.in_dim,
            conv_3_params.I, conv_3_params.K,
            conv_2_out_pooled, conv_3_in, &conv_3_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_3_in, conv_3_w, conv_3_b, conv_3_out,
            RELU, conv_3_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_3");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto_largeC(
            conv_3_params.batch_size, conv_3_params.in_dim, conv_3_params.in_channels,
            conv_3_params.out_channels, conv_3_params.out_dim,
            conv_3_params.stride, conv_3_params.padding, conv_3_params.kernel_size,

            (elem_t*)conv_2_out_pooled, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

            RELU, conv_3_params.output_scale, 0,
            conv_3_params.pool_size, 0, conv_3_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    printf("conv_3 cycles: %llu \n", end - start);


    // conv_4
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_3_params.I, conv_3_params.J,
            conv_4_params.I, conv_4_params.K,
            conv_3_out, conv_4_in, &conv_4_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_4_in, conv_4_w, conv_4_b, conv_4_out,
            RELU, conv_4_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_4");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto_largeC(
            conv_4_params.batch_size, conv_4_params.in_dim, conv_4_params.in_channels,
            conv_4_params.out_channels, conv_4_params.out_dim,
            conv_4_params.stride, conv_4_params.padding, conv_4_params.kernel_size,

            (elem_t*)conv_3_out, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out,

            RELU, conv_4_params.output_scale, 0,
            conv_4_params.pool_size, 0, conv_4_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    printf("conv_4 cycles: %llu \n", end - start);


    // conv_5
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_4_params.I, conv_4_params.J,
            conv_5_params.I, conv_5_params.K,
            conv_4_out, conv_5_in, &conv_5_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_5_in, conv_5_w, conv_5_b, conv_5_out,
            RELU, conv_5_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_5");

        end = read_cycles();
        matmul_cycles += end - start;

      start = read_cycles();

        pool_with_col2im(conv_5_params.I, conv_5_params.J,
            conv_5_params.batch_size, conv_5_params.out_channels, conv_5_params.out_dim_pooled,
            conv_5_out, conv_5_out_pooled, &conv_5_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto_largeC(
            conv_5_params.batch_size, conv_5_params.in_dim, conv_5_params.in_channels,
            conv_5_params.out_channels, conv_5_params.out_dim,
            conv_5_params.stride, conv_5_params.padding, conv_5_params.kernel_size,

            (elem_t*)conv_4_out, (elem_t*)conv_5_w, (acc_t*)conv_5_b, (elem_t*)conv_5_out_pooled,

            RELU, conv_5_params.output_scale, 0,
            conv_5_params.pool_size, conv_5_params.pool_stride, conv_5_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    printf("conv_5 cycles: %llu \n", end - start);

    // Global averaging
    static elem_t average[9216][4] row_align(1);

    start = read_cycles();

    for (int batch = 0; batch < conv_5_params.batch_size; batch++) {
        for (int channel = 0; channel < conv_5_params.out_channels; channel++) {
            int sum = 0;
	    int channel_axis = channel * conv_5_params.out_dim_pooled * conv_5_params.out_dim_pooled;
            for (int row = 0; row < conv_5_params.out_dim_pooled; row++) {
		int row_axis = row * conv_5_params.out_dim_pooled;
                for (int col = 0; col < conv_5_params.out_dim_pooled; col++) {
                    //sum += conv_5_out_pooled[batch][row][col][channel];
		    average[col+row_axis+channel_axis][batch] = conv_5_out_pooled[batch][row][col][channel];
		}
            }
        }
    }

    end = read_cycles();
    other_cycles += end - start;

      // fc_6
    start = read_cycles();

    tiled_matmul_nn_auto(fc_6_params.I, fc_6_params.J, fc_6_params.K,
        fc_6_w, average, fc_6_b, fc_6_out,
        RELU, fc_6_params.output_scale, 0, false,
        tiled_matmul_type, check, "fc_6");

    end = read_cycles();
    matmul_cycles += end - start;

    printf("fc_6 cycles: %llu \n", end - start);
 
    // fc_7
    start = read_cycles();

    tiled_matmul_nn_auto(fc_7_params.I, fc_7_params.J, fc_7_params.K,
        fc_7_w, fc_6_out, fc_7_b, fc_7_out,
        RELU, fc_7_params.output_scale, 0, false,
        tiled_matmul_type, check, "fc_7");

    end = read_cycles();
    matmul_cycles += end - start;

    printf("fc_7 cycles: %llu \n", end - start);
 
    // fc_8
    start = read_cycles();

    tiled_matmul_nn_auto(fc_8_params.I, fc_8_params.J, fc_8_params.K,
        fc_8_w, fc_7_out, fc_8_b, fc_8_out,
        NO_ACTIVATION, fc_8_params.output_scale, 0, false,
        tiled_matmul_type, check, "fc_8");

    end = read_cycles();
    matmul_cycles += end - start;

    printf("fc_8 cycles: %llu \n", end - start);
 
    // Find highest probs
    int preds[fc_8_params.batch_size]; 
    for (int batch = 0; batch < fc_8_params.batch_size; batch++) {
        elem_t max_prob = fc_8_out[0][batch];
        size_t max_idx = 0;

        for (int i = 1; i < fc_8_params.out_features; i++) {
            if (fc_8_out[i][batch] > max_prob) {
                max_prob = fc_8_out[i][batch];
                max_idx = i;
            }
        }
	preds[batch] = max_idx;
        printf("Prediction: %u (score: %d)\n", max_idx, max_prob);
    }

    uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_dw_cycles + res_add_cycles + other_cycles + conv_cycles;

    printf("\nTotal cycles: %llu (100%%)\n", total_cycles);
    printf("Matmul cycles: %llu (%d%%)\n", matmul_cycles, (matmul_cycles * 100) / total_cycles);
    printf("Im2col cycles: %llu (%d%%)\n", im2col_cycles, (im2col_cycles * 100) / total_cycles);
    printf("Conv cycles: %llu (%d%%)\n", conv_cycles, (conv_cycles * 100) / total_cycles);
    printf("Pooling cycles: %llu (%d%%)\n", pool_cycles, (pool_cycles * 100) / total_cycles);
    printf("Depthwise convolution cycles: %llu (%d%%)\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
    printf("Res add cycles: %llu (%d%%)\n", res_add_cycles, (res_add_cycles * 100) / total_cycles);
    printf("Other cycles: %llu (%d%%)\n", other_cycles, (other_cycles * 100) / total_cycles);

    int correct[] = {824, 725, 135, 646};
    for (int i = 0; i < fc_8_params.batch_size; i++) {
        if (preds[i] != correct[i] && fc_8_out[preds[i]][i] != fc_8_out[correct[i]][i]) {
            printf("Prediction %d is incorrect!\nFAIL\n", i+1);
            exit(1);
        }
    }

    printf("PASS\n");
 
    exit(0);
}

