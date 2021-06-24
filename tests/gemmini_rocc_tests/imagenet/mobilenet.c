#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "mobilenet_params.h"
#include "images.h"

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

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

    bool conv = true;
    
    if (argc < 3) {
        conv = false;
    } else if (strcmp(argv[2], "conv") == 0) {
        conv = true;
    } else if (strcmp(argv[2], "matmul") == 0) {
        conv = false;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check] [conv]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check = false;
    
    if (argc < 4) {
        check = false;
    } else if (strcmp(argv[3], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
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

    } else {
        start = read_cycles();

        tiled_conv_auto_first(
            conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
            conv_1_params.out_channels, conv_1_params.out_dim,
            conv_1_params.stride, conv_1_params.padding, conv_1_params.kernel_size,

            (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out,

            RELU, conv_1_params.output_scale, 0,
            conv_1_params.pool_size, 0, conv_1_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
    }

    // conv_dw_2
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_1_params.I, conv_1_params.J, conv_dw_2_params.I, conv_dw_2_params.J,
            conv_dw_2_params.batch_size, conv_dw_2_params.in_channels, conv_dw_2_params.out_dim, conv_dw_2_params.kernel_size,
            conv_1_out, conv_dw_2_w, conv_dw_2_b, conv_dw_2_out, &conv_dw_2_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_2_params.batch_size, conv_dw_2_params.in_dim, conv_dw_2_params.in_channels,
            conv_dw_2_params.out_channels, conv_dw_2_params.out_dim,
            conv_dw_2_params.stride, conv_dw_2_params.padding, conv_dw_2_params.kernel_size,

            (elem_t*)conv_1_out, (elem_t*)conv_dw_2_w_flat, (acc_t*)conv_dw_2_b, (elem_t*)conv_dw_2_out,

            RELU, conv_dw_2_params.output_scale, 0,
            conv_dw_2_params.pool_size, 0, conv_dw_2_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_2: %llu \n", end - start);

    // conv_3
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_dw_2_out, conv_3_w, conv_3_b, conv_3_out,
            NO_ACTIVATION, conv_3_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_3");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_dw_2_out, conv_3_w, conv_3_b, conv_3_out,
            NO_ACTIVATION, conv_3_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_3");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_4
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_3_out, conv_4_w, conv_4_b, conv_4_out,
            RELU, conv_4_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_4");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_3_out, conv_4_w, conv_4_b, conv_4_out,
            RELU, conv_4_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_4");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_5
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_4_params.I, conv_4_params.J, conv_dw_5_params.I, conv_dw_5_params.J,
            conv_dw_5_params.batch_size, conv_dw_5_params.in_channels, conv_dw_5_params.out_dim, conv_dw_5_params.kernel_size,
            conv_4_out, conv_dw_5_w, conv_dw_5_b, conv_dw_5_out, &conv_dw_5_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_5_params.batch_size, conv_dw_5_params.in_dim, conv_dw_5_params.in_channels,
            conv_dw_5_params.out_channels, conv_dw_5_params.out_dim,
            conv_dw_5_params.stride, conv_dw_5_params.padding, conv_dw_5_params.kernel_size,

            (elem_t*)conv_4_out, (elem_t*)conv_dw_5_w_flat, (acc_t*)conv_dw_5_b, (elem_t*)conv_dw_5_out,

            RELU, conv_dw_5_params.output_scale, 0,
            conv_dw_5_params.pool_size, 0, conv_dw_5_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;

    printf("conv_dw_5: %llu \n", end - start);

    // conv_6
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_dw_5_out, conv_6_w, conv_6_b, conv_6_out,
            NO_ACTIVATION, conv_6_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_6");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_dw_5_out, conv_6_w, conv_6_b, conv_6_out,
            NO_ACTIVATION, conv_6_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_6");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_7
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_6_out, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_7");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_6_out, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_7");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_8
    start = read_cycles();
    if (!conv) {
        conv_dw_with_col2im(conv_7_params.I, conv_7_params.J, conv_dw_8_params.I, conv_dw_8_params.J,
            conv_dw_8_params.batch_size, conv_dw_8_params.in_channels, conv_dw_8_params.out_dim, conv_dw_8_params.kernel_size,
            conv_7_out, conv_dw_8_w, conv_dw_8_b, conv_dw_8_out, &conv_dw_8_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_8_params.batch_size, conv_dw_8_params.in_dim, conv_dw_8_params.in_channels,
            conv_dw_8_params.out_channels, conv_dw_8_params.out_dim,
            conv_dw_8_params.stride, conv_dw_8_params.padding, conv_dw_8_params.kernel_size,

            (elem_t*)conv_7_out, (elem_t*)conv_dw_8_w_flat, (acc_t*)conv_dw_8_b, (elem_t*)conv_dw_8_out,

            RELU, conv_dw_8_params.output_scale, 0,
            conv_dw_8_params.pool_size, 0, conv_dw_8_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;

    printf("conv_dw_8: %llu \n", end - start);
    // conv_9
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_dw_8_out, conv_9_w, conv_9_b, conv_9_out,
            NO_ACTIVATION, conv_9_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_9");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_dw_8_out, conv_9_w, conv_9_b, conv_9_out,
            NO_ACTIVATION, conv_9_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_9");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_9_params.I, conv_9_params.J,
        conv_9_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_6_out,
        conv_9_out,
        conv_9_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_10
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            conv_9_out, conv_10_w, conv_10_b, conv_10_out,
            RELU, conv_10_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_10");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            conv_9_out, conv_10_w, conv_10_b, conv_10_out,
            RELU, conv_10_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_10");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_11
    start = read_cycles();
    if (!conv) {
        conv_dw_with_col2im(conv_10_params.I, conv_10_params.J, conv_dw_11_params.I, conv_dw_11_params.J,
            conv_dw_11_params.batch_size, conv_dw_11_params.in_channels, conv_dw_11_params.out_dim, conv_dw_11_params.kernel_size,
            conv_10_out, conv_dw_11_w, conv_dw_11_b, conv_dw_11_out, &conv_dw_11_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_11_params.batch_size, conv_dw_11_params.in_dim, conv_dw_11_params.in_channels,
            conv_dw_11_params.out_channels, conv_dw_11_params.out_dim,
            conv_dw_11_params.stride, conv_dw_11_params.padding, conv_dw_11_params.kernel_size,

            (elem_t*)conv_10_out, (elem_t*)conv_dw_11_w_flat, (acc_t*)conv_dw_11_b, (elem_t*)conv_dw_11_out,

            RELU, conv_dw_11_params.output_scale, 0,
            conv_dw_11_params.pool_size, 0, conv_dw_11_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_11: %llu \n", end - start);

    // conv_12
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_dw_11_out, conv_12_w, conv_12_b, conv_12_out,
            NO_ACTIVATION, conv_12_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_12");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_dw_11_out, conv_12_w, conv_12_b, conv_12_out,
            NO_ACTIVATION, conv_12_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_12");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_13
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_13_params.I, conv_13_params.J, conv_13_params.K,
            conv_12_out, conv_13_w, conv_13_b, conv_13_out,
            RELU, conv_13_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_13");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_13_params.I, conv_13_params.J, conv_13_params.K,
            conv_12_out, conv_13_w, conv_13_b, conv_13_out,
            RELU, conv_13_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_13");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_14
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_13_params.I, conv_13_params.J, conv_dw_14_params.I, conv_dw_14_params.J,
            conv_dw_14_params.batch_size, conv_dw_14_params.in_channels, conv_dw_14_params.out_dim, conv_dw_14_params.kernel_size,
            conv_13_out, conv_dw_14_w, conv_dw_14_b, conv_dw_14_out, &conv_dw_14_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_14_params.batch_size, conv_dw_14_params.in_dim, conv_dw_14_params.in_channels,
            conv_dw_14_params.out_channels, conv_dw_14_params.out_dim,
            conv_dw_14_params.stride, conv_dw_14_params.padding, conv_dw_14_params.kernel_size,

            (elem_t*)conv_13_out, (elem_t*)conv_dw_14_w_flat, (acc_t*)conv_dw_14_b, (elem_t*)conv_dw_14_out,

            RELU, conv_dw_14_params.output_scale, 0,
            conv_dw_14_params.pool_size, 0, conv_dw_14_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_14: %llu \n", end - start);


    // conv_15
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_15_params.I, conv_15_params.J, conv_15_params.K,
            conv_dw_14_out, conv_15_w, conv_15_b, conv_15_out,
            NO_ACTIVATION, conv_15_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_15");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_15_params.I, conv_15_params.J, conv_15_params.K,
            conv_dw_14_out, conv_15_w, conv_15_b, conv_15_out,
            NO_ACTIVATION, conv_15_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_15");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_15_params.I, conv_15_params.J,
        conv_15_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_12_out,
        conv_15_out,
        conv_15_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_16
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_15_out, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_16");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_15_out, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_16");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_17
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_16_params.I, conv_16_params.J, conv_dw_17_params.I, conv_dw_17_params.J,
            conv_dw_17_params.batch_size, conv_dw_17_params.in_channels, conv_dw_17_params.out_dim, conv_dw_17_params.kernel_size,
            conv_16_out, conv_dw_17_w, conv_dw_17_b, conv_dw_17_out, &conv_dw_17_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_17_params.batch_size, conv_dw_17_params.in_dim, conv_dw_17_params.in_channels,
            conv_dw_17_params.out_channels, conv_dw_17_params.out_dim,
            conv_dw_17_params.stride, conv_dw_17_params.padding, conv_dw_17_params.kernel_size,

            (elem_t*)conv_16_out, (elem_t*)conv_dw_17_w_flat, (acc_t*)conv_dw_17_b, (elem_t*)conv_dw_17_out,

            RELU, conv_dw_17_params.output_scale, 0,
            conv_dw_17_params.pool_size, 0, conv_dw_17_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_17: %llu \n", end - start);


    // conv_18
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_dw_17_out, conv_18_w, conv_18_b, conv_18_out,
            NO_ACTIVATION, conv_18_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_18");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_dw_17_out, conv_18_w, conv_18_b, conv_18_out,
            NO_ACTIVATION, conv_18_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_18");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_18_params.I, conv_18_params.J,
        conv_18_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_15_out,
        conv_18_out,
        conv_18_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_19
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_19");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_19");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_20
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_19_params.I, conv_19_params.J, conv_dw_20_params.I, conv_dw_20_params.J,
            conv_dw_20_params.batch_size, conv_dw_20_params.in_channels, conv_dw_20_params.out_dim, conv_dw_20_params.kernel_size,
            conv_19_out, conv_dw_20_w, conv_dw_20_b, conv_dw_20_out, &conv_dw_20_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_20_params.batch_size, conv_dw_20_params.in_dim, conv_dw_20_params.in_channels,
            conv_dw_20_params.out_channels, conv_dw_20_params.out_dim,
            conv_dw_20_params.stride, conv_dw_20_params.padding, conv_dw_20_params.kernel_size,

            (elem_t*)conv_19_out, (elem_t*)conv_dw_20_w_flat, (acc_t*)conv_dw_20_b, (elem_t*)conv_dw_20_out,

            RELU, conv_dw_20_params.output_scale, 0,
            conv_dw_20_params.pool_size, 0, conv_dw_20_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_20: %llu \n", end - start);


    // conv_21
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_dw_20_out, conv_21_w, conv_21_b, conv_21_out,
            NO_ACTIVATION, conv_21_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_21");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_dw_20_out, conv_21_w, conv_21_b, conv_21_out,
            NO_ACTIVATION, conv_21_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_21");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_22
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_22");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_22");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_23
    start = read_cycles();
    if (!conv) {
        conv_dw_with_col2im(conv_22_params.I, conv_22_params.J, conv_dw_23_params.I, conv_dw_23_params.J,
            conv_dw_23_params.batch_size, conv_dw_23_params.in_channels, conv_dw_23_params.out_dim, conv_dw_23_params.kernel_size,
            conv_22_out, conv_dw_23_w, conv_dw_23_b, conv_dw_23_out, &conv_dw_23_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_23_params.batch_size, conv_dw_23_params.in_dim, conv_dw_23_params.in_channels,
            conv_dw_23_params.out_channels, conv_dw_23_params.out_dim,
            conv_dw_23_params.stride, conv_dw_23_params.padding, conv_dw_23_params.kernel_size,

            (elem_t*)conv_22_out, (elem_t*)conv_dw_23_w_flat, (acc_t*)conv_dw_23_b, (elem_t*)conv_dw_23_out,

            RELU, conv_dw_23_params.output_scale, 0,
            conv_dw_23_params.pool_size, 0, conv_dw_23_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_23: %llu \n", end - start);


    // conv_24
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_dw_23_out, conv_24_w, conv_24_b, conv_24_out,
            NO_ACTIVATION, conv_24_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_24");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_dw_23_out, conv_24_w, conv_24_b, conv_24_out,
            NO_ACTIVATION, conv_24_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_24");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_24_params.I, conv_24_params.J,
        conv_24_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_21_out,
        conv_24_out,
        conv_24_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_25
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_25");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_25");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_26
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_25_params.I, conv_25_params.J, conv_dw_26_params.I, conv_dw_26_params.J,
            conv_dw_26_params.batch_size, conv_dw_26_params.in_channels, conv_dw_26_params.out_dim, conv_dw_26_params.kernel_size,
            conv_25_out, conv_dw_26_w, conv_dw_26_b, conv_dw_26_out, &conv_dw_26_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_26_params.batch_size, conv_dw_26_params.in_dim, conv_dw_26_params.in_channels,
            conv_dw_26_params.out_channels, conv_dw_26_params.out_dim,
            conv_dw_26_params.stride, conv_dw_26_params.padding, conv_dw_26_params.kernel_size,

            (elem_t*)conv_25_out, (elem_t*)conv_dw_26_w_flat, (acc_t*)conv_dw_26_b, (elem_t*)conv_dw_26_out,

            RELU, conv_dw_26_params.output_scale, 0,
            conv_dw_26_params.pool_size, 0, conv_dw_26_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_26: %llu \n", end - start);


    // conv_27
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_dw_26_out, conv_27_w, conv_27_b, conv_27_out,
            NO_ACTIVATION, conv_27_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_27");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_dw_26_out, conv_27_w, conv_27_b, conv_27_out,
            NO_ACTIVATION, conv_27_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_27");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_27_params.I, conv_27_params.J,
        conv_27_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_24_out,
        conv_27_out,
        conv_27_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_28
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_28_params.I, conv_28_params.J, conv_28_params.K,
            conv_27_out, conv_28_w, conv_28_b, conv_28_out,
            RELU, conv_28_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_28");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_28_params.I, conv_28_params.J, conv_28_params.K,
            conv_27_out, conv_28_w, conv_28_b, conv_28_out,
            RELU, conv_28_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_28");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_29
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_28_params.I, conv_28_params.J, conv_dw_29_params.I, conv_dw_29_params.J,
            conv_dw_29_params.batch_size, conv_dw_29_params.in_channels, conv_dw_29_params.out_dim, conv_dw_29_params.kernel_size,
            conv_28_out, conv_dw_29_w, conv_dw_29_b, conv_dw_29_out, &conv_dw_29_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_29_params.batch_size, conv_dw_29_params.in_dim, conv_dw_29_params.in_channels,
            conv_dw_29_params.out_channels, conv_dw_29_params.out_dim,
            conv_dw_29_params.stride, conv_dw_29_params.padding, conv_dw_29_params.kernel_size,

            (elem_t*)conv_28_out, (elem_t*)conv_dw_29_w_flat, (acc_t*)conv_dw_29_b, (elem_t*)conv_dw_29_out,

            RELU, conv_dw_29_params.output_scale, 0,
            conv_dw_29_params.pool_size, 0, conv_dw_29_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_29: %llu \n", end - start);


    // conv_30
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_30_params.I, conv_30_params.J, conv_30_params.K,
            conv_dw_29_out, conv_30_w, conv_30_b, conv_30_out,
            NO_ACTIVATION, conv_30_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_30");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_30_params.I, conv_30_params.J, conv_30_params.K,
            conv_dw_29_out, conv_30_w, conv_30_b, conv_30_out,
            NO_ACTIVATION, conv_30_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_30");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_30_params.I, conv_30_params.J,
        conv_30_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_27_out,
        conv_30_out,
        conv_30_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_31
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            RELU, conv_31_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_31");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            RELU, conv_31_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_31");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_32
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_31_params.I, conv_31_params.J, conv_dw_32_params.I, conv_dw_32_params.J,
            conv_dw_32_params.batch_size, conv_dw_32_params.in_channels, conv_dw_32_params.out_dim, conv_dw_32_params.kernel_size,
            conv_31_out, conv_dw_32_w, conv_dw_32_b, conv_dw_32_out, &conv_dw_32_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_32_params.batch_size, conv_dw_32_params.in_dim, conv_dw_32_params.in_channels,
            conv_dw_32_params.out_channels, conv_dw_32_params.out_dim,
            conv_dw_32_params.stride, conv_dw_32_params.padding, conv_dw_32_params.kernel_size,

            (elem_t*)conv_31_out, (elem_t*)conv_dw_32_w_flat, (acc_t*)conv_dw_32_b, (elem_t*)conv_dw_32_out,

            RELU, conv_dw_32_params.output_scale, 0,
            conv_dw_32_params.pool_size, 0, conv_dw_32_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_32: %llu \n", end - start);


    // conv_33
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_33_params.I, conv_33_params.J, conv_33_params.K,
            conv_dw_32_out, conv_33_w, conv_33_b, conv_33_out,
            NO_ACTIVATION, conv_33_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_33");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_33_params.I, conv_33_params.J, conv_33_params.K,
            conv_dw_32_out, conv_33_w, conv_33_b, conv_33_out,
            NO_ACTIVATION, conv_33_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_33");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_34
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            RELU, conv_34_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_34");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            RELU, conv_34_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_34");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_35
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_34_params.I, conv_34_params.J, conv_dw_35_params.I, conv_dw_35_params.J,
            conv_dw_35_params.batch_size, conv_dw_35_params.in_channels, conv_dw_35_params.out_dim, conv_dw_35_params.kernel_size,
            conv_34_out, conv_dw_35_w, conv_dw_35_b, conv_dw_35_out, &conv_dw_35_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_35_params.batch_size, conv_dw_35_params.in_dim, conv_dw_35_params.in_channels,
            conv_dw_35_params.out_channels, conv_dw_35_params.out_dim,
            conv_dw_35_params.stride, conv_dw_35_params.padding, conv_dw_35_params.kernel_size,

            (elem_t*)conv_34_out, (elem_t*)conv_dw_35_w_flat, (acc_t*)conv_dw_35_b, (elem_t*)conv_dw_35_out,

            RELU, conv_dw_35_params.output_scale, 0,
            conv_dw_35_params.pool_size, 0, conv_dw_35_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_35: %llu \n", end - start);


    // conv_36
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_36_params.I, conv_36_params.J, conv_36_params.K,
            conv_dw_35_out, conv_36_w, conv_36_b, conv_36_out,
            NO_ACTIVATION, conv_36_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_36");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_36_params.I, conv_36_params.J, conv_36_params.K,
            conv_dw_35_out, conv_36_w, conv_36_b, conv_36_out,
            NO_ACTIVATION, conv_36_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_36");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_36_params.I, conv_36_params.J,
        conv_36_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_33_out,
        conv_36_out,
        conv_36_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_37
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            RELU, conv_37_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_37");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            RELU, conv_37_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_37");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_38
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_37_params.I, conv_37_params.J, conv_dw_38_params.I, conv_dw_38_params.J,
        conv_dw_38_params.batch_size, conv_dw_38_params.in_channels, conv_dw_38_params.out_dim, conv_dw_38_params.kernel_size,
        conv_37_out, conv_dw_38_w, conv_dw_38_b, conv_dw_38_out, &conv_dw_38_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_38_params.batch_size, conv_dw_38_params.in_dim, conv_dw_38_params.in_channels,
            conv_dw_38_params.out_channels, conv_dw_38_params.out_dim,
            conv_dw_38_params.stride, conv_dw_38_params.padding, conv_dw_38_params.kernel_size,

            (elem_t*)conv_37_out, (elem_t*)conv_dw_38_w_flat, (acc_t*)conv_dw_38_b, (elem_t*)conv_dw_38_out,

            RELU, conv_dw_38_params.output_scale, 0,
            conv_dw_38_params.pool_size, 0, conv_dw_38_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_38: %llu \n", end - start);


    // conv_39
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_39_params.I, conv_39_params.J, conv_39_params.K,
            conv_dw_38_out, conv_39_w, conv_39_b, conv_39_out,
            NO_ACTIVATION, conv_39_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_39");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_39_params.I, conv_39_params.J, conv_39_params.K,
            conv_dw_38_out, conv_39_w, conv_39_b, conv_39_out,
            NO_ACTIVATION, conv_39_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_39");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_39_params.I, conv_39_params.J,
        conv_39_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_36_out,
        conv_39_out,
        conv_39_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_40
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            RELU, conv_40_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_40");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            RELU, conv_40_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_40");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_41
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_40_params.I, conv_40_params.J, conv_dw_41_params.I, conv_dw_41_params.J,
            conv_dw_41_params.batch_size, conv_dw_41_params.in_channels, conv_dw_41_params.out_dim, conv_dw_41_params.kernel_size,
            conv_40_out, conv_dw_41_w, conv_dw_41_b, conv_dw_41_out, &conv_dw_41_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_41_params.batch_size, conv_dw_41_params.in_dim, conv_dw_41_params.in_channels,
            conv_dw_41_params.out_channels, conv_dw_41_params.out_dim,
            conv_dw_41_params.stride, conv_dw_41_params.padding, conv_dw_41_params.kernel_size,

            (elem_t*)conv_40_out, (elem_t*)conv_dw_41_w_flat, (acc_t*)conv_dw_41_b, (elem_t*)conv_dw_41_out,

            RELU, conv_dw_41_params.output_scale, 0,
            conv_dw_41_params.pool_size, 0, conv_dw_41_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_41: %llu \n", end - start);


    // conv_42
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_42_params.I, conv_42_params.J, conv_42_params.K,
            conv_dw_41_out, conv_42_w, conv_42_b, conv_42_out,
            NO_ACTIVATION, conv_42_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_42");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_42_params.I, conv_42_params.J, conv_42_params.K,
            conv_dw_41_out, conv_42_w, conv_42_b, conv_42_out,
            NO_ACTIVATION, conv_42_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_42");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_43
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            RELU, conv_43_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_43");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            RELU, conv_43_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_43");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_44
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_43_params.I, conv_43_params.J, conv_dw_44_params.I, conv_dw_44_params.J,
            conv_dw_44_params.batch_size, conv_dw_44_params.in_channels, conv_dw_44_params.out_dim, conv_dw_44_params.kernel_size,
            conv_43_out, conv_dw_44_w, conv_dw_44_b, conv_dw_44_out, &conv_dw_44_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_44_params.batch_size, conv_dw_44_params.in_dim, conv_dw_44_params.in_channels,
            conv_dw_44_params.out_channels, conv_dw_44_params.out_dim,
            conv_dw_44_params.stride, conv_dw_44_params.padding, conv_dw_44_params.kernel_size,

            (elem_t*)conv_43_out, (elem_t*)conv_dw_44_w_flat, (acc_t*)conv_dw_44_b, (elem_t*)conv_dw_44_out,

            RELU, conv_dw_44_params.output_scale, 0,
            conv_dw_44_params.pool_size, 0, conv_dw_44_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_44: %llu \n", end - start);


    // conv_45
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_45_params.I, conv_45_params.J, conv_45_params.K,
            conv_dw_44_out, conv_45_w, conv_45_b, conv_45_out,
            NO_ACTIVATION, conv_45_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_45");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_45_params.I, conv_45_params.J, conv_45_params.K,
            conv_dw_44_out, conv_45_w, conv_45_b, conv_45_out,
            NO_ACTIVATION, conv_45_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_45");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_45_params.I, conv_45_params.J,
        conv_45_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_42_out,
        conv_45_out,
        conv_45_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_46
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_45_out, conv_46_w, conv_46_b, conv_46_out,
            RELU, conv_46_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_46");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_45_out, conv_46_w, conv_46_b, conv_46_out,
            RELU, conv_46_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_46");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_47
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_46_params.I, conv_46_params.J, conv_dw_47_params.I, conv_dw_47_params.J,
            conv_dw_47_params.batch_size, conv_dw_47_params.in_channels, conv_dw_47_params.out_dim, conv_dw_47_params.kernel_size,
            conv_46_out, conv_dw_47_w, conv_dw_47_b, conv_dw_47_out, &conv_dw_47_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_47_params.batch_size, conv_dw_47_params.in_dim, conv_dw_47_params.in_channels,
            conv_dw_47_params.out_channels, conv_dw_47_params.out_dim,
            conv_dw_47_params.stride, conv_dw_47_params.padding, conv_dw_47_params.kernel_size,

            (elem_t*)conv_46_out, (elem_t*)conv_dw_47_w_flat, (acc_t*)conv_dw_47_b, (elem_t*)conv_dw_47_out,

            RELU, conv_dw_47_params.output_scale, 0,
            conv_dw_47_params.pool_size, 0, conv_dw_47_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_47: %llu \n", end - start);


    // conv_48
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_dw_47_out, conv_48_w, conv_48_b, conv_48_out,
            NO_ACTIVATION, conv_48_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_48");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_dw_47_out, conv_48_w, conv_48_b, conv_48_out,
            NO_ACTIVATION, conv_48_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_48");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_48_params.I, conv_48_params.J,
        conv_48_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_45_out,
        conv_48_out,
        conv_48_out,
        false,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_49
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_49_params.I, conv_49_params.J, conv_49_params.K,
            conv_48_out, conv_49_w, conv_49_b, conv_49_out,
            RELU, conv_49_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_49");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_49_params.I, conv_49_params.J, conv_49_params.K,
            conv_48_out, conv_49_w, conv_49_b, conv_49_out,
            RELU, conv_49_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_49");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_dw_50
    start = read_cycles();

    if (!conv) {
        conv_dw_with_col2im(conv_49_params.I, conv_49_params.J, conv_dw_50_params.I, conv_dw_50_params.J,
            conv_dw_50_params.batch_size, conv_dw_50_params.in_channels, conv_dw_50_params.out_dim, conv_dw_50_params.kernel_size,
            conv_49_out, conv_dw_50_w, conv_dw_50_b, conv_dw_50_out, &conv_dw_50_params);
    } else {
        tiled_conv_auto_dw(
            conv_dw_50_params.batch_size, conv_dw_50_params.in_dim, conv_dw_50_params.in_channels,
            conv_dw_50_params.out_channels, conv_dw_50_params.out_dim,
            conv_dw_50_params.stride, conv_dw_50_params.padding, conv_dw_50_params.kernel_size,

            (elem_t*)conv_49_out, (elem_t*)conv_dw_50_w_flat, (acc_t*)conv_dw_50_b, (elem_t*)conv_dw_50_out,

            RELU, conv_dw_50_params.output_scale, 0,
            conv_dw_50_params.pool_size, 0, conv_dw_50_params.pool_padding,

            tiled_matmul_type);
    }

    end = read_cycles();
    conv_dw_cycles += end - start;
    printf("conv_dw_50: %llu \n", end - start);


    // conv_51
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_dw_50_out, conv_51_w, conv_51_b, conv_51_out,
            NO_ACTIVATION, conv_51_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_51");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_dw_50_out, conv_51_w, conv_51_b, conv_51_out,
            NO_ACTIVATION, conv_51_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_51");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // conv_52
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_52_params.I, conv_52_params.J, conv_52_params.K,
            conv_51_out, conv_52_w, conv_52_b, conv_52_out,
            RELU, conv_52_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_52");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_52_params.I, conv_52_params.J, conv_52_params.K,
            conv_51_out, conv_52_w, conv_52_b, conv_52_out,
            RELU, conv_52_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_52");

        end = read_cycles();
        matmul_cycles += end - start;
    }

    // Global averaging
    static elem_t average[1280][4] row_align(1);

    start = read_cycles();

    for (int batch = 0; batch < conv_52_params.batch_size; batch++) {
        for (int channel = 0; channel < conv_52_params.out_channels; channel++) {
            int sum = 0;
            for (int row = 0; row < conv_52_params.out_dim; row++) {
                for (int col = 0; col < conv_52_params.out_dim; col++) {
                    size_t r = batch * conv_52_params.out_dim * conv_52_params.out_dim + row * conv_52_params.out_dim + col;

                    sum += conv_52_out[r][channel];
                }
            }
            const int count = conv_52_params.out_dim * conv_52_params.out_dim;

            average[channel][batch] = (sum + count/2) / count;
        }
    }

    end = read_cycles();
    other_cycles += end - start;

    // fc_53
    start = read_cycles();

    tiled_matmul_nn_auto(fc_53_params.I, fc_53_params.J, fc_53_params.K,
        fc_53_w, average, fc_53_b, fc_53_out,
        NO_ACTIVATION, fc_53_params.output_scale, 0, false,
        tiled_matmul_type, check, "fc_53");

    end = read_cycles();
    matmul_cycles += end - start;

    // Find highest probs
    int preds[fc_53_params.batch_size];
    for (int batch = 0; batch < fc_53_params.batch_size; batch++) {
        elem_t max_prob = fc_53_out[0][batch];
        size_t max_idx = 0;

        for (int i = 1; i < fc_53_params.out_features; i++) {
            if (fc_53_out[i][batch] > max_prob) {
                max_prob = fc_53_out[i][batch];
                max_idx = i;
            }
        }

        preds[batch] = max_idx;
        printf("Prediction: %u (score: %d)\n", max_idx, max_prob);
    }

    uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_cycles + conv_dw_cycles + res_add_cycles + other_cycles;

    printf("\nTotal cycles: %llu (100%%)\n", total_cycles);
    printf("Matmul cycles: %llu (%d%%)\n", matmul_cycles, (matmul_cycles * 100) / total_cycles);
    printf("Im2col cycles: %llu (%d%%)\n", im2col_cycles, (im2col_cycles * 100) / total_cycles);
    printf("Conv cycles: %llu (%d%%)\n", conv_cycles, (conv_cycles * 100) / total_cycles);
    printf("Pooling cycles: %llu (%d%%)\n", pool_cycles, (pool_cycles * 100) / total_cycles);
    printf("Depthwise convolution cycles: %llu (%d%%)\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
    printf("Res add cycles: %llu (%d%%)\n", res_add_cycles, (res_add_cycles * 100) / total_cycles);
    printf("Other cycles: %llu (%d%%)\n", other_cycles, (other_cycles * 100) / total_cycles);

    int correct[] = {75, 900, 125, 897};
    for (int i = 0; i < fc_53_params.batch_size; i++) {
        if (preds[i] != correct[i] && fc_53_out[preds[i]][i] != fc_53_out[correct[i]][i]) {
            printf("Prediction %d is incorrect! Actual class has score of %d\nFAIL\n", i+1, fc_53_out[correct[i]][i]);
            exit(1);
        }
    }

    printf("PASS\n");
    exit(0);
}

