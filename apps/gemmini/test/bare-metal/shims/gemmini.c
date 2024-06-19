#include <include/gemmini.h>

#include "gemmini_lib.h"

void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
    const elem_t *A, const elem_t *B, const void *D, void *C, size_t stride_A,
    size_t stride_B, size_t stride_D, size_t stride_C, scale_t A_scale_factor,
    scale_t B_scale_factor, scale_acc_t D_scale_factor, int act,
    acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias,
    bool transpose_A, bool transpose_B, bool full_C, bool low_D,
    uint8_t weightA, enum tiled_matmul_type_t tiled_matmul_type) {

  float c_scale = (float)scale;
  bool act_ = (bool)act;

  // gemm_init_mem();
  // gemm_acc_init_mem();
  // gemmini_flush(0);
  // gemmini_fence();

  if (dim_I == 512 && dim_J == 512 && dim_K == 512) {
    // 512x512x512
    printf("Calling 512 matmul\n");
    gemmini_lib_Context *ctxt;
    matmul_512x512x512(ctxt, &c_scale, act_, A, B, C);
  } else if (dim_I == 12544 && dim_J == 256 && dim_K == 64) {
    // matmul_4
    printf("Calling matmul_4\n");
    gemmini_lib_Context *ctxt;
    matmul_4(ctxt, &c_scale, act_, A, B, C);
  } else if (dim_I == 12544 && dim_J == 64 && dim_K == 256) {
    // matmul_6
    printf("Calling matmul_6\n");
    gemmini_lib_Context *ctxt;
    matmul_6(ctxt, &c_scale, act_, A, B, C);
  } else if (dim_I == 3136 && dim_J == 512 && dim_K == 128) {
    // matmul_14
    printf("Calling matmul_14\n");
    gemmini_lib_Context *ctxt;
    matmul_14(ctxt, &c_scale, act_, A, B, C);
  } else if (dim_I == 3136 && dim_J == 128 && dim_K == 512) {
    // matmul_16
    printf("Calling matmul_16\n");
    gemmini_lib_Context *ctxt;
    matmul_16(ctxt, &c_scale, act_, A, B, C);
  } else if (dim_I == 784 && dim_J == 1024 && dim_K == 256) {
    // matmul_27
    printf("Calling matmul_27\n");
    gemmini_lib_Context *ctxt;
    matmul_27(ctxt, &c_scale, act_, A, B, C);
  } else {
    printf("Calling original matmul auto\n");
    orig_tiled_matmul_auto(dim_I, dim_J, dim_K, A, B, D, C, stride_A, stride_B,
        stride_D, stride_C, A_scale_factor, B_scale_factor, D_scale_factor, act,
        scale, bert_scale, repeating_bias, transpose_A, transpose_B, full_C,
        low_D, weightA, tiled_matmul_type);
  }
}

void tiled_conv_auto(int batch_size, int in_row_dim, int in_col_dim,
    int in_channels, int out_channels, int out_row_dim, int out_col_dim,
    int stride, int input_dilation, int kernel_dilation, int padding,
    int kernel_dim, bool wrot180, bool trans_output_1203, bool trans_input_3120,
    bool trans_weight_1203, bool trans_weight_0132,

    const elem_t *input, const elem_t *weights, const acc_t *bias,
    elem_t *output,

    int act, acc_scale_t scale, int pool_size, int pool_stride,
    int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type) {

  if (input_dilation != 1) {
    printf("input_dilation should be 1\n");
    exit(1);
  }
  if (kernel_dilation != 1) {
    printf("kernel_dilation should be 1\n");
    exit(1);
  }
  if (wrot180 || trans_output_1203 || trans_input_3120 || trans_weight_1203 ||
      trans_weight_0132) {
    printf("transpose should not happen in inference\n");
    exit(1);
  }
  if (out_row_dim != out_col_dim || in_row_dim != in_col_dim) {
    printf("Row dimensions should be the same!\n");
    exit(1);
  }

  /*
  //printf("pool_size: %d\n", pool_size);
  //printf("pool_stride: %d\n", pool_stride);
  //printf("pool_padding: %d\n", pool_padding);
  //printf("act: %d\n", act);
  //printf("scale: %d\n", (int)scale);

  if (padding < 0 || padding >= 16) {
  printf("padding should be 0 to 15!\n");
  exit(1);
  }
  if (padding >= out_dim) {
  printf("padding should be less than out_dim!\n");
  exit(1);
  }
  if (padding != 1) {
  printf("padding should be 1!\n");
  exit(1);
  }
  */

  // gemm_init_mem();
  // gemm_acc_init_mem();
  // gemmini_flush(0);
  // gemmini_fence();
  float c_scale = (float)scale;
  bool act_ = (bool)act;

  if (out_row_dim == 56 & out_channels == 64 & stride == 1) {
    printf("calling conv_3\n");
    gemmini_lib_Context *ctxt;
    conv_3(ctxt, output, bias, input, weights, act_, &c_scale);
  } else if (out_row_dim == 28 & out_channels == 128 & stride == 1) {
    printf("calling conv_17\n");
    gemmini_lib_Context *ctxt;
    conv_17(ctxt, output, bias, input, weights, act_, &c_scale);
  } else if (out_row_dim == 14 & out_channels == 256 & stride == 1) {
    printf("calling conv_30\n");
    gemmini_lib_Context *ctxt;
    conv_30(ctxt, output, bias, input, weights, act_, &c_scale);
  } else {
    printf("Calling original conv auto\n");
    orig_tiled_conv_auto(batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim, stride, input_dilation,
        kernel_dilation, padding, kernel_dim, wrot180, trans_output_1203,
        trans_input_3120, trans_weight_1203, trans_weight_0132, input, weights,
        bias, output, act, scale, pool_size, pool_stride, pool_padding,
        tiled_conv_type);
  }
}
