#include "Halide.h"

namespace {

using namespace Halide;

class ConvolutionLayer : public Halide::Generator<ConvolutionLayer> {
public:
  Input<Buffer<float>> input{"input", 4};
  Input<Buffer<float>> filter{"filter", 4};
  Input<Buffer<float>> bias{"bias", 1};

  Output<Buffer<float>> relu{"relu", 4};

  void generate() {
    const int N = 5, CI = 128, CO = 128, W = 100, H = 80;

    /* THE ALGORITHM */

    Var x("x"), y("y"), c("c"), n("n");

    Func conv("conv");
    RDom r({{0, CI}, {0, 3}, {0, 3}}, "k");

    RVar kc = r.x;
    RVar kx = r.y;
    RVar ky = r.z;

    conv(c, x, y, n) = bias(c);
    conv(c, x, y, n) += filter(c, kx, ky, kc) * input(kc, x + kx, y + ky, n);

    relu(c, x, y, n) = max(0, conv(c, x, y, n));

    /* THE SCHEDULE */

    // MKL JITs code for the specific size and strides, so we'll
    // do the same and ask Halide to compile for this specific
    // size:

    relu.dim(0).set_bounds(0, CO).set_stride(1);
    relu.dim(1).set_bounds(0, W).set_stride(CO);
    relu.dim(2).set_bounds(0, H).set_stride(CO * W);
    relu.dim(3).set_bounds(0, N).set_stride(CO * H * W);

    input.dim(0).set_bounds(0, CI).set_stride(1);
    input.dim(1).set_bounds(0, W + 2).set_stride(CI);
    input.dim(2).set_bounds(0, H + 2).set_stride(CI * (W + 2));
    input.dim(3).set_bounds(0, N).set_stride(CI * (W + 2) * (H + 2));

    filter.dim(0).set_bounds(0, CO).set_stride(1);
    filter.dim(1).set_bounds(0, 3).set_stride(CO);
    filter.dim(2).set_bounds(0, 3).set_stride(CO * 3);
    filter.dim(3).set_bounds(0, CI).set_stride(CO * 3 * 3);

    bias.dim(0).set_bounds(0, CO).set_stride(1);

    // 4.06ms on an Intel i9-9960X using 16 threads at 3.0 GHz,
    // which is 94.5% of peak flops assuming the math below is correct:

    // 16 cores times 2 FMAs per cycle times 3G cycles per
    // second times 16 vector lanes is a peak throughput of
    // 1.536 TFlops.

    // This conv does N * CI * CO * W * H * 3 * 3 = 5 * 128 *
    // 128 * 100 * 80 * 3 * 3 FMAs in 4.06ms is 1.453 TFlops.

    // The ratio of actual to theoretical flops hit is 0.9458

    int tile_w = 4;
    int tile_h = 5;
    const int vec = natural_vector_size<float>();

    Var co{"co"}, ci{"ci"}, xo{"xo"}, xi{"xi"}, yo{"yo"}, yi{"yi"}, t{"t"};
    relu.split(c, co, ci, vec * tile_w)
        .split(x, xo, xi, tile_h)
        .reorder(ci, xi, xo, y, n, co)
        .vectorize(ci, vec)
        .unroll(ci)
        .unroll(xi)
        .parallel(y)
        .parallel(n)
        .parallel(co);
    conv.compute_at(relu, xo)
        .vectorize(c, vec)
        .unroll(c)
        .unroll(x)
        .unroll(y)
        .update()
        .reorder(c, x, y, kc, kx, ky, n)
        .vectorize(c, vec)
        .unroll(c)
        .unroll(x)
        .unroll(y)
        .unroll(kc, 2);
    filter.in().compute_at(conv, kc).vectorize(_0, vec).unroll(_0).unroll(_3);
    input.in().compute_at(conv, x).unroll(_0);
  }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ConvolutionLayer, halide_conv_kernel)
