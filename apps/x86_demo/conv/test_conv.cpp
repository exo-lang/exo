// largely adapted from
// https://oneapi-src.github.io/oneDNN/v2/convolution_example_cpp.html licensed
// by Intel Corporation, 2020 under Apache 2.0

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "conv.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

inline dnnl::memory::dim product(const dnnl::memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1,
                           std::multiplies<>());
}

// Read from memory, write to handle
inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) {
        throw std::runtime_error("handle is nullptr.");
    }

    assert(eng.get_kind() == dnnl::engine::kind::cpu);

    auto src = static_cast<uint8_t *>(mem.get_data_handle());
    if (!src) {
        throw std::runtime_error("get_data_handle returned nullptr.");
    }

    for (size_t i = 0; i < size; ++i) {
        ((uint8_t *)handle)[i] = src[i];
    }
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) {
        throw std::runtime_error("handle is nullptr.");
    }

    assert(eng.get_kind() == dnnl::engine::kind::cpu);

    auto dst = static_cast<uint8_t *>(mem.get_data_handle());
    if (!dst) {
        throw std::runtime_error("get_data_handle returned nullptr.");
    }

    for (size_t i = 0; i < size; ++i) {
        dst[i] = ((uint8_t *)handle)[i];
    }
}

struct conv_instance {
    const memory::dim N = 4;  // batch size
    const memory::dim IC,     // input channels
        IH,                   // input height
        IW,                   // input width
        OC,                   // output channels
        KH,                   // weights height
        KW,                   // weights width
        PH_L,                 // height padding: left (top)
        PH_R,                 // height padding: right (bottom)
        PW_L,                 // width padding: left
        PW_R,                 // width padding: right
        SH,                   // height-wise stride
        SW,                   // width-wise stride
        OH,                   // output height
        OW;                   // output width

    const memory::dims src_dims;
    const memory::dims weights_dims;
    const memory::dims bias_dims;
    const memory::dims dst_dims;

    const memory::dims strides_dims;
    const memory::dims padding_dims_l;
    const memory::dims padding_dims_r;

    std::vector<float> src_data;
    std::vector<float> weights_data;
    std::vector<float> bias_data;
    std::vector<float> dst_data;

    conv_instance(memory::dim in_dim, memory::dim in_chan, memory::dim out_chan,
                  memory::dim kern_sz, memory::dim pad, memory::dim stride)
        : IC(in_chan),
          IH(in_dim),
          IW(in_dim),
          OC(out_chan),
          KH(kern_sz),
          KW(kern_sz),
          PH_L(pad),
          PH_R(pad),
          PW_L(pad),
          PW_R(pad),
          SH(stride),
          SW(stride),
          OH((IH - KH + PH_L + PH_R) / SH + 1),
          OW((IW - KW + PW_L + PW_R) / SW + 1),
          src_dims({N, IC, IH, IW}),
          weights_dims({OC, IC, KH, KW}),
          bias_dims({OC}),
          dst_dims({N, OC, OH, OW}),
          strides_dims({SH, SW}),
          padding_dims_l({PH_L, PW_L}),
          padding_dims_r({PH_R, PW_R}),
          src_data(product(src_dims)),
          weights_data(product(weights_dims)),
          bias_data(product(bias_dims)),
          dst_data(product(dst_dims)) {
        // Initialize src, weights, and dst tensors.
        std::generate(src_data.begin(), src_data.end(),
                      [i = 0.0f]() mutable { return std::cos(i++ / 10.f); });

        std::generate(weights_data.begin(), weights_data.end(),
                      [i = 0.0f]() mutable { return std::sin(i++ * 2.f); });

        std::generate(bias_data.begin(), bias_data.end(),
                      [i = 0.0f]() mutable { return std::tanh(i++); });
    }
};

class OneDNN_Conv {
    conv_instance &ci;
    dnnl::engine engine{engine::kind::cpu, 0};
    dnnl::stream engine_stream{engine};
    memory user_dst_mem;
    memory conv_dst_mem;
    convolution_forward::primitive_desc conv_pd;
    std::unordered_map<int, memory> conv_args;
    dnnl::convolution_forward conv_prim;

   public:
    OneDNN_Conv(conv_instance &ci) : ci(ci) {
        // Create memory objects for tensor data (src, weights, dst). In this
        // example, NHWC layout is assumed for src and dst, and HWIO for
        // weights.
        auto user_src_mem = memory({ci.src_dims, dt::f32, tag::nhwc}, engine);
        auto user_weights_mem =
            memory({ci.weights_dims, dt::f32, tag::hwio}, engine);
        user_dst_mem = memory({ci.dst_dims, dt::f32, tag::nhwc}, engine);

        // Create memory descriptors with format_tag::any for the primitive.
        // This enables the convolution primitive to choose memory layouts for
        // an optimized primitive implementation, and these layouts may differ
        // from the ones provided by the user.
        auto conv_src_md = memory::desc(ci.src_dims, dt::f32, tag::any);
        auto conv_weights_md = memory::desc(ci.weights_dims, dt::f32, tag::any);
        auto conv_dst_md = memory::desc(ci.dst_dims, dt::f32, tag::any);

        // Create memory descriptor and memory object for input bias.
        auto user_bias_md = memory::desc(ci.bias_dims, dt::f32, tag::a);
        auto user_bias_mem = memory(user_bias_md, engine);

        // Write data to memory object's handle.
        write_to_dnnl_memory(ci.src_data.data(), user_src_mem);
        write_to_dnnl_memory(ci.weights_data.data(), user_weights_mem);
        write_to_dnnl_memory(ci.bias_data.data(), user_bias_mem);

        // Create operation descriptor.
        auto conv_desc = convolution_forward::desc(
            prop_kind::forward_training, algorithm::convolution_direct,
            conv_src_md, conv_weights_md, user_bias_md, conv_dst_md,
            ci.strides_dims, ci.padding_dims_l, ci.padding_dims_r);

        // Create primitive post-ops (ReLU).
        const float scale = 1.f;
        const float alpha = 0.f;
        const float beta = 0.f;
        post_ops conv_ops;
        conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
        primitive_attr conv_attr;
        conv_attr.set_post_ops(conv_ops);

        // Create primitive descriptor.
        conv_pd =
            convolution_forward::primitive_desc(conv_desc, conv_attr, engine);

        // For now, assume that the src, weights, and dst memory layouts
        // generated by the primitive and the ones provided by the user are
        // identical.
        auto conv_src_mem = user_src_mem;
        auto conv_weights_mem = user_weights_mem;
        conv_dst_mem = user_dst_mem;

        // Reorder the data in case the src and weights memory layouts generated
        // by the primitive and the ones provided by the user are different. In
        // this case, we create additional memory objects with internal buffers
        // that will contain the reordered data. The data in dst will be
        // reordered after the convolution computation has finalized.
        if (conv_pd.src_desc() != user_src_mem.get_desc()) {
            conv_src_mem = memory(conv_pd.src_desc(), engine);
            reorder(user_src_mem, conv_src_mem)
                .execute(engine_stream, user_src_mem, conv_src_mem);
        }

        if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
            conv_weights_mem = memory(conv_pd.weights_desc(), engine);
            reorder(user_weights_mem, conv_weights_mem)
                .execute(engine_stream, user_weights_mem, conv_weights_mem);
        }

        if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
            conv_dst_mem = memory(conv_pd.dst_desc(), engine);
        }

        // Create the primitive.
        conv_prim = convolution_forward(conv_pd);

        // Primitive arguments.
        conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
        conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
        conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
        conv_args.insert({DNNL_ARG_DST, conv_dst_mem});
    }

    void run() {
        // Primitive execution: convolution with ReLU.
        conv_prim.execute(engine_stream, conv_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Reorder the data in case the dst memory descriptor generated by the
        // primitive and the one provided by the user are different.
        if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
            reorder(conv_dst_mem, user_dst_mem)
                .execute(engine_stream, conv_dst_mem, user_dst_mem);
        } else {
            user_dst_mem = conv_dst_mem;
        }

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        read_from_dnnl_memory(ci.dst_data.data(), user_dst_mem);
    }
};

void conv_SYS_ATL(conv_instance &ci) {
    assert(ci.IW == ci.IH);
    assert(ci.OW == ci.OH);
    assert(ci.KW == ci.KH);

    float scale = 1.0f;
    constexpr int batch_size = 4;

    conv(nullptr, (int)ci.OW, (int)ci.OC, (int)ci.KW, (int)ci.IC, (int)ci.IW,
         &scale, batch_size, ci.src_data.data(), ci.dst_data.data(),
         ci.weights_data.data(), ci.bias_data.data());
}

int main() {
    conv_instance ci_onednn{56, 64, 64, 3, 0, 1};
    conv_instance ci_sys_atl{56, 64, 64, 3, 0, 1};

    OneDNN_Conv reference{ci_onednn};
    reference.run();

    conv_SYS_ATL(ci_sys_atl);

    if (ci_onednn.dst_data.size() != ci_sys_atl.dst_data.size()) {
        fprintf(stderr, "Sizes do not match!\n");
        return 1;
    }

    auto n = ci_onednn.dst_data.size();
    for (int i = 0; i < n; ++i) {
        double expected = ci_onednn.dst_data[i];
        double actual = ci_sys_atl.dst_data[i];
        double relerr = fabs((actual - expected) / expected);
        if (relerr > 1e-3) {
            fprintf(stderr,
                    "Bad value at index %d - relative error = %.6f - actual = "
                    "%.6f - "
                    "expected = %.6f\n",
                    i, relerr, actual, expected);
        }
    }
}
