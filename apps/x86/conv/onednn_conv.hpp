#ifndef ONEDNN_CONV_H
#define ONEDNN_CONV_H

#include <oneapi/dnnl/dnnl.hpp>

#include "conv_instance.hpp"

class OneDNN_Conv {
  conv_instance &ci;

  dnnl::engine engine{dnnl::engine::kind::cpu, 0};
  dnnl::stream engine_stream{engine};
  dnnl::memory user_dst_mem;
  dnnl::memory conv_dst_mem;
  dnnl::convolution_forward::primitive_desc conv_pd;
  std::unordered_map<int, dnnl::memory> conv_args;
  dnnl::convolution_forward conv_prim;

public:
  OneDNN_Conv(conv_instance &ci);
  void run();
};

#endif
