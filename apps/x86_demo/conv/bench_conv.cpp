// largely adapted from
// https://oneapi-src.github.io/oneDNN/v2/convolution_example_cpp.html licensed
// by Intel Corporation, 2020 under Apache 2.0

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

#include "exo_conv.hpp"
#include "halide_conv.hpp"
#include "onednn_conv.hpp"

static long num_fmas(conv_instance &ci) {
  return ci.N * ci.OH * ci.OW * ci.OC * ci.KH * ci.KW * ci.IC;
}

void conv_oneDNN(benchmark::State &state) {
  const long                        // Benchmark inputs.
      batch_size = state.range(0),  // e.g. 4
      in_h = state.range(1),        // e.g. 224
      in_w = state.range(2),        // e.g. 224
      in_chan = state.range(3),     // e.g. 3
      out_chan = state.range(4),    // e.g. 64
      kern_sz = state.range(5),     // e.g. 7
      pad = state.range(6),         // e.g. 3
      stride = state.range(7);      // e.g. 2

  conv_instance ci{
      batch_size, in_h, in_w, in_chan, out_chan, kern_sz, pad, stride};

  OneDNN_Conv reference{ci};
  for ([[maybe_unused]] auto _ : state) {
    reference.run();
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * num_fmas(ci),  //
      benchmark::Counter::kIsRate,                             //
      benchmark::Counter::kIs1000                              //
  );
}

BENCHMARK(conv_oneDNN)  // N in-dim in-chan out-chan kern-dim pad str
    ->Args({4, 224, 224, 3, 64, 7, 3, 2})            // conv1
    ->Args({4, 56, 56, 64, 64, 3, 1, 1})             // conv3/7/10
    ->Args({4, 28, 28, 128, 128, 3, 1, 2})           // conv13
    ->Args({4, 56, 56, 64, 64, 3, 0, 1})             // test size
    ->Args({5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1})  // Halide size
    ;

void conv_exo(benchmark::State &state) {
  const long                        // Benchmark inputs.
      batch_size = state.range(0),  // e.g. 4
      in_h = state.range(1),        // e.g. 224
      in_w = state.range(2),        // e.g. 224
      in_chan = state.range(3),     // e.g. 3
      out_chan = state.range(4),    // e.g. 64
      kern_sz = state.range(5),     // e.g. 7
      pad = state.range(6),         // e.g. 3
      stride = state.range(7);      // e.g. 2

  conv_instance ci{
      batch_size, in_h, in_w, in_chan, out_chan, kern_sz, pad, stride};

  for ([[maybe_unused]] auto _ : state) {
    exo_conv(ci);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * num_fmas(ci),  //
      benchmark::Counter::kIsRate,                             //
      benchmark::Counter::kIs1000                              //
  );
}

BENCHMARK(conv_exo)  // N in-dim in-chan out-chan kern-dim pad str
                     //    ->Args({4, 56, 56, 64, 64, 3, 0, 1}) // test size
    ->Args({5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1})  // Halide size
    ;

void conv_Halide(benchmark::State &state) {
  const long                        // Benchmark inputs.
      batch_size = state.range(0),  // e.g. 4
      in_h = state.range(1),        // e.g. 224
      in_w = state.range(2),        // e.g. 224
      in_chan = state.range(3),     // e.g. 3
      out_chan = state.range(4),    // e.g. 64
      kern_sz = state.range(5),     // e.g. 7
      pad = state.range(6),         // e.g. 3
      stride = state.range(7);      // e.g. 2

  conv_instance ci{
      batch_size, in_h, in_w, in_chan, out_chan, kern_sz, pad, stride};

  for ([[maybe_unused]] auto _ : state) {
    halide_conv(ci);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * num_fmas(ci),  //
      benchmark::Counter::kIsRate,                             //
      benchmark::Counter::kIs1000                              //
  );
}

BENCHMARK(conv_Halide)  // N in-dim in-chan out-chan kern-dim pad str
    ->Args({5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1})  // Halide size
    ;
