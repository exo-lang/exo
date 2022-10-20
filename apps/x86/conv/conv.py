from __future__ import annotations

from exo import *
from exo.builtins import *
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *


@proc
def conv(
    out_h: size,
    out_w: size,
    out_channel: size,
    in_h: size,
    in_w: size,
    in_channel: size,
    kernel_dim: size,
    batch_size: size,
    inp: f32[batch_size, in_h, in_w, in_channel],
    output: f32[batch_size, out_h, out_w, out_channel],
    weights: f32[in_channel, kernel_dim, kernel_dim, out_channel],
    bias: f32[out_channel],
):
    assert out_h == in_h - kernel_dim + 1
    assert out_w == in_w - kernel_dim + 1

    for oc in seq(0, out_channel):
        for n in seq(0, batch_size):
            for oy in seq(0, out_h):
                for ox in seq(0, out_w):
                    res: f32
                    res = bias[oc]
                    for ky in seq(0, kernel_dim):
                        for kx in seq(0, kernel_dim):
                            for kc in seq(0, in_channel):
                                # todo: add padding, stride
                                res += (
                                    weights[kc, ky, kx, oc]
                                    * inp[n, oy + ky, ox + kx, kc]
                                )

                    relu_v: f32
                    relu_v = relu(res)
                    output[n, oy, ox, oc] = relu_v


VEC_W = 16
H, W, C, K, N = 80, 100, 128, 3, 5
TILE_W, TILE_H = 4, 5


def do_specialization(p):
    p = rename(p, "conv_specialized")
    p = p.partial_eval(H, W, C, H + 2, W + 2, C, K, N)
    p = simplify(p)
    #
    p = divide_loop(p, "oc", TILE_W * VEC_W, ["oc_o", "oc_i"], perfect=True)
    p = reorder_loops(p, "oc_i n")
    p = reorder_loops(p, "oc_i oy")
    p = reorder_loops(p, "oc_i ox")
    #
    p = divide_loop(p, "ox", TILE_H, ["ox_o", "ox_i"], perfect=True)
    #
    p = divide_loop(p, "oc_i", VEC_W, ["oc_u", "oc_v"], perfect=True)
    #
    p = autolift_alloc(p, "res: _", n_lifts=3, keep_dims=True)
    p = set_memory(p, "res", AVX512)
    p = fission(p, p.find("res[_] = bias[_]").after(), n_lifts=3)
    p = replace(p, "for oc_v in _: _ #0", mm512_loadu_ps)
    #
    p = fission(p, p.find("for ky in _: _").after(), n_lifts=3)
    #
    p = autolift_alloc(p, "relu_v: _", keep_dims=True)
    p = set_memory(p, "relu_v", AVX512)
    p = fission(p, p.find("relu_v = _").after())
    p = replace(p, "for oc_v in _: _ #2", mm512_storeu_ps)
    #
    p = repeat(reorder_loops)(p, "ox_i oc_u")
    p = reorder_loops(p, "ox_i oc_v")
    p = reorder_loops(p, "ox_i ky")
    p = reorder_loops(p, "ox_i kx")
    #
    p = reorder_loops(p, "oc_v ky")
    p = reorder_loops(p, "oc_v kx")
    p = reorder_loops(p, "oc_u ky")
    p = reorder_loops(p, "oc_u kx")
    #
    p = reorder_loops(p, "ox_i kc")
    p = reorder_loops(p, "oc_v kc")
    p = reorder_loops(p, "oc_u kc")
    #
    p = reorder_loops(p, "oc_v ox_i")
    p = repeat(reorder_loops)(p, "oc_u ox_i")
    #
    def stage_input(p, read_expr, name):
        p = bind_expr(p, read_expr, name)
        p = expand_dim(p, name, 16, "oc_v")
        p = lift_alloc(p, name)
        p = set_memory(p, name, AVX512)
        p = fission(p, p.find(f"{name}[_] = _").after())
        return p

    p = stage_input(p, "weights[_]", "wt_vec")
    p = stage_input(p, "inp[_]", "in_vec")
    #
    p = replace_all(p, mm512_relu_ps)
    p = replace_all(p, mm512_set1_ps)
    p = replace_all(p, mm512_fmadd_ps)
    p = replace_all(p, mm512_loadu_ps)
    #
    p = divide_loop(p, "kc", 2, ["kc_o", "kc_i"], perfect=True)
    #
    p = simplify(p)
    return p


conv_specialized = do_specialization(conv)

if __name__ == "__main__":
    print(conv_specialized)

__all__ = ["conv_specialized"]
