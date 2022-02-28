from __future__ import annotations

from exo import *
from exo.builtins import *
from exo.platforms.x86 import *
from exo.syntax import *


@proc
def conv(
        out_h: size, out_w: size, out_channel: size, in_h: size, in_w: size,
        in_channel: size, kernel_dim: size, batch_size: size,
        inp: f32[batch_size, in_h, in_w, in_channel],
        output: f32[batch_size, out_h, out_w, out_channel],
        weights: f32[in_channel, kernel_dim, kernel_dim, out_channel],
        bias: f32[out_channel],
):
    assert out_h == in_h - kernel_dim + 1
    assert out_w == in_w - kernel_dim + 1

    for oc in par(0, out_channel):
        for n in par(0, batch_size):
            for oy in par(0, out_h):
                for ox in par(0, out_w):
                    res: f32
                    res = bias[oc]
                    for ky in par(0, kernel_dim):
                        for kx in par(0, kernel_dim):
                            for kc in par(0, in_channel):
                                # todo: add padding, stride
                                res += (weights[kc, ky, kx, oc] *
                                        inp[n, oy + ky, ox + kx, kc])

                    relu_v: f32
                    relu_v = relu(res)
                    output[n, oy, ox, oc] = relu_v


VEC_W = 16
H, W, C, K, N = 80, 100, 128, 3, 5
TILE_W, TILE_H = 4, 5

conv_specialized = (
    conv
        .rename('conv_specialized')
        .partial_eval(H, W, C, H + 2, W + 2, C, K, N)
        .simplify()
        #
        .split('oc', TILE_W * VEC_W, ['oc_o', 'oc_i'], perfect=True)
        .reorder('oc_i', 'n')
        .reorder('oc_i', 'oy')
        .reorder('oc_i', 'ox')
        #
        .split('ox', TILE_H, ['ox_o', 'ox_i'], perfect=True)
        #
        .split('oc_i', VEC_W, ['oc_u', 'oc_v'], perfect=True)
        #
        .lift_alloc('res: _', n_lifts=3)
        .set_memory('res', AVX512)
        .fission_after('res[_] = bias[_]', n_lifts=3)
        .replace(mm512_loadu_ps, 'for oc_v in _: _ #0')
        #
        .fission_after('for ky in _: _', n_lifts=3)
        #
        .lift_alloc('relu_v: _')
        .set_memory('relu_v', AVX512)
        .fission_after('relu_v = _')
        .replace(mm512_storeu_ps, 'for oc_v in _: _ #2')
        #
        .reorder('ox_i', 'oc_u')
        .reorder('ox_i', 'oc_v')
        .reorder('ox_i', 'ky')
        .reorder('ox_i', 'kx')
        #
        .reorder('oc_v', 'ky')
        .reorder('oc_v', 'kx')
        .reorder('oc_u', 'ky')
        .reorder('oc_u', 'kx')
        #
        .reorder('ox_i', 'kc')
        .reorder('oc_v', 'kc')
        .reorder('oc_u', 'kc')
        #
        .reorder('oc_v', 'ox_i')
        .reorder('oc_u', 'ox_i')
        #
        .stage_expr('wt_vec', 'weights[_]', memory=AVX512)
        .stage_expr('in_vec', 'inp[_]', memory=AVX512)
        #
        .replace_all(mm512_relu_ps)
        .replace_all(mm512_set1_ps)
        .replace_all(mm512_fmadd_ps)
        .replace_all(mm512_loadu_ps)
        #
        .split('kc', 2, ['kc_o', 'kc_i'], perfect=True)
        #
        .simplify()
)

if __name__ == '__main__':
    print(conv_specialized)

__all__ = ['conv_specialized']
