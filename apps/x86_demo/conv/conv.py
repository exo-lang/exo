from __future__ import annotations

from exo import *
from exo.builtins import *
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *



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

def do_specialization(p):
    p = rename(p, 'conv_specialized')
#conv_specialized = (
#    conv
#        .rename('conv_specialized')
    p = p.partial_eval(H, W, C, H + 2, W + 2, C, K, N)
    p = simplify(p)
        #
    p = p.split('oc', TILE_W * VEC_W, ['oc_o', 'oc_i'], perfect=True)
    p = p.reorder('oc_i', 'n')
    p = p.reorder('oc_i', 'oy')
    p = p.reorder('oc_i', 'ox')
        #
    p = p.split('ox', TILE_H, ['ox_o', 'ox_i'], perfect=True)
        #
    p = p.split('oc_i', VEC_W, ['oc_u', 'oc_v'], perfect=True)
        #
    p = p.lift_alloc('res: _', n_lifts=3)
    p = set_memory(p, 'res', AVX512)
    p = old_fission_after(p, 'res[_] = bias[_]', n_lifts=3)
    p = p.replace(mm512_loadu_ps, 'for oc_v in _: _ #0')
        #
    p = old_fission_after(p, 'for ky in _: _', n_lifts=3)
        #
    p = p.lift_alloc('relu_v: _')
    p = set_memory(p, 'relu_v', AVX512)
    p = old_fission_after(p, 'relu_v = _')
    p = p.replace(mm512_storeu_ps, 'for oc_v in _: _ #2')
        #
    p = p.reorder('ox_i', 'oc_u')
    p = p.reorder('ox_i', 'oc_v')
    p = p.reorder('ox_i', 'ky')
    p = p.reorder('ox_i', 'kx')
        #
    p = p.reorder('oc_v', 'ky')
    p = p.reorder('oc_v', 'kx')
    p = p.reorder('oc_u', 'ky')
    p = p.reorder('oc_u', 'kx')
        #
    p = p.reorder('ox_i', 'kc')
    p = p.reorder('oc_v', 'kc')
    p = p.reorder('oc_u', 'kc')
        #
    p = p.reorder('oc_v', 'ox_i')
    p = p.reorder('oc_u', 'ox_i')
        #
    p = p.stage_expr('wt_vec', 'weights[_]', memory=AVX512)
    p = p.stage_expr('in_vec', 'inp[_]', memory=AVX512)
        #
    p = p.replace_all(mm512_relu_ps)
    p = p.replace_all(mm512_set1_ps)
    p = p.replace_all(mm512_fmadd_ps)
    p = p.replace_all(mm512_loadu_ps)
        #
    p = p.split('kc', 2, ['kc_o', 'kc_i'], perfect=True)
        #
    p = simplify(p)
    return p

conv_specialized = do_specialization(conv)

if __name__ == '__main__':
    print(conv_specialized)

__all__ = ['conv_specialized']
