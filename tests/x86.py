import sys

sys.path.append(sys.path[0] + "/..")
sys.path.append(sys.path[0] + "/.")

from SYS_ATL import instr, DRAM


# --------------------------------------------------------------------------- #
#   Instructions
# --------------------------------------------------------------------------- #

@instr('''
_mm256_storeu_ps(
  {dst}.data, 
  _mm256_fmadd_ps(
    _mm256_loadu_ps({src1}.data), 
    _mm256_loadu_ps({src2}.data), 
    _mm256_loadu_ps({dst}.data)
  )
);
''')
def fma(
    dst: [f32][8] @ DRAM,
    src1: [f32][8] @ DRAM,
    src2: [f32][8] @ DRAM,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] += src1[i] * src2[i]


@instr('_mm256_storeu_ps({dst}.data, _mm256_broadcast_ss({value}));')
def broadcast(
    value: f32,
    dst: [f32][8] @ DRAM,
):
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] = value
