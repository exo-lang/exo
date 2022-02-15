import os
from pathlib import Path

from ..memory import Memory, DRAM, MemGenError


def _is_const_size(sz, c):
    return sz.isdecimal() and int(sz) == c


def _configure_file(path: Path, **kwargs):
    def transform(lines):
        for line in lines:
            if line.startswith('#define'):
                line = line.format(**kwargs)
            yield line

    return '\n'.join(transform(path.read_text().split('\n')))


# ----------- DRAM using custom malloc ----------------

class MDRAM(DRAM):
    @classmethod
    def global_(cls):
        _here_ = os.path.dirname(os.path.abspath(__file__))
        return _configure_file(Path(_here_) / 'malloc.c', heap_size=100000)

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"

        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        return (f"{prim_type} *{new_name} = "
                f"({prim_type}*) malloc_dram ({size_str} * sizeof({prim_type}));")

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""

        return f"free_dram({new_name});"


# ----------- DRAM using static memory ----------------

class DRAM_STATIC(DRAM):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # Error checking only
        for extent in shape:
            try:
                int(extent)
            except ValueError as e:
                raise MemGenError(f'DRAM_STATIC requires constant shapes. '
                                  f'Saw: {extent}') from e

        return f'static {prim_type} {new_name}[{" * ".join(shape)}];'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ''


# ----------- GEMMINI scratchpad ----------------

class GEMM_SCRATCH(Memory):
    @classmethod
    def global_(cls):
        _here_ = os.path.dirname(os.path.abspath(__file__))
        return _configure_file(Path(_here_) / 'gemm_malloc.c',
                               heap_size=100000,
                               dim=16)

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"

        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        if not _is_const_size(shape[-1], 16):
            raise MemGenError(f"{srcinfo}: "
                              "Cannot allocate GEMMINI Scratchpad Memory "
                              "unless innermost dimension is exactly 16.  "
                              f"got {shape[-1]}")
        return (f"{prim_type} *{new_name} = "
                f"({prim_type}*) ((uint64_t)gemm_malloc ({size_str} * sizeof({prim_type})));")

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""

        return f"gemm_free((uint64_t)({new_name}));"

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        # assume that strides[-1] == 1
        #    and that strides[-2] == 16 (if there is a strides[-2])
        assert len(indices) == len(strides) and len(strides) >= 2
        prim_type = basetyp.basetype().ctype()
        offset = " + ".join(
            [f"({i}) * ({s})" for i, s in zip(indices, strides)])
        return (f"*({prim_type}*)((uint64_t)( "
                f"((uint32_t)((uint64_t){baseptr})) + "
                f"({offset})/16))")


# ----------- GEMMINI accumulator scratchpad ----------------


class GEMM_ACCUM(Memory):
    @classmethod
    def global_(cls):
        _here_ = os.path.dirname(os.path.abspath(__file__))
        return _configure_file(Path(_here_) / 'gemm_acc_malloc.c',
                               heap_size=100000,
                               dim=16)

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"

        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        if not _is_const_size(shape[-1], 16):
            raise MemGenError(f"{srcinfo}: "
                              "Cannot allocate GEMMINI Accumulator Memory "
                              "unless innermost dimension is exactly 16.  "
                              f"got {shape[-1]}")
        return (f"{prim_type} *{new_name} = "
                f"({prim_type}*) ((uint32_t)gemm_acc_malloc ({size_str} * sizeof({prim_type})));")

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""
        return f"gemm_acc_free((uint32_t)({new_name}));"

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        # assume that strides[-1] == 1
        #    and that strides[-2] == 16 (if there is a strides[-2])
        assert len(indices) == len(strides) and len(strides) >= 2
        prim_type = basetyp.basetype().ctype()
        offset = " + ".join([f"({i}) * ({s})"
                             for i, s in zip(indices, strides)])
        return (f"*({prim_type}*)((uint64_t)( "
                f"((uint32_t)((uint64_t){baseptr})) + "
                f"({offset})/16))")


# ----------- AVX2 registers ----------------

class AVX2(Memory):
    @classmethod
    def global_(cls):
        return '#include <immintrin.h>'

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f'{srcinfo}: AVX2 vectors are not scalar values')
        if not prim_type == 'float':
            raise MemGenError(f'{srcinfo}: AVX2 vectors must be f32 (for now)')
        if not _is_const_size(shape[-1], 8):
            raise MemGenError(f'{srcinfo}: AVX2 vectors must be 8-wide')
        shape = shape[:-1]
        if shape:
            result = f'__m256 {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f'__m256 {new_name};'
        return result

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ''

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        assert strides[-1] == '1'
        idxs = indices[:-1] or ''
        if idxs:
            idxs = '[' + ']['.join(idxs) + ']'
        return f'{baseptr}{idxs}'


# ----------- AVX-512 registers ----------------

class AVX512(Memory):
    @classmethod
    def global_(cls):
        return '#include <immintrin.h>'

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(
                f'{srcinfo}: AVX512 vectors are not scalar values')
        if not prim_type == 'float':
            raise MemGenError(
                f'{srcinfo}: AVX512 vectors must be f32 (for now)')
        if not _is_const_size(shape[-1], 16):
            raise MemGenError(f'{srcinfo}: AVX512 vectors must be 16-wide')
        shape = shape[:-1]
        if shape:
            result = f'__m512 {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f'__m512 {new_name};'
        return result

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ''

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        assert strides[-1] == '1'
        idxs = indices[:-1] or ''
        if idxs:
            idxs = '[' + ']['.join(idxs) + ']'
        return f'{baseptr}{idxs}'


# ----------- AMX tile! ----------------

num_amx_tiles_alloced = 0
class AMX_TILE(Memory):
    @classmethod
    def global_(cls):
        return '#include <immintrin.h>'

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        global num_amx_tiles_alloced
        num_amx_tiles_alloced += 1
        return f"#define {new_name} {num_amx_tiles_alloced-1}"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        global num_amx_tiles_alloced
        num_amx_tiles_alloced -= 1
        return f"#undef {new_name}"
