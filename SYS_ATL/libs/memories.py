import os

from SYS_ATL.memory import Memory, DRAMBase
from pathlib import Path

from memory import MemGenError


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

class MDRAMBase(DRAMBase):
    @property
    def global_(self):
        _here_ = os.path.dirname(os.path.abspath(__file__))
        return _configure_file(Path(_here_) / 'malloc.c', heap_size=100000)

    def alloc(self, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return f"{prim_type} {new_name};"

        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        return (f"{prim_type} *{new_name} = "
                f"({prim_type}*) malloc_dram ({size_str} * sizeof({prim_type}));")

    def free(self, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""

        return f"free_dram({new_name});"


MDRAM = MDRAMBase()


# ----------- GEMMINI scratchpad ----------------

class GEMM_SCRATCH_BASE(Memory):
    @property
    def global_(self):
        _here_ = os.path.dirname(os.path.abspath(__file__))
        return _configure_file(Path(_here_) / 'gemm_malloc.c',
                               heap_size=100000,
                               dim=16)

    def alloc(self, new_name, prim_type, shape, srcinfo):
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

    def free(self, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""

        return f"gemm_free((uint64_t)({new_name}));"

    def window(self, basetyp, baseptr, idx_expr, indices, strides, srcinfo):
        # assume that strides[-1] == 1
        #    and that strides[-2] == 16 (if there is a strides[-2])
        assert len(indices) == len(strides) and len(strides) >= 2
        prim_type = basetyp.basetype().ctype()
        offset = " + ".join(
            [f"({i}) * ({s})" for i, s in zip(indices, strides)])
        return (f"({prim_type}*)((uint64_t)( "
                f"((uint32_t)((uint64_t){baseptr})) + "
                f"({offset})/16 ))")

    @property
    def can_read(self):
        return False

    @property
    def can_write(self):
        return False

    @property
    def can_reduce(self):
        return False


GEMM_SCRATCH = GEMM_SCRATCH_BASE()


# ----------- GEMMINI accumulator scratchpad ----------------


class GEMM_ACCUM_BASE(Memory):
    @property
    def global_(self):
        _here_ = os.path.dirname(os.path.abspath(__file__))
        return _configure_file(Path(_here_) / 'gemm_acc_malloc.c',
                               heap_size=100000,
                               dim=16)

    def alloc(self, new_name, prim_type, shape, srcinfo):
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

    def free(self, new_name, prim_type, shape, srcinfo):
        if len(shape) == 0:
            return ""
        return f"gemm_acc_free((uint32_t)({new_name}));"

    def window(self, basetyp, baseptr, idx_expr, indices, strides, srcinfo):
        # assume that strides[-1] == 1
        #    and that strides[-2] == 16 (if there is a strides[-2])
        assert len(indices) == len(strides) and len(strides) >= 2
        prim_type = basetyp.basetype().ctype()
        offset = " + ".join([f"({i}) * ({s})"
                             for i, s in zip(indices, strides)])
        return (f"({prim_type}*)((uint64_t)( "
                f"((uint32_t)((uint64_t){baseptr})) + "
                f"({offset})/16 ))")

    @property
    def can_read(self):
        return False

    @property
    def can_write(self):
        return False

    @property
    def can_reduce(self):
        return False


GEMM_ACCUM = GEMM_ACCUM_BASE()


# ----------- AVX2 registers ----------------

class AVX2Base(Memory):
    @property
    def global_(self):
        return '#include <immintrin.h>'

    def alloc(self, new_name, prim_type, shape, srcinfo):
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

    def free(self, new_name, prim_type, shape, srcinfo):
        return ''

    def window(self, basetyp, baseptr, idx_expr, indices, strides, srcinfo):
        assert strides[-1] == '1'
        prim_type = basetyp.basetype().ctype()
        return f'({prim_type}*)&{baseptr}[{"][".join(indices[:-1])}]'

    @property
    def can_read(self):
        return False

    @property
    def can_write(self):
        return False

    @property
    def can_reduce(self):
        return False


AVX2 = AVX2Base()


# ----------- AVX-512 registers ----------------

class AVX512Base(Memory):
    @property
    def global_(self):
        return '#include <immintrin.h>'

    def alloc(self, new_name, prim_type, shape, srcinfo):
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

    def free(self, new_name, prim_type, shape, srcinfo):
        return ''

    def window(self, basetyp, baseptr, idx_expr, indices, strides, srcinfo):
        assert strides[-1] == '1'
        prim_type = basetyp.basetype().ctype()
        idxs = indices[:-1] if len(indices) > 1 else ["0"]
        return f'({prim_type}*)&{baseptr}[{"][".join(idxs)}]'

    @property
    def can_read(self):
        return False

    @property
    def can_write(self):
        return False

    @property
    def can_reduce(self):
        return False


AVX512 = AVX512Base()


# ----------- AMX tile! ----------------

class AMX_TILE_BASE(Memory):
    @property
    def global_(self):
        return '#include <immintrin.h>'

    def alloc(self, new_name, prim_type, shape, srcinfo):
        return f"{prim_type} *{new_name} = ({prim_type}*) 0;"

    def free(self, new_name, prim_type, shape, srcinfo):
        return ""

    @property
    def can_read(self):
        return False

    @property
    def can_write(self):
        return False

    @property
    def can_reduce(self):
        return False


AMX_TILE = AMX_TILE_BASE()
