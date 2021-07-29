from SYS_ATL import Memory
import os


def _is_const_size(sz, c):
    return sz.isdecimal() and int(sz) == c


# ----------- DRAM using custom malloc ----------------

def _mdram_alloc(new_name, prim_type, shape, srcinfo):
    if len(shape) == 0:
        return (f"{prim_type} {new_name};")
    else:
        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        return (f"{prim_type} *{new_name} = " +
                f"({prim_type}*) malloc_dram ({size_str} * sizeof({prim_type}));")


def _mdram_free(new_name, prim_type, shape, srcinfo):
    if len(shape) == 0:
        return ""
    else:
        return f"free_dram({new_name});"


def _mdram_globl():
    _here_ = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(_here_, 'malloc.c'), 'r') as fp:
        line = fp.readline()
        malloc = line.format(heap_size=100000)
        while line:
            line = fp.readline()
            malloc += line

    return malloc


MDRAM = Memory(
    "MDRAM",
    globl=_mdram_globl(),
    alloc=_mdram_alloc,
    free=_mdram_free,
    read=True,
    write=True,
    red=True
)


# ----------- GEMMINI scratchpad ----------------

def _gemm_alloc(new_name, prim_type, shape, srcinfo):
    if len(shape) == 0:
        return (f"{prim_type} {new_name};")
    else:
        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        if not _is_const_size(shape[-1], 16):
            raise MemGenError(f"{srcinfo}: " +
                              "Cannot allocate GEMMINI Scratchpad Memory " +
                              "unless innermost dimension is exactly 16.  " +
                              f"got {shape[-1]}")
        return (f"{prim_type} *{new_name} = " +
                f"({prim_type}*) ((uint64_t)gemm_malloc ({size_str} * sizeof({prim_type})));")


def _gemm_free(new_name, prim_type, shape, srcinfo):
    if len(shape) == 0:
        return ""
    else:
        return f"gemm_free((uint64_t)({new_name}));"


def _gemm_window(prim_type, baseptr, indices, strides, srcinfo):
    # assume that strides[-1] == 1
    #    and that strides[-2] == 16 (if there is a strides[-2])
    assert len(indices) == len(strides) and len(strides) >= 2
    offset = " + ".join([f"({i}) * ({s})" for i, s in zip(indices, strides)])
    return (f"({prim_type}*)((uint64_t)( " +
            f"((uint32_t)((uint64_t){baseptr})) + " +
            f"({offset})/16 ))")


def _gemm_global():
    _here_ = os.path.dirname(os.path.abspath(__file__))

    malloc = []
    with open(os.path.join(_here_, 'gemm_malloc.c'), 'r') as fp:
        line = fp.readline()
        malloc.append(line.format(heap_size=100000))
        line = fp.readline()
        malloc.append(line.format(dim=16))
        while line:
            line = fp.readline()
            malloc.append(line)

    return ''.join(malloc)


GEMM_SCRATCH = Memory(
    "GEMM_SCRATCH",
    globl=_gemm_global(),
    alloc=_gemm_alloc,
    free=_gemm_free,
    window=_gemm_window,
    read=False,
    write=False,
    red=False,
)


# ----------- GEMMINI accumulator scratchpad ----------------

def _gemm_accum_alloc(new_name, prim_type, shape, srcinfo):
    if len(shape) == 0:
        return (f"{prim_type} {new_name};")
    else:
        size_str = shape[0]
        for s in shape[1:]:
            size_str = f"{s} * {size_str}"
        if not _is_const_size(shape[-1], 16):
            raise MemGenError(f"{srcinfo}: " +
                              "Cannot allocate GEMMINI Accumulator Memory " +
                              "unless innermost dimension is exactly 16.  " +
                              f"got {shape[-1]}")
        return (f"{prim_type} *{new_name} = " +
                f"({prim_type}*) ((uint32_t)gemm_acc_malloc ({size_str} * sizeof({prim_type})));")


def _gemm_accum_free(new_name, prim_type, shape, srcinfo):
    if len(shape) == 0:
        return ""
    else:
        return f"gemm_acc_free((uint32_t)({new_name}));"


def _gemm_accum_window(prim_type, baseptr, indices, strides, srcinfo):
    # assume that strides[-1] == 1
    #    and that strides[-2] == 16 (if there is a strides[-2])
    assert len(indices) == len(strides) and len(strides) >= 2
    offset = " + ".join([f"({i}) * ({s})" for i, s in zip(indices, strides)])
    return (f"({prim_type}*)((uint64_t)( " +
            f"((uint32_t)((uint64_t){baseptr})) + " +
            f"({offset})/16 ))")


def _gemm_accum_global():
    _here_ = os.path.dirname(os.path.abspath(__file__))

    malloc = []
    with open(os.path.join(_here_, 'gemm_acc_malloc.c'), 'r') as fp:
        line = fp.readline()
        malloc.append(line.format(heap_size=100000))
        line = fp.readline()
        malloc.append(line.format(dim=16))
        while line:
            line = fp.readline()
            malloc.append(line)

    return ''.join(malloc)


GEMM_ACCUM = Memory(
    "GEMM_ACCUM",
    globl=_gemm_accum_global(),
    alloc=_gemm_accum_alloc,
    free=_gemm_accum_free,
    window=_gemm_accum_window,
    read=False,
    write=False,
    red=False,
)
