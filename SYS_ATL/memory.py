from .prelude import *
from . import shared_types as T

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Helper Functions


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory Objects

"""
#   A memory consists of
#     - a global C-code block
#     - C-code to initialize the memory
#     - C-code to:
#           * allocate from the memory
#           * free an allocation
#           * read from the memory (optional)
#           * write to the memory (optional)
#           * reduce to the memory (optional)
"""

# This MetaData will be compiled into
# struct GEMM_req_metadata {
class MetaData:
    def __init__(self,
        row_size,
        col_size,
        buf_type,
        buf_bits,
    ):
    
    self.row_size = row_size
    self.col_size = col_size
    self.buf_type = buf_type
    self.buf_bits = buf_bits


class Memory:
    def __init__(self, name,
        metadata= None,
        alloc   = None,
        free    = None,
        read    = None,
        write   = None,
        red     = None,
    ):

    self._name  = name
    self._meta  = metadata
    self._alloc = alloc
    self._free  = free
    self._read  = read
    self._write = write
    self._reduce= red

# TODO: How to encode size??
# Or assume that alloc/free code must not change by types?
# How to handle different types? e.g. float, double,..
#
#if s.type is T.R:
#    self.add_line(f"float {name};")
#else:
#    size = _type_size(s.type, self.env)
#    self.add_line(f"float *{name} = " +
#            f"(float*) malloc ({size} * sizeof(float));")
# Check this in typecheck.py, isinstance(Memory)
# We should at least create TL_float

# Write some compiler-ish code here?
def dram_alloc(x) :
    if type(x) is TL_float:
        return ("float {x};")
    elif type(x) is TL_double:
        return ("double {x};")
def dram_free(x):
    if type(x) is TL_float:
        return ("")
    else:
        return ("free {x};")
DRAM = Memory("DRAM",
        alloc = dram_alloc,
        free = dram_free,
        ...
    )


"""
Example Memory:

Python Code written to Specify the memory

GEMM = Memory("GEMM",
    alloc = ""
)



GENERATED code

struct GEMM_req_metadata {
    // encoding of SYS_TL buffer shape
    // element type of buffer in bytes
};

// returns 0xFFFFFFFF (to full bitwidth) on allocation error
void * GEMM_alloc( req_metadata req ) {
   {{alloc}}
}

void GEMM_free( void *buf, req_metadata req ) {
   {{free}}
}

"""
