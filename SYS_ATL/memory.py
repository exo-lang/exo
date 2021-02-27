from .prelude import *
#from .LoopIR import UAST, LoopIR, front_ops, bin_ops, LoopIR_Rewrite
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

class Memory:
    def __init__(self, name,
        alloc   = None,
        free    = None,
        read    = None,
        write   = None,
        reduce  = None,
    ):
        self._name = name
        self._alloc = alloc




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


A memory is
