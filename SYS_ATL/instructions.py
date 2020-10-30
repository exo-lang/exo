
from .prelude import *

#from . import shared_types as T
#from .LoopIR import LoopIR

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# List of valid memories

MEM_NAMES = {
    "GEMM",
    "HEAP",
}

def is_valid_mem(x):
    return x in MEM_NAMES
