# fmt: off
"""Top-level module for Exo-GPU wrappers of ThunderKittens

We have focused on wrapping tile ops and tile-to-vec reductions.
Vector support is limited but could be added.

For register (RMEM) storage, we wrap the ThunderKittens tile and vec types
as Exo-GPU Memory classes. Use CudaTkWarpTile(rows, cols, layout="row")
and its row_vec and col_vec attributes. These are allocated at warp scope.

For SMEM and GMEM, we continue to use Exo-GPU's low-level memory classes,
for easy inter-operation with non-ThunderKittens code:

    CudaGmemLinear, for GMEM
    Sm90_SmemSwizzled(swizzle), for SMEM tiles (usually swizzle=128)
    CudaSmemLinear, for SMEM vectors

The instructions have to do a bunch of C++ casts to make this work.

WGMMA and TMA are not exposed here. Use the Sm90 module.
Sm90_tk_ instructions use ThunderKittens RMEM tiles. TMA is completely independent.
In particular, we do NOT use kittens::GL and its automatic tensorMap dict.
This is a smart abstraction for C++, but too high-level to adapt cleanly to Exo.

"""

# fmt: on

# Register tile and vector kittens types
from .kittens_impl.tk_types import *

# Register-to-register tile copy, conversion, maps, reduction-to-vector
from .kittens_impl.tk_tile_ops import *

# RMEM/SMEM vec copies
# Note, we currently don't support GMEM vectors.
from .kittens_impl.tk_shared_to_register_vec import *

# RMEM/SMEM tile copies
from .kittens_impl.tk_shared_to_register import *

# RMEM/GMEM tile copies
from .kittens_impl.tk_global_to_register import *

# SMEM/GMEM tile copies
from .kittens_impl.tk_global_to_shared import *

from .cuda_fwd import CudaSmemLinear
from .Sm90 import Sm90_SmemSwizzled
