# Register tile and vector kittens types
from .kittens.tk_types import *

# Register-to-register copy/conversion instructions
from .kittens.tk_register_copy import *

# Register tile non-trivial operations
from .kittens.tk_tile_ops import *

# RMEM/SMEM instructions
from .kittens.tk_shared_to_register import *

# RMEM/GMEM instructions
from .kittens.tk_global_to_register import *

# SMEM/GMEM instructions
from .kittens.tk_global_to_shared import *

# TODO MMA template, support transpose f16/bf16, and wgmma A
# Also support tf32 for SMEM only
# TODO TMA template
