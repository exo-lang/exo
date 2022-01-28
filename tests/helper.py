import numpy as np


def nparray(arg, typ=np.float32):
    return np.array(arg, dtype=typ)


__all__ = [
    'nparray',
]
