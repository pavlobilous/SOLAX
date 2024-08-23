"""
Manipulations with indices.
"""

import numpy as np
from numbers import Integral
from collections.abc import Sequence

    
def int_to_slice(l, i):
    if (i >= l) or (i < -l):
        raise IndexError("Index out of range.")
    if i < 0:
        i += l
    return slice(i, i + 1)


def make_1d_index(l, s):
    if isinstance(s, slice):
        return s
    if isinstance(s, Integral):
        return int_to_slice(l, s)
    if isinstance(s, Sequence):
        s = np.array(s)
    if isinstance(s, np.ndarray):
        if s.ndim == 1:
            return s
    raise TypeError("Cannot create a 1D index from this.")