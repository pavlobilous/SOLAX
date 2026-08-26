"""
Manipulations with indices.
"""

import numpy as np
from numbers import Integral
from collections.abc import Sequence

    
def int_to_slice(length, i):
    """
    Converts a single (possibly negative) integer index "i" into a
    length-1 slice(i, i + 1) that selects the same element from a
    sequence of length "length", normalizing negative "i" as Python
    indexing does. This keeps single-index access uniform with
    slice/array indexing for classes whose __getitem__ always returns a
    new instance rather than a bare element. Raises IndexError if "i" is
    out of range for "length".
    """
    if (i >= length) or (i < -length):
        raise IndexError("Index out of range.")
    if i < 0:
        i += length
    return slice(i, i + 1)


def make_1d_index(length, s):
    """
    Normalizes an index/selector "s" for a sequence of length "length"
    into a form directly usable to index a NumPy array along its first
    axis, for use in __getitem__ implementations. Accepts:

        - a slice, returned unchanged;
        - a single integer (Integral), converted via int_to_slice();
        - a Sequence (e.g. a list/tuple of ints or booleans), converted
            to a NumPy array;
        - a 1D NumPy array (fancy/boolean indexing), returned as is.

    Raises TypeError if "s" is none of the above (or a NumPy array of
    dimension other than 1).
    """
    if isinstance(s, slice):
        return s
    if isinstance(s, Integral):
        return int_to_slice(length, s)
    if isinstance(s, Sequence):
        s = np.array(s)
    if isinstance(s, np.ndarray):
        if s.ndim == 1:
            return s
    raise TypeError("Cannot create a 1D index from this.")