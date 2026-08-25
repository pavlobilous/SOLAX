"""
type_hinting: a shorthand type-hint alias used throughout SOLAX to
annotate NumPy arrays by dimensionality and element type.
"""
import numpy as np


class NDArray:
    """
    A type-hint-only helper: NDArray[ndims, dtype] is shorthand for
    np.ndarray[ndims, dtype] (e.g. NDArray[2, np.uint8] for a 2D array
    of bytes). Not an actual array class -- only usable in the
    subscript position of a type annotation.
    """

    @classmethod
    def __class_getitem__(cls, s):
        ndims, tps = s                
        return np.ndarray[ndims, tps]