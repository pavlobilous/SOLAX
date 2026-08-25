"""
ndarray_tools: low-level NumPy array helpers used across quantum_core --
byte-packed hashing/indexing of determinant rows via pandas.Index
(pandas_boost) and the NDArray[ndims, dtype] type-hint shorthand
(type_hinting).
"""
from .type_hinting import NDArray

from .pandas_boost import (
    create_byte_pdindex,
    sum_by_indexer,
    squeeze_array,
    array_is_squeezed,
    array_difference_bmask
)