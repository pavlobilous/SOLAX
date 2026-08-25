"""
Validation and normalization of OperatorTerm's constructor arguments.
"""
import numpy as np

what_is_posits = '"posits" must be a 2D NumPy array of integers indicating '\
                            'for each ladder operator product (=length) '\
                            'positions where this product acts (=width).'

what_is_coeffs = '"coeffs" must be a 1D NumPy numeric array indicating '\
                            'a coefficient in front of the corresponding '\
                            'ladder operator product.'

def cleanup_input(daggers, posits, coeffs):
    """
    Validates and normalizes the ("daggers", "posits", "coeffs")
    arguments of OperatorTerm.__init__/__post_init__.
    Input:
        - "daggers": any iterable of 0/1, coerced to a tuple; must be
            non-empty and contain only 0s and 1s.
        - "posits": a 2D NumPy integer array, one row of spin-orbital
            positions per ladder-operator product, its width matching
            len(daggers); must contain only non-negative integers. A
            width-0 array (no rows worth of positions, i.e. an empty
            second axis) is reshaped to (0, len(daggers)) so an empty
            OperatorTerm can still be built.
        - "coeffs": a 1D NumPy numeric array of per-row coefficients
            (or a scalar/0D array, promoted to 1D via
            np.atleast_1d), one entry per row of "posits".
    Raises TypeError/ValueError with a descriptive message if any of
    the above is violated, including length mismatches between
    "daggers"/"posits" or "posits"/"coeffs".
    Output:
        The normalized (daggers, posits, coeffs) tuple.
    """
    daggers = tuple(daggers)
    if not daggers:
        raise ValueError('"daggers" must have positive length.')
    if  set(daggers) - {0, 1}:
        raise ValueError('"daggers" can only be encoded by 0 and 1.')
    if not isinstance(posits, np.ndarray):
        raise TypeError(what_is_posits)
    if posits.shape[-1] == 0:
        posits = posits.astype(np.int_)
        posits = posits.reshape(0, len(daggers))      
    if posits.ndim != 2:
        raise TypeError(what_is_posits)
    if (posits < 0).any():
        raise ValueError('"posits" must contain only non-negative integer numbers.')
    if not isinstance(coeffs, np.ndarray) or coeffs.ndim != 1:
        raise TypeError(what_is_coeffs)
    if len(daggers) != posits.shape[-1]:
        raise ValueError('"daggers" must be of the same length as the width of the "posits" array.')
    coeffs = np.atleast_1d(coeffs)
    if len(coeffs) != len(posits):
        raise ValueError('"coeffs" must be of the same length as the length of the "posits" array.')
    return daggers, posits, coeffs