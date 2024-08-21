import numpy as np

what_is_posits = '"posits" must be a 2D NumPy array of integers indicating '\
                            'for each ladder operator product (=length) '\
                            'positions where this product acts (=width).'

what_is_coeffs = '"coeffs" must be a 1D NumPy numeric array indicating '\
                            'a coefficient in front of the corresponding '\
                            'ladder operator product.'

def cleanup_input(daggers, posits, coeffs):
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