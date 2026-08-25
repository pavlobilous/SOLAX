"""
Helpers turning packed determinant encodings (and, optionally, their
coefficients) into human-readable strings, for Basis.__str__/State.__str__.
"""
import numpy as np

from ..mode_ctrl.printing import print_params
from ...ndarray_tools import *
from ..bit_level_primitives import *


def dets_to_strs(encoding: NDArray[2, np.uint8],
                 bitlen: int
                ) -> tuple[list[str], bool]:
    """
    Decodes a packed determinant encoding into one "0"/"1" bitstring per
    determinant, e.g. "1100", stopping after the current
    print_params["DETS_PRINTING_LIMIT"] (see dets_printing_limit()) if
    it is not None.
    Input:
        - "encoding": packed determinant bytes, as stored in
            Basis._encoding.
        - "bitlen": number of spin-orbitals (bits) per determinant.
    Output:
        Tuple (det_strs, overflow), where "det_strs" is the list of
        bitstrings actually produced and "overflow" is True if
        "encoding" held more determinants than were rendered.
    """
    lns_np = det_to_bits(
                    encoding[:print_params["DETS_PRINTING_LIMIT"]],
                    bitlen,
                    module=np
                )
    det_strs = [
        "".join([str(c) for c in ln_np])
        for ln_np in lns_np
    ]
    overflow = (print_params["DETS_PRINTING_LIMIT"] is not None) \
            and (len(encoding) > print_params["DETS_PRINTING_LIMIT"])
    return det_strs, overflow


def dets_with_coeffs_to_strs(encoding: NDArray[2, np.uint8],
                             bitlen: int,
                             coeffs: NDArray[1, np.uint8]
                             ) -> tuple[list[str], bool]:
    """
    Like dets_to_strs(), but pairs each rendered determinant with its
    coefficient, formatted as e.g. "|1100>  *  (1+0j)".
    Input:
        - "encoding"/"bitlen": as in dets_to_strs().
        - "coeffs": 1D array of coefficients, one per determinant in
            "encoding" (only as many as are actually rendered are used).
    Output:
        Tuple (det_coeff_strs, overflow), as in dets_to_strs().
    """
    det_strs, overflow = dets_to_strs(encoding, bitlen)
    det_coeff_strs = [f"|{ds}>  *  {c}" for (ds, c) in zip(det_strs, coeffs)]
    return det_coeff_strs, overflow