import numpy as np

from ..mode_ctrl.printing import print_params
from ...ndarray_tools import *
from ..bit_level_primitives import *


def dets_to_strs(encoding: NDArray[2, np.uint8],
                 bitlen: int
                ) -> tuple[list[str], bool]:
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
    det_strs, overflow = dets_to_strs(encoding, bitlen)
    det_coeff_strs = [f"|{ds}>  *  {c}" for (ds, c) in zip(det_strs, coeffs)]
    return det_coeff_strs, overflow