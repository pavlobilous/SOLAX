import numpy as np

from ...det_based_classes import *
from ....ndarray_tools import *


def eval_mat_elems(op,
                   basis_rows: Basis,
                   basis_cols: Basis,
                   *,
                   det_batch_size: int | None = None,
                   op_batch_size: int | None = None,
                   multiple_devices: bool = True):
    
    state_cols = State(basis_cols, np.ones(len(basis_cols)))
    
    res_state, track = op(state_cols,
                          det_batch_size=det_batch_size,
                          op_batch_size=op_batch_size,
                          multiple_devices=multiple_devices,
                          det_tracking=True)
    
    pdi_res = create_byte_pdindex(res_state.basis._encoding)
    pdi_rows = create_byte_pdindex(basis_rows._encoding)
    index_res_by_rows = pdi_rows.get_indexer(pdi_res)
    mask_res_in_rows = (index_res_by_rows >= 0)

    mat_rows = index_res_by_rows[mask_res_in_rows]
    mat_cols = track[mask_res_in_rows]
    mat_vals = res_state.coeffs[mask_res_in_rows]
    
    coords = np.vstack([mat_rows, mat_cols]).T
    coords, mat_vals = squeeze_array(coords, mat_vals)
    
    return coords, mat_vals