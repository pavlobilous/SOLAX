from numbers import Number
import numpy as np

from ...det_based_classes import *
from ....ndarray_tools import *
from ..operator_term.act_in_batches import *


def concat_batches(batches_lst):
    if len(batches_lst) > 1:
        return np.concatenate(batches_lst)
    if len(batches_lst) == 1:
        return batches_lst[0]
    else:
        return None


def act_with_scal_on_basis(basis, scal):
    enc_batch = basis._encoding
    cfs_batch = np.full(len(basis), scal)
    track_batch = np.arange(len(basis))
    yield enc_batch, cfs_batch, track_batch
    

def eval_mat_elems(term,
                   basis_rows: Basis,
                   basis_cols: Basis,
                   *,
                   det_batch_size: int | None = None,
                   op_batch_size: int | None = None,
                   multiple_devices: bool = True):

    if isinstance(term, Number):
        g = act_with_scal_on_basis(basis_cols, term)
    else:
        state_coeffs = np.ones(len(basis_cols))
        g = act_in_batches_generator(basis_cols, state_coeffs, term,
                     det_batch_size, op_batch_size,
                     multiple_devices,
                     det_tracking=True
                )

    coords_list = []
    mat_vals_list = []
                       
    pdi_rows = create_byte_pdindex(basis_rows._encoding)

    for enc_batch, cfs_batch, track_batch in g:
        pdi_res = create_byte_pdindex(enc_batch)
        index_res_by_rows = pdi_rows.get_indexer(pdi_res)
        mask_res_in_rows = (index_res_by_rows >= 0)

        mat_rows = index_res_by_rows[mask_res_in_rows]
        mat_cols = track_batch[mask_res_in_rows]
        mat_vals_batch = cfs_batch[mask_res_in_rows]
        mat_vals_list.append(mat_vals_batch)

        coords_batch = np.vstack([mat_rows, mat_cols]).T
        coords_list.append(coords_batch)

    coords = concat_batches(coords_list)
    mat_vals = concat_batches(mat_vals_list)
    if (coords is not None) and (mat_vals is not None):
        return squeeze_array(coords, mat_vals)
    else:
        return None
