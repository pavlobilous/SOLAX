"""
Computation of the coordinate-sparse (row, col, value) matrix elements
of a single OperatorTerm (or a plain scalar) between two bases, the
core numerical routine behind OperatorMatrix.from_opterm_or_scal().
"""
from numbers import Number
import numpy as np

from ...det_based_classes import *
from ....ndarray_tools import *
from ..operator_term.act_in_batches import *


def concat_batches(batches_lst):
    """
    Concatenates a list of NumPy arrays collected batch by batch into
    one array. Returns the single element unchanged if "batches_lst"
    has exactly one entry (avoiding an unnecessary copy), or None if
    "batches_lst" is empty.
    """
    if len(batches_lst) > 1:
        return np.concatenate(batches_lst)
    if len(batches_lst) == 1:
        return batches_lst[0]
    else:
        return None


def act_with_scal_on_basis(basis, scal):
    """
    Generator mimicking the (encoding, coeffs, det_tracking) batch
    interface of act_in_batches_generator(), for the special case of
    acting with a plain scalar (identity operator) instead of an
    OperatorTerm. Yields a single batch covering the whole "basis"
    unchanged: its packed encoding, a coefficient array filled with
    "scal" (one per determinant), and a det-tracking array mapping each
    determinant to itself (np.arange(len(basis))).
    """
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
    """
    Computes the matrix elements <basis_rows[row]| term |basis_cols[col]>
    of a single OperatorTerm (or scalar) "term", in coordinate sparse
    format (similarly to scipy.sparse.coo_matrix).

    Acts with "term" on "basis_cols" (with unit coefficients and
    det_tracking=True, batched per "det_batch_size"/"op_batch_size"/
    "multiple_devices" -- see OperatorTerm.__call__ for their meaning),
    then, for each resulting determinant, looks up its row position in
    "basis_rows" (both bases are assumed squeezed by the caller); hits
    become (row, col) coordinate pairs with the corresponding
    coefficient as value, duplicate coordinates are summed via
    squeeze_array().

    Returns a (coords, vals) tuple -- "coords" an (N, 2) int array,
    "vals" an (N,) array -- or None if no matrix elements were found
    (e.g. "basis_cols" is empty, or none of the resulting determinants
    lie in "basis_rows").
    """
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
