"""
Data-level implementation of OperatorMatrix.shrink_basis(): re-indexes a
matrix's stored coordinates from an initial basis down to a sub-basis
along one or both axes, dropping entries that fall outside it.
"""
import pandas as pd
import numpy as np

from ....ndarray_tools import *


def get_new_coord(pd_inds, old_coord, *, axis):
    """
    Re-indexes one axis ("axis": 0 for rows, 1 for columns) of a
    (N, 2) coordinate array "old_coord" through the position mapping
    "pd_inds" (a pandas.Index of old-basis positions, indexed by
    new-basis position -- see shrink_basis(), which builds it), leaving
    the other axis ("spectator") untouched. Positions on "axis" that
    have no match in "pd_inds" become -1, to be filtered out by the
    caller.
    """
    old_entries = old_coord[:, axis]
    spectator = old_coord[:, 1 - axis]
    pd_crd = pd.Index(old_entries)
    new_entries = pd_inds.get_indexer(pd_crd)
    if axis == 0:
        new_coord = np.vstack([new_entries, spectator]).T
    elif axis == 1:
        new_coord = np.vstack([spectator, new_entries]).T
    else:
        raise ValueError
    return new_coord


def shrink_basis(matrix, init_basis, fin_basis, axis: int | None = None):
    """
    Data-level implementation of OperatorMatrix.shrink_basis(). Builds
    the position mapping from "fin_basis" (a sub-basis of "init_basis")
    to "init_basis", uses it (via get_new_coord()) to re-index
    "matrix"'s stored coordinates on the requested "axis" (0=rows,
    1=columns, None=both), and drops entries whose position on the
    re-indexed axis/axes fell outside "fin_basis".

    Raises ValueError if "fin_basis" is not a sub-basis of "init_basis",
    or if "axis" is not 0, 1, or None.

    Returns a (coord, val, size) tuple, ready to build a new
    OperatorMatrix from -- "size" keeps "matrix"'s own extent on the
    axis/axes NOT shrunk, and becomes len(fin_basis) on the axis/axes
    that were.
    """
    pdi_init = create_byte_pdindex(init_basis._encoding)
    pdi_fin = create_byte_pdindex(fin_basis._encoding)
    pd_inds = pd.Index(pdi_init.get_indexer(pdi_fin))
    if (pd_inds == -1).any():
        raise ValueError('"fin_basis" must be a sub-basis of "init_basis".')
        
    if axis is None:
        coord = get_new_coord(pd_inds, matrix._coord, axis=0)
        coord = get_new_coord(pd_inds, coord, axis=1)
    elif axis == 0 or axis == 1:
        coord = get_new_coord(pd_inds, matrix._coord, axis=axis)
    else:
        raise ValueError('"axis" must be 0 (rows), 1 (columns) or None (both).')

    mask = (coord >= 0).all(axis=1)
    coord = coord[mask]
    val = matrix._val[mask]

    rows_num = matrix.size[0] if axis == 1 else len(fin_basis)
    cols_num = matrix.size[1] if axis == 0 else len(fin_basis)
    size = np.array([rows_num, cols_num])
    return coord, val, size