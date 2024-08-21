import pandas as pd
import numpy as np

from ....ndarray_tools import *


def get_new_coord(pd_inds, old_coord, *, axis):
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