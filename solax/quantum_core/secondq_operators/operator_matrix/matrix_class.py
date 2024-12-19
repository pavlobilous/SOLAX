import numpy as np
import scipy as sp
from dataclasses import dataclass
from numbers import Number, Integral, Real

from .eval_mat_elems import *
from .shrink_basis import *
from ...det_based_classes import *
from ....save_load import *


WindowCorner = tuple[Integral | None, Integral | None]


def check_rows_cols_squeezed(basis_rows, basis_cols):
    if not (basis_rows.is_squeezed and (basis_cols is None or basis_cols.is_squeezed)):
        raise ValueError(
            'Operator matrix can be built only on "squeezed" Basis objects, '\
            'i. e. those which have unique determinants.'
        )


@dataclass
class OperatorMatrix:
    _coord: np.ndarray[np.ndarray[int]]
    _val: np.ndarray[float | complex]
    _size: np.ndarray[int]
    
    
    __array_ufunc__ = None
    
    
    @property
    def size(self):
        return tuple(self._size)
    
    
    @property
    def num_nonzero(self):
        return len(self._val)
    
    
    def __str__(self):
        s = repr(self)
        s = s.replace("_coord", "coord")
        s = s.replace("_val", "val")
        s = s.replace("_size", "size")
        return s

    
    @classmethod
    def zero(cls, num_rows: int, num_cols: int | None = None):
        if num_cols is None:
            num_cols = num_rows
        coord = np.array([], dtype=int).reshape(0, 2)
        val = np.array([], dtype=float)
        size = np.array([num_rows, num_cols])
        return cls(coord, val, size)
            
    
    @classmethod
    def from_opterm_or_scal(cls,
                            term,
                            basis_rows: Basis,
                            basis_cols: Basis = None,
                            *,
                            det_batch_size: int | None,
                            op_batch_size: int | None,
                            multiple_devices: bool,
                            check_squeezed: bool = True):
        
        if check_squeezed:
            check_rows_cols_squeezed(basis_rows, basis_cols)
                    
        if basis_cols is None:
            basis_cols = basis_rows

        min_len = min(len(basis_rows), len(basis_cols))
        if min_len == 0:
            return cls.zero(len(basis_rows), len(basis_cols))
            
        kwargs = dict(
            det_batch_size=det_batch_size,
            op_batch_size=op_batch_size,
            multiple_devices=multiple_devices
        )
        
        if isinstance(term, Number) or (len(basis_cols) <= len(basis_rows)):
            coord_val = eval_mat_elems(
                term, basis_rows, basis_cols, **kwargs
            )
            if coord_val is None:
                mat = cls.zero(len(basis_rows), len(basis_cols))
            else:
                size = np.array([len(basis_rows), len(basis_cols)])
                mat = cls(*coord_val, size)
        else:
            mat = cls.from_opterm_or_scal(term.hconj, basis_cols, basis_rows,
                                          check_squeezed=False, **kwargs).hconj
        return mat
    
        
    @classmethod
    def from_operator(cls,
                      op,
                      basis_rows: Basis,
                      basis_cols: Basis = None,
                      *,
                      det_batch_size: int | None,
                      op_batch_size: int | None,
                      multiple_devices: bool,
                      check_squeezed: bool = True):
        
        if check_squeezed:
            check_rows_cols_squeezed(basis_rows, basis_cols)
            
        kwargs = dict(
            det_batch_size=det_batch_size,
            op_batch_size=op_batch_size,
            multiple_devices=multiple_devices
        )
        
        if basis_cols is None:
            basis_cols = basis_rows
            
        mat = cls.zero(len(basis_rows), len(basis_cols))
        
        for key, term in op.items():
            mat_term = cls.from_opterm_or_scal(
                term, basis_rows, basis_cols,
                check_squeezed=False, **kwargs
            )
            mat += mat_term

        return mat
        
        
    def shrink_basis(self,
                     init_basis: Basis, fin_basis: Basis,
                     axis: int | None = None):
        """
        Extract the sub-Matrix of "matrix"
            corresponding to the sub-Basis "fin_basis" of the "init_basis".
        Note that "init_basis" must be the construction Basis for the current Matrix,
            so avoid usage of this function after basis-relevant matrix transformations.
        Argument "axis" can be 0 (rows), 1 (columns) or None (both).
        """
        if not (init_basis.is_squeezed and fin_basis.is_squeezed):
             raise ValueError('This operation works only on "squeezed" Basis objects, '\
                              'i. e. those which have unique determinants.')        
        return OperatorMatrix(*shrink_basis(self, init_basis, fin_basis, axis))
        
            
    def __add__(self, other):
        if not isinstance(other, OperatorMatrix):
            return NotImplemented
        coord = np.concatenate([self._coord, other._coord])
        val = np.concatenate([self._val, other._val])
        coord, val = squeeze_array(coord, val)
        size = np.vstack([self._size, other._size]).max(axis=0)
        return OperatorMatrix(coord, val, size)
    
    
    def __mul__(self, scalar: Number):
        if not isinstance(scalar, Number):
            return NotImplemented
        return OperatorMatrix(self._coord, self._val * scalar, self._size)
    
    
    from ....utils.arithm_amends import (
        __rmul__, __truediv__, __neg__, __sub__
    )
    
    
    @property
    def hconj(self):
        coord = self._coord[:, ::-1]
        val = self._val.conj()
        size = self._size[::-1]
        return OperatorMatrix(coord, val, size)
    
    
    def displace(self, row_shift: Integral, col_shift: Integral):
        if not isinstance(row_shift, Integral) or not isinstance(col_shift, Integral):
            raise TypeError("Wrong argument passed. Row and column shift must be both integer.")    
        shift = np.array([row_shift, col_shift])     
        coord = self._coord + shift
        where_nonneg = (coord >= 0).all(axis=1)
        coord = coord[where_nonneg]
        val = self._val[where_nonneg] 
        size = self._size + shift
        if (size < 0).any():
            size = np.array([0, 0])
        return OperatorMatrix(coord, val, size)
    
    
    def window(self, left_top_incl: WindowCorner, right_bottom_excl: WindowCorner):
        
        check_int_or_none = lambda v: isinstance(v, Integral) or (v is None)
        check_max_limit_point = lambda p: len(p) == 2 and check_int_or_none(p[0]) and check_int_or_none(p[1])
        
        if not check_max_limit_point(left_top_incl) or not check_max_limit_point(right_bottom_excl):
            raise TypeError('Wrong argument passed. '\
                            '"Window corners" must each contain 2 elements which are integer or None.')
        
        def get_mask(i):
            start = left_top_incl[i] if left_top_incl[i] is not None else 0
            end = right_bottom_excl[i] if right_bottom_excl[i] is not None else np.inf
            return (self._coord[:, i] >= start) & (self._coord[:, i] < end)
        
        mask = get_mask(0) & get_mask(1)
        coord = self._coord[mask]
        val = self._val[mask]
        
        return OperatorMatrix(coord, val, self._size)
    
    
    def __eq__(self, other):
        raise AttributeError('Equality == is not implemented for the OperatorMatrix class.')
    
    
    def chop(self, abs_val_cut: Real) -> "Self":
        """
        Drops all entries with abs(val) < abs_val_cut.
        """
        where_not_small = np.abs(self._val) >= abs_val_cut
        coord = self._coord[where_not_small]
        val = self._val[where_not_small]
        return OperatorMatrix(coord, val, self._size)
    
    
    def to_scipy(self):
        mat_scipy = sp.sparse.coo_array(
            (self._val, (self._coord[:, 0], self._coord[:, 1])),
            shape=self._size
        )
        return mat_scipy
    
    
    
save_load_registry.register("OperatorMatrix", OperatorMatrix, OperatorMatrix)