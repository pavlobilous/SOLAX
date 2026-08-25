"""
OperatorMatrix: a sparse (COO-style) matrix representation of an
Operator/OperatorTerm in a given basis (or pair of row/column bases).
"""
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
    """Raises ValueError unless both bases (or just "basis_rows" if
    "basis_cols" is None) are squeezed -- a matrix's rows/columns must
    correspond to unique determinants."""
    if not (basis_rows.is_squeezed and (basis_cols is None or basis_cols.is_squeezed)):
        raise ValueError(
            'Operator matrix can be built only on "squeezed" Basis objects, '\
            'i. e. those which have unique determinants.'
        )


@dataclass
class OperatorMatrix:
    """
    A sparse (COO-style) matrix: "_coord" is an (N, 2) array of
    (row, col) index pairs, "_val" an (N,) array of the corresponding
    values, and "_size" the (num_rows, num_cols) shape -- see
    build_matrix() on Operator/OperatorTerm for how one is normally
    constructed, rather than instantiating this class directly.

    Supports scalar arithmetic (``+``, ``-``, ``*``, ``/``, unary ``-``)
    with other OperatorMatrix instances/numbers, hconj, and the basis-relative
    reshaping operations displace()/window()/shrink_basis() (see each
    for how they differ). Equality ("==") is deliberately unsupported
    (raises AttributeError), for the same reason as for the other
    quantum_core classes. Use to_scipy() to get a real
    scipy.sparse.coo_array for further numerical work (diagonalization,
    etc.).
    """
    _coord: np.ndarray[np.ndarray[int]]
    _val: np.ndarray[float | complex]
    _size: np.ndarray[int]


    __array_ufunc__ = None


    @property
    def size(self):
        """The (num_rows, num_cols) shape, as a plain tuple."""
        return tuple(self._size)


    @property
    def num_nonzero(self):
        """Number of stored (nonzero) entries."""
        return len(self._val)


    def __str__(self):
        """Like repr(), but with the leading underscores stripped from
        the "_coord"/"_val"/"_size" field names for a more readable
        rendering."""
        s = repr(self)
        s = s.replace("_coord", "coord")
        s = s.replace("_val", "val")
        s = s.replace("_size", "size")
        return s

    
    @classmethod
    def zero(cls, num_rows: int, num_cols: int | None = None):
        """An all-zero (num_rows, num_cols) matrix (no stored entries).
        "num_cols" defaults to "num_rows" (a square matrix)."""
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
        """
        Builds the matrix of a single OperatorTerm (or a plain scalar,
        treated as a multiple of the identity) in "basis_rows" x
        "basis_cols" (defaulting "basis_cols" to "basis_rows"). Both
        bases must be squeezed. For efficiency, computes matrix
        elements in whichever of the two bases is smaller and takes
        hconj if that was "basis_cols" rather than "basis_rows"; pass
        check_squeezed=False internally to skip a redundant recheck
        during that recursive hconj step.
        """
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
        """Builds the matrix of a full Operator by summing
        from_opterm_or_scal() over each of its terms (see that method
        for the argument semantics)."""
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
        
        for _key, term in op.items():
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
        Extract the sub-Matrix of "matrix" corresponding to the
        sub-Basis "fin_basis" of the "init_basis". Note that
        "init_basis" must be the construction Basis for the current
        Matrix, so avoid usage of this function after basis-relevant
        matrix transformations. Argument "axis" can be 0 (rows), 1
        (columns) or None (both).
        """
        if not (init_basis.is_squeezed and fin_basis.is_squeezed):
             raise ValueError('This operation works only on "squeezed" Basis objects, '\
                              'i. e. those which have unique determinants.')        
        return OperatorMatrix(*shrink_basis(self, init_basis, fin_basis, axis))
        
            
    def __add__(self, other):
        """Adds two matrices entry-wise (values at shared coordinates
        summed), with the result "size" the element-wise max of both
        operands' sizes -- so operands of different sizes may be added,
        the smaller effectively zero-padded."""
        if not isinstance(other, OperatorMatrix):
            return NotImplemented
        coord = np.concatenate([self._coord, other._coord])
        val = np.concatenate([self._val, other._val])
        coord, val = squeeze_array(coord, val)
        size = np.vstack([self._size, other._size]).max(axis=0)
        return OperatorMatrix(coord, val, size)


    def __mul__(self, scalar: Number):
        """Scales all stored values by "scalar"."""
        if not isinstance(scalar, Number):
            return NotImplemented
        return OperatorMatrix(self._coord, self._val * scalar, self._size)
    
    
    from ....utils.arithm_amends import (
        __rmul__, __truediv__, __neg__, __sub__
    )
    
    
    @property
    def hconj(self):
        """Hermitian conjugate: swaps (row, col) to (col, row) for
        every stored entry, conjugates the values, and swaps the
        (rows, cols) size to (cols, rows)."""
        coord = self._coord[:, ::-1]
        val = self._val.conj()
        size = self._size[::-1]
        return OperatorMatrix(coord, val, size)


    def displace(self, row_shift: Integral, col_shift: Integral):
        """Returns a new matrix with every entry's coordinates shifted
        by (row_shift, col_shift) and the size grown/shrunk to match;
        entries whose shifted coordinates would go negative are
        dropped, and the size is clamped to (0, 0) rather than going
        negative. "row_shift"/"col_shift" must both be integers."""
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
        """
        Returns a new matrix of the SAME size as self, with entries
        outside the [left_top_incl, right_bottom_excl) coordinate
        window dropped (zeroed) rather than the matrix being resized --
        contrast with shrink_basis(), which extracts an actually
        smaller sub-matrix. Each corner is a (row, col) pair of ints or
        None (None means unbounded on that side/that end).
        """
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
        """Always raises AttributeError: equality is deliberately
        unsupported for OperatorMatrix (see the class docstring)."""
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
        """Converts to a real scipy.sparse.coo_array of the same shape
        and entries, for use with scipy's sparse linear algebra (e.g.
        scipy.sparse.linalg.eigsh)."""
        mat_scipy = sp.sparse.coo_array(
            (self._val, (self._coord[:, 0], self._coord[:, 1])),
            shape=self._size
        )
        return mat_scipy
    
    
    
save_load_registry.register("OperatorMatrix", OperatorMatrix, OperatorMatrix)