import numpy as np
from dataclasses import dataclass
from collections.abc import Sequence
from numbers import Real, Number

from ...ndarray_tools import *
from ...utils.index_manips import *
from ...save_load import *
from .printing import *
from .basis_class import *
from ..mode_ctrl import *
from ..mode_ctrl.squeezing import squeeze_params


get_data = lambda state: (state.basis._encoding, state.coeffs)
get_bitlen = lambda state: state.basis.bitlen


def mod_on_data(s, o):
    s_encoding, s_coeffs = s
    o_encoding, _ = o
    bmask = array_difference_bmask(s_encoding, o_encoding)
    encoding = s_encoding[bmask]
    coeffs = s_coeffs[bmask]
    return encoding, coeffs


@dataclass
class FakeArr:
    length: int
    def __len__(self):
        return self.length


@dataclass
class State(Sequence):
    basis: Basis
    coeffs: NDArray[1, Number]
    
    __array_ufunc__ = None
    

    def __post_init__(self):
        if len(self.coeffs) != len(self.basis):
            raise ValueError('"coeffs" and "basis" must be of the same length.')
            
        
    def __len__(self):
        return len(self.basis)
    
                   
    def __getitem__(self, s):
        try:
            s = make_1d_index(len(self), s)
        except NotImplementedError:
            return NotImplemented
        return State(self.basis[s], self.coeffs[s])
    
    
    def __reversed__(self):
        return self[::-1]

    
    def __eq__(self, other):
        raise AttributeError('Equality operator == is not implemented for the State class.')
        

    def __str__(self):
        det_coeff_strs, overflow = dets_with_coeffs_to_strs(
            self.basis._encoding, self.basis.bitlen, self.coeffs
        )
        if overflow:
            det_coeff_strs.append("...")
        return "\n".join(det_coeff_strs)

    
    def squeeze(self) -> "Self":
        sq_encoding, sq_coeffs = squeeze_array(self.basis._encoding, self.coeffs)
        basis = Basis._from_attrs(sq_encoding, self.basis.bitlen)
        return State(basis, sq_coeffs)


    @property
    def is_squeezed(self) -> bool:
        return self.basis.is_squeezed

    
    def __add__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        with manual_squeezing():
            basis = self.basis + other.basis
        coeffs = np.concatenate([self.coeffs, other.coeffs])
        state = State(basis, coeffs)
        if squeeze_params["SQUEEZE_DETS_AFTER_ADD"]:
            state = state.squeeze()
        return state


    def __mod__(self, other):
        if not isinstance(other, Basis):
            return NotImplemented
        other_fake_state = State(other, FakeArr(len(other)))
        (encoding, coeffs), bitlen = op_on_cls(
            mod_on_data, self, other_fake_state, self, self, get_data, get_bitlen
        )
        basis = Basis._from_attrs(encoding, bitlen)
        state = State(basis, coeffs)
        return state


    def __mul__(self, other):
        if isinstance(other, State): 
            if self.basis.bitlen != other.basis.bitlen:
                return 0
            pd_s = create_byte_pdindex(self.basis._encoding)
            pd_o = create_byte_pdindex(other.basis._encoding)
            pd_i = pd_s.intersection(pd_o)
            cf_s = sum_by_indexer(self.coeffs, pd_i.get_indexer(pd_s))
            cf_o = sum_by_indexer(other.coeffs, pd_i.get_indexer(pd_o))
            return (cf_s.conj() * cf_o).sum().item()       
        elif isinstance(other, Number):
            new_coeffs = self.coeffs * other
            return State(self.basis, new_coeffs)    
        else:
            return NotImplemented
        

    from ...utils.arithm_amends import (
        __rmul__, __truediv__, __neg__, __sub__
    )
    

    def chop(self, abs_coeff_cut: Real) -> "Self":
        """
        Drops all determinants with abs(coeff) < abs_coeff_cut.
        Note: Each coeff is treated individually even if there are repeated determinants.
        """
        where_not_small = np.abs(self.coeffs) >= abs_coeff_cut
        basis = Basis._from_attrs(self.basis._encoding[where_not_small], self.basis.bitlen)
        state = State(basis, self.coeffs[where_not_small])
        return state
        

    def normalize(self) -> "Self":
        """
        Returns normalizes version of the State.
        """
        norm = self * self
        if norm == 0:
            raise ValueError("Vector of norm 0 is not normalizable.")
        return State(self.basis, self.coeffs / np.sqrt(norm))


save_load_registry.register("State", State, State)