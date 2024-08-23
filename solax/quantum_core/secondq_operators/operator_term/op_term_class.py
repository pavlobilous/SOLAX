import numpy as np
from dataclasses import dataclass
from collections.abc import Sequence
from numbers import Integral, Number, Real

from ....ndarray_tools import *
from ....utils.index_manips import *
from ...mode_ctrl import *
from ...mode_ctrl.squeezing import squeeze_params
from .cleanup_input import *
from ...det_based_classes import *
from .act_in_batches import *
from ....save_load import *


@dataclass
class OperatorTerm(Sequence):
    daggers: tuple[int]
    posits: NDArray[2, Integral]
    coeffs: NDArray[1, Number]
    
    __array_ufunc__ = None
    

    def __post_init__(self):
        self.daggers, self.posits, self.coeffs = cleanup_input(
            self.daggers, self.posits, self.coeffs
        )
        if squeeze_params["SQUEEZE_OPTERM_AFTER_INIT"]:
            self.posits, self.coeffs = squeeze_array(
                self.posits, self.coeffs
            )
            
            
    def __repr__(self):
        vars_str = ",\n".join(
            f"{str(k)}={repr(v)}" for k, v in vars(self).items()
        )
        vars_str = ("\n" + vars_str).replace("\n", "\n\t")
        return f"{self.__class__.__name__}({vars_str:>4s}\n)"
            
            
    def __str__(self):
        repr_str = repr(self)
        str_str = repr_str.replace(" posits", "\n  posits")
        str_str = str_str.replace(" coeffs", "\n  coeffs")
        return str_str
            

    def __len__(self):
        return len(self.posits)


    def __getitem__(self, s):
        s = make_1d_index(len(self), s)
        with manual_squeezing():
            op_term = OperatorTerm(self.daggers, self.posits[s], self.coeffs[s])
        return op_term


    @property
    def hconj(self):
        daggers = tuple(1 - d for d in reversed(self.daggers))
        posits = self.posits[..., ::-1]
        coeffs = self.coeffs.conjugate()
        with manual_squeezing():
            op_term = OperatorTerm(daggers, posits, coeffs)
        return op_term


    @property
    def is_squeezed(self) -> bool:
        return array_is_squeezed(self.posits)
    

    def squeeze(self) -> "Self":
        posits, coeffs = squeeze_array(self.posits, self.coeffs)
        return OperatorTerm(self.daggers, posits, coeffs)


    def __add__(self, other):
        if isinstance(other, OperatorTerm) and self.daggers == other.daggers:
            posits = np.concatenate([self.posits, other.posits])
            coeffs = np.concatenate([self.coeffs, other.coeffs])
            op_term = OperatorTerm(self.daggers, posits, coeffs)
            if squeeze_params["SQUEEZE_OPTERM_AFTER_ADD"]:
                op_term = op_term.squeeze()
            return op_term
        else:
            from ..operator_dict import Operator
            if not isinstance(other, Operator):
                try:
                    other = Operator(other)
                except TypeError:
                    raise TypeError("Cannot perform addition since the objects are incompatible.")
            return other + self


    def __mul__(self, scalar: Number):
        if not isinstance(scalar, Number):
            return NotImplemented
        coeffs = self.coeffs * scalar
        with manual_squeezing():
            op_term = OperatorTerm(self.daggers, self.posits, coeffs)
        return op_term


    from ....utils.arithm_amends import (
            __rmul__, __radd__, __truediv__, __neg__, __sub__
        )


    def chop(self, abs_coeff_cut: Real) -> "Self":
        """
        Drops all entries with abs(coeff) < abs_coeff_cut.
        Note: Each coeff is treated individually even if there are repeated entries.
        """
        where_not_small = np.abs(self.coeffs) >= abs_coeff_cut
        posits = self.posits[where_not_small]
        coeffs = self.coeffs[where_not_small]
        with manual_squeezing():
            op_term = OperatorTerm(self.daggers, posits, coeffs)
        return op_term


    def __eq__(self, other):
        raise AttributeError('Equality == is not implemented for the OperatorTerm class.')


    def __call__(self,
                 arg: Basis | State,
                 *,
                 det_batch_size: int | None = None,
                 op_batch_size: int | None = None,
                 multiple_devices: bool = False,
                 det_tracking: bool = False
                 ) -> Basis | State:
        
        if len(self) == 0:
            raise ValueError("OperatorTerm is empty. Cannot act with it.")
        
        if isinstance(arg, Basis):
            basis = arg
            state_coeffs = None
        elif isinstance(arg, State):
            basis = arg.basis
            state_coeffs = arg.coeffs
        else:
            return NotImplemented
        
        if (self.posits >= basis.bitlen).any():
            raise ValueError("OperatorTerm contains positions beyond the determinant.")
        
        encoding, coeffs, det_track = act_in_batches(
            basis, state_coeffs, self, det_batch_size, op_batch_size,
            multiple_devices, det_tracking
        )
        
        basis = Basis._from_attrs(encoding, basis.bitlen)
        result = basis if isinstance(arg, Basis) else State(basis, coeffs)
        
        if det_tracking:
            return result, det_track
        
        if squeeze_params["SQUEEZE_DETS_AFTER_OPTERM"]:
            result = result.squeeze()
        return result
    
    
    def build_matrix(self, *args, **kwargs):
        from ..operator_dict import Operator
        return Operator(self).build_matrix(*args, **kwargs)

    
    
def init_from_attr(*args, **kwargs):
    with manual_squeezing():
        op_term = OperatorTerm(*args, **kwargs)
    return op_term
 
    
save_load_registry.register("OperatorTerm", OperatorTerm, init_from_attr)