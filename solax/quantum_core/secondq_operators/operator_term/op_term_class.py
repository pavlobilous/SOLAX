"""
OperatorTerm: a single second-quantized monomial (a fixed pattern of
creation/annihilation operators) as a batch of position/coefficient rows.
"""
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
    """
    A single second-quantized monomial: a fixed sequence of
    creation/annihilation ladder operators ("daggers", 1=creation
    a-dagger/0=annihilation a, applied right to left as usual) at given
    spin-orbital positions, summed over a batch of position rows, each
    with its own prefactor:
        sum_i coeffs[i] * (op_{daggers[0]} at posits[i, 0]) ... (op_{daggers[-1]} at posits[i, -1])
    "daggers" is a tuple of 0/1 shared by all rows (which specific
    physical operator this represents is entirely the caller's choice
    of pattern); "posits" is a 2D integer array (one row of
    spin-orbital positions per term); "coeffs" is a 1D array of
    prefactors, one per row -- rows with identical "posits" are
    automatically merged on construction, with their coefficients
    summed. Behaves like a read-only Sequence over its rows: len(),
    indexing/slicing, and iteration. See SciPost Phys. Codebases 51
    Sec. 2.4.

    Equality ("==") is deliberately unsupported (raises AttributeError),
    for the same reason as for State. Call an OperatorTerm on a Basis or
    State to apply it (see __call__); use build_matrix() to get its
    matrix representation directly.
    """
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
        """Constructor-style representation:
        "OperatorTerm(daggers=..., posits=..., coeffs=...)", each field
        on its own indented line via repr() of its value."""
        vars_str = ",\n".join(
            f"{str(k)}={repr(v)}" for k, v in vars(self).items()
        )
        vars_str = ("\n" + vars_str).replace("\n", "\n\t")
        return f"{self.__class__.__name__}({vars_str:>4s}\n)"


    def __str__(self):
        """Like repr(), with an extra newline inserted before the
        "posits"/"coeffs" fields for a more readable rendering."""
        repr_str = repr(self)
        str_str = repr_str.replace(" posits", "\n  posits")
        str_str = str_str.replace(" coeffs", "\n  coeffs")
        return str_str
            

    def __len__(self):
        """Number of position/coefficient rows."""
        return len(self.posits)


    def __getitem__(self, s):
        """Indexes/slices the rows like a sequence. Accepts an int, a
        slice, a tuple of ints (fancy indexing), or a boolean mask of
        length len(self). Always returns a new OperatorTerm with the
        same "daggers"."""
        s = make_1d_index(len(self), s)
        with manual_squeezing():
            op_term = OperatorTerm(self.daggers, self.posits[s], self.coeffs[s])
        return op_term


    @property
    def hconj(self):
        """Hermitian conjugate: reverses the operator sequence order
        and flips each dagger (1<->0), reverses the spin-orbital positions of
        each row to match, and conjugates the coefficients."""
        daggers = tuple(1 - d for d in reversed(self.daggers))
        posits = self.posits[..., ::-1]
        coeffs = self.coeffs.conjugate()
        with manual_squeezing():
            op_term = OperatorTerm(daggers, posits, coeffs)
        return op_term


    @property
    def is_squeezed(self) -> bool:
        """True if no two rows share the same "posits"."""
        return array_is_squeezed(self.posits)


    def squeeze(self) -> "Self":
        """Returns a new OperatorTerm with rows sharing the same
        "posits" merged (their coefficients summed)."""
        posits, coeffs = squeeze_array(self.posits, self.coeffs)
        return OperatorTerm(self.daggers, posits, coeffs)


    def __add__(self, other):
        """If "other" is an OperatorTerm with the same "daggers"
        pattern, concatenates their rows (deduplicating unless inside a
        manual_squeezing() context). Otherwise promotes both sides to
        Operator and delegates to Operator.__add__ (e.g. adding a plain
        Number introduces/updates the "scalar" term)."""
        if isinstance(other, OperatorTerm) and self.daggers == other.daggers:
            posits = np.concatenate([self.posits, other.posits])
            coeffs = np.concatenate([self.coeffs, other.coeffs])
            op_term = OperatorTerm(self.daggers, posits, coeffs)
            if squeeze_params["SQUEEZE_OPTERM_AFTER_ADD"]:
                op_term = op_term.squeeze()
            return op_term
        else:
            from ..operator_class import Operator
            if not isinstance(other, Operator):
                try:
                    other = Operator(other)
                except TypeError:
                    raise TypeError("Cannot perform addition since the objects are incompatible.") from None
            return other + self


    def __mul__(self, scalar: Number):
        """Scales all coefficients by "scalar"."""
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
        """Always raises AttributeError: equality is deliberately
        unsupported for OperatorTerm (rows can represent the same
        operator in more than one way, e.g. before/after squeeze())."""
        raise AttributeError('Equality == is not implemented for the OperatorTerm class.')


    def _call_on_batches(self, basis, state_coeffs,
                        det_batch_size, op_batch_size,
                        multiple_devices: bool,
                        det_tracking: bool):
        """Internal: runs __call__'s core computation in batches over
        determinants ("det_batch_size") and/or operator rows
        ("op_batch_size") via act_in_batches_generator, then collects
        the resulting encoding/coeffs/det-tracking chunks into lists
        (substituting a correctly-shaped empty array if a list would
        otherwise be empty, e.g. when every batch's result is empty)."""

        g = act_in_batches_generator(
            basis, state_coeffs, self,
            det_batch_size, op_batch_size,
            multiple_devices,
            det_tracking
        )

        encoding_list, coeffs_list, det_track_list = zip(*g)
        
        if len(encoding_list) == 0:
            det_code_len = basis._encoding.shape[-1]
            empty_enc = np.array([], dtype=np.uint8).reshape(0, det_code_len)
            encoding_list.append(empty_enc)
        if (state_coeffs is not None) and (len(coeffs_list) == 0):
            empty_cfs =  np.array([], dtype=state_coeffs.dtype)
            coeffs_list.append(empty_cfs)
        if det_tracking and (len(det_track_list) == 0):
            empty_det_track = np.array([], dtype=int)
            det_track_list.append(empty_det_track)
            
        return encoding_list, coeffs_list, det_track_list


    def __call__(self,
                 arg: Basis | State,
                 *,
                 det_batch_size: int | None = None,
                 op_batch_size: int | None = None,
                 multiple_devices: bool = False,
                 det_tracking: bool = False
                 ) -> Basis | State:
        """
        Applies this OperatorTerm to a Basis or a State, returning the
        same kind of object: acting on a Basis returns the Basis of
        resulting determinants (invalid/annihilated results dropped);
        acting on a State additionally propagates and combines
        coefficients, and includes ladder-operator sign phases.

        "det_batch_size"/"op_batch_size" process determinants/rows in
        batches instead of all at once (memory/performance knob only;
        the batched result equals the unbatched one). "multiple_devices"
        spreads batches across local JAX devices via pmap (see the
        `multi_device` pytest marker in the test suite for how to
        exercise this). "det_tracking", if True, additionally returns a
        1D integer array mapping each output determinant back to the
        index of the input determinant it came from.

        Raises ValueError if this OperatorTerm is empty (len(self) == 0)
        or if any of its "posits" is >= arg's "bitlen". Returns "arg"
        unchanged (with an empty det-tracking array, if requested) when
        "arg" is an empty Basis/State.
        """
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

        if len(basis) == 0:
            return (arg, np.array([], dtype=int)) if det_tracking else arg
        
        encoding_list, coeffs_list, det_track_list = self._call_on_batches(
            basis, state_coeffs, det_batch_size, op_batch_size,
            multiple_devices, det_tracking
        )
        
        basis = Basis._from_attrs(
            np.concatenate(encoding_list), basis.bitlen
        )
        result = basis \
            if isinstance(arg, Basis) \
            else State(basis, np.concatenate(coeffs_list))
        if det_tracking:
            return result, np.concatenate(det_track_list)
        
        if squeeze_params["SQUEEZE_DETS_AFTER_OPTERM"]:
            result = result.squeeze()
        return result
    
    
    def build_matrix(self, *args, **kwargs):
        """Convenience shortcut for Operator(self).build_matrix(...);
        see Operator.build_matrix for the accepted arguments."""
        from ..operator_class import Operator
        return Operator(self).build_matrix(*args, **kwargs)

    
    
def init_from_attr(*args, **kwargs):
    """Reconstructs an OperatorTerm from already-cleaned attributes
    without re-squeezing; used when loading one back with
    solax.load()."""
    with manual_squeezing():
        op_term = OperatorTerm(*args, **kwargs)
    return op_term
 
    
save_load_registry.register("OperatorTerm", OperatorTerm, init_from_attr)
