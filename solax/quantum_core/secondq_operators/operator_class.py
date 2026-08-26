"""
Operator: a second-quantized operator as a sum of OperatorTerm monomials
(one per distinct daggers pattern) plus an optional scalar (identity) term.
"""
import numpy as np
from numbers import Number, Integral, Real
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .operator_term import *
from ..det_based_classes import *
from ...save_load import *
from ..mode_ctrl import *
from .operator_matrix import *


def get_default(op, key):
    """The value Operator "op" would report for "key" if it were present:
    a zero-length OperatorTerm with that daggers pattern, or 0 for
    "scalar". Used by __add__ to add two Operators key-by-key even
    where one side is missing a key."""
    default = OperatorTerm(key, np.array([]), np.ones(0)) if key != "scalar" else 0
    return op._d.get(key, default)


def op_from_dict(_d):
    """Builds an Operator directly from an already-prepared internal
    dict, bypassing __init__'s argument parsing."""
    op = Operator()
    op._d = _d
    return op


def key_to_str(key):
    """Encodes an Operator key (a daggers tuple, or the string
    "scalar") as a JSON-safe string, for __pre_dictify__."""
    if isinstance(key, tuple):
        return "_" + "".join(str(v) for v in key)
    elif key == "scalar":
        return "_"
    else:
        return key


def str_to_key(s):
    """Inverse of key_to_str, for __post_undictify__."""
    return tuple(int(v) for v in s[1:]) if (len(s) > 1) else "scalar"
    
    
msg_failed_init = """Could not construct an Operator object from the provided arguments.
One of the following must be passed:
    - (0) nothing;
    - (1) a scalar argument;
    - (2) an OperatorTerm object;
    - (3) arguments for construction of (2)."""

 
@dataclass
class Operator(Mapping):
    """
    A second-quantized operator, represented as a read-only Mapping from
    a "daggers" tuple (as in OperatorTerm) to the OperatorTerm holding
    all monomials with that pattern, plus an optional "scalar" key for
    a pure-number (identity) term. Supports dict-like access (indexing
    by key, ``in``, keys()/values()/items(), iteration, len()) as well as
    arithmetic (``+``, ``-``, ``*``, ``/``, unary ``-``) with other
    Operators, OperatorTerms, and plain numbers.

    Equality ("==") is deliberately unsupported (raises AttributeError).
    This class holds floating-point (real/complex) coefficients (in its
    OperatorTerm values and its optional "scalar" entry), and
    floating-point arithmetic is not exact, so an exact/bitwise equality
    check would depend on incidental rounding rather than genuine
    mathematical equality -- e.g. 0.1 + 0.1 == 0.2 is True, but
    0.1 + 0.1 + 0.1 == 0.3 is False, purely due to rounding. Comparing
    two such objects with "==" would therefore give results that look
    arbitrary rather than meaningful.

    For State/OperatorTerm, this can be checked via
    "len((a - b).chop(delta)) == 0" for a chosen precision delta > 0
    (OperatorMatrix similarly, via "(a - b).chop(delta).num_nonzero == 0",
    since it has no len()). Operator itself has no single "chop the whole
    thing" call: chop() takes a "key" and only chops one term at a time,
    and doesn't apply to the "scalar" key at all -- so check each
    underlying OperatorTerm the same way, plus the scalar entry
    separately::

        diff = a - b
        (
            all(len(diff[key].chop(delta)) == 0 for key in diff if key != "scalar")
            and abs(diff.get("scalar", 0)) < delta
        )

    Call an Operator on a Basis or State to apply it (see __call__);
    use build_matrix() to get its matrix representation directly.
    """
    _d : dict[ Literal["scalar"] | tuple, Number | OperatorTerm ]

    __array_ufunc__ = None


    def __init__(self, *args, **kwargs):
        """
        Flexible constructor, accepting one of:

        - nothing: an empty Operator;
        - a single Number: a pure scalar Operator;
        - a single OperatorTerm: an Operator with just that term;
        - the same (daggers, posits, coeffs) arguments OperatorTerm
          itself accepts, forwarded to build one term.

        Raises TypeError with a detailed message otherwise.
        """
        try:
            op_term = OperatorTerm(*args, **kwargs)
            args = [op_term]
            kwargs = {}
        except (TypeError, ValueError) as e:
            msg_failed_fin = msg_failed_init + \
                "\nOperatorTerm construction was attempted and failed with message: " + str(e)
        
        if kwargs or len(args) > 1:
            raise TypeError(msg_failed_fin)
            
        self._d = {}
        
        if args:
            term = args[0]
            if isinstance(term, Number):
                self._d["scalar"] = term
            elif isinstance(term, OperatorTerm):
                if len(term) > 0:
                    self._d[term.daggers] = term
            else:
                raise TypeError(msg_failed_fin)
                
                
    def __repr__(self):
        """Shows the constructor-style dict of daggers-key -> repr(term)
        (and "scalar" -> its value, if present), one entry per line;
        "Operator({})" for an empty Operator."""
        if self:
            d_repr = ",\n".join(
                f"{str(k)}: {repr(v)}" for k, v in self._d.items()
            )
            d_repr = ("\n" + d_repr).replace("\n", "\n\t") + "\n"
        else:
            d_repr = ""
        return f"Operator({{{d_repr}}})"
                
                
    def __len__(self):
        """Number of terms, including the "scalar" term if present."""
        return len(self._d)


    def __getitem__(self, s):
        """Looks up a term by its daggers tuple. A bare 0 or 1 is
        normalized to (0,)/(1,). Raises KeyError if absent."""
        if s == 0 or s == 1:
            s = (s,)
        return self._d[s]


    def __iter__(self):
        return iter(self._d)


    def __reversed__(self):
        return reversed(self._d)


    def keys(self):
        """Keys (daggers tuples, plus "scalar" if present); see Mapping.keys()."""
        return self._d.keys()


    def values(self):
        """Terms (OperatorTerm/Number values); see Mapping.values()."""
        return self._d.values()


    def items(self):
        """(key, term) pairs; see Mapping.items()."""
        return self._d.items()


    def drop(self, *key):
        """Returns a new Operator without the term at "key" (a daggers
        tuple, its individual ints, or "scalar"). Raises KeyError if the
        key isn't present."""
        key = tuple(key)
        if len(key) == 1 and not isinstance(key[0], Integral):
            key = key[0]
        if key not in self:
            raise KeyError("Could not find the provided key.")
        d = {k: v for k, v in self.items() if k != key}
        return op_from_dict(d)


    def chop(self, key, abs_coeff_cut: Real) -> "Self":
        """Returns a new Operator with the term at "key" chopped (see
        OperatorTerm.chop), other terms unchanged. Raises KeyError if
        "key" is absent, or TypeError for the "scalar" key (a single
        number has no per-entry coefficients to chop)."""
        if key not in self:
            raise KeyError("Could not find the provided key.")
        if key == "scalar":
            raise TypeError("Chopping is not supported for the scalar term.")
        op_term = self[key].chop(abs_coeff_cut)
        op_without = self.drop(key)
        return op_without + op_term


    def __eq__(self, other):
        """Always raises AttributeError: equality is deliberately
        unsupported for Operator (see the class docstring)."""
        raise AttributeError('Equality == is not implemented for the Operator class.')


    def __mul__(self, scalar: Number):
        """Scales every term (including "scalar", if present) by
        "scalar"."""
        if not isinstance(scalar, Number):
            return NotImplemented
        d = {key: term * scalar for key, term in self.items()}
        return op_from_dict(d)


    def __add__(self, other):
        """Adds "other" (an Operator, OperatorTerm, or Number) term by
        term, unioning the set of keys; a key present on only one side
        is combined with the appropriate default (a zero-length
        OperatorTerm, or 0 for "scalar") from the other."""
        if isinstance(other, OperatorTerm | Number):
            other = Operator(other)
        elif not isinstance(other, Operator):
            return NotImplemented
        d = {
            key: get_default(self, key) + get_default(other, key)
            for key in self.keys() | other.keys()
        }
        return op_from_dict(d)
    
    
    from ...utils.arithm_amends import (
            __rmul__, __radd__, __truediv__, __neg__, __sub__
        )
    
    
    @property
    def hconj(self):
        """Hermitian conjugate: conjugates the "scalar" term (if any)
        and, for each other term, takes OperatorTerm.hconj and re-keys
        it by its (reversed) daggers pattern."""
        d = {}
        for key, val in self.items():
            if key != "scalar":
                opterm_hconj = val.hconj
                d[opterm_hconj.daggers] = opterm_hconj
            else:
                d[key] = val.conjugate() 
        return op_from_dict(d)
    
    
    def __call__(self,
                 arg: Basis | State,
                 *,
                 det_batch_size: int | None = None,
                 op_batch_size: int | None = None,
                 multiple_devices: bool = False,
                 det_tracking: bool = False
                 ) -> Basis | State:
        """
        Applies this Operator to a Basis or a State, summing the
        contributions of all its terms. On a State, the "scalar" term
        (if present) contributes that number times "arg", as expected.
        On a bare Basis, the "scalar" term instead acts effectively as
        the unity operator -- it contributes "arg" unchanged, regardless
        of the scalar's actual value, even 0 -- since a Basis carries no
        coefficients to scale (per SciPost Phys. Codebases 51 Sec. 2.5).
        Returns the same kind of object as "arg"; the other keyword
        arguments behave exactly as in OperatorTerm.__call__, which see.
        Raises ValueError if this Operator is empty (len(self) == 0).
        """
        if not isinstance(arg, Basis | State):
            return NotImplemented
        if len(self) == 0:
            raise ValueError("Operator is empty. Cannot act with it.")
        for i, (key, term) in enumerate(self.items()):
            if key != "scalar":
                res_from_term = term(arg,
                                     det_batch_size=det_batch_size,
                                     op_batch_size=op_batch_size,
                                     multiple_devices=multiple_devices,
                                     det_tracking=det_tracking)
                if det_tracking:
                    obj_from_term, track_from_term = res_from_term
                else:
                    obj_from_term = res_from_term
            else:
                obj_from_term = term * arg if isinstance(arg, State) else arg
                if det_tracking:
                    track_from_term = np.arange(len(arg))
            
            if det_tracking:
                with manual_squeezing():
                    obj = obj + obj_from_term if i > 0 else obj_from_term
                track = np.concatenate([track, track_from_term]) if i > 0 else track_from_term
            else:
                obj = obj + obj_from_term if i > 0 else obj_from_term
            
        return (obj, track) if det_tracking else obj
    
    
    def build_matrix(self,
                     basis_rows: Basis,
                     basis_cols: Basis = None,
                     *,
                     det_batch_size: int | None = None,
                     op_batch_size: int | None = None,
                     multiple_devices: bool = False
                    ) -> OperatorMatrix:
        """
        Builds the matrix representation of this Operator in the given
        basis/bases: the (row, col) entry is the matrix element of self
        between basis_rows[row] (bra) and basis_cols[col] (ket). If
        "basis_cols" is omitted, it defaults to
        "basis_rows" (a square matrix). Both bases must be squeezed
        (no duplicate determinants); "det_batch_size"/"op_batch_size"/
        "multiple_devices" are the same batching/device knobs as in
        OperatorTerm.__call__.
        """
        mat = OperatorMatrix.from_operator(
            self, basis_rows, basis_cols,
            det_batch_size=det_batch_size,
            op_batch_size=op_batch_size,
            multiple_devices=multiple_devices
        )
        return mat
    
    
    def __pre_dictify__(self):
        """Hook used by solax.save(): re-keys terms with JSON-safe
        string keys (tuples/"scalar" aren't valid dict keys in JSON)."""
        d = {key_to_str(k): v for k, v in self.items()}
        return op_from_dict(d)


    def __post_undictify__(self):
        """Inverse of __pre_dictify__, used by solax.load()."""
        d = {str_to_key(s): v for s, v in self.items()}
        return op_from_dict(d)
    
    
    
save_load_registry.register("Operator", Operator, op_from_dict)