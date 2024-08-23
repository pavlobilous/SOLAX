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
    default = OperatorTerm(key, np.array([]), np.ones(0)) if key != "scalar" else 0
    return op._d.get(key, default)


def op_from_dict(_d):
    op = Operator()
    op._d = _d
    return op


def key_to_str(key):
    if isinstance(key, tuple):
        return "_" + "".join(str(v) for v in key)
    elif key == "scalar":
        return "_"
    else:
        return key
        

def str_to_key(s):
    return tuple(int(v) for v in s[1:]) if (len(s) > 1) else "scalar"
    
    
msg_failed_init = """Could not construct an Operator object from the provided arguments.
One of the following must be passed:
    - (0) nothing;
    - (1) a scalar argument;
    - (2) an OperatorTerm object;
    - (3) arguments for construction of (2)."""

 
@dataclass
class Operator(Mapping):
    _d : dict[ Literal["scalar"] | tuple, Number | OperatorTerm ]
    
    __array_ufunc__ = None
    
    
    def __init__(self, *args, **kwargs):
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
        d_repr = ",\n".join(
            f"{str(k)}: {repr(v)}" for k, v in self._d.items()
        )
        d_repr = ("\n" + d_repr).replace("\n", "\n\t")
        return f"Operator({{{d_repr}\n}})"
                
                
    def __len__(self):
        return len(self._d)
            
                
    def __getitem__(self, s):
        if s == 0 or s == 1:
            s = (s,)
        return self._d[s]
    
    
    def __iter__(self):
        return iter(self._d)
    
    
    def __reversed__(self):
        return reversed(self._d)
    
    
    def keys(self):
        return self._d.keys()
    
    
    def values(self):
        return self._d.values()
    
    
    def items(self):
        return self._d.items()
    
    
    def drop(self, *key):
        key = tuple(key)
        if len(key) == 1 and not isinstance(key[0], Integral):
            key = key[0]
        if key not in self:
            raise KeyError("Could not find the provided key.")
        d = {k: v for k, v in self.items() if k != key}
        return op_from_dict(d)
    
    
    def chop(self, key, abs_coeff_cut: Real) -> "Self":
        if key not in self:
            raise KeyError("Could not find the provided key.")     
        if key == "scalar":
            raise TypeError("Chopping is not supported for the scalar term.")
        op_term = self[key].chop(abs_coeff_cut)
        op_without = self.drop(key)
        return op_without + op_term
    
    
    def __eq__(self, other):
        raise AttributeError('Equality == is not implemented for the Operator class.')
        
        
    def __mul__(self, scalar: Number):
        if not isinstance(scalar, Number):
            return NotImplemented
        d = {key: term * scalar for key, term in self.items()}
        return op_from_dict(d)
    
    
    def __add__(self, other):
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
        m = OperatorMatrix.from_operator(
            self, basis_rows, basis_cols,
            det_batch_size=det_batch_size,
            op_batch_size=op_batch_size,
            multiple_devices=multiple_devices
        )
        return m
    
    
    def __pre_dictify__(self):
        d = {key_to_str(k): v for k, v in self.items()}
        return op_from_dict(d)
    
    
    def __post_undictify__(self):
        d = {str_to_key(s): v for s, v in self.items()}
        return op_from_dict(d)
    
    
    
save_load_registry.register("Operator", Operator, op_from_dict)