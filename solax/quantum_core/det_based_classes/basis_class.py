import numpy as np
from dataclasses import dataclass
from collections.abc import Sequence, Iterable
from numbers import Integral

from ..bit_level_primitives import *
from .arr_op_wraps import *
from ...ndarray_tools import *
from ...utils.index_manips import *
from ...save_load import *
from .printing import *
from ..mode_ctrl.squeezing import squeeze_params


def attrs_from_bits(bits: NDArray[1, Integral] | NDArray[2, Integral]
                    ) -> tuple[NDArray[2, np.uint8], int]:
    bits = np.atleast_2d(bits)
    encoding, bitlen = det_from_bits(bits, module=np)
    if len(encoding[0]) == 0:
        encoding = encoding.reshape(0, 0)
    return encoding, bitlen


def det_strings_to_bits(det_strings):
    det_strings = [s for s in det_strings if s]
    lengths = {len(s) for s in det_strings}
    if len(lengths) > 1:
        raise ValueError("All determinant strings must have equal length.")
    bits = []
    for det_string in det_strings:
        bits_ln = []
        for c in det_string:
            c_int = int(c)
            if (c_int != 0) and (c_int != 1):
                raise ValueError("Determinant strings must contain only 0 and 1.") 
            bits_ln.append(c_int)
        bits.append(bits_ln)
    return np.array(bits, dtype=np.int8)


get_data = lambda basis: basis._encoding
get_bitlen = lambda basis: basis.bitlen



@dataclass
class Basis(Sequence):
    _encoding: NDArray[2, np.uint8]
    _bitlen: int
    
    __array_ufunc__ = None

    
    @property
    def bitlen(self):
        return self._bitlen


    def __init__(self, det_strings: Iterable[str]):
        bits = det_strings_to_bits(det_strings)
        self._encoding, self._bitlen = attrs_from_bits(bits)
        if squeeze_params["SQUEEZE_BASIS_AFTER_INIT"]:
            self._encoding, _ = squeeze_array(self._encoding)


    @classmethod
    def _from_attrs(cls, _encoding, _bitlen):
        instance = cls([])
        instance._encoding = _encoding
        instance._bitlen = _bitlen
        return instance
            

    @classmethod
    def from_bits(cls, bits: NDArray[1, Integral] | NDArray[2, Integral]):
        instance = cls._from_attrs(*attrs_from_bits(bits))
        if squeeze_params["SQUEEZE_BASIS_AFTER_INIT"]:
            instance = instance.squeeze()
        return instance


    def to_bits(self, *, module=np):
        return det_to_bits(self._encoding, self.bitlen, module=module)


    def __str__(self):
        det_strs, overflow = dets_to_strs(self._encoding, self.bitlen)
        if overflow:
            det_strs.append("...")
        return "\n".join(det_strs)


    def __len__(self):
        return len(self._encoding)
    
    
    def __getitem__(self, s):
        s = make_1d_index(len(self), s)
        return Basis._from_attrs(self._encoding[s], self.bitlen)


    def __eq__(self, other):
        if isinstance(other, Basis):
            return bool(
                self.bitlen == other.bitlen
                and len((self % other) + (other % self)) == 0
            )
        else:
            return NotImplemented


    @property
    def is_squeezed(self) -> bool:
        return array_is_squeezed(self._encoding)
    

    def squeeze(self) -> "Self":
        encoding, _ = squeeze_array(self._encoding)
        return Basis._from_attrs(encoding, self.bitlen)
    

    def __add__(self, other):
        if not isinstance(other, Basis):
            return NotImplemented
        add_on_data = lambda s, o: np.concatenate([s, o])
        encoding, bitlen = op_on_cls(
            add_on_data, self, other, other, self, get_data, get_bitlen
        )
        basis = Basis._from_attrs(encoding, bitlen)
        if squeeze_params["SQUEEZE_DETS_AFTER_ADD"]:
            basis = basis.squeeze()
        return basis


    def __mod__(self, other):
        if not isinstance(other, Basis):
            return NotImplemented
        mod_on_data = lambda s, o: s[array_difference_bmask(s, o)]
        encoding, bitlen = op_on_cls(
            mod_on_data, self, other, self, self, get_data, get_bitlen
        )
        basis = Basis._from_attrs(encoding, bitlen)
        return basis



save_load_registry.register("Basis", Basis, Basis._from_attrs)