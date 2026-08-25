"""
Basis: an ordered collection of Slater determinants, in occupation-number
(01 bitstring) representation, stored packed as bytes for compactness
and fast set-like operations. See SciPost Phys. Codebases 51 Sec. 2.2.
"""
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
    """
    Builds the ("_encoding", "_bitlen") attribute pair of a Basis
    from a 1D (single det) or 2D (batch of dets) array of bits 01.
    """
    bits = np.atleast_2d(bits)
    encoding, bitlen = det_from_bits(bits, module=np)
    if len(encoding[0]) == 0:
        encoding = encoding.reshape(0, 0)
    return encoding, bitlen


def det_strings_to_bits(det_strings):
    """
    Converts an iterable of equal-length "0"/"1" determinant strings
    (e.g. "1100") into a 2D array of bits. Empty strings are dropped.
    Raises ValueError if the strings have unequal length or contain
    characters other than "0"/"1".
    """
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
    """
    An ordered collection of Slater determinants of a fixed number of
    spin-orbitals ("bitlen"), each determinant a 01 occupation-number
    bitstring (Pauli exclusion: at most one particle per spin-orbital).

    Internally, determinants are stored packed as bytes in "_encoding"
    (one packed row per determinant); use to_bits()/from_bits() to
    convert to/from a plain 01 bit array. Basis behaves like a
    read-only Sequence: len(), indexing/slicing (including boolean
    masks and tuples of positions), and iteration all work as expected.

    By default, a Basis auto-deduplicates ("squeezes") its determinants
    on construction and after "+"; wrap construction in the
    manual_squeezing() context manager to keep duplicates instead. Two
    Basis instances compare equal ("==") if they hold the same set of
    determinants, independent of order or duplicate count; see
    is_squeezed()/squeeze() to inspect/enforce deduplication directly.
    """
    _encoding: NDArray[2, np.uint8]
    _bitlen: int

    __array_ufunc__ = None


    @property
    def bitlen(self):
        """Number of spin-orbitals (bits) per determinant. Read-only."""
        return self._bitlen


    def __init__(self, det_strings: Iterable[str]):
        """
        Builds a Basis from an iterable of equal-length "0"/"1" strings,
        e.g. Basis(["1100", "1010"]). Raises ValueError if the strings
        have unequal length or contain characters other than "0"/"1".
        """
        bits = det_strings_to_bits(det_strings)
        self._encoding, self._bitlen = attrs_from_bits(bits)
        if squeeze_params["SQUEEZE_BASIS_AFTER_INIT"]:
            self._encoding, _ = squeeze_array(self._encoding)


    @classmethod
    def _from_attrs(cls, _encoding, _bitlen):
        """Low-level constructor from already-packed attributes, bypassing
        string parsing/validation. Used internally and to reconstruct a
        Basis when loading it back with solax.load()."""
        instance = cls([])
        instance._encoding = _encoding
        instance._bitlen = _bitlen
        return instance


    @classmethod
    def from_bits(cls, bits: NDArray[1, Integral] | NDArray[2, Integral]):
        """
        Builds a Basis from a 1D (single det) or 2D (batch of dets)
        array of bits 01, as an alternative to the string-based
        constructor.
        """
        instance = cls._from_attrs(*attrs_from_bits(bits))
        if squeeze_params["SQUEEZE_BASIS_AFTER_INIT"]:
            instance = instance.squeeze()
        return instance


    def to_bits(self, *, module=np):
        """Decodes the Basis back to a 2D array of bits 01, one row per
        determinant. "module" is numpy or jax.numpy, selecting the
        array type of the result."""
        return det_to_bits(self._encoding, self.bitlen, module=module)


    def __str__(self):
        """Renders the determinants as one "0"/"1" bitstring per line
        (up to the current dets_printing_limit()), appending an "..."
        line if determinants were omitted."""
        det_strs, overflow = dets_to_strs(self._encoding, self.bitlen)
        if overflow:
            det_strs.append("...")
        return "\n".join(det_strs)


    def __len__(self):
        """Number of determinants in the Basis."""
        return len(self._encoding)


    def __getitem__(self, s):
        """Indexes/slices the Basis like a sequence. Accepts an int, a
        slice, a tuple of ints (fancy indexing), or a boolean mask of
        length len(self). Always returns a new Basis (even for a single
        int index)."""
        s = make_1d_index(len(self), s)
        return Basis._from_attrs(self._encoding[s], self.bitlen)


    def __eq__(self, other):
        """True if "other" is a Basis holding the same set of
        determinants (same "bitlen", same determinants up to order and
        duplicate count); NotImplemented for non-Basis "other"."""
        if isinstance(other, Basis):
            return bool(
                self.bitlen == other.bitlen
                and len((self % other) + (other % self)) == 0
            )
        else:
            return NotImplemented


    @property
    def is_squeezed(self) -> bool:
        """True if the Basis holds no duplicate determinants."""
        return array_is_squeezed(self._encoding)


    def squeeze(self) -> "Self":
        """Returns a new Basis with duplicate determinants removed,
        keeping the first occurrence of each."""
        encoding, _ = squeeze_array(self._encoding)
        return Basis._from_attrs(encoding, self.bitlen)


    def __add__(self, other):
        """Concatenates the determinants of "self" and "other" (order
        preserved), then deduplicates unless inside a manual_squeezing()
        context. Raises ValueError if "bitlen" differs between non-empty
        operands; an empty-basis operand is treated as neutral regardless
        of its own "bitlen"."""
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
        """Set difference: the determinants of "self" that are NOT
        present in "other". Raises ValueError if "bitlen" differs
        between non-empty operands."""
        if not isinstance(other, Basis):
            return NotImplemented
        mod_on_data = lambda s, o: s[array_difference_bmask(s, o)]
        encoding, bitlen = op_on_cls(
            mod_on_data, self, other, self, self, get_data, get_bitlen
        )
        basis = Basis._from_attrs(encoding, bitlen)
        return basis



save_load_registry.register("Basis", Basis, Basis._from_attrs)