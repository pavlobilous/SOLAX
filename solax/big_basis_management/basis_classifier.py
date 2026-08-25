"""
BasisClassifier: a SoftmaxClassifier specialized to classify determinants
of a big basis (see solax.big_basis_management.manager_class.BigBasisManager)
directly from a Basis, converting determinants to their bit-encoded
occupation representation internally. See SciPost Phys. Codebases 51
Sec. 3.
"""
from collections.abc import Callable
import jax.numpy as jnp

from solax.neural_framework import *
from solax.quantum_core.bit_level_primitives import *


class BasisClassifier(SoftmaxClassifier):
    """
    A binary SoftmaxClassifier (important vs. unimportant) over
    determinants, initialized directly from a Basis: the network
    architecture "nn_call_on_bits" passed to __init__ is written to act
    on plain 01 occupation-number bit arrays, and BasisClassifier wraps
    it so it can instead be called directly on packed determinant
    encodings, decoding them to bits internally via det_to_bits.

    Per SciPost Phys. Codebases 51 Sec. 3, a BasisClassifier is meant to
    be reused across successive big bases/BigBasisManager instances,
    "transferring in this way the NN experience from case to case" --
    unlike BigBasisManager, it is not bound to any one big basis.

    BasisClassifier inherits save_state()/load_state() (Orbax-based)
    from NeuralModel; this is a separate persistence mechanism from
    solax.save()/solax.load(), not interchangeable with it. To reload a
    saved BasisClassifier, reconstruct a fresh instance with the same
    architecture function, then call initialize() with a prototype
    Basis of the right bitlen and an optimizer (typically passing
    RandomKeys.fake_key() as the throwaway init key, since load_state()
    immediately overwrites whatever weights initialize() produced),
    and finally call load_state().
    """

    def __init__(self, nn_call_on_bits: Callable):
        """
        Wraps a bit-level network architecture "nn_call_on_bits"
        (call_on_entry(bits) -> logits, acting on a single, non-batched
        01 bit array) so the resulting BasisClassifier can instead be
        called on packed determinant encodings; encodings are decoded
        to bits via det_to_bits using "_bitlen", which is set by
        initialize().
        """
        def call_on_entry(x):
            x = det_to_bits(x, self._bitlen, module=jnp)
            x = nn_call_on_bits(x)
            return x
        super().__init__(call_on_entry)


    def initialize(self, key, dummy_basis, optimizer):
        """
        Initializes the network parameters. "dummy_basis" is a Basis
        used only to determine the number of spin-orbitals per
        determinant ("bitlen", stored for later bit-decoding) and to
        supply dummy input features (its packed encoding) for shape
        inference; its actual determinants are otherwise irrelevant, so
        any Basis of the intended bitlen will do. "key" is a JAX PRNG
        key and "optimizer" an optax optimizer, as in
        NeuralModel.initialize.
        """
        self._bitlen = dummy_basis.bitlen
        super().initialize(key, dummy_basis._encoding, optimizer)