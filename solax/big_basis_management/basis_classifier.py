from collections.abc import Callable
import jax.numpy as jnp

from solax.neural_framework import *
from solax.quantum_core.bit_level_primitives import *


class BasisClassifier(SoftmaxClassifier):
    
    def __init__(self, nn_call_on_bits: Callable):
        def call_on_entry(x):
            x = det_to_bits(x, self._bitlen, module=jnp)
            x = nn_call_on_bits(x)
            return x
        super().__init__(call_on_entry)
        
    
    def initialize(self, key, dummy_basis, optimizer):
        self._bitlen = dummy_basis.bitlen
        super().initialize(key, dummy_basis._encoding, optimizer)