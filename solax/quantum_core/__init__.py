"""
quantum_core: the core quantum-mechanical data model of SOLAX --
Slater-determinant bases and states (det_based_classes), second-
quantized operators and their sparse matrix representation
(secondq_operators), and shared global switches for auto-deduplication
and printing (mode_ctrl), all built on the low-level bit manipulations
in bit_level_primitives.
"""
from .det_based_classes import *
from .secondq_operators import *
from .mode_ctrl import *
