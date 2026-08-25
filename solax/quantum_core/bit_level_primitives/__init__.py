"""
bit_level_primitives: low-level, JAX/NumPy-friendly operations on
determinants encoded as packed 01 occupation-number bitstrings --
packing/unpacking bits (det_encoding), applying a ladder-operator
sequence and computing its sign phase (ladder_mappings, commut_phases),
and their vmap/pmap-vectorized batch counterparts (vectorizations).
"""
from .det_encoding import det_from_bits, det_to_bits, locate_bit, extract_bit
from .ladder_mappings import map_with_ladder, map_with_ladseq
from .commut_phases import ladseq_phase

from .vectorizations import map_with_ladseq_pvDet_vOpt, ladseq_phase_pvDet_vOpt
from .vectorizations import map_with_ladseq_vDet_vOpt, ladseq_phase_vDet_vOpt