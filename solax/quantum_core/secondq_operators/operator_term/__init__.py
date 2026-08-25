"""
operator_term: OperatorTerm, a single second-quantized monomial (a
fixed pattern of creation/annihilation ladder operators applied at a
batch of spin-orbital position rows), and its supporting input
validation (cleanup_input) and batched/multi-device application
machinery (act_in_batches).
"""
from .op_term_class import OperatorTerm