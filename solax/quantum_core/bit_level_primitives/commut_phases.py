import numpy as np
import jax.numpy as jnp

from .det_encoding import *


def pos_ndisords(posits):
    """
    Finds number of disorders in a sequence of positions.
    """
    posits_tl = jnp.tril(
        jnp.broadcast_to(posits.reshape(-1, 1), (posits.size, posits.size)),
        k=-1
    )
    return (posits - posits_tl < 0).sum()


def build_ladseq_pmask(ladseq_bit_posits, det_bit_len):
    """
    Returns a "phase mask" for a sequence of ladder operators.
        "Phase mask" is an array of bits 01 with 1 at bits relevant for the phase.
    Input:
        - ladseq_bit_posits: bits where the ladder operator sequence acts;
        - det_bit_len: bit length of determinants the ladder operator sequence acts on.
    """
    arng = jnp.arange(det_bit_len)
    stack = arng > ladseq_bit_posits[:, jnp.newaxis]
    pmask_bits = stack.sum(axis=0) % 2
    return pmask_bits


def ladseq_phase(det_code, det_bit_len, ladseq_bit_posits):
    """
    Returns a phase factor +1 or -1
        from action of a ladder operator sequence on a determinant.
    Input:
        - det_code: an encoded determinant;
        - det_bit_len: the bit length of the determinant;
        - ladseq_bit_posits: bits where the ladder operator sequence acts.
    """
    det_bits = det_to_bits(det_code, det_bit_len, module=jnp)
    pmask_bits = build_ladseq_pmask(ladseq_bit_posits, det_bit_len)
    num_ones = (det_bits * pmask_bits).sum()
    phase_01 = (num_ones + pos_ndisords(ladseq_bit_posits)) % 2    
    phase = 1 - 2 * phase_01
    return phase.astype(jnp.int8)