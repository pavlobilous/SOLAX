import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

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


def build_ladseq_pmask(ladseq_bit_posits, det_code_len):
    """
    Returns a "phase mask" for a sequence of ladder operators.
        "Phase mask" is an array of bits 01 with 1 at bits relevant for the phase.
    Input:
        - ladseq_bit_posits: bits where the ladder operator sequence acts;
        - det_code_len: length of the encoded determinant.
    """    
    @partial(jax.vmap, in_axes=(0, None))
    def create_swapper(indx, det_code_len):
        col, res = locate_bit(indx)
        swp = jnp.zeros(2 * det_code_len, dtype=jnp.uint8)
        swp = swp.at[:det_code_len].set(~0)
        swp = jnp.roll(swp, col + 1)[:det_code_len]
        swp = swp.at[col].set(2**res - 1)
        return swp
        
    swp = create_swapper(ladseq_bit_posits, det_code_len)
    pmask = jax.lax.reduce(swp, jnp.array(0, dtype=jnp.uint8),
                            jnp.bitwise_xor, (0,))
    return pmask


def ladseq_phase(det_code, det_bit_len, ladseq_bit_posits):
    """
    Returns a phase factor +1 or -1
        from action of a ladder operator sequence on a determinant.
    Input:
        - det_code: an encoded determinant;
        - det_bit_len: the bit length of the determinant;
        - ladseq_bit_posits: bits where the ladder operator sequence acts.
    """
    pmask = build_ladseq_pmask(ladseq_bit_posits, len(det_code))
    num_ones = jnp.bitwise_count(det_code & pmask).sum()
    phase_01 = (num_ones + pos_ndisords(ladseq_bit_posits)) % 2    
    phase = 1 - 2 * phase_01
    return phase.astype(jnp.int8)