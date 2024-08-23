import jax
import jax.numpy as jnp

from .det_encoding import *


def map_with_ladder(det_code, bit_posit, dagger):
    """
    Acts with a ladder operator on an encoded determinant "det_code"
            at a bit position "bit_posit".
        dagger = 0 --- annihilation;
        dagger = 1 --- creation.
    The function returns (det_code, valid) where:
        "det_code" is the resulting encoded determinant;
        "valid" = 1 or 0 shows if the determinant survived
            (e.g. for a|0> we get valid==0).
    Note that the phase factor is not evaluated here.
    """
    col, res = locate_bit(bit_posit)
    value = extract_bit(det_code, (col, res))
    original_subdet = det_code[col]
    flipped_subdet = (1 << res) ^ original_subdet
    res_det = det_code.at[col].set(flipped_subdet)
    valid = dagger ^ value
    return res_det, valid


def map_with_ladseq(det_code, bit_posits, daggers):
    """
    Acts with a sequence of ladder operators (as usually, from right to left)
        on an encoded determinant at given bits.
    See also help(map_with_ladder).
    """
    valid = 1
    daggers = jnp.array(daggers, dtype=jnp.uint8)
    
    lower = 1
    upper = len(daggers) + 1
    init_val = (det_code, valid)
    
    def body_fun(i, val):
        det_code, valid = val
        bit_posit = bit_posits[-i]
        dagger = daggers[-i]
        det_code, det_valid = map_with_ladder(det_code, bit_posit, dagger)
        valid = valid & det_valid
        return det_code, valid
    
    det_code, valid = jax.lax.fori_loop(lower, upper, body_fun, init_val)
        
    return det_code, valid