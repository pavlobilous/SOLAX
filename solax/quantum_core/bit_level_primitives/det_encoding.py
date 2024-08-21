import jax.numpy as jnp
import numpy as np


def det_from_bits(det_bits, *, module):
    """
    Encodes dets from bits 01.
    Input:
        - "det_bits" can be a 1D array (for 1 det)
            or a 2D array (for a batch of dets)
        - "module" is either numpy or jax.numpy
    Output:
        Tuple (encoded dets, bitlen),
            where bitlen stores the number of bits for decoding.
    """
    det_code = module.packbits(det_bits, axis=-1)
    bitlen = det_bits.shape[-1]
    return det_code, bitlen


def det_to_bits(det_code, bitlen, *, module):
    """
    Decodes dets to bits 01.
    Input:
        - "det_code" can be a 1D array (for 1 det)
            or a 2D array (for a batch of dets)
        - "bitlen" is the number of bits in dets
            (it had to be stored at the encoding stage)
        - "module" is either numpy or jax.numpy
    Output:
        Decoded dets.
    """
    det_bits = module.unpackbits(det_code, axis=-1, count=bitlen)
    return det_bits


def locate_bit(bit_pos):
    """
    Gets internal coordinates of a bit in an encoded determinant based on its position.
    """
    col = bit_pos // 8
    res = jnp.array(7 - bit_pos % 8, dtype=jnp.uint8)
    bit_coord = (col, res)
    return bit_coord


def extract_bit(det_code, bit_coord):
    """
    Extracts a bit value with given internal coordinates in an encoded determinant.
    "det_code" is a 1D array corresponding to 1 det.
    Coordinates "bit_coord" can be obtained using the function "locate_bit".
    """
    col, res = bit_coord
    return 1 & (det_code[col] >> res)