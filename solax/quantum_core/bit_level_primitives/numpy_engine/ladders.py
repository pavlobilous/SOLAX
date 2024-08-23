import numpy as np


def locate_bit(bit_pos):
    col = bit_pos // 8
    res = np.array(7 - bit_pos % 8, dtype=np.uint8)
    bit_coord = (col, res)
    return bit_coord


def map_with_ladder(det_code_vv, bit_posit_v, dagger):
    col, res = locate_bit(bit_posit_v)
    arng = np.arange(len(bit_posit_v))
    original_subdet = det_code_vv[:, arng, col]
    value = 1 & (original_subdet >> res)
    flipped_subdet = (1 << res) ^ original_subdet
    det_code_vv[:, arng, col] = flipped_subdet
    valid = dagger ^ value
    return valid


def map_with_ladseq(det_code, bit_posits, daggers):
    det_code = np.swapaxes(np.tile(det_code, (len(bit_posits), 1, 1)), 0, 1)
    valid = np.ones(shape=(len(det_code), len(bit_posits)), dtype=np.uint8)
    daggers = np.array(daggers, dtype=np.uint8)
    
    for i in range(1, len(daggers) + 1):
        bit_posit = bit_posits[:, -i]
        dagger = daggers[-i]
        det_valid = map_with_ladder(det_code, bit_posit, dagger)
        valid = valid & det_valid
        
    return det_code, valid