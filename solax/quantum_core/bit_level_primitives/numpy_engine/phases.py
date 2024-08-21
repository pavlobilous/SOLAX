import numpy as np

from ..det_encoding import det_to_bits


def pos_ndisords(posits):
    bks = np.broadcast_to(
        posits[..., np.newaxis],
        (len(posits), posits.shape[-1], posits.shape[-1])
    )
    posits_tl = np.tril(bks, k=-1)
    return (np.swapaxes(bks, -1, -2) - posits_tl < 0).reshape(len(posits), -1).sum(axis=1)


def build_ladseq_pmask(ladseq_bit_posits, det_bit_len):
    arng = np.arange(det_bit_len)
    stack = arng > ladseq_bit_posits[..., np.newaxis]
    pmask_bits = stack.sum(axis=-2) % 2
    return pmask_bits


def ladseq_phase(det_code, det_bit_len, ladseq_bit_posits):
    det_bits = det_to_bits(det_code, det_bit_len, module=np)
    pmask_bits = build_ladseq_pmask(ladseq_bit_posits, det_bit_len)
    num_ones = (det_bits[:, np.newaxis, :] * pmask_bits).sum(axis=-1)
    phase_01 = (num_ones + pos_ndisords(ladseq_bit_posits)) % 2    
    phase = 1 - 2 * phase_01
    return phase.astype(np.int8)