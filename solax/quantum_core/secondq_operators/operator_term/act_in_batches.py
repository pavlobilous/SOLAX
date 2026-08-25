import jax
import numpy as np
from typing import Generator, Tuple

from ....utils.multi_batching import *
from ...bit_level_primitives import *


def gen_det_batches(basis_len, det_batch_size, multiple_devices: bool):
    n_devices = jax.local_device_count() if multiple_devices else 1
    det_batch_size = det_batch_size or basis_len
    return gen_pack_edges(basis_len, det_batch_size, n_devices)


def gen_op_batches(op_term_len, op_batch_size):
    op_batch_size = op_batch_size or op_term_len
    return gen_pack_edges(op_term_len, op_batch_size, 1)


def ladders_func(enc_batch, posits_batch, daggers):
    enc, valid = map_with_ladseq_pvDet_vOpt(enc_batch, posits_batch, daggers)
    enc = np.asarray(enc)
    valid = np.where(valid.reshape(-1))[0]
    return enc, valid


def phases_func(enc_batch, bitlen, posits_batch):
    phs = ladseq_phase_pvDet_vOpt(enc_batch, bitlen, posits_batch)
    return np.asarray(phs)


def act_in_batches_generator(
    basis, state_coeffs, op_term,
    det_batch_size, op_batch_size,
    multiple_devices: bool,
    det_tracking: bool
) -> Generator[Tuple[np.ndarray, np.ndarray | None, np.ndarray | None], None, None]:

    if len(basis) == 0:
        raise ValueError("This generator works with Basis objects of len > 0.")

    det_code_len = basis._encoding.shape[-1]

    det_batches = gen_det_batches(len(basis), det_batch_size, multiple_devices)
    for det_batch_start, det_batch_end, num_pbatches in det_batches:
        enc_batch = basis._encoding[det_batch_start:det_batch_end]
        enc_batch = enc_batch.reshape(num_pbatches, -1, det_code_len)
        
        if state_coeffs is not None:
            state_cfs_batch = state_coeffs[det_batch_start:det_batch_end]
        if det_tracking:
            det_arange_batch = np.arange(det_batch_start, det_batch_end)

        op_batches = gen_op_batches(len(op_term), op_batch_size)
        for op_batch_start, op_batch_end, _ in op_batches:
            posits_batch = op_term.posits[op_batch_start:op_batch_end]
            res_enc_batch, valid = ladders_func(enc_batch, posits_batch, op_term.daggers)
            res_enc_batch = res_enc_batch.reshape(-1, det_code_len)[valid]

            det_track_batch = None
            if det_tracking:
                det_track_batch = np.broadcast_to(
                    det_arange_batch[:, np.newaxis],
                    shape=(len(det_arange_batch), len(posits_batch))
                ).reshape(-1)[valid]

            res_cfs_batch = None
            if state_coeffs is not None:
                op_cfs_batch = op_term.coeffs[op_batch_start:op_batch_end]
                res_phs_batch = phases_func(enc_batch, basis.bitlen, posits_batch)
                res_cfs_batch = (op_cfs_batch * state_cfs_batch[:, np.newaxis]).reshape(-1)
                res_cfs_batch = res_cfs_batch[valid] * np.array(res_phs_batch.reshape(-1)[valid])

            yield res_enc_batch, res_cfs_batch, det_track_batch