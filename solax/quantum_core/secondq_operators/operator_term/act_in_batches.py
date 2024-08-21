import jax
import numpy as np

from ....utils.multi_batching import *
from ...bit_level_primitives import *
from ...mode_ctrl.backend import backend_params


def gen_det_batches(basis_len, det_batch_size, multiple_devices: bool):
    if backend_params["ENGINE"] == "numpy":
        n_devices = 1
    else:
        n_devices = jax.local_device_count() if multiple_devices else 1
    det_batch_size = det_batch_size or basis_len
    return gen_pack_edges(basis_len, det_batch_size, n_devices)


def gen_op_batches(op_term_len, op_batch_size):
    op_batch_size = op_batch_size or op_term_len
    return gen_pack_edges(op_term_len, op_batch_size, 1)


def ladders_func(enc_batch, posits_batch, daggers):
    if backend_params["ENGINE"] == "numpy":
        enc_batch = enc_batch[0]
        f = map_with_ladseq_vDet_vOpt_np
    else:
        f = map_with_ladseq_pvDet_vOpt
    return f(enc_batch, posits_batch, daggers)


def phases_func(enc_batch, bitlen, posits_batch):
    if backend_params["ENGINE"] == "numpy":
        enc_batch = enc_batch[0]
        f = ladseq_phase_vDet_vOpt_np
    else:
        f = ladseq_phase_pvDet_vOpt
    return f(enc_batch, bitlen, posits_batch)    


def act_in_batches(basis, state_coeffs, op_term,
                   det_batch_size, op_batch_size,
                   multiple_devices: bool,
                   det_tracking: bool):
    
    det_code_len = basis._encoding.shape[-1]
    res_enc = np.array([], dtype=np.uint8).reshape(0,  det_code_len)
    res_cfs =  np.array([], dtype=state_coeffs.dtype) \
                if (state_coeffs is not None) else None
    det_track = np.array([], dtype=int) \
                if det_tracking else None
    
    if det_tracking:
        det_arange = np.arange(len(basis))
    
    det_batches = gen_det_batches(len(basis), det_batch_size, multiple_devices)
    for det_batch_start, det_batch_end, num_pbatches in det_batches:
        
        enc_batch = basis._encoding[det_batch_start : det_batch_end]
        enc_batch = enc_batch.reshape(num_pbatches, -1, det_code_len)
        
        if state_coeffs is not None:
            state_cfs_batch = state_coeffs[det_batch_start : det_batch_end]
        if det_tracking:
            det_arange_batch = det_arange[det_batch_start : det_batch_end]
        
        op_batches = gen_op_batches(len(op_term), op_batch_size)
        for op_batch_start, op_batch_end, _ in op_batches:
            
            posits_batch = op_term.posits[op_batch_start : op_batch_end]  
            res_enc_batch, valid = ladders_func(enc_batch, posits_batch, op_term.daggers)
            res_enc_batch = res_enc_batch.reshape(-1, det_code_len)
            valid = valid.reshape(-1).astype(bool)
            res_enc_batch = res_enc_batch[valid]
            res_enc = np.concatenate([res_enc, res_enc_batch])
            
            if det_tracking:
                det_track_batch = np.broadcast_to(
                    det_arange_batch[:, np.newaxis],
                    shape=(len(det_arange_batch), len(posits_batch))
                )
                det_track_batch = det_track_batch.reshape(-1)
                det_track_batch = det_track_batch[valid]
                det_track = np.concatenate([det_track, det_track_batch])

            if state_coeffs is not None:
                op_cfs_batch = op_term.coeffs[op_batch_start:op_batch_end]
                res_phs_batch = phases_func(enc_batch, basis.bitlen, posits_batch)
                res_cfs_batch = op_cfs_batch * state_cfs_batch[:, np.newaxis]
                res_phs_batch = res_phs_batch.reshape(-1)
                res_cfs_batch = res_cfs_batch.reshape(-1)
                res_phs_batch = res_phs_batch[valid]
                res_cfs_batch = res_cfs_batch[valid]
                res_cfs_batch = res_cfs_batch * np.array(res_phs_batch)
                res_cfs = np.concatenate([res_cfs, res_cfs_batch])
            
    return res_enc, res_cfs, det_track