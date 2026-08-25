"""
Batched core of OperatorTerm.__call__(): splits the determinants of a
Basis (and, in lockstep, the rows of an OperatorTerm) into chunks sized
by det_batch_size/op_batch_size, optionally packed across local JAX
devices via pmap (multiple_devices=True), applies the term's ladder
operator sequence to each chunk, and yields the resulting encodings/
coefficients/det-tracking arrays batch by batch.
"""
import jax
import numpy as np
from typing import Generator, Tuple

from ....utils.multi_batching import *
from ...bit_level_primitives import *


def gen_det_batches(basis_len, det_batch_size, multiple_devices: bool):
    """
    Generates the (start, end, num_pbatches) batch edges (see
    gen_pack_edges()) that chunk a length-"basis_len" sequence of
    determinants into pieces of "det_batch_size" (defaulting to the
    whole "basis_len" if falsy/None -- i.e. unbatched), further packed
    "num_pbatches"-wide for jax.pmap when "multiple_devices" is True
    (one sub-batch per local JAX device), or left unpacked (1 device)
    otherwise.
    """
    n_devices = jax.local_device_count() if multiple_devices else 1
    det_batch_size = det_batch_size or basis_len
    return gen_pack_edges(basis_len, det_batch_size, n_devices)


def gen_op_batches(op_term_len, op_batch_size):
    """
    Generates the (start, end, num_pbatches) batch edges (see
    gen_pack_edges()) that chunk a length-"op_term_len" sequence of
    OperatorTerm rows into pieces of "op_batch_size" (defaulting to the
    whole "op_term_len" if falsy/None -- i.e. unbatched). Operator-term
    rows are never spread across multiple devices, so packing is always
    1-wide here.
    """
    op_batch_size = op_batch_size or op_term_len
    return gen_pack_edges(op_term_len, op_batch_size, 1)


def ladders_func(enc_batch, posits_batch, daggers):
    """
    Applies the fixed ladder-operator sequence "daggers" at each row of
    positions in "posits_batch" to each determinant in "enc_batch"
    (shaped (num_pbatches, ..., code_len) for jax.pmap over the leading
    axis), via map_with_ladseq_pvDet_vOpt (pmap over devices, vmap over
    determinants, vmap over position rows). Converts the resulting
    encodings to a plain NumPy array, and turns the per-result boolean
    validity mask (False where the sequence is annihilated by the Pauli
    exclusion principle) into a flat array of valid linear indices into
    the flattened (device, determinant, position-row) result.
    Returns (enc, valid).
    """
    enc, valid = map_with_ladseq_pvDet_vOpt(enc_batch, posits_batch, daggers)
    enc = np.asarray(enc)
    valid = np.where(valid.reshape(-1))[0]
    return enc, valid


def phases_func(enc_batch, bitlen, posits_batch):
    """
    Computes the ladder-operator sign phase for each determinant in
    "enc_batch" acted on with each position row of "posits_batch", via
    ladseq_phase_pvDet_vOpt (pmap over devices, vmap over determinants,
    vmap over position rows). Returns a plain NumPy array of phases,
    same shape as the corresponding "enc"/"valid" from ladders_func().
    """
    phs = ladseq_phase_pvDet_vOpt(enc_batch, bitlen, posits_batch)
    return np.asarray(phs)


def act_in_batches_generator(
    basis, state_coeffs, op_term,
    det_batch_size, op_batch_size,
    multiple_devices: bool,
    det_tracking: bool
) -> Generator[Tuple[np.ndarray, np.ndarray | None, np.ndarray | None], None, None]:
    """
    Generator implementing the batched core of OperatorTerm.__call__():
    iterates "basis"'s determinants in chunks of "det_batch_size"
    (packed for jax.pmap across local devices if "multiple_devices"),
    and within each determinant chunk, "op_term"'s rows in chunks of
    "op_batch_size", applying the ladder-operator sequence to every
    (determinant, row) pair in the chunk via ladders_func() and
    dropping invalid (Pauli-excluded) results.

    "state_coeffs" (a 1D array aligned with "basis", or None when
    acting on a bare Basis rather than a State) scales each result by
    the product of the originating determinant's state coefficient, the
    row's OperatorTerm coefficient, and the ladder-operator sign phase
    (via phases_func()); pass None to skip this and get coefficients of
    None back (a Basis-only application needs no coefficients).
    "det_tracking", if True, additionally yields for each chunk a 1D
    integer array mapping each result determinant back to the index (in
    "basis") of the determinant it came from.

    Raises ValueError if "basis" is empty. Yields one (encoding, coeffs,
    det_track) tuple per (determinant chunk, operator-row chunk) pair,
    "coeffs"/"det_track" being None where not requested/applicable; the
    concatenation of all yielded chunks is exactly what an unbatched
    call would produce.
    """
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