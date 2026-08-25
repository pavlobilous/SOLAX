"""
Generation of batch packs for multidevice computations.
"""


def gen_pack_edges(data_len, batch_size, n_devices):
    """
    Generates (pack_start, pack_end, width) triples that chunk a
    length-"data_len" sequence into slices ready for (optionally)
    jax.pmap-distributed processing across "n_devices" devices, each
    handling up to "batch_size" elements.

    Each full "pack" covers "batch_size" * "n_devices" elements and is
    yielded with width="n_devices" (meant to be reshaped into
    "n_devices" sub-batches of "batch_size" each, one per device). Once
    fewer than a full pack remains, as many complete "batch_size"-sized
    sub-batches as fit are yielded together as one shorter pack (width =
    however many fit, less than "n_devices"); any final remainder
    smaller than "batch_size" is yielded on its own with width=1. Yields
    nothing for the corresponding tail portion if it is empty (e.g. when
    "data_len" is an exact multiple of "batch_size" * "n_devices"). Used
    by gen_det_batches()/gen_op_batches() to drive the batched/
    multi-device evaluation in act_in_batches_generator().
    """
    pack_size = batch_size * n_devices
    n_packs = data_len // pack_size
    pack_start = 0
    for _i in range(n_packs):
        pack_end = pack_start + pack_size
        yield pack_start, pack_end, n_devices
        pack_start = pack_end
        
    tail_pack_len = data_len % pack_size
    if tail_pack_len > 0:
        n_batches_tail = tail_pack_len // batch_size
        if n_batches_tail > 0:
            pack_end = pack_start + n_batches_tail * batch_size
            yield pack_start, pack_end, n_batches_tail
            pack_start = pack_end
        
        tail_batch_len = tail_pack_len % batch_size
        if tail_batch_len > 0:
            yield pack_start, data_len, 1