"""
Generation of batch packs for multidevice computations.
"""


def gen_pack_edges(data_len, batch_size, n_devices):
    pack_size = batch_size * n_devices
    n_packs = data_len // pack_size
    pack_start = 0
    for i in range(n_packs):
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